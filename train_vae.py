import os
import torch
import wandb
import torch.nn.functional as F

from PIL import Image

from accelerate import Accelerator
from tqdm.auto import tqdm

from diffusers import AutoencoderKL

from zizi_pipeline import (
    TrainingConfig,
    get_ddpm,
    get_adamw,
    get_lr_scheduler,
    get_dataloader,
)
from zizi_vae_pipeline import ZiziVaePipeline, get_vae_unet

config = TrainingConfig(
    "data/pink-me/", "output/pink-me-vae-128/", image_size=256, train_batch_size=32
)


def get_pretrained_vae():
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1", subfolder="vae"
    )
    vae.config.sample_size = 512
    return vae


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config: TrainingConfig, epoch, pipeline, condition: torch.FloatTensor):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        condition,
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        num_inference_steps=50,
    ).images

    wandb.log({"examples": [wandb.Image(img) for img in images]})

    # Make a grid out of the images
    image_grid = make_grid(images, rows=2, cols=2)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, vae):
    # Initialize accelerator
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(project_name="train_vae", config=config)

    train_dataloader = get_dataloader(config)

    vae = vae.to(accelerator.device)
    model = get_vae_unet(config)
    noise_scheduler = get_ddpm()
    optimizer = get_adamw(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer, train_dataloader)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler, vae = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, vae
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]

            with torch.no_grad():
                latent = vae.encode(clean_images)

            latent_vector = latent.latent_dist.sample() * vae.config.scaling_factor

            # Sample noise to add to the latents
            noise = torch.randn(latent_vector.shape).to(accelerator.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each latent
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=accelerator.device,
            ).long()

            # Add noise to the clean latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latent_vector, noise, timesteps)

            poses = batch["poses"].reshape((bs, 1, 75))

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_latents, timesteps, poses, return_dict=False)[
                    0
                ]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = ZiziVaePipeline(
                vae=vae,
                unet_cond=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
            ).to(accelerator.device)

            if (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(
                    config,
                    epoch,
                    pipeline,
                    train_dataloader.dataset[0]["poses"]
                    .unsqueeze(0)
                    .to(accelerator.device),
                )

            if (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(f"{config.output_dir}/checkpoint-{str(epoch)}")


if __name__ == "__main__":
    train_loop(config, get_pretrained_vae())

import os
import argparse

import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from accelerate import Accelerator
from tqdm.auto import tqdm

from zizi_pipeline import (
    ZiziPipeline,
    TrainingConfig,
    get_unet,
    get_unet_crossattn,
    get_ddpm,
    get_adamw,
    get_lr_scheduler,
    get_dataloader,
    get_subset_dataloader,
)

from utils import make_grid

config = TrainingConfig(
    "data/pink-cape-me/",
    "output/pink-me-cross-attn-128/",
    image_size=128,
    train_batch_size=8,
    save_model_epochs=10,
    lr_warmup_steps=0,
    num_epochs=100,
    save_image_epochs=1,
)


def load_from_checkpoint(checkpoint_dir):
    print(f"Resuming training from checkpoint {checkpoint_dir}")
    pipeline = ZiziPipeline.from_pretrained(checkpoint_dir)
    return pipeline.unet_cond, pipeline.scheduler


def evaluate(
    config: TrainingConfig, epoch, pipeline: ZiziPipeline, condition: torch.FloatTensor
):
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


def train_loop(config, checkpoint_dir=None, epoch_offset=0):
    # Initialize accelerator
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    train_dataloader = get_dataloader(config)

    if checkpoint_dir is not None:
        model, noise_scheduler = load_from_checkpoint(checkpoint_dir)
    else:
        model = get_unet_crossattn(config)
        noise_scheduler = get_ddpm()
    optimizer = get_adamw(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer, train_dataloader)
    # from torch.optim.lr_scheduler import ConstantLR
    # lr_scheduler = ConstantLR(optimizer, factor=1) # TEMP HACK

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("zizi-pink-full-pose")

    # Now you train the model
    for raw_epoch in range(config.num_epochs):
        epoch = raw_epoch + epoch_offset
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            poses = batch["poses"].reshape((bs, 1, 411))

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, poses, return_dict=False)[0]
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
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = ZiziPipeline(
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

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train zizi with cross attn")
    parser.add_argument(
        "--resume_from", type=str, help="Directory to resume a checkpoint from"
    )
    parser.add_argument("--offset", help="Previous epoch count", type=int, default=0)
    args = parser.parse_args()

    train_loop(config, args.resume_from, args.offset)

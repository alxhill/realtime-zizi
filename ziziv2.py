import math
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image
from diffusers import DDPMScheduler
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils.torch_utils import randn_tensor
from diffusers import UNet2DConditionModel

from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 512  # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 4
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 5
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    input_dir = "data/pink-me/"
    output_dir = "output/zizi-pink-512"
    overwrite_output_dir = True
    seed = 56738

config = TrainingConfig()

class ConditionalZiziDataset(Dataset):
    def __init__(self, input_dir, image_size):
        self.img_dir = input_dir + "train_img"
        self.pose_dir = input_dir + "train_openpose"
        self.img_files = os.listdir(self.img_dir)
        self.img_files.sort()
        self.image_size = image_size
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        )

    def __len__(self):
        return len(self.img_files)

    def _get_img(self, idx):
        img_dir = os.path.join(self.img_dir, self.img_files[idx])
        with Image.open(img_dir) as img:
            return self.preprocess(img.convert("RGB"))

    def _get_json(self, idx):
        input_split = os.path.splitext(self.img_files[idx])
        pose_json_name = input_split[0] + "_keypoints.json"
        json_dir = os.path.join(self.pose_dir, pose_json_name)
        with open(json_dir, 'r') as f:
            data = json.load(f)
        return torch.tensor(data['people'][0]['pose_keypoints_2d'])
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*idx.indices(len(self)))]

        input_img = self._get_img(idx)
        json_data = self._get_json(idx)

        return {
            "images": input_img,
            "poses": json_data
        }

class ZiziPipeline(DiffusionPipeline):
    def __init__(self, unet_conditional, scheduler):
        super().__init__()
        self.register_modules(unet_conditional=unet_conditional, scheduler=scheduler)
    
    @torch.no_grad()
    def __call__(
        self,
        condition: torch.FloatTensor,
        batch_size: int = 1,
        generator: torch.Generator = None,
        num_inference_steps: int = 50
    ) -> ImagePipelineOutput:
        image_shape = (
                batch_size,
                self.unet_conditional.config.in_channels,
                self.unet_conditional.config.sample_size,
                self.unet_conditional.config.sample_size,
            )
        image = randn_tensor(image_shape, generator=generator, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        condition = condition.view((1, 1, 75)).repeat((batch_size, 1, 1))

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet_conditional(image, t, condition).sample
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)

        return ImagePipelineOutput(images=image)

def get_dataset():
    return ConditionalZiziDataset(config.input_dir, config.image_size)

def get_dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

def get_model(config: TrainingConfig):
    return UNet2DConditionModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        encoder_hid_dim=75,
        cross_attention_dim=512,
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

def get_scheduler():
    return DDPMScheduler(num_train_timesteps=1000)

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def evaluate(config, epoch, pipeline, condition):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        condition,
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        num_inference_steps=50
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=2, cols=2)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config):
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    dataset = get_dataset()
    train_dataloader = get_dataloader(dataset)

    model = get_model()
    noise_scheduler = get_scheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            poses = batch["poses"].reshape((bs, 1, 75))
            
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
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = ZiziPipeline(unet_conditional=accelerator.unwrap_model(model), scheduler=noise_scheduler).to(accelerator.device)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, dataset[0]["poses"].unsqueeze(0).to(accelerator.device))

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(f"{config.output_dir}-{str(epoch)}")

if __name__ == "__main__":
    train_loop(config)

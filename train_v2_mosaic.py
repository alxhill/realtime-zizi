import os
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from composer.models import ComposerModel
from composer import Trainer

from PIL import Image

from accelerate import Accelerator
from tqdm.auto import tqdm

from zizi_pipeline import (
    ZiziPipeline,
    TrainingConfig,
    get_unet,
    get_ddpm,
    get_adamw,
    get_lr_scheduler,
    get_dataloader,
    get_subset_dataloader,
)

from utils import make_grid

config = TrainingConfig("data/pink-me/", "output/pink-full-pose-256/", image_size=256, train_batch_size=4)

class ComposerUnet(ComposerModel):
    def __init__(self, unet, noise_scheduler):
        super().__init__()
        self.unet = unet
        self.noise_scheduler = noise_scheduler

    def forward(self, batch):
        clean_images = batch["images"]
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()

        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        poses = batch["poses"].reshape((bs, 1, 411))

        noise_pred = self.unet(noisy_images, timesteps, poses, return_dict=False)[0]
        
        return (noise_pred, noise)

    def loss(self, outputs, batch):
        noise_pred, noise = outputs
        return F.mse_loss(noise_pred, noise)

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


def get_trainer(config: TrainingConfig):
    model = get_unet(config)
    optim = get_adamw(config, model)
    dataloader = get_subset_dataloader(config, 64)
    noise_scheduler = get_ddpm()
    return Trainer(
        model=ComposerUnet(model, noise_scheduler),
        train_dataloader=dataloader,
        # eval_dataloader=eval_dataloader,
        optimizers=optim,
        max_duration=config.num_epochs,
        device='mps',
        save_folder=config.output_dir,
        save_interval='1ep',
    )

if __name__ == "__main__":
    trainer = get_trainer(config)
    trainer.fit()

import math
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms
from PIL import Image
from diffusers import DDPMScheduler
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils.torch_utils import randn_tensor
from diffusers import UNet2DConditionModel

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    input_dir: str
    output_dir: str
    image_size: int = 512  # the generated image resolution
    train_batch_size: int = 4
    eval_batch_size: int = 4
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 1
    save_model_epochs: int = 5
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    overwrite_output_dir: bool = True
    seed: int = 56738


class ConditionalZiziDataset(Dataset):
    pose_keys = ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d")
    
    def __init__(self, input_dir, image_size, img_dir_name="train_img", pose_dir_name="train_openpose"):
        self.img_dir = input_dir + img_dir_name
        self.pose_dir = input_dir + pose_dir_name
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

    def _get_img_path(self, idx):
        return os.path.join(self.img_dir, self.img_files[idx])

    def _get_img(self, idx):
        img_dir = self._get_img_path(idx)
        with Image.open(img_dir) as img:
            return self.preprocess(img.convert("RGB"))

    def _get_json(self, idx):
        input_split = os.path.splitext(self.img_files[idx])
        pose_json_name = input_split[0] + "_keypoints.json"
        json_dir = os.path.join(self.pose_dir, pose_json_name)
        with open(json_dir, 'r') as f:
            pose_data = json.load(f)
        pose_points = [points for key in self.pose_keys for points in pose_data['people'][-1][key]]
        return torch.tensor(pose_points)
    
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
    def __init__(self, unet_cond: UNet2DConditionModel, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(unet_cond=unet_cond, scheduler=scheduler)
    
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
                self.unet_cond.config.in_channels,
                self.unet_cond.config.sample_size,
                self.unet_cond.config.sample_size,
            )
        image = randn_tensor(image_shape, generator=generator, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        condition = condition.view((1, 1, 75)).repeat((batch_size, 1, 1))

        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet_cond(image, t, condition).sample
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)

        return ImagePipelineOutput(images=image)

def get_unet(config: TrainingConfig):
    return UNet2DConditionModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        encoder_hid_dim=411,
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


def get_unet_crossattn(config: TrainingConfig):
    """
    UNet with cross attn blocks instead of just self attn blocks.
    Does not seem to work with the config below (loss stays at 1).
    """
    return UNet2DConditionModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        encoder_hid_dim=411,
        cross_attention_dim=512,
        block_out_channels=(128, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

def get_ddpm():
    return DDPMScheduler(num_train_timesteps=1000)

def get_adamw(config: TrainingConfig, model: UNet2DConditionModel):
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

def get_lr_scheduler(config: TrainingConfig, optimizer, dataloader):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )

def get_dataloader(config: TrainingConfig):
    dataset = ConditionalZiziDataset(config.input_dir, config.image_size)
    return DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

def get_subset_dataloader(config: TrainingConfig, count=8):
    dataset = ConditionalZiziDataset(config.input_dir, config.image_size)
    return DataLoader(Subset(dataset, range(0,count)), batch_size=config.train_batch_size, shuffle=True)
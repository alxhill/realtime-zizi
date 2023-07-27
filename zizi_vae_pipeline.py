from diffusers import *
from zizi_pipeline import *

def get_vae_unet(config: TrainingConfig):
    return UNet2DConditionModel(
        sample_size=64,  # resolution of the inner layers
        in_channels=4,  # the number of input channels, 3 for RGB images
        out_channels=4,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        encoder_hid_dim=75,
        cross_attention_dim=512,
        block_out_channels=(128, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "CrossAttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

class ZiziVaePipeline(DiffusionPipeline):
    def __init__(self, unet_cond: UNet2DConditionModel, scheduler: DDPMScheduler, vae: AutoencoderKL):
        super().__init__()
        self.register_modules(unet_cond=unet_cond, scheduler=scheduler, vae=vae)
    
    @torch.no_grad()
    def __call__(
        self,
        condition: torch.FloatTensor,
        batch_size: int = 1,
        generator: torch.Generator = None,
        num_inference_steps: int = 50
    ) -> ImagePipelineOutput:
        latent_shape = (
                batch_size,
                self.unet_cond.config.in_channels,
                self.unet_cond.config.sample_size,
                self.unet_cond.config.sample_size,
            )
        
        latent = randn_tensor(latent_shape, generator=generator, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        condition = condition.view((1, 1, 75)).repeat((batch_size, 1, 1))

        for t in self.progress_bar(self.scheduler.timesteps):
            latent_noise = self.unet_cond(latent, t, condition).sample
            latent = self.scheduler.step(latent_noise, t, latent, generator=generator).prev_sample

        latent = (1 / self.vae.config.scaling_factor) * latent
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)

        return ImagePipelineOutput(images=image)
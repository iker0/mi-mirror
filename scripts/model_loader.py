"""Load Stable Diffusion 1.5 for T4/free Colab GPUs."""

import torch
from diffusers import StableDiffusionPipeline
from scripts.config import MODEL_ID


def load_pipeline(device: str = "cuda") -> StableDiffusionPipeline:
    """Load SD1.5 pipeline. Fits comfortably on a T4 in float16.

    Args:
        device: Target device ("cuda" or "cpu").

    Returns:
        Configured StableDiffusionPipeline ready for inference.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    return pipe


def get_unet(pipe: StableDiffusionPipeline):
    """Return the UNet2DConditionModel from the pipeline."""
    return pipe.unet

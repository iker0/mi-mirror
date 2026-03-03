"""Generation wrapper with attention extraction support for SD1.5."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from scripts.attention_extraction import (
    AttnBlockInfo,
    AttentionData,
    install_cra_processors,
    install_storage_processors,
    restore_default_processors,
    set_all_timesteps,
)
from scripts.config import NUM_INFERENCE_STEPS, RESOLUTION, SEEDS
from scripts.roi import ROI


def _make_timestep_callback(processors: Dict):
    """Create a callback that updates the timestep counter on all processors."""
    step_counter = [0]

    def callback(pipe, step_index, timestep, callback_kwargs):
        set_all_timesteps(processors, step_counter[0])
        step_counter[0] += 1
        return callback_kwargs

    return callback


def generate_with_cra(
    pipe: StableDiffusionPipeline,
    prompt: str,
    seed: int,
    obj_roi: ROI,
    ref_roi: ROI,
    attention_data: Optional[AttentionData] = None,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    resolution: int = RESOLUTION,
) -> Tuple[Image.Image, AttentionData, List[AttnBlockInfo]]:
    """Generate an image and extract CRA scalars from all self-attention blocks.

    Returns:
        (generated_image, attention_data, block_infos)
    """
    if attention_data is None:
        attention_data = AttentionData()

    unet = pipe.unet
    block_infos, custom_processors = install_cra_processors(
        unet,
        obj_roi=obj_roi,
        ref_roi=ref_roi,
        attention_data=attention_data,
    )

    callback = _make_timestep_callback(custom_processors)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=num_inference_steps,
        generator=generator,
        callback_on_step_end=callback,
    ).images[0]

    restore_default_processors(unet)
    return image, attention_data, block_infos


def generate_with_full_attention(
    pipe: StableDiffusionPipeline,
    prompt: str,
    seed: int,
    obj_roi: ROI,
    ref_roi: ROI,
    candidate_linear_indices: List[int],
    attention_data: Optional[AttentionData] = None,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    resolution: int = RESOLUTION,
) -> Tuple[Image.Image, AttentionData, List[AttnBlockInfo]]:
    """Generate with full attention maps stored for candidate blocks.

    Args:
        candidate_linear_indices: Linear indices of blocks to store full maps for.

    Returns:
        (generated_image, attention_data, block_infos)
    """
    if attention_data is None:
        attention_data = AttentionData()

    unet = pipe.unet
    block_infos, custom_processors = install_storage_processors(
        unet,
        candidate_linear_indices=candidate_linear_indices,
        obj_roi=obj_roi,
        ref_roi=ref_roi,
        attention_data=attention_data,
    )

    callback = _make_timestep_callback(custom_processors)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=num_inference_steps,
        generator=generator,
        callback_on_step_end=callback,
    ).images[0]

    restore_default_processors(unet)
    return image, attention_data, block_infos


def generate_baseline(
    pipe: StableDiffusionPipeline,
    prompt: str,
    seed: int,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    resolution: int = RESOLUTION,
) -> Image.Image:
    """Generate an image without attention extraction."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    return image


def save_image(image: Image.Image, path: Path, prompt: str = "", seed: int = 0):
    """Save image and create parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

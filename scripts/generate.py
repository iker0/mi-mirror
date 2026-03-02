"""Generation wrapper with attention extraction support."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from diffusers import FluxPipeline

from scripts.attention_extraction import (
    AttentionData,
    FluxAttnProcessorCRAOnly,
    FluxAttnProcessorWithStorage,
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
    pipe: FluxPipeline,
    prompt: str,
    seed: int,
    obj_roi: ROI,
    ref_roi: ROI,
    attention_data: Optional[AttentionData] = None,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    resolution: int = RESOLUTION,
) -> Tuple[Image.Image, AttentionData]:
    """Generate an image and extract CRA scalars from all blocks.

    This is the screening pass — computes CRA per head but doesn't store
    full attention maps.

    Returns:
        (generated_image, attention_data)
    """
    if attention_data is None:
        attention_data = AttentionData()

    transformer = pipe.transformer
    processors = install_cra_processors(
        transformer,
        obj_indices=obj_roi.token_indices,
        ref_indices=ref_roi.token_indices,
        attention_data=attention_data,
    )

    callback = _make_timestep_callback(processors)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=num_inference_steps,
        generator=generator,
        callback_on_step_end=callback,
    ).images[0]

    restore_default_processors(transformer)
    return image, attention_data


def generate_with_full_attention(
    pipe: FluxPipeline,
    prompt: str,
    seed: int,
    obj_roi: ROI,
    ref_roi: ROI,
    candidate_blocks: List[int],
    attention_data: Optional[AttentionData] = None,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    resolution: int = RESOLUTION,
) -> Tuple[Image.Image, AttentionData]:
    """Generate an image and extract full attention maps from candidate blocks.

    This is the detailed pass — stores full attention maps for candidate blocks
    and CRA scalars for all blocks.

    Args:
        candidate_blocks: Block indices to extract full attention maps from.

    Returns:
        (generated_image, attention_data)
    """
    if attention_data is None:
        attention_data = AttentionData()

    transformer = pipe.transformer
    storage_processors = install_storage_processors(
        transformer,
        candidate_blocks=candidate_blocks,
        obj_indices=obj_roi.token_indices,
        ref_indices=ref_roi.token_indices,
        attention_data=attention_data,
    )

    # Collect all processors (both CRA-only and storage) for timestep management
    all_processors = {
        k: v for k, v in transformer.attn_processors.items()
        if hasattr(v, "set_timestep")
    }
    callback = _make_timestep_callback(all_processors)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=num_inference_steps,
        generator=generator,
        callback_on_step_end=callback,
    ).images[0]

    restore_default_processors(transformer)
    return image, attention_data


def generate_baseline(
    pipe: FluxPipeline,
    prompt: str,
    seed: int,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    resolution: int = RESOLUTION,
) -> Image.Image:
    """Generate an image without attention extraction (for equivalence testing)."""
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


def generate_dataset(
    pipe: FluxPipeline,
    prompts: List[str],
    seeds: List[int],
    output_dir: Path,
    obj_roi: ROI,
    ref_roi: ROI,
) -> List[Tuple[Path, AttentionData]]:
    """Generate a full dataset with CRA extraction.

    Returns list of (image_path, attention_data) for each prompt/seed pair.
    """
    output_dir = Path(output_dir)
    results = []

    for i, prompt in enumerate(prompts):
        for seed in seeds:
            attn_data = AttentionData()
            image, attn_data = generate_with_cra(
                pipe, prompt, seed, obj_roi, ref_roi, attn_data
            )
            filename = f"prompt{i:02d}_seed{seed}.png"
            path = output_dir / filename
            save_image(image, path, prompt, seed)
            results.append((path, attn_data))
            print(f"  Generated: {filename}")

    return results

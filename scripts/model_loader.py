"""Load Flux-schnell with NF4 quantization for T4/A100 GPUs."""

import torch
from diffusers import FluxPipeline
from scripts.config import MODEL_ID, NUM_INFERENCE_STEPS


def load_flux_pipeline(quantize_nf4: bool = True, cpu_offload: bool = True) -> FluxPipeline:
    """Load Flux-schnell pipeline with memory optimizations.

    Args:
        quantize_nf4: Use NF4 quantization via bitsandbytes (saves ~6GB VRAM).
        cpu_offload: Enable model CPU offloading for low-VRAM GPUs (T4).

    Returns:
        Configured FluxPipeline ready for inference.
    """
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
    }

    if quantize_nf4:
        from diffusers import BitsAndBytesConfig, PipelineQuantizationConfig
        nf4_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer"],
        )
        load_kwargs["quantization_config"] = nf4_config

    pipe = FluxPipeline.from_pretrained(MODEL_ID, **load_kwargs)

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    return pipe


def get_transformer(pipe: FluxPipeline):
    """Return the FluxTransformer2DModel from the pipeline."""
    return pipe.transformer

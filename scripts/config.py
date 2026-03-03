"""Central configuration for Mirror Reflection Mechanistic Interpretability."""

from pathlib import Path

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
RESOLUTION = 512
NUM_INFERENCE_STEPS = 20

# ── SD1.5 architecture constants ──────────────────────────────────────────────
# SD1.5 UNet has cross-attention blocks at multiple resolutions.
# Each block has self-attention (attn1) and cross-attention (attn2).
# We only analyze self-attention (image-to-image).
#
# Latent space: 512/8 = 64x64
# Down blocks with attention: 3 (at 64x64, 32x32, 16x16), 2 layers each = 6
# Mid block: 1 layer at 8x8 = 1
# Up blocks with attention: 3 (at 16x16, 32x32, 64x64), 3 layers each = 9
# Total self-attention blocks: 16
# Heads: 8 per block

VAE_DOWNSCALE = 8
LATENT_SIZE = RESOLUTION // VAE_DOWNSCALE  # 64

NUM_HEADS = 8  # SD1.5 uses 8 attention heads at all resolutions

# Block output channels per down block level
BLOCK_OUT_CHANNELS = (320, 640, 1280, 1280)

# ── Seeds ──────────────────────────────────────────────────────────────────────
SEEDS = [42, 137, 256]

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
MIRROR_DIR = GENERATED_DIR / "mirror"
NON_MIRROR_DIR = GENERATED_DIR / "non_mirror"
ROI_DIR = DATA_DIR / "rois"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORT_DIR = PROJECT_ROOT / "report"

# ── Prompts ────────────────────────────────────────────────────────────────────
# Mirror prompts: spatial-separation guidance to keep object and reflection distinct
MIRROR_PROMPTS = [
    "a cat on the left side looking at its reflection in a large mirror on the right side",
    "a woman standing on the left facing a tall mirror on the right, seeing her reflection",
    "a red ball on the left of the scene with a mirror on the right showing the ball's reflection",
    "a dog sitting on the left looking at a mirror on the right side of the image",
    "a vase of flowers on the left with a wall mirror on the right reflecting the flowers",
    "a person on the left raising their hand, mirror on the right showing the raised hand reflection",
    "a teddy bear on the left side next to a standing mirror on the right showing its reflection",
    "a bird perched on the left with a round mirror on the right reflecting the bird",
]

# Non-mirror prompts: matched scenes without mirrors
NON_MIRROR_PROMPTS = [
    "a cat sitting on the left side of the scene, plain wall on the right side",
    "a woman standing on the left side of the room, bookshelf on the right side",
    "a red ball on the left of the scene with a window on the right side",
    "a dog sitting on the left side with a wooden door on the right side",
    "a vase of flowers on the left with a painting on the right side of the wall",
    "a person on the left raising their hand, curtains on the right side",
    "a teddy bear on the left side next to a shelf on the right side",
    "a bird perched on the left with a clock on the right side of the wall",
]

# ── Experiment parameters ──────────────────────────────────────────────────────
TOP_K_CANDIDATES = 10  # Number of candidate heads to investigate in detail
# Capture CRA at every timestep (scalars are cheap)
TIMESTEPS_TO_CAPTURE = list(range(NUM_INFERENCE_STEPS))

"""Central configuration for Mirror Reflection Mechanistic Interpretability."""

from pathlib import Path

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
RESOLUTION = 512
NUM_INFERENCE_STEPS = 4  # Schnell's distilled schedule

# ── Flux architecture constants ────────────────────────────────────────────────
NUM_DUAL_STREAM_BLOCKS = 19   # MMDiT blocks (joint text+image attention)
NUM_SINGLE_STREAM_BLOCKS = 38 # Single-stream DiT blocks
NUM_BLOCKS = NUM_DUAL_STREAM_BLOCKS + NUM_SINGLE_STREAM_BLOCKS  # 57 total
NUM_HEADS = 24
HEAD_DIM = 128  # 3072 / 24

# ── Token-space geometry ───────────────────────────────────────────────────────
# Flux patchifies: 2x2 patches on top of 8x VAE downsampling = 16x total
PATCH_SIZE = 2
VAE_DOWNSCALE = 8
TOTAL_DOWNSCALE = PATCH_SIZE * VAE_DOWNSCALE  # 16
GRID_H = RESOLUTION // TOTAL_DOWNSCALE  # 32
GRID_W = RESOLUTION // TOTAL_DOWNSCALE  # 32
NUM_IMAGE_TOKENS = GRID_H * GRID_W  # 1024
MAX_TEXT_TOKENS = 512  # T5-XXL max sequence length for Flux

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
TIMESTEPS_TO_CAPTURE = list(range(NUM_INFERENCE_STEPS))  # [0, 1, 2, 3]

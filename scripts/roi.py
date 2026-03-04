"""ROI utilities for mapping between token-space and pixel-space.

Flux token grid: 32x32 (at 512px resolution) due to 2x2 patchify + 8x VAE = 16x total.
Token indices are row-major: token_idx = row * GRID_W + col.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from scripts.config import GRID_H, GRID_W, TOTAL_DOWNSCALE, RESOLUTION, NUM_IMAGE_TOKENS

# ── CLIPSeg lazy-loaded globals ───────────────────────────────────────────────
_clipseg_processor = None
_clipseg_model = None


@dataclass
class ROI:
    """Region of interest in token space."""
    name: str
    token_indices: List[int]

    @property
    def size(self) -> int:
        return len(self.token_indices)

    def to_pixel_mask(self, resolution: int = RESOLUTION) -> np.ndarray:
        """Convert token indices to a binary pixel-space mask."""
        mask = np.zeros((resolution, resolution), dtype=bool)
        for idx in self.token_indices:
            row = idx // GRID_W
            col = idx % GRID_W
            y0 = row * TOTAL_DOWNSCALE
            y1 = y0 + TOTAL_DOWNSCALE
            x0 = col * TOTAL_DOWNSCALE
            x1 = x0 + TOTAL_DOWNSCALE
            mask[y0:y1, x0:x1] = True
        return mask


def pixel_to_token(px_x: int, px_y: int) -> int:
    """Convert pixel coordinate to token index."""
    row = px_y // TOTAL_DOWNSCALE
    col = px_x // TOTAL_DOWNSCALE
    return row * GRID_W + col


def token_to_pixel_center(token_idx: int) -> Tuple[int, int]:
    """Convert token index to pixel coordinate (center of patch)."""
    row = token_idx // GRID_W
    col = token_idx % GRID_W
    px_x = col * TOTAL_DOWNSCALE + TOTAL_DOWNSCALE // 2
    px_y = row * TOTAL_DOWNSCALE + TOTAL_DOWNSCALE // 2
    return px_x, px_y


def split_vertical(split_frac: float = 0.5) -> Tuple[ROI, ROI]:
    """Split image into left and right ROIs at a given fraction.

    For mirror prompts with "object on left, mirror on right", use default 0.5.
    Returns (left_roi, right_roi) = (object_roi, reflection_roi).
    """
    split_col = int(GRID_W * split_frac)
    left_indices = []
    right_indices = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            idx = row * GRID_W + col
            if col < split_col:
                left_indices.append(idx)
            else:
                right_indices.append(idx)
    return (
        ROI(name="left_object", token_indices=left_indices),
        ROI(name="right_reflection", token_indices=right_indices),
    )


def split_horizontal(split_frac: float = 0.5) -> Tuple[ROI, ROI]:
    """Split image into top and bottom ROIs."""
    split_row = int(GRID_H * split_frac)
    top_indices = []
    bottom_indices = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            idx = row * GRID_W + col
            if row < split_row:
                top_indices.append(idx)
            else:
                bottom_indices.append(idx)
    return (
        ROI(name="top", token_indices=top_indices),
        ROI(name="bottom", token_indices=bottom_indices),
    )


def bbox_roi(name: str, x0: int, y0: int, x1: int, y1: int) -> ROI:
    """Create an ROI from pixel-space bounding box coordinates.

    Args:
        name: Label for this ROI.
        x0, y0: Top-left corner in pixels.
        x1, y1: Bottom-right corner in pixels.
    """
    col0 = x0 // TOTAL_DOWNSCALE
    row0 = y0 // TOTAL_DOWNSCALE
    col1 = min((x1 + TOTAL_DOWNSCALE - 1) // TOTAL_DOWNSCALE, GRID_W)
    row1 = min((y1 + TOTAL_DOWNSCALE - 1) // TOTAL_DOWNSCALE, GRID_H)
    indices = []
    for row in range(row0, row1):
        for col in range(col0, col1):
            indices.append(row * GRID_W + col)
    return ROI(name=name, token_indices=indices)


def get_default_rois() -> Tuple[ROI, ROI]:
    """Get default object/reflection ROIs (left/right vertical split)."""
    return split_vertical(0.5)


# ── CLIPSeg-based segmentation ────────────────────────────────────────────────

def _load_clipseg():
    """Lazy-load CLIPSeg model and processor (cached as module globals)."""
    global _clipseg_processor, _clipseg_model
    if _clipseg_model is None:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        _clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        _clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        _clipseg_model.eval()
    return _clipseg_processor, _clipseg_model


def get_roi(
    object_name: str,
    image: Image.Image,
    threshold: float = 0.5,
    roi_name: Optional[str] = None,
) -> ROI:
    """Segment an object in an image using CLIPSeg and return token-space ROI.

    Args:
        object_name: Text query for CLIPSeg (e.g. "cat", "red ball").
        image: PIL image to segment.
        threshold: Fraction of max probability to threshold at.
        roi_name: Name for the ROI (defaults to object_name).

    Returns:
        ROI with token indices where the object was detected.

    Raises:
        ValueError: If no tokens survive thresholding.
    """
    processor, model = _load_clipseg()

    inputs = processor(text=[object_name], images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, H, W)

    # Sigmoid → probability map
    prob_map = torch.sigmoid(logits[0])  # (H, W)

    # Resize to token grid via bilinear interpolation
    prob_grid = torch.nn.functional.interpolate(
        prob_map.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        size=(GRID_H, GRID_W),
        mode="bilinear",
        align_corners=False,
    ).squeeze()  # (GRID_H, GRID_W)

    # Threshold at threshold * max_prob
    max_prob = prob_grid.max().item()
    binary_mask = prob_grid >= (threshold * max_prob)

    token_indices = binary_mask.nonzero(as_tuple=False)  # (N, 2) — row, col
    if token_indices.numel() == 0:
        raise ValueError(f"No tokens survived thresholding for '{object_name}'")

    indices = (token_indices[:, 0] * GRID_W + token_indices[:, 1]).tolist()

    return ROI(name=roi_name or object_name, token_indices=sorted(indices))


def get_object_and_reflection_rois(
    image: Image.Image,
    object_name: str,
    threshold: float = 0.5,
) -> Tuple[ROI, ROI]:
    """Get separate object and reflection ROIs by splitting CLIPSeg mask at midpoint.

    Tokens in the left half (col < 16) are assigned to the object ROI,
    tokens in the right half (col >= 16) to the reflection ROI.
    Falls back to a 50/50 vertical split if one side is empty.

    Args:
        image: PIL image to segment.
        object_name: Text query for CLIPSeg.
        threshold: Fraction of max probability to threshold at.

    Returns:
        (object_roi, reflection_roi) tuple.
    """
    full_roi = get_roi(object_name, image, threshold=threshold, roi_name=object_name)

    mid_col = GRID_W // 2
    left_indices = []
    right_indices = []
    for idx in full_roi.token_indices:
        col = idx % GRID_W
        if col < mid_col:
            left_indices.append(idx)
        else:
            right_indices.append(idx)

    # Fall back to half-split if one side is empty
    if not left_indices or not right_indices:
        return split_vertical(0.5)

    return (
        ROI(name=f"{object_name}_object", token_indices=left_indices),
        ROI(name=f"{object_name}_reflection", token_indices=right_indices),
    )

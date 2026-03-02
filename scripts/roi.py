"""ROI utilities for mapping between token-space and pixel-space.

Flux token grid: 32x32 (at 512px resolution) due to 2x2 patchify + 8x VAE = 16x total.
Token indices are row-major: token_idx = row * GRID_W + col.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from scripts.config import GRID_H, GRID_W, TOTAL_DOWNSCALE, RESOLUTION, NUM_IMAGE_TOKENS


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

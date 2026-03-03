"""ROI utilities for mapping between token-space and pixel-space.

SD1.5 has self-attention at multiple resolutions (64x64, 32x32, 16x16, 8x8).
ROIs are defined in pixel space and can be projected to any token grid resolution.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from scripts.config import VAE_DOWNSCALE, RESOLUTION, LATENT_SIZE


@dataclass
class ROI:
    """Region of interest defined by a fractional bounding box (resolution-independent)."""
    name: str
    col_start: float  # 0.0 to 1.0
    col_end: float
    row_start: float  # 0.0 to 1.0
    row_end: float

    def token_indices(self, grid_size: int) -> List[int]:
        """Get token indices for this ROI at a given grid resolution.

        Args:
            grid_size: Spatial resolution of the token grid (e.g. 64, 32, 16, 8).

        Returns:
            List of token indices (row-major) within this ROI.
        """
        c0 = int(self.col_start * grid_size)
        c1 = int(self.col_end * grid_size)
        r0 = int(self.row_start * grid_size)
        r1 = int(self.row_end * grid_size)
        indices = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                indices.append(r * grid_size + c)
        return indices

    def token_count(self, grid_size: int) -> int:
        return len(self.token_indices(grid_size))

    def to_pixel_mask(self, resolution: int = RESOLUTION) -> np.ndarray:
        """Convert to a binary pixel-space mask."""
        mask = np.zeros((resolution, resolution), dtype=bool)
        x0 = int(self.col_start * resolution)
        x1 = int(self.col_end * resolution)
        y0 = int(self.row_start * resolution)
        y1 = int(self.row_end * resolution)
        mask[y0:y1, x0:x1] = True
        return mask


def split_vertical(split_frac: float = 0.5) -> Tuple[ROI, ROI]:
    """Split image into left and right ROIs.

    For mirror prompts with "object on left, mirror on right", use default 0.5.
    Returns (left_roi, right_roi) = (object_roi, reflection_roi).
    """
    return (
        ROI(name="left_object", col_start=0.0, col_end=split_frac, row_start=0.0, row_end=1.0),
        ROI(name="right_reflection", col_start=split_frac, col_end=1.0, row_start=0.0, row_end=1.0),
    )


def split_horizontal(split_frac: float = 0.5) -> Tuple[ROI, ROI]:
    """Split image into top and bottom ROIs."""
    return (
        ROI(name="top", col_start=0.0, col_end=1.0, row_start=0.0, row_end=split_frac),
        ROI(name="bottom", col_start=0.0, col_end=1.0, row_start=split_frac, row_end=1.0),
    )


def bbox_roi(name: str, x0: int, y0: int, x1: int, y1: int,
             resolution: int = RESOLUTION) -> ROI:
    """Create an ROI from pixel-space bounding box coordinates."""
    return ROI(
        name=name,
        col_start=x0 / resolution,
        col_end=x1 / resolution,
        row_start=y0 / resolution,
        row_end=y1 / resolution,
    )


def get_default_rois() -> Tuple[ROI, ROI]:
    """Get default object/reflection ROIs (left/right vertical split)."""
    return split_vertical(0.5)

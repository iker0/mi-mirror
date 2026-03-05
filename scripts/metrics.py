"""Metrics for analyzing reflection attention patterns.

Step 1: Selectivity Score — CRA_mirror - CRA_nonmirror
Step 2: Attention Entropy + HIES score
Step 3: Temporal profiling (CRA vs timestep)
Step 5: Reflection quality degradation (causal validation)
Step 6: Superadditivity (circuit composition)
"""

from typing import Dict, List, Tuple

import torch
import numpy as np
from PIL import Image

from scripts.attention_extraction import AttentionData
from scripts.config import NUM_BLOCKS, NUM_HEADS, NUM_INFERENCE_STEPS


# ── Step 1: Cross-Region Attention & Selectivity ───────────────────────────────

def compute_cra(
    attn_map: torch.Tensor,
    obj_indices: List[int],
    ref_indices: List[int],
    txt_len: int,
) -> torch.Tensor:
    """Compute CRA per head from a full attention map.

    Args:
        attn_map: (heads, seq, seq) attention weights.
        obj_indices: Object ROI token indices (image-space, 0-indexed).
        ref_indices: Reflection ROI token indices (image-space, 0-indexed).
        txt_len: Number of text tokens (to offset into image quadrant).

    Returns:
        (heads,) tensor of CRA values.
    """
    # Shift indices to account for text tokens
    ref_abs = [r + txt_len for r in ref_indices]
    obj_abs = [o + txt_len for o in obj_indices]

    ref_idx = torch.tensor(ref_abs)
    obj_idx = torch.tensor(obj_abs)

    # CRA: mean attention from reflection queries to object keys
    cross_attn = attn_map[:, ref_idx][:, :, obj_idx]  # (heads, n_ref, n_obj)
    return cross_attn.mean(dim=(1, 2))  # (heads,)


def compute_selectivity_matrix(
    mirror_data: List[AttentionData],
    nonmirror_data: List[AttentionData],
    n_blocks: int = NUM_BLOCKS,
    n_heads: int = NUM_HEADS,
    n_timesteps: int = NUM_INFERENCE_STEPS,
) -> torch.Tensor:
    """Compute selectivity score S(block, head) averaged over timesteps and samples.

    S = mean(CRA_mirror) - mean(CRA_nonmirror)

    Args:
        mirror_data: List of AttentionData from mirror images.
        nonmirror_data: List of AttentionData from non-mirror images.

    Returns:
        (n_blocks, n_heads) selectivity matrix.
    """
    def _avg_cra_matrix(data_list):
        matrices = []
        for data in data_list:
            # Average over timesteps
            m = data.get_all_cra_matrices(n_blocks, n_heads, n_timesteps)
            matrices.append(m.mean(dim=0))  # (blocks, heads)
        return torch.stack(matrices).mean(dim=0)

    mirror_avg = _avg_cra_matrix(mirror_data)
    nonmirror_avg = _avg_cra_matrix(nonmirror_data)
    return mirror_avg - nonmirror_avg


def rank_candidates(
    selectivity_matrix: torch.Tensor,
    top_k: int = 10,
) -> List[Tuple[int, int, float]]:
    """Rank heads by selectivity score.

    Returns list of (block_idx, head_idx, score) sorted descending.
    """
    flat = selectivity_matrix.flatten()
    top_indices = torch.argsort(flat, descending=True)[:top_k]

    candidates = []
    n_heads = selectivity_matrix.shape[1]
    for idx in top_indices:
        block = idx.item() // n_heads
        head = idx.item() % n_heads
        score = selectivity_matrix[block, head].item()
        candidates.append((block, head, score))

    return candidates


# ── Step 2: Attention Entropy & HIES ───────────────────────────────────────────

def compute_attention_entropy(attn_map: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute entropy of attention distribution per head.

    H(h) = -sum_ij A * log(A)

    Args:
        attn_map: (heads, seq, seq) attention weights (probabilities along last dim).

    Returns:
        (heads,) entropy values.
    """
    attn_clamped = attn_map.clamp(min=eps)
    entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)  # (heads, seq)
    return entropy.mean(dim=-1)  # average over query positions -> (heads,)


def compute_entropy_from_data(
    data_list: List[AttentionData],
    candidate_blocks: List[int],
    n_heads: int = NUM_HEADS,
) -> dict:
    """Compute mean entropy per candidate block/head from stored full attention maps.

    Returns dict[(block_idx, head_idx)] -> mean_entropy
    """
    entropy_sums = {}
    entropy_counts = {}

    for data in data_list:
        for (block_idx, timestep), attn_map in data.full_maps.items():
            if block_idx not in candidate_blocks:
                continue
            # attn_map: (heads, seq, seq)
            H = compute_attention_entropy(attn_map.float())
            for h in range(n_heads):
                key = (block_idx, h)
                entropy_sums[key] = entropy_sums.get(key, 0.0) + H[h].item()
                entropy_counts[key] = entropy_counts.get(key, 0) + 1

    return {k: v / entropy_counts[k] for k, v in entropy_sums.items()}


def compute_hies(
    selectivity: float,
    H_mirror: float,
    H_nonmirror: float,
    H_max: float = None,
) -> float:
    """Compute HIES score: S × (H_nonmirror - H_mirror) / H_max.

    Reflection heads should have high selectivity AND lower entropy on mirror images
    (more focused attention), yielding positive HIES.

    Args:
        selectivity: S(h, l) = CRA_mirror - CRA_nonmirror
        H_mirror: Mean entropy on mirror images
        H_nonmirror: Mean entropy on non-mirror images
        H_max: Maximum possible entropy (log(seq_len)). If None, uses max(H_mirror, H_nonmirror).
    """
    if H_max is None:
        H_max = max(H_mirror, H_nonmirror, 1e-10)
    return selectivity * (H_nonmirror - H_mirror) / H_max


# ── Step 3: Temporal Profiling ─────────────────────────────────────────────────

def compute_temporal_cra(
    data: AttentionData,
    block_idx: int,
    head_idx: int,
    n_timesteps: int = NUM_INFERENCE_STEPS,
) -> List[float]:
    """Get CRA values across timesteps for a specific head.

    Returns list of CRA values, one per timestep.
    """
    return [
        data.cra_scalars.get((block_idx, head_idx, t), 0.0)
        for t in range(n_timesteps)
    ]


def compute_temporal_profiles(
    data_list: List[AttentionData],
    candidates: List[Tuple[int, int, float]],
    n_timesteps: int = NUM_INFERENCE_STEPS,
) -> dict:
    """Compute mean temporal CRA profiles for candidate heads.

    Returns dict[(block_idx, head_idx)] -> list of mean CRA per timestep.
    """
    profiles = {}
    for block_idx, head_idx, _ in candidates:
        all_profiles = []
        for data in data_list:
            profile = compute_temporal_cra(data, block_idx, head_idx, n_timesteps)
            all_profiles.append(profile)
        mean_profile = np.mean(all_profiles, axis=0).tolist()
        profiles[(block_idx, head_idx)] = mean_profile
    return profiles


def identify_peak_timestep(
    temporal_profiles: dict,
) -> int:
    """Find the timestep where CRA is highest on average across candidates."""
    n_timesteps = len(next(iter(temporal_profiles.values())))
    timestep_means = np.zeros(n_timesteps)
    for profile in temporal_profiles.values():
        timestep_means += np.array(profile)
    timestep_means /= len(temporal_profiles)
    return int(np.argmax(timestep_means))


# ── Step 5: Causal Validation — Reflection Quality Degradation ──────────────

def _image_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL image to float64 numpy array in [0, 1]."""
    return np.asarray(img).astype(np.float64) / 255.0


def _ssim_simple(
    img1: np.ndarray,
    img2: np.ndarray,
    C1: float = (0.01 * 255) ** 2,
    C2: float = (0.03 * 255) ** 2,
) -> float:
    """Compute mean SSIM between two images (H, W, C) in [0, 255] scale.

    Uses the standard SSIM formula with 8x8 block means.
    """
    # Work in [0, 255] range for standard C1/C2
    a = img1 * 255.0
    b = img2 * 255.0

    mu1 = a.mean()
    mu2 = b.mean()
    sigma1_sq = ((a - mu1) ** 2).mean()
    sigma2_sq = ((b - mu2) ** 2).mean()
    sigma12 = ((a - mu1) * (b - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim)


def compute_reflection_quality(
    original_img: Image.Image,
    ablated_img: Image.Image,
    ref_roi: "ROI",
    obj_roi: "ROI",
    resolution: int = None,
) -> Dict[str, float]:
    """Compare original vs ablated image quality in reflection and object regions.

    Args:
        original_img: Baseline image (no ablation).
        ablated_img: Image generated with head(s) ablated.
        ref_roi: Reflection region ROI.
        obj_roi: Object region ROI.
        resolution: Image resolution (inferred from image if None).

    Returns:
        Dict with ssim_reflection, ssim_object, ssim_full, mse_reflection,
        mse_object, mse_full, and degradation_ratio.
    """
    if resolution is None:
        resolution = original_img.size[0]

    orig = _image_to_array(original_img.resize((resolution, resolution)))
    ablated = _image_to_array(ablated_img.resize((resolution, resolution)))

    ref_mask = ref_roi.to_pixel_mask(resolution)
    obj_mask = obj_roi.to_pixel_mask(resolution)

    # Extract masked regions
    ref_orig = orig[ref_mask]
    ref_ablated = ablated[ref_mask]
    obj_orig = orig[obj_mask]
    obj_ablated = ablated[obj_mask]

    # MSE
    mse_ref = float(((ref_orig - ref_ablated) ** 2).mean())
    mse_obj = float(((obj_orig - obj_ablated) ** 2).mean())
    mse_full = float(((orig - ablated) ** 2).mean())

    # SSIM (computed on the full masked region as flat arrays reshaped to pseudo-image)
    ssim_ref = _ssim_simple(ref_orig.reshape(1, -1, 1), ref_ablated.reshape(1, -1, 1))
    ssim_obj = _ssim_simple(obj_orig.reshape(1, -1, 1), obj_ablated.reshape(1, -1, 1))
    ssim_full = _ssim_simple(orig, ablated)

    # Degradation ratio: how much worse is reflection degradation vs object degradation
    # Higher = more specific to reflection (what we want for reflection heads)
    eps = 1e-10
    degradation_ratio = mse_ref / (mse_obj + eps)

    return {
        "ssim_reflection": ssim_ref,
        "ssim_object": ssim_obj,
        "ssim_full": ssim_full,
        "mse_reflection": mse_ref,
        "mse_object": mse_obj,
        "mse_full": mse_full,
        "degradation_ratio": degradation_ratio,
    }


# ── Step 6: Circuit Composition — Superadditivity ───────────────────────────

def compute_superadditivity(
    individual_effects: List[float],
    joint_effect: float,
) -> float:
    """Compute superadditivity: joint effect minus sum of individual effects.

    Positive = heads form a synergistic circuit (joint ablation is worse than
    the sum of individual ablations).
    Near-zero = heads work independently.

    Args:
        individual_effects: List of degradation values from ablating each head alone.
        joint_effect: Degradation value from ablating all heads simultaneously.

    Returns:
        Superadditivity score.
    """
    return joint_effect - sum(individual_effects)

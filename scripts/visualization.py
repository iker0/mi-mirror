"""Visualization utilities for mirror reflection analysis."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from scripts.attention_extraction import AttnBlockInfo
from scripts.config import NUM_HEADS, NUM_INFERENCE_STEPS


def plot_selectivity_heatmap(
    selectivity_matrix: torch.Tensor,
    block_infos: List[AttnBlockInfo] = None,
    title: str = "Reflection Selectivity Score (S = CRA_mirror - CRA_nonmirror)",
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot selectivity heatmap: blocks (y) x heads (x)."""
    fig, ax = plt.subplots(figsize=figsize)
    data = selectivity_matrix.numpy()

    vmax = max(abs(data.min()), abs(data.max()), 1e-8)
    im = ax.imshow(
        data, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )

    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Block Index", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Annotate block positions if block_infos provided
    if block_infos:
        # Find boundaries between down/mid/up
        positions = [b.position for b in block_infos]
        for i in range(1, len(positions)):
            if positions[i] != positions[i - 1]:
                ax.axhline(y=i - 0.5, color="green", linewidth=2, linestyle="--")

        # Label y-axis with block info
        labels = []
        for b in block_infos:
            labels.append(f"{b.position[0].upper()}{b.block_idx}.{b.layer_idx} ({b.resolution})")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)

    plt.colorbar(im, ax=ax, label="Selectivity Score")
    fig.tight_layout()
    return fig


def plot_top_candidates(
    candidates: List[Tuple[int, int, float]],
    block_infos: List[AttnBlockInfo] = None,
    title: str = "Top Candidate Reflection Heads",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Bar chart of top candidate heads by selectivity score."""
    fig, ax = plt.subplots(figsize=figsize)

    if block_infos:
        info_map = {b.linear_idx: b for b in block_infos}
        labels = []
        colors = []
        pos_colors = {"down": "#e74c3c", "mid": "#2ecc71", "up": "#3498db"}
        for b, h, _ in candidates:
            info = info_map.get(b)
            if info:
                labels.append(f"{info.position[0].upper()}{info.block_idx}.{info.layer_idx} H{h}")
                colors.append(pos_colors.get(info.position, "#95a5a6"))
            else:
                labels.append(f"B{b}H{h}")
                colors.append("#95a5a6")
    else:
        labels = [f"B{b}H{h}" for b, h, _ in candidates]
        colors = ["#3498db"] * len(candidates)

    scores = [s for _, _, s in candidates]
    ax.barh(range(len(candidates)), scores, color=colors)
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Selectivity Score")
    ax.set_title(title)
    ax.invert_yaxis()

    if block_infos:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="Down blocks"),
            Patch(facecolor="#2ecc71", label="Mid block"),
            Patch(facecolor="#3498db", label="Up blocks"),
        ]
        ax.legend(handles=legend_elements)

    fig.tight_layout()
    return fig


def plot_entropy_comparison(
    mirror_entropy: Dict[Tuple[int, int], float],
    nonmirror_entropy: Dict[Tuple[int, int], float],
    candidates: List[Tuple[int, int, float]],
    title: str = "Attention Entropy: Mirror vs Non-Mirror",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Bar chart comparing entropy for candidate heads."""
    fig, ax = plt.subplots(figsize=figsize)

    labels, mirror_vals, nonmirror_vals = [], [], []
    for b, h, _ in candidates:
        key = (b, h)
        if key in mirror_entropy and key in nonmirror_entropy:
            labels.append(f"B{b}H{h}")
            mirror_vals.append(mirror_entropy[key])
            nonmirror_vals.append(nonmirror_entropy[key])

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, mirror_vals, width, label="Mirror", color="#e74c3c", alpha=0.8)
    ax.bar(x + width / 2, nonmirror_vals, width, label="Non-mirror", color="#3498db", alpha=0.8)
    ax.set_xlabel("Head")
    ax.set_ylabel("Entropy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_hies_scores(
    hies_scores: Dict[Tuple[int, int], float],
    title: str = "HIES Score (Selectivity x Entropy Difference)",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Bar chart of HIES scores for candidate heads."""
    fig, ax = plt.subplots(figsize=figsize)

    sorted_items = sorted(hies_scores.items(), key=lambda x: x[1], reverse=True)
    labels = [f"B{b}H{h}" for (b, h), _ in sorted_items]
    scores = [s for _, s in sorted_items]
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in scores]

    ax.barh(range(len(labels)), scores, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("HIES Score")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_temporal_profiles(
    profiles: Dict[Tuple[int, int], List[float]],
    n_timesteps: int = NUM_INFERENCE_STEPS,
    title: str = "Temporal CRA Profile",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Line plot of CRA across timesteps for candidate heads."""
    fig, ax = plt.subplots(figsize=figsize)

    timesteps = list(range(n_timesteps))
    for (b, h), profile in profiles.items():
        ax.plot(timesteps, profile, marker="o", label=f"B{b}H{h}", linewidth=2)

    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Cross-Region Attention (CRA)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spatial_attention(
    attn_map: torch.Tensor,
    head_idx: int,
    query_indices: List[int],
    grid_size: int,
    image: Optional[Image.Image] = None,
    title: str = "Spatial Attention Pattern",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Visualize spatial attention for self-attention.

    Args:
        attn_map: (heads, seq, seq) attention weights.
        head_idx: Which head to visualize.
        query_indices: Token indices to use as queries.
        grid_size: Spatial resolution of this attention block.
        image: Optional generated image to overlay.
    """
    head_attn = attn_map[head_idx]  # (seq, seq)
    avg_attn = head_attn[query_indices].mean(dim=0).float().numpy()  # (seq,)
    attn_grid = avg_attn.reshape(grid_size, grid_size)

    if image is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(image)
        axes[0].set_title("Generated Image")
        axes[0].axis("off")

        im = axes[1].imshow(attn_grid, cmap="hot", interpolation="bilinear")
        axes[1].set_title(f"Attention (Head {head_idx})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        img_resized = image.resize((grid_size * 4, grid_size * 4))
        axes[2].imshow(img_resized, alpha=0.6)
        from scipy.ndimage import zoom
        attn_upsampled = zoom(attn_grid, 4, order=1)
        axes[2].imshow(attn_upsampled, cmap="hot", alpha=0.5, interpolation="bilinear")
        axes[2].set_title("Overlay")
        axes[2].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(attn_grid, cmap="hot", interpolation="bilinear")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

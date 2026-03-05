"""Visualization utilities for mirror reflection analysis.

All functions return matplotlib figures for embedding in notebooks.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image

from scripts.config import (
    GRID_H, GRID_W, NUM_BLOCKS, NUM_DUAL_STREAM_BLOCKS,
    NUM_HEADS, NUM_INFERENCE_STEPS, TOTAL_DOWNSCALE, RESOLUTION,
)


def plot_selectivity_heatmap(
    selectivity_matrix: torch.Tensor,
    title: str = "Reflection Selectivity Score (S = CRA_mirror - CRA_nonmirror)",
    figsize: Tuple[int, int] = (20, 8),
) -> plt.Figure:
    """Plot selectivity heatmap: blocks (y) x heads (x).

    Args:
        selectivity_matrix: (n_blocks, n_heads) tensor.
    """
    fig, ax = plt.subplots(figsize=figsize)
    data = selectivity_matrix.numpy()

    # Diverging colormap centered at 0
    vmax = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(
        data, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )

    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Block Index", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Mark dual-stream / single-stream boundary
    ax.axhline(y=NUM_DUAL_STREAM_BLOCKS - 0.5, color="green", linewidth=2,
               linestyle="--", label="Dual→Single boundary")
    ax.legend(loc="upper right")

    plt.colorbar(im, ax=ax, label="Selectivity Score")

    # Tick labels
    ax.set_xticks(range(0, NUM_HEADS, 2))
    ax.set_yticks(range(0, NUM_BLOCKS, 5))

    fig.tight_layout()
    return fig


def plot_top_candidates(
    candidates: List[Tuple[int, int, float]],
    title: str = "Top Candidate Reflection Heads",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Bar chart of top candidate heads by selectivity score."""
    fig, ax = plt.subplots(figsize=figsize)

    labels = [f"B{b}H{h}" for b, h, _ in candidates]
    scores = [s for _, _, s in candidates]
    colors = ["#e74c3c" if b < NUM_DUAL_STREAM_BLOCKS else "#3498db"
              for b, _, _ in candidates]

    bars = ax.barh(range(len(candidates)), scores, color=colors)
    ax.set_yticks(range(len(candidates)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Selectivity Score")
    ax.set_title(title)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Dual-stream (0-18)"),
        Patch(facecolor="#3498db", label="Single-stream (19-56)"),
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
    """Bar chart comparing entropy for candidate heads on mirror vs non-mirror."""
    fig, ax = plt.subplots(figsize=figsize)

    labels = []
    mirror_vals = []
    nonmirror_vals = []

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
    title: str = "HIES Score (Selectivity × Entropy Difference)",
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
    ax.set_xticks(timesteps)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_spatial_attention(
    attn_map: torch.Tensor,
    head_idx: int,
    query_indices: List[int],
    txt_len: int,
    image: Optional[Image.Image] = None,
    title: str = "Spatial Attention Pattern",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Visualize spatial attention: for query tokens, show which keys are attended.

    Args:
        attn_map: (heads, seq, seq) attention weights.
        head_idx: Which head to visualize.
        query_indices: Image-space token indices to use as queries.
        txt_len: Number of text tokens.
        image: Optional generated image to overlay.
    """
    # Get attention from query tokens to all image tokens
    query_abs = [q + txt_len for q in query_indices]
    head_attn = attn_map[head_idx]  # (seq, seq)

    # Average attention across query tokens, looking at image key tokens
    img_key_attn = head_attn[query_abs][:, txt_len:]  # (n_query, n_img_keys)
    avg_attn = img_key_attn.mean(dim=0).float().numpy()  # (n_img_keys,)

    # Reshape to spatial grid
    attn_grid = avg_attn.reshape(GRID_H, GRID_W)

    if image is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Generated Image")
        axes[0].axis("off")

        # Attention heatmap
        im = axes[1].imshow(attn_grid, cmap="hot", interpolation="bilinear")
        axes[1].set_title(f"Attention (Head {head_idx})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        img_resized = image.resize((GRID_W * 4, GRID_H * 4))
        axes[2].imshow(img_resized, alpha=0.6)
        # Upsample attention to match
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


def plot_attention_grid(
    attn_map: torch.Tensor,
    txt_len: int,
    heads_to_show: List[int] = None,
    title: str = "Image-to-Image Attention",
    figsize: Tuple[int, int] = (20, 10),
) -> plt.Figure:
    """Show attention matrices for multiple heads side by side."""
    if heads_to_show is None:
        heads_to_show = list(range(min(8, attn_map.shape[0])))

    n = len(heads_to_show)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1) if n > 1 else np.array([[axes]])

    for i, h in enumerate(heads_to_show):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        # Show image-to-image quadrant
        img_attn = attn_map[h, txt_len:, txt_len:].float().numpy()
        im = ax.imshow(img_attn, cmap="hot", aspect="auto")
        ax.set_title(f"Head {h}")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


# ── Step 5: Ablation Comparison Visualizations ──────────────────────────────

def plot_ablation_comparison(
    original: Image.Image,
    ablated_images: Dict[Tuple[int, int], Image.Image],
    title: str = "Ablation Comparison",
    figsize: Tuple[int, int] = (20, 4),
) -> plt.Figure:
    """Side-by-side comparison: original vs ablated images.

    Args:
        original: Baseline image.
        ablated_images: Dict of (block_idx, head_idx) -> ablated image.
    """
    n = len(ablated_images) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for i, ((b, h), img) in enumerate(ablated_images.items(), 1):
        axes[i].imshow(img)
        axes[i].set_title(f"Ablate B{b}H{h}", fontsize=11)
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_degradation_bars(
    metrics_per_head: Dict[Tuple[int, int], dict],
    title: str = "Reflection vs Object Degradation (MSE)",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Bar chart of MSE for reflection vs object region per ablated head.

    Args:
        metrics_per_head: Dict of (block_idx, head_idx) -> metrics dict from
            compute_reflection_quality.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    labels = [f"B{b}H{h}" for b, h in metrics_per_head]
    ref_mse = [m["mse_reflection"] for m in metrics_per_head.values()]
    obj_mse = [m["mse_object"] for m in metrics_per_head.values()]
    ref_ssim = [m["ssim_reflection"] for m in metrics_per_head.values()]
    obj_ssim = [m["ssim_object"] for m in metrics_per_head.values()]

    x = np.arange(len(labels))
    width = 0.35

    # MSE
    axes[0].bar(x - width / 2, ref_mse, width, label="Reflection", color="#e74c3c", alpha=0.8)
    axes[0].bar(x + width / 2, obj_mse, width, label="Object", color="#3498db", alpha=0.8)
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MSE by Region")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].legend()

    # SSIM
    axes[1].bar(x - width / 2, ref_ssim, width, label="Reflection", color="#e74c3c", alpha=0.8)
    axes[1].bar(x + width / 2, obj_ssim, width, label="Object", color="#3498db", alpha=0.8)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM by Region")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


# ── Step 6: Circuit Composition Visualizations ──────────────────────────────

def plot_superadditivity_heatmap(
    pairwise_scores: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
    title: str = "Pairwise Superadditivity",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Heatmap of pairwise superadditivity scores.

    Args:
        pairwise_scores: Dict of ((b1,h1), (b2,h2)) -> superadditivity score.
    """
    # Collect unique heads
    heads = sorted(set(
        h for pair in pairwise_scores for h in pair
    ))
    n = len(heads)
    head_to_idx = {h: i for i, h in enumerate(heads)}
    matrix = np.zeros((n, n))

    for (h1, h2), score in pairwise_scores.items():
        i, j = head_to_idx[h1], head_to_idx[h2]
        matrix[i, j] = score
        matrix[j, i] = score

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(abs(matrix.min()), abs(matrix.max())) or 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    labels = [f"B{b}H{h}" for b, h in heads]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] != 0:
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Superadditivity")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_circuit_diagram(
    pairwise_scores: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
    threshold: float = 0.0,
    title: str = "Reflection Head Circuit",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Network diagram showing head interactions (edges for superadditive pairs).

    Args:
        pairwise_scores: Dict of ((b1,h1), (b2,h2)) -> superadditivity score.
        threshold: Only draw edges for scores above this threshold.
    """
    heads = sorted(set(h for pair in pairwise_scores for h in pair))

    fig, ax = plt.subplots(figsize=figsize)

    # Position nodes by block index (y) and head index (x)
    blocks = sorted(set(b for b, h in heads))
    block_to_y = {b: i for i, b in enumerate(blocks)}

    # Count heads per block for x spacing
    heads_per_block = {}
    for b, h in heads:
        heads_per_block.setdefault(b, []).append(h)

    positions = {}
    for b, h in heads:
        y = block_to_y[b]
        hlist = sorted(heads_per_block[b])
        x = hlist.index(h) - (len(hlist) - 1) / 2
        positions[(b, h)] = (x, -y)

    # Draw edges
    for (h1, h2), score in pairwise_scores.items():
        if score > threshold:
            x1, y1 = positions[h1]
            x2, y2 = positions[h2]
            linewidth = 1 + 5 * abs(score) / (max(abs(s) for s in pairwise_scores.values()) or 1)
            color = "#2ecc71" if score > 0 else "#e74c3c"
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.6, zorder=1)

    # Draw nodes
    for (b, h), (x, y) in positions.items():
        is_dual = b < NUM_DUAL_STREAM_BLOCKS
        color = "#e74c3c" if is_dual else "#3498db"
        ax.scatter(x, y, s=300, c=color, zorder=2, edgecolors="black", linewidth=1)
        ax.annotate(f"B{b}H{h}", (x, y), ha="center", va="center", fontsize=7,
                    fontweight="bold", zorder=3)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Head offset within block")
    ax.set_ylabel("Block (depth)")
    ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor="#e74c3c", label="Dual-stream"),
        Patch(facecolor="#3498db", label="Single-stream"),
        Line2D([0], [0], color="#2ecc71", linewidth=2, label="Synergistic (>0)"),
        Line2D([0], [0], color="#e74c3c", linewidth=2, label="Redundant (<0)"),
    ]
    ax.legend(handles=legend, loc="upper right")

    fig.tight_layout()
    return fig

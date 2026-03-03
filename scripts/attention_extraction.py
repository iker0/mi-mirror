"""Custom attention processors for extracting attention patterns from SD1.5.

SD1.5 UNet has self-attention (attn1) and cross-attention (attn2) in each block.
We only intercept self-attention for CRA analysis.

Two processors:
1. SD15AttnProcessorCRAOnly — screening: computes CRA scalar per head.
2. SD15AttnProcessorWithStorage — detailed: stores full self-attention maps on CPU.

SD1.5 attention is simple: Q/K/V projections + scaled dot product. No QK-norm, no RoPE.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from scripts.roi import ROI


@dataclass
class AttnBlockInfo:
    """Metadata for a self-attention block."""
    key: str            # processor dict key
    position: str       # "down", "mid", "up"
    block_idx: int      # index within position
    layer_idx: int      # attention layer within block
    resolution: int     # spatial grid size (H = W)
    linear_idx: int     # sequential index for heatmaps


@dataclass
class AttentionData:
    """Container for extracted attention data across a full generation."""
    # CRA scalars: dict[(linear_idx, head_idx, timestep)] -> float
    cra_scalars: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    # Full attention maps: dict[(linear_idx, timestep)] -> tensor (heads, seq, seq) on CPU float16
    full_maps: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def get_cra_matrix(self, timestep: int, n_blocks: int, n_heads: int) -> torch.Tensor:
        """Return (n_blocks, n_heads) matrix of CRA values for a given timestep."""
        matrix = torch.zeros(n_blocks, n_heads)
        for (b, h, t), v in self.cra_scalars.items():
            if t == timestep:
                matrix[b, h] = v
        return matrix

    def get_all_cra_matrices(self, n_blocks: int, n_heads: int, n_timesteps: int) -> torch.Tensor:
        """Return (n_timesteps, n_blocks, n_heads) tensor of CRA values."""
        matrices = torch.zeros(n_timesteps, n_blocks, n_heads)
        for (b, h, t), v in self.cra_scalars.items():
            if t < n_timesteps:
                matrices[t, b, h] = v
        return matrices


def discover_self_attn_blocks(unet) -> List[AttnBlockInfo]:
    """Discover all self-attention (attn1) blocks in the SD1.5 UNet.

    Returns a sorted list of AttnBlockInfo with linear indices for heatmaps.
    """
    from scripts.config import LATENT_SIZE

    blocks = []
    for key in sorted(unet.attn_processors.keys()):
        if ".attn1.processor" not in key:
            continue

        parts = key.split(".")
        if parts[0] == "mid_block":
            position = "mid"
            block_idx = 0
            layer_idx = int(parts[2])  # mid_block.attentions.{idx}...
        else:
            position = parts[0].replace("_blocks", "")  # "down" or "up"
            block_idx = int(parts[1])
            layer_idx = int(parts[3])  # {}_blocks.{}.attentions.{idx}...

        # Compute resolution from block position
        n_down = len(unet.config.down_block_types)  # 4 for SD1.5
        if position == "down":
            resolution = LATENT_SIZE // (2 ** block_idx)
        elif position == "mid":
            resolution = LATENT_SIZE // (2 ** (n_down - 1))
        else:  # up
            resolution = LATENT_SIZE // (2 ** (n_down - 1 - block_idx))

        blocks.append(AttnBlockInfo(
            key=key,
            position=position,
            block_idx=block_idx,
            layer_idx=layer_idx,
            resolution=resolution,
            linear_idx=-1,  # assigned below
        ))

    # Assign linear indices (down → mid → up order)
    order = {"down": 0, "mid": 1, "up": 2}
    blocks.sort(key=lambda b: (order[b.position], b.block_idx, b.layer_idx))
    for i, block in enumerate(blocks):
        block.linear_idx = i

    return blocks


class SD15AttnProcessorCRAOnly:
    """Screening pass: compute CRA scalar per head for self-attention.

    For cross-attention (encoder_hidden_states is not None), falls back to
    standard SDPA without interception.
    """

    def __init__(
        self,
        linear_idx: int,
        resolution: int,
        obj_roi: ROI,
        ref_roi: ROI,
        attention_data: AttentionData,
    ):
        self.linear_idx = linear_idx
        self.resolution = resolution
        self.obj_roi = obj_roi
        self.ref_roi = ref_roi
        self.attention_data = attention_data
        self._current_timestep = 0

    def set_timestep(self, t: int):
        self._current_timestep = t

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None

        # For cross-attention, use standard SDPA (no interception needed)
        if is_cross:
            return self._standard_attn(attn, hidden_states, encoder_hidden_states, attention_mask)

        # Self-attention: manual computation to extract attention weights
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape to (B, heads, seq, head_dim)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        batch = query.shape[0]

        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

        # Manual attention: Q @ K^T / sqrt(d) + softmax
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Extract CRA
        self._extract_cra(attn_weights)

        # Compute output
        hidden_states = torch.matmul(attn_weights.to(value.dtype), value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _extract_cra(self, attn_weights: torch.Tensor):
        """Extract CRA scalar from self-attention weights."""
        obj_idx = torch.tensor(
            self.obj_roi.token_indices(self.resolution),
            device=attn_weights.device,
        )
        ref_idx = torch.tensor(
            self.ref_roi.token_indices(self.resolution),
            device=attn_weights.device,
        )

        for h in range(attn_weights.shape[1]):
            # CRA: mean attention from reflection queries to object keys
            cross_attn = attn_weights[0, h][ref_idx][:, obj_idx]
            cra = cross_attn.mean().item()
            self.attention_data.cra_scalars[
                (self.linear_idx, h, self._current_timestep)
            ] = cra

    @staticmethod
    def _standard_attn(attn, hidden_states, encoder_hidden_states, attention_mask):
        """Standard SDPA for cross-attention (no interception)."""
        residual = hidden_states
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        batch = query.shape[0]

        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SD15AttnProcessorWithStorage(SD15AttnProcessorCRAOnly):
    """Detailed pass: stores full self-attention maps on CPU in float16."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None

        if is_cross:
            return self._standard_attn(attn, hidden_states, encoder_hidden_states, attention_mask)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        batch = query.shape[0]

        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Extract CRA
        self._extract_cra(attn_weights)

        # Store full attention map on CPU
        self.attention_data.full_maps[
            (self.linear_idx, self._current_timestep)
        ] = attn_weights[0].to(torch.float16).cpu()

        # Compute output
        hidden_states = torch.matmul(attn_weights.to(value.dtype), value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def install_cra_processors(
    unet,
    obj_roi: ROI,
    ref_roi: ROI,
    attention_data: AttentionData,
) -> Tuple[List[AttnBlockInfo], Dict]:
    """Install CRA-only processors on all self-attention blocks.

    Cross-attention blocks (attn2) get default SDPA processors.

    Returns:
        (block_infos, processors_dict) for timestep management.
    """
    block_infos = discover_self_attn_blocks(unet)
    block_info_map = {b.key: b for b in block_infos}

    processors = {}
    custom_processors = {}

    for key in unet.attn_processors.keys():
        if key in block_info_map:
            info = block_info_map[key]
            proc = SD15AttnProcessorCRAOnly(
                linear_idx=info.linear_idx,
                resolution=info.resolution,
                obj_roi=obj_roi,
                ref_roi=ref_roi,
                attention_data=attention_data,
            )
            processors[key] = proc
            custom_processors[key] = proc
        else:
            # Cross-attention or non-attention: use default
            from diffusers.models.attention_processor import AttnProcessor2_0
            processors[key] = AttnProcessor2_0()

    unet.set_attn_processor(processors)
    return block_infos, custom_processors


def install_storage_processors(
    unet,
    candidate_linear_indices: List[int],
    obj_roi: ROI,
    ref_roi: ROI,
    attention_data: AttentionData,
) -> Tuple[List[AttnBlockInfo], Dict]:
    """Install storage processors on candidate blocks, CRA-only on the rest."""
    block_infos = discover_self_attn_blocks(unet)
    block_info_map = {b.key: b for b in block_infos}

    processors = {}
    custom_processors = {}

    for key in unet.attn_processors.keys():
        if key in block_info_map:
            info = block_info_map[key]
            if info.linear_idx in candidate_linear_indices:
                proc = SD15AttnProcessorWithStorage(
                    linear_idx=info.linear_idx,
                    resolution=info.resolution,
                    obj_roi=obj_roi,
                    ref_roi=ref_roi,
                    attention_data=attention_data,
                )
            else:
                proc = SD15AttnProcessorCRAOnly(
                    linear_idx=info.linear_idx,
                    resolution=info.resolution,
                    obj_roi=obj_roi,
                    ref_roi=ref_roi,
                    attention_data=attention_data,
                )
            processors[key] = proc
            custom_processors[key] = proc
        else:
            from diffusers.models.attention_processor import AttnProcessor2_0
            processors[key] = AttnProcessor2_0()

    unet.set_attn_processor(processors)
    return block_infos, custom_processors


def restore_default_processors(unet):
    """Restore original SD1.5 attention processors."""
    from diffusers.models.attention_processor import AttnProcessor2_0
    unet.set_attn_processor(AttnProcessor2_0())


def set_all_timesteps(processors: Dict, timestep: int):
    """Set the current timestep on all custom processors."""
    for proc in processors.values():
        if hasattr(proc, "set_timestep"):
            proc.set_timestep(timestep)

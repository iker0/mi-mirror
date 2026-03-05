"""Custom attention processors for extracting attention patterns from Flux.

Two processors:
1. FluxAttnProcessorCRAOnly — screening pass: computes CRA scalar per head, no storage.
2. FluxAttnProcessorWithStorage — detailed pass: stores full attention maps on CPU.

Both replicate Flux's QK-norm (RMSNorm) + rotary position embeddings exactly,
replacing F.scaled_dot_product_attention with manual Q@K^T/sqrt(d) + softmax.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from scripts.config import NUM_HEADS, HEAD_DIM


def _get_qkv(attn, hidden_states, encoder_hidden_states=None):
    """Replicate Flux's Q/K/V projection + QK-norm + concatenation."""
    if attn.fused_projections:
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
            enc_q, enc_k, enc_v = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
        else:
            enc_q = enc_k = enc_v = None
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)
        else:
            enc_q = enc_k = enc_v = None

    # Reshape to multi-head: (B, seq, heads, head_dim)
    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    # QK-norm (RMSNorm per head)
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    # Dual-stream: normalize and concatenate encoder tokens
    if enc_q is not None:
        enc_q = enc_q.unflatten(-1, (attn.heads, -1))
        enc_k = enc_k.unflatten(-1, (attn.heads, -1))
        enc_v = enc_v.unflatten(-1, (attn.heads, -1))
        enc_q = attn.norm_added_q(enc_q)
        enc_k = attn.norm_added_k(enc_k)
        # Concat: [text_tokens, image_tokens]
        query = torch.cat([enc_q, query], dim=1)
        key = torch.cat([enc_k, key], dim=1)
        value = torch.cat([enc_v, value], dim=1)

    return query, key, value


def _apply_rope_and_compute_attn_weights(query, key, image_rotary_emb):
    """Apply rotary embeddings and compute attention weights (no value multiply).

    Returns attention weights: (B, heads, seq, seq) in float32.
    """
    # Import Flux's rotary embedding function
    from diffusers.models.transformers.transformer_flux import apply_rotary_emb

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # Transpose to (B, heads, seq, head_dim) for matmul
    q = query.transpose(1, 2)  # (B, heads, seq_q, d)
    k = key.transpose(1, 2)    # (B, heads, seq_k, d)

    scale = 1.0 / math.sqrt(q.shape[-1])
    # Compute attention weights in float32 for numerical stability
    attn_weights = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)

    return attn_weights


def _compute_attn_output(attn_weights, value):
    """Multiply attention weights by values to get output."""
    # value: (B, seq, heads, head_dim) -> (B, heads, seq, head_dim)
    v = value.transpose(1, 2)
    # attn_weights: (B, heads, seq_q, seq_k)
    out = torch.matmul(attn_weights.to(v.dtype), v)
    # (B, heads, seq, head_dim) -> (B, seq, heads, head_dim) -> (B, seq, inner_dim)
    out = out.transpose(1, 2).flatten(2, 3)
    return out.to(value.dtype)


@dataclass
class CRAResult:
    """CRA scalar for one head at one timestep."""
    block_idx: int
    head_idx: int
    timestep: int
    cra_value: float


@dataclass
class AttentionData:
    """Container for extracted attention data across a full generation."""
    # CRA scalars: dict[(block_idx, head_idx, timestep)] -> float
    cra_scalars: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    # Full attention maps: dict[(block_idx, timestep)] -> tensor (heads, seq, seq) on CPU float16
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


class FluxAttnProcessorCRAOnly:
    """Screening pass: compute CRA scalar per head, discard full attention map.

    Replaces SDPA with manual attention computation. Computes the attention
    output identically to the original processor, but also extracts CRA.

    Args:
        block_idx: Index of this block (0-56).
        obj_indices: Token indices for the object ROI (in image-token space, 0-indexed).
        ref_indices: Token indices for the reflection ROI (in image-token space, 0-indexed).
        attention_data: Shared container to store CRA scalars.
    """

    def __init__(
        self,
        block_idx: int,
        obj_indices: List[int],
        ref_indices: List[int],
        attention_data: AttentionData,
    ):
        self.block_idx = block_idx
        self.obj_indices = obj_indices
        self.ref_indices = ref_indices
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
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        is_dual_stream = encoder_hidden_states is not None

        # Get Q, K, V (with QK-norm and concatenation)
        query, key, value = _get_qkv(attn, hidden_states, encoder_hidden_states)

        # Compute attention weights manually
        attn_weights = _apply_rope_and_compute_attn_weights(query, key, image_rotary_emb)
        # attn_weights: (B, heads, seq, seq)

        # Extract CRA from image-to-image quadrant
        if is_dual_stream:
            txt_len = encoder_hidden_states.shape[1]
        else:
            # Single-stream: text+image already concatenated in hidden_states
            # We need to figure out txt_len from the sequence length
            total_seq = hidden_states.shape[1]
            from scripts.config import NUM_IMAGE_TOKENS
            txt_len = total_seq - NUM_IMAGE_TOKENS

        self._extract_cra(attn_weights, txt_len)

        # Compute attention output (identical to original processor)
        out = _compute_attn_output(attn_weights, value)

        if is_dual_stream:
            encoder_hidden_states_out, hidden_states_out = out.split(
                [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
            )
            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
            return hidden_states_out, encoder_hidden_states_out
        else:
            return out

    def _extract_cra(self, attn_weights: torch.Tensor, txt_len: int):
        """Extract CRA scalar from image-to-image quadrant of attention."""
        # attn_weights: (B, heads, seq, seq) where seq = txt_len + img_tokens
        # Image-to-image quadrant: rows [txt_len:], cols [txt_len:]
        img_attn = attn_weights[:, :, txt_len:, txt_len:]  # (B, heads, img, img)

        obj_idx = torch.tensor(self.obj_indices, device=attn_weights.device)
        ref_idx = torch.tensor(self.ref_indices, device=attn_weights.device)

        for h in range(attn_weights.shape[1]):
            # CRA: average attention from reflection tokens to object tokens
            # ref_tokens (query) attending to obj_tokens (key)
            cross_attn = img_attn[0, h][ref_idx][:, obj_idx]  # (n_ref, n_obj)
            cra = cross_attn.mean().item()
            self.attention_data.cra_scalars[
                (self.block_idx, h, self._current_timestep)
            ] = cra


class FluxAttnProcessorWithStorage(FluxAttnProcessorCRAOnly):
    """Detailed pass: stores full attention maps on CPU in float16.

    Only install on top-K candidate blocks at peak timestep.
    One full map per block: (heads, seq, seq) ≈ 108MB at float16.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        is_dual_stream = encoder_hidden_states is not None

        query, key, value = _get_qkv(attn, hidden_states, encoder_hidden_states)
        attn_weights = _apply_rope_and_compute_attn_weights(query, key, image_rotary_emb)

        if is_dual_stream:
            txt_len = encoder_hidden_states.shape[1]
        else:
            total_seq = hidden_states.shape[1]
            from scripts.config import NUM_IMAGE_TOKENS
            txt_len = total_seq - NUM_IMAGE_TOKENS

        # Extract CRA
        self._extract_cra(attn_weights, txt_len)

        # Store full attention map on CPU (float16 to save memory)
        self.attention_data.full_maps[
            (self.block_idx, self._current_timestep)
        ] = attn_weights[0].to(torch.float16).cpu()

        # Compute output
        out = _compute_attn_output(attn_weights, value)

        if is_dual_stream:
            encoder_hidden_states_out, hidden_states_out = out.split(
                [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
            )
            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
            return hidden_states_out, encoder_hidden_states_out
        else:
            return out


def get_processor_key(block_idx: int) -> str:
    """Get the processor dict key for a given block index.

    Blocks 0-18: dual-stream (transformer_blocks)
    Blocks 19-56: single-stream (single_transformer_blocks)
    """
    from scripts.config import NUM_DUAL_STREAM_BLOCKS
    if block_idx < NUM_DUAL_STREAM_BLOCKS:
        return f"transformer_blocks.{block_idx}.attn.processor"
    else:
        single_idx = block_idx - NUM_DUAL_STREAM_BLOCKS
        return f"single_transformer_blocks.{single_idx}.attn.processor"


def install_cra_processors(
    transformer,
    obj_indices: List[int],
    ref_indices: List[int],
    attention_data: AttentionData,
) -> Dict[str, FluxAttnProcessorCRAOnly]:
    """Install CRA-only processors on all attention blocks.

    Returns dict of processor_key -> processor for timestep management.
    """
    from scripts.config import NUM_BLOCKS
    processors = {}
    for block_idx in range(NUM_BLOCKS):
        key = get_processor_key(block_idx)
        proc = FluxAttnProcessorCRAOnly(
            block_idx=block_idx,
            obj_indices=obj_indices,
            ref_indices=ref_indices,
            attention_data=attention_data,
        )
        processors[key] = proc

    transformer.set_attn_processor(processors)
    return processors


def install_storage_processors(
    transformer,
    candidate_blocks: List[int],
    obj_indices: List[int],
    ref_indices: List[int],
    attention_data: AttentionData,
) -> Dict[str, FluxAttnProcessorWithStorage]:
    """Install storage processors on candidate blocks, CRA-only on the rest.

    Args:
        candidate_blocks: Block indices to store full attention maps for.
    """
    from scripts.config import NUM_BLOCKS
    processors = {}
    storage_processors = {}

    for block_idx in range(NUM_BLOCKS):
        key = get_processor_key(block_idx)
        if block_idx in candidate_blocks:
            proc = FluxAttnProcessorWithStorage(
                block_idx=block_idx,
                obj_indices=obj_indices,
                ref_indices=ref_indices,
                attention_data=attention_data,
            )
            storage_processors[key] = proc
        else:
            proc = FluxAttnProcessorCRAOnly(
                block_idx=block_idx,
                obj_indices=obj_indices,
                ref_indices=ref_indices,
                attention_data=attention_data,
            )
        processors[key] = proc

    transformer.set_attn_processor(processors)
    return storage_processors


class FluxAttnProcessorWithAblation:
    """Ablation processor: zeros out specified heads' output contributions.

    Used for causal validation — if ablating a head degrades reflection quality
    but not overall image quality, the head is causally involved in reflection.

    Args:
        block_idx: Index of this block (0-56).
        ablate_heads: Head indices to zero out within this block.
    """

    def __init__(self, block_idx: int, ablate_heads: List[int]):
        self.block_idx = block_idx
        self.ablate_heads = ablate_heads

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        is_dual_stream = encoder_hidden_states is not None

        query, key, value = _get_qkv(attn, hidden_states, encoder_hidden_states)
        attn_weights = _apply_rope_and_compute_attn_weights(query, key, image_rotary_emb)
        out = _compute_attn_output_with_ablation(attn_weights, value, self.ablate_heads)

        if is_dual_stream:
            encoder_hidden_states_out, hidden_states_out = out.split(
                [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
            )
            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
            return hidden_states_out, encoder_hidden_states_out
        else:
            return out


def _compute_attn_output_with_ablation(
    attn_weights: torch.Tensor,
    value: torch.Tensor,
    ablate_heads: List[int],
) -> torch.Tensor:
    """Like _compute_attn_output but zeros ablated heads before flattening.

    Args:
        attn_weights: (B, heads, seq_q, seq_k)
        value: (B, seq, heads, head_dim)
        ablate_heads: Head indices to zero out.
    """
    v = value.transpose(1, 2)  # (B, heads, seq, head_dim)
    out = torch.matmul(attn_weights.to(v.dtype), v)  # (B, heads, seq, head_dim)
    # Zero ablated heads
    out[:, ablate_heads, :, :] = 0
    # (B, heads, seq, head_dim) -> (B, seq, heads, head_dim) -> (B, seq, inner_dim)
    out = out.transpose(1, 2).flatten(2, 3)
    return out.to(value.dtype)


def install_ablation_processors(
    transformer,
    ablation_targets: List[Tuple[int, List[int]]],
) -> None:
    """Install ablation processors on specified blocks, default Flux processors elsewhere.

    Args:
        transformer: The Flux transformer model.
        ablation_targets: List of (block_idx, [head_indices]) to ablate.
    """
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    from scripts.config import NUM_BLOCKS

    target_map = {block_idx: heads for block_idx, heads in ablation_targets}
    processors = {}

    for block_idx in range(NUM_BLOCKS):
        key = get_processor_key(block_idx)
        if block_idx in target_map:
            processors[key] = FluxAttnProcessorWithAblation(
                block_idx=block_idx,
                ablate_heads=target_map[block_idx],
            )
        else:
            processors[key] = FluxAttnProcessor()

    transformer.set_attn_processor(processors)


def restore_default_processors(transformer):
    """Restore original Flux attention processors."""
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    transformer.set_attn_processor(FluxAttnProcessor())


def set_all_timesteps(processors: Dict, timestep: int):
    """Set the current timestep on all custom processors."""
    for proc in processors.values():
        if hasattr(proc, "set_timestep"):
            proc.set_timestep(timestep)

"""μ-Zero fused decode-time attention forward pass.

Phase-3 fix: the original `zeroanchor_decode_attention_forward_opt` contained
a heuristic that fell back to dense dequantise for head_dim >= 128 and
seq >= 2048 — exactly the regime of Llama-3.1-8B and Qwen3-8B where the
fused kernels provide the largest bandwidth savings.  That heuristic is
removed here: mu_zero_decode_attention_forward ALWAYS uses the fused CUDA
kernels when available.
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch
import torch.nn.functional as F


_MU_ZERO_DECODE_STATS = defaultdict(int)
_MU_ZERO_CUDA_EXTENSION = None


def reset_mu_zero_decode_stats() -> None:
    _MU_ZERO_DECODE_STATS.clear()


def get_mu_zero_decode_stats() -> dict[str, int]:
    return dict(_MU_ZERO_DECODE_STATS)


def _get_mu_zero_cuda_extension():
    global _MU_ZERO_CUDA_EXTENSION
    if _MU_ZERO_CUDA_EXTENSION is None:
        from ..kernels.loader import load_mu_zero_cuda_extension

        _MU_ZERO_CUDA_EXTENSION = load_mu_zero_cuda_extension()
    return _MU_ZERO_CUDA_EXTENSION


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden
    b, kv_h, s, d = hidden.shape
    return hidden[:, :, None, :, :].expand(b, kv_h, n_rep, s, d).reshape(b, kv_h * n_rep, s, d)


def mu_zero_decode_attention_forward(
    module,
    query: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    kv_cache_state: dict,
    *,
    key_chunk_groups: int = 32,
    value_chunk_tokens: int = 256,
):
    """Custom attention forward for μ-Zero quantised KV cache.

    Returns (attn_output, attn_weights) in the same format as standard SDPA,
    or None if the fast path cannot be used (signals the caller to fall back
    to default attention).

    Fast-path conditions:
      - query.shape[-2] == 1  (single-token decode)
      - query.device is CUDA
      - query.dtype in {float16, bfloat16}
      - alpha_key and alpha_value have been selected
    """
    layer = kv_cache_state.get("layer")
    if layer is None:
        _MU_ZERO_DECODE_STATS["missing_layer"] += 1
        return None
    if query.shape[-2] != 1:
        # Multi-token (prefill) — let the model handle it normally
        _MU_ZERO_DECODE_STATS["non_decode_query"] += 1
        return None
    _MU_ZERO_DECODE_STATS["calls"] += 1

    key_prefix_length   = int(kv_cache_state.get("key_prefix_length",   0))
    value_prefix_length = int(kv_cache_state.get("value_prefix_length", 0))

    can_fuse = (
        query.is_cuda
        and query.dtype in {torch.float16, torch.bfloat16}
        and query.shape[0] > 0
        and layer._selected_alpha_key   is not None
        and layer._selected_alpha_value is not None
    )

    if (
        os.environ.get("MUZERO_FUSED_PREFIX_DECODE", "0") == "1"
        and
        can_fuse
        and attention_mask is None
        and key_prefix_length > 0
        and key_prefix_length == value_prefix_length
        and layer.nbits == 4
        and layer.q_group_size == 64
        and layer._packed_key_q is not None
        and layer._packed_value_q is not None
    ):
        try:
            cuda_ext = _get_mu_zero_cuda_extension()

            batch_size, num_heads, _, head_dim = query.shape
            prefix_accum = torch.empty((batch_size, num_heads, head_dim), dtype=torch.float32, device=query.device)
            prefix_m = torch.empty((batch_size, num_heads), dtype=torch.float32, device=query.device)
            prefix_l = torch.empty((batch_size, num_heads), dtype=torch.float32, device=query.device)
            if os.environ.get("MUZERO_FUSED_PREFIX_DECODE_LEGACY", "0") == "1":
                cuda_ext.prefix_decode(
                    query.contiguous(),
                    layer._packed_key_q.contiguous(),
                    layer._key_mn.contiguous(),
                    layer._key_scale.contiguous(),
                    layer._packed_value_q[:, :, :value_prefix_length, ...].contiguous(),
                    layer._value_mn[:, :, :value_prefix_length, ...].contiguous(),
                    layer._value_scale[:, :, :value_prefix_length, ...].contiguous(),
                    prefix_accum,
                    prefix_m,
                    prefix_l,
                    float(layer._selected_alpha_key),
                    float(layer._selected_alpha_value),
                    module.num_key_value_groups,
                    scaling,
                )
            else:
                chunk_groups = int(os.environ.get("MUZERO_PREFIX_CHUNK_GROUPS", "2"))
                num_chunks = max(1, (layer._packed_key_q.shape[2] + chunk_groups - 1) // chunk_groups)
                partial_accum = torch.empty((batch_size * num_heads, num_chunks, head_dim), dtype=torch.float32, device=query.device)
                partial_m = torch.empty((batch_size * num_heads, num_chunks), dtype=torch.float32, device=query.device)
                partial_l = torch.empty((batch_size * num_heads, num_chunks), dtype=torch.float32, device=query.device)
                cuda_ext.prefix_decode_chunked(
                    query.contiguous(),
                    layer._packed_key_q.contiguous(),
                    layer._key_mn.contiguous(),
                    layer._key_scale.contiguous(),
                    layer._packed_value_q[:, :, :value_prefix_length, ...].contiguous(),
                    layer._value_mn[:, :, :value_prefix_length, ...].contiguous(),
                    layer._value_scale[:, :, :value_prefix_length, ...].contiguous(),
                    partial_accum,
                    partial_m,
                    partial_l,
                    prefix_accum,
                    prefix_m,
                    prefix_l,
                    float(layer._selected_alpha_key),
                    float(layer._selected_alpha_value),
                    module.num_key_value_groups,
                    scaling,
                )

            dense_keys = _repeat_kv(layer.keys, module.num_key_value_groups)
            dense_vals = _repeat_kv(layer.values, module.num_key_value_groups)
            if dense_keys.shape[-2] > 0:
                dense_logits = (torch.matmul(query, dense_keys.transpose(2, 3)) * scaling).squeeze(2).float()
                dense_m = dense_logits.amax(dim=-1)
                dense_exp = torch.exp(dense_logits - dense_m.unsqueeze(-1))
                dense_l = dense_exp.sum(dim=-1)
                dense_accum = torch.matmul(dense_exp.unsqueeze(2), dense_vals.float()).squeeze(2)

                merged_m = torch.maximum(prefix_m, dense_m)
                prefix_factor = torch.exp(prefix_m - merged_m)
                dense_factor = torch.exp(dense_m - merged_m)
                merged_l = prefix_l * prefix_factor + dense_l * dense_factor
                merged_accum = prefix_accum * prefix_factor.unsqueeze(-1) + dense_accum * dense_factor.unsqueeze(-1)
                out = merged_accum / merged_l.clamp_min(1e-20).unsqueeze(-1)
            else:
                out = prefix_accum / prefix_l.clamp_min(1e-20).unsqueeze(-1)

            _MU_ZERO_DECODE_STATS["fused_prefix_decode_success"] += 1
            if os.environ.get("MUZERO_FUSED_PREFIX_DECODE_LEGACY", "0") != "1":
                _MU_ZERO_DECODE_STATS["fused_prefix_decode_chunked_success"] += 1
            _MU_ZERO_DECODE_STATS["returns"] += 1
            return out.to(query.dtype).unsqueeze(1).contiguous(), None
        except Exception:
            _MU_ZERO_DECODE_STATS["fused_prefix_decode_fallback"] += 1

    logits_parts: list[torch.Tensor] = []
    key_cursor = 0

    # ------- Quantised key prefix ----------------------------------------
    if key_prefix_length > 0:
        if can_fuse:
            try:
                cuda_ext = _get_mu_zero_cuda_extension()
                prefix_logits = torch.empty(
                    (query.shape[0], query.shape[1], key_prefix_length),
                    dtype=query.dtype, device=query.device,
                )
                cuda_ext.qk(
                    query.contiguous(),
                    layer._packed_key_q.contiguous(),
                    layer._key_mn.contiguous(),
                    layer._key_scale.contiguous(),
                    prefix_logits,
                    layer.nbits,
                    float(layer._selected_alpha_key),
                    module.num_key_value_groups,
                    scaling,
                )
                prefix_logits = prefix_logits.unsqueeze(2)
                if attention_mask is not None:
                    prefix_logits = prefix_logits + attention_mask[..., :key_prefix_length]
                logits_parts.append(prefix_logits)
                key_cursor += key_prefix_length
            except Exception:
                can_fuse = False
                _MU_ZERO_DECODE_STATS["fused_qk_fallback"] += 1
            else:
                _MU_ZERO_DECODE_STATS["fused_qk_success"] += 1

        if not can_fuse:
            # Chunked dequant fallback
            _MU_ZERO_DECODE_STATS["key_dequant_fallback"] += 1
            total_groups = key_prefix_length // layer.q_group_size
            for sg in range(0, total_groups, key_chunk_groups):
                eg = min(sg + key_chunk_groups, total_groups)
                key_chunk = layer._dequantize_key_range(sg, eg)
                key_chunk = _repeat_kv(key_chunk, module.num_key_value_groups)
                chunk_logits = torch.matmul(query, key_chunk.transpose(2, 3)) * scaling
                cl = key_chunk.shape[-2]
                if attention_mask is not None:
                    chunk_logits = chunk_logits + attention_mask[..., key_cursor:key_cursor + cl]
                logits_parts.append(chunk_logits)
                key_cursor += cl

    # ------- Dense residual keys -----------------------------------------
    dense_keys = _repeat_kv(layer.keys, module.num_key_value_groups)
    if dense_keys.shape[-2] > 0:
        dense_logits = torch.matmul(query, dense_keys.transpose(2, 3)) * scaling
        if attention_mask is not None:
            dense_logits = dense_logits + attention_mask[..., key_cursor:key_cursor + dense_keys.shape[-2]]
        logits_parts.append(dense_logits)

    # ------- Softmax -------------------------------------------------------
    attn_weights = F.softmax(
        torch.cat(logits_parts, dim=-1), dim=-1, dtype=torch.float32
    ).to(query.dtype)

    # ------- Quantised value prefix ---------------------------------------
    attn_output = None
    value_cursor = 0

    if value_prefix_length > 0:
        if can_fuse:
            try:
                cuda_ext = _get_mu_zero_cuda_extension()
                prefix_output = torch.empty(
                    (query.shape[0], query.shape[1], layer.v_head_dim),
                    dtype=query.dtype, device=query.device,
                )
                cuda_ext.av(
                    attn_weights[..., :value_prefix_length].squeeze(2).contiguous(),
                    layer._packed_value_q[:, :, :value_prefix_length, ...].contiguous(),
                    layer._value_mn[:, :, :value_prefix_length, ...].contiguous(),
                    layer._value_scale[:, :, :value_prefix_length, ...].contiguous(),
                    prefix_output,
                    layer.nbits,
                    float(layer._selected_alpha_value),
                    module.num_key_value_groups,
                    layer.q_group_size,
                )
                attn_output = prefix_output.unsqueeze(2)
                value_cursor += value_prefix_length
            except Exception:
                can_fuse = False
                attn_output = None
                value_cursor = 0
                _MU_ZERO_DECODE_STATS["fused_av_fallback"] += 1
            else:
                _MU_ZERO_DECODE_STATS["fused_av_success"] += 1

        if not can_fuse:
            _MU_ZERO_DECODE_STATS["value_dequant_fallback"] += 1
            for st in range(0, value_prefix_length, value_chunk_tokens):
                et = min(st + value_chunk_tokens, value_prefix_length)
                val_chunk = layer._dequantize_value_range(st, et)
                val_chunk = _repeat_kv(val_chunk, module.num_key_value_groups)
                wt = attn_weights[..., value_cursor:value_cursor + val_chunk.shape[-2]]
                co = torch.matmul(wt, val_chunk)
                attn_output = co if attn_output is None else attn_output + co
                value_cursor += val_chunk.shape[-2]

    # ------- Dense residual values ----------------------------------------
    dense_vals = _repeat_kv(layer.values, module.num_key_value_groups)
    if dense_vals.shape[-2] > 0:
        dense_wt   = attn_weights[..., value_cursor:value_cursor + dense_vals.shape[-2]]
        dense_out  = torch.matmul(dense_wt, dense_vals)
        attn_output = dense_out if attn_output is None else attn_output + dense_out
    if attn_output is None:
        raise RuntimeError("MuZero attention produced no output; both quantized prefix and dense residual are empty")

    _MU_ZERO_DECODE_STATS["returns"] += 1
    return attn_output.transpose(1, 2).contiguous(), None

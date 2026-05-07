"""Triton fallback for μ-Zero quantisation (used when CUDA extension unavailable)."""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_triton():
    import triton
    import triton.language as tl
    return triton, tl


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def triton_is_available() -> bool:
    try:
        _load_triton()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _define_pack_kernel():
    triton, tl = _load_triton()

    @triton.jit
    def _pack_kernel(
        q_ptr, packed_ptr,
        stride_q_row, stride_packed_row,
        group_size, q_bytes,
        bits: tl.constexpr,
        BLOCK_Q_BYTES: tl.constexpr,
    ):
        row = tl.program_id(0)
        byte_offsets = tl.arange(0, BLOCK_Q_BYTES)
        mask_bytes = byte_offsets < q_bytes
        packed = tl.zeros((BLOCK_Q_BYTES,), dtype=tl.int32)
        q_row = q_ptr + row * stride_q_row
        for bit_idx in tl.static_range(0, 8):
            global_bit = byte_offsets * 8 + bit_idx
            value_idx = global_bit // bits
            bit_in_value = global_bit % bits
            mask_values = mask_bytes & (value_idx < group_size)
            q_values = tl.load(q_row + value_idx, mask=mask_values, other=0).to(tl.int32)
            packed |= ((q_values >> bit_in_value) & 1) << bit_idx
        tl.store((packed_ptr + row * stride_packed_row) + byte_offsets, packed.to(tl.uint8), mask=mask_bytes)

    return _pack_kernel


def mu_zero_pack_triton(q: torch.Tensor, bits: int) -> torch.Tensor:
    if not q.is_cuda or q.dtype != torch.uint8 or q.dim() != 2:
        raise ValueError("`q` must be a 2-D contiguous CUDA uint8 tensor")
    q = q.contiguous()
    rows, group_size = q.shape
    q_bytes = (group_size * bits + 7) // 8
    packed_q = torch.empty((rows, q_bytes), dtype=torch.uint8, device=q.device)
    _define_pack_kernel()[(rows,)](
        q, packed_q,
        q.stride(0), packed_q.stride(0),
        group_size, q_bytes,
        bits=bits,
        BLOCK_Q_BYTES=_next_pow2(q_bytes),
    )
    return packed_q


def mu_zero_quantize_triton(
    grouped: torch.Tensor,
    bits: int,
    alpha: float,
    eps: float,
    amax_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grouped.dim() != 2 or not grouped.is_cuda:
        raise ValueError("`grouped` must be a 2-D contiguous CUDA tensor")
    qmax = float((1 << bits) - 1)
    mn = grouped.amin(dim=-1, keepdim=True)
    mx = grouped.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp_min(eps)
    normalized = ((grouped - mn) / scale).clamp(0.0, 1.0)
    zero_anchor = ((0.0 - mn) / scale).clamp(0.0, 1.0)
    zero_level = torch.round(zero_anchor * qmax)
    pos_bins = (qmax - zero_level).clamp_min(1.0)
    neg_bins = zero_level.clamp_min(1.0)
    pos_extent = (1.0 - zero_anchor).clamp_min(eps)
    neg_extent = zero_anchor.clamp_min(eps)
    pos_norm = ((normalized - zero_anchor) / pos_extent).clamp(0.0, 1.0)
    neg_norm = ((zero_anchor - normalized) / neg_extent).clamp(0.0, 1.0)
    lb = torch.log1p(torch.tensor(alpha, device=grouped.device, dtype=torch.float32))
    pos_companded = torch.log1p(alpha * pos_norm) / lb
    neg_companded = torch.log1p(alpha * neg_norm) / lb
    q_pos = zero_level + torch.round(pos_companded * pos_bins)
    q_neg = zero_level - torch.round(neg_companded * neg_bins)
    q = torch.where(normalized >= zero_anchor, q_pos, q_neg).clamp(0.0, qmax).to(torch.uint8)
    packed_q = mu_zero_pack_triton(q, bits)
    return packed_q, mn.to(amax_dtype), scale.to(amax_dtype)

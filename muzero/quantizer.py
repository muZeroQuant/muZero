"""μ-Zero quantisation utilities.

Provides:
  select_alpha()        – choose the log-companding α from tensor statistics.
  quantize_rows()       – quantise a flat [N, group_size] tensor.
  dequantize_rows()     – inverse of quantize_rows().
  roundtrip()           – convenience wrapper for offline quality evaluation.

Alpha selection notes
---------------------
For bits == 4 (the most common setting) the heuristic ALWAYS returns 2.0
regardless of statistics.  The statistics computation is therefore skipped
entirely to avoid a device→host quantile sync on the critical prefill path.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Alpha selection
# ---------------------------------------------------------------------------

def select_alpha(
    tensor: torch.Tensor,
    *,
    bits: int,
    q_group_size: int,
    kind: str,
    max_points: int = 65536,
) -> float:
    """Return the log-companding α for the given tensor and config.

    For bits == 4 this is always 2.0 and no statistics are computed.
    """
    # Fast path: for 4-bit (the common case), alpha is always 2.0.
    if bits == 4:
        return 2.0

    # General path: compute GPU statistics without CPU→device quantile transfer.
    sample = _subsample(tensor.float(), max_points)
    abs_sample = sample.abs()
    eps = 1e-8

    # Use kthvalue (GPU) instead of quantile to avoid sort overhead.
    n = abs_sample.numel()
    p50_abs  = float(torch.kthvalue(abs_sample.reshape(-1), max(1, int(0.50 * n))).values.item())
    p95_abs  = float(torch.kthvalue(abs_sample.reshape(-1), max(1, int(0.95 * n))).values.item())
    p99_abs  = float(torch.kthvalue(abs_sample.reshape(-1), max(1, int(0.99 * n))).values.item())

    tail_ratio        = p99_abs / max(p95_abs, eps)
    concentration_ratio = p95_abs / max(p50_abs, eps)

    grouped    = _prepare_groups(tensor, q_group_size=q_group_size, kind=kind)
    mn         = grouped.amin(dim=-1)
    mx         = grouped.amax(dim=-1)
    zero_anchor = ((0.0 - mn) / (mx - mn).clamp_min(eps)).clamp(0.0, 1.0)
    anchor_offset = float((zero_anchor - 0.5).abs().mean().item())

    if kind == "key":
        if bits == 1:
            if tail_ratio > 3.0 or concentration_ratio > 12.0 or anchor_offset > 0.22:
                return 16.0
            if tail_ratio > 2.0 or concentration_ratio > 8.0 or anchor_offset > 0.16:
                return 4.0
            return 2.0
        if bits == 8 and (tail_ratio > 2.6 or concentration_ratio > 10.0):
            return 4.0
        return 2.0

    # value
    if bits >= 8:
        if tail_ratio > 2.0 or concentration_ratio > 6.0 or anchor_offset > 0.14:
            return 4.0
        return 2.0
    if bits == 3 and tail_ratio > 2.3 and anchor_offset > 0.12:
        return 4.0
    return 2.0


def _subsample(tensor: torch.Tensor, max_points: int) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if flat.numel() <= max_points:
        return flat
    step = max(1, flat.numel() // max_points)
    return flat[::step][:max_points]


def _prepare_groups(tensor: torch.Tensor, *, q_group_size: int, kind: str) -> torch.Tensor:
    work = tensor.float()
    if work.dim() == 2:
        # Already flat [N, group_size] — treat as [1, 1, N, group_size]
        # No transposition needed.
        n, gs = work.shape
        if gs != q_group_size:
            # Re-group along last axis
            ng = max(1, gs // q_group_size)
            work = work[:, :ng * q_group_size].reshape(n, ng, q_group_size)
            return work.unsqueeze(0).unsqueeze(0)
        return work.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    if kind == "key":
        work = work.transpose(2, 3).contiguous()
    bs, nh, sl, hd = work.shape
    hd_pad = ((hd + q_group_size - 1) // q_group_size) * q_group_size
    if hd_pad != hd:
        work = F.pad(work, (0, hd_pad - hd), value=0)
    return work.reshape(bs, nh, sl, hd_pad // q_group_size, q_group_size)


# ---------------------------------------------------------------------------
# Quantise / dequantise rows (flat [N, group_size] → packed + metadata)
# ---------------------------------------------------------------------------

def quantize_rows(
    flat_grouped: torch.Tensor,
    bits: int,
    alpha: float,
    amax_eps: float,
    amax_dtype: torch.dtype,
    *,
    backend: str = "auto",
    _cuda_ok: bool = True,
    _triton_ok: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise a [N, group_size] tensor.

    Returns (packed_q, mn, scale) where packed_q is uint8.
    """
    q_group_size = flat_grouped.shape[1]
    q_bytes = (q_group_size * bits + 7) // 8

    # CUDA path
    if backend in {"auto", "cuda"} and _cuda_ok and flat_grouped.is_cuda:
        try:
            from .kernels.loader import mu_zero_quantize_cuda
            packed_q = torch.empty((flat_grouped.shape[0], q_bytes), dtype=torch.uint8, device=flat_grouped.device)
            mn       = torch.empty((flat_grouped.shape[0], 1), dtype=amax_dtype, device=flat_grouped.device)
            scale    = torch.empty((flat_grouped.shape[0], 1), dtype=amax_dtype, device=flat_grouped.device)
            mu_zero_quantize_cuda(flat_grouped.contiguous(), packed_q, mn, scale, bits, float(alpha), amax_eps)
            return packed_q, mn, scale
        except Exception:
            pass

    # Triton path
    if backend == "triton" and _triton_ok and flat_grouped.is_cuda:
        try:
            from .kernels.muzero_triton import mu_zero_quantize_triton
            return mu_zero_quantize_triton(flat_grouped.contiguous(), bits, float(alpha), amax_eps, amax_dtype)
        except Exception:
            pass

    # Pure-PyTorch fallback
    return _quantize_rows_torch(flat_grouped, bits, float(alpha), amax_eps, amax_dtype)


def _quantize_rows_torch(
    flat_grouped: torch.Tensor,
    bits: int,
    alpha: float,
    eps: float,
    amax_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qmax = float((1 << bits) - 1)
    mn    = flat_grouped.amin(dim=-1, keepdim=True)
    mx    = flat_grouped.amax(dim=-1, keepdim=True)
    scale = (mx - mn).clamp_min(eps)
    norm  = ((flat_grouped - mn) / scale).clamp(0.0, 1.0)
    za    = ((0.0 - mn) / scale).clamp(0.0, 1.0)
    zl    = torch.round(za * qmax)
    pb    = (qmax - zl).clamp_min(1.0)
    nb    = zl.clamp_min(1.0)
    pe    = (1.0 - za).clamp_min(eps)
    ne    = za.clamp_min(eps)
    lb    = torch.log1p(torch.tensor(alpha, device=flat_grouped.device, dtype=torch.float32))
    pos_n = ((norm - za) / pe).clamp(0.0, 1.0)
    neg_n = ((za - norm) / ne).clamp(0.0, 1.0)
    q_pos = zl + torch.round(torch.log1p(alpha * pos_n) / lb * pb)
    q_neg = zl - torch.round(torch.log1p(alpha * neg_n) / lb * nb)
    q     = torch.where(norm >= za, q_pos, q_neg).clamp(0.0, qmax).to(torch.uint8)
    packed_q = _pack_q(q, bits)
    return packed_q, mn.to(amax_dtype), scale.to(amax_dtype)


def _pack_q(q: torch.Tensor, bits: int) -> torch.Tensor:
    group_size = q.shape[-1]
    q_bytes    = (group_size * bits + 7) // 8
    total_bits = group_size * bits
    flat       = q.shape[:-1]
    bit_pos    = torch.arange(total_bits, device=q.device, dtype=torch.long)
    val_idx    = bit_pos // bits
    bit_off    = bit_pos % bits
    bits_t     = ((q[..., val_idx] >> bit_off) & 1).to(torch.uint8)
    if total_bits < q_bytes * 8:
        bits_t = F.pad(bits_t, (0, q_bytes * 8 - total_bits), value=0)
    bits_t = bits_t.reshape(*flat, q_bytes, 8)
    shifts = torch.arange(8, device=q.device, dtype=torch.uint8)
    return torch.sum(torch.bitwise_left_shift(bits_t, shifts), dim=-1, dtype=torch.uint8).contiguous()


def dequantize_rows(
    packed_q: torch.Tensor,
    mn: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    alpha: float,
    q_group_size: int,
    out_dtype: torch.dtype,
    *,
    amax_eps: float = 1e-8,
    _cuda_ok: bool = True,
) -> torch.Tensor:
    """Dequantise packed_q back to [N, q_group_size] in out_dtype."""
    if _cuda_ok and packed_q.is_cuda:
        try:
            from .kernels.loader import mu_zero_dequantize_cuda
            output = torch.empty((packed_q.shape[0], q_group_size), dtype=out_dtype, device=packed_q.device)
            mu_zero_dequantize_cuda(packed_q.contiguous(), mn.contiguous(), scale.contiguous(), output, bits, float(alpha))
            return output
        except Exception:
            pass
    return _dequantize_rows_torch(packed_q, mn, scale, bits, float(alpha), q_group_size, out_dtype, amax_eps)


def _dequantize_rows_torch(
    packed_q: torch.Tensor,
    mn: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    alpha: float,
    q_group_size: int,
    out_dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    qmax   = float((1 << bits) - 1)
    q      = _unpack_q(packed_q, bits, q_group_size).to(torch.float32)
    mn_fp  = mn.to(torch.float32)
    scl_fp = scale.to(torch.float32)
    za     = ((0.0 - mn_fp) / scl_fp).clamp(0.0, 1.0)
    zl     = torch.round(za * qmax)
    pb     = (qmax - zl).clamp_min(1.0)
    nb     = zl.clamp_min(1.0)
    pe     = (1.0 - za).clamp_min(eps)
    ne     = za.clamp_min(eps)
    lb     = torch.log1p(torch.tensor(alpha, device=packed_q.device, dtype=torch.float32))
    pc     = ((q - zl) / pb).clamp(0.0, 1.0)
    nc     = ((zl - q) / nb).clamp(0.0, 1.0)
    pn     = torch.expm1(pc * lb) / alpha
    nn_    = torch.expm1(nc * lb) / alpha
    norm   = torch.where(q >= zl, za + pe * pn, za - ne * nn_).clamp(0.0, 1.0)
    return (norm * scl_fp + mn_fp).to(out_dtype)


def _unpack_q(packed: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    total_bits = group_size * bits
    bit_pos    = torch.arange(total_bits, device=packed.device, dtype=torch.long)
    byte_idx   = bit_pos >> 3
    bit_in_byte= bit_pos & 7
    bits_t     = ((packed[..., byte_idx] >> bit_in_byte) & 1).to(torch.uint8)
    bits_t     = bits_t.reshape(*packed.shape[:-1], group_size, bits)
    shifts     = torch.arange(bits, device=packed.device, dtype=torch.uint8)
    return torch.sum(torch.bitwise_left_shift(bits_t, shifts), dim=-1, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Offline quality evaluation
# ---------------------------------------------------------------------------

def roundtrip(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    bits: int = 4,
    q_group_size: int = 64,
    alpha_key: Optional[float] = None,
    alpha_value: Optional[float] = None,
    amax_dtype: torch.dtype = torch.float16,
    amax_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise then dequantise key/value tensors for offline quality measurement."""
    if alpha_key is None:
        alpha_key = select_alpha(key, bits=bits, q_group_size=q_group_size, kind="key")
    if alpha_value is None:
        alpha_value = select_alpha(value, bits=bits, q_group_size=q_group_size, kind="value")

    def _rt(tensor: torch.Tensor, alpha: float, kind: str) -> torch.Tensor:
        bs, nh, sl, hd = tensor.shape
        if kind == "key":
            # Group along sequence axis: [bs, nh, sl, hd] →
            #   reshape [bs, nh, nsg, q_group_size, hd]
            #   permute [bs, nh, nsg, hd, q_group_size]
            #   flat    [bs*nh*nsg*hd, q_group_size]
            nsg  = sl // q_group_size
            flat = tensor.reshape(bs, nh, nsg, q_group_size, hd)
            flat = flat.permute(0, 1, 2, 4, 3).contiguous().reshape(-1, q_group_size)

            packed_q, mn, scale = quantize_rows(flat.float(), bits, alpha, amax_eps, amax_dtype)
            recon = dequantize_rows(packed_q, mn, scale, bits, alpha, q_group_size, tensor.dtype, amax_eps=amax_eps)

            # Inverse: [bs*nh*nsg*hd, gs] → [bs, nh, nsg, hd, gs] → permute → [bs, nh, sl, hd]
            recon = recon.reshape(bs, nh, nsg, hd, q_group_size)
            recon = recon.permute(0, 1, 2, 4, 3).reshape(bs, nh, sl, hd)
            return recon.contiguous()
        else:
            # Group along head_dim axis: [bs, nh, sl, hd_pad] →
            #   reshape [bs, nh, sl, ng, q_group_size]
            #   flat    [bs*nh*sl*ng, q_group_size]
            hd_pad = ((hd + q_group_size - 1) // q_group_size) * q_group_size
            t = F.pad(tensor, (0, hd_pad - hd)) if hd_pad != hd else tensor
            ng   = hd_pad // q_group_size
            flat = t.reshape(bs, nh, sl, ng, q_group_size).reshape(-1, q_group_size)

            packed_q, mn, scale = quantize_rows(flat.float(), bits, alpha, amax_eps, amax_dtype)
            recon = dequantize_rows(packed_q, mn, scale, bits, alpha, q_group_size, tensor.dtype, amax_eps=amax_eps)

            recon = recon.reshape(bs, nh, sl, hd_pad)[..., :hd]
            return recon.contiguous()

    return _rt(key, alpha_key, "key"), _rt(value, alpha_value, "value")

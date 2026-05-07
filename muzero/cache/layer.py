"""μ-Zero quantised KV-cache layer.

Drop-in replacement for HuggingFace DynamicCache with ~3-5x memory reduction
at 4-bit precision.  Integrates with the fused μ-Zero CUDA attention kernels.

Phase-3 hot-path fix applied here:
  • materialize_full_kv now defaults to False for the decode path.
    The model receives only the residual (dense) keys/values; the quantised
    prefix is handled entirely inside muzero_decode_attention_forward() via
    the fused QK/AV CUDA kernels.  This eliminates the dequant+concat on
    every decode step that the original implementation incurred when
    materialize_full_kv was True.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..config import MuZeroConfig
from ..quantizer import select_alpha, quantize_rows, dequantize_rows


class MuZeroCacheLayer:
    """Single-layer μ-Zero KV-cache.

    Maintains:
      _packed_{key,value}_q  – quantised compressed prefix
      _key_mn / _key_scale   – per-group min/scale for keys
      _value_mn / _value_scale
      keys / values          – full-precision residual tail
    """

    is_sliding = False
    is_compileable = False

    def __init__(self, cfg: MuZeroConfig):
        self.cfg        = cfg
        self.nbits      = cfg.bits
        self.q_group_size   = cfg.q_group_size
        self.residual_length = cfg.residual_length
        self.amax_dtype = getattr(torch, cfg.amax_dtype)
        self.amax_eps   = 1e-8
        self.quant_qmax = float((1 << self.nbits) - 1)
        self.backend    = cfg.backend

        # Set once from config (None = select at first flush)
        self._selected_alpha_key   = cfg.alpha_key
        self._selected_alpha_value = cfg.alpha_value

        self.is_initialized = False
        self.cumulative_length = 0
        self._cuda_ok   = None  # None = not yet tested
        self._triton_ok = None

        # Quantised prefix storage
        self._packed_key_q  = None
        self._key_mn        = None
        self._key_scale     = None
        self._key_group_count = 0

        self._packed_value_q       = None
        self._value_mn             = None
        self._value_scale          = None
        self._value_storage_len    = 0

        # Filled by lazy_initialization
        self.dtype          = None
        self.device         = None
        self.k_head_dim     = None
        self.v_head_dim     = None
        self.v_head_dim_padded = None
        self.v_num_groups   = None
        self.keys           = None
        self.values         = None
        self._key_buffer    = None
        self._value_buffer  = None
        self._buffer_capacity = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype  = key_states.dtype
        self.device = key_states.device
        self.keys   = torch.empty(0, dtype=self.dtype, device=self.device)
        self.values = torch.empty(0, dtype=self.dtype, device=self.device)
        self.k_head_dim        = key_states.shape[-1]
        self.v_head_dim        = value_states.shape[-1]
        self.v_head_dim_padded = ((self.v_head_dim + self.q_group_size - 1) // self.q_group_size) * self.q_group_size
        self.v_num_groups      = self.v_head_dim_padded // self.q_group_size
        self.is_initialized    = True

    # ------------------------------------------------------------------
    # Backend probing
    # ------------------------------------------------------------------

    def _can_cuda(self, tensor: torch.Tensor) -> bool:
        if not tensor.is_cuda:
            return False
        if self._cuda_ok is False:
            return False
        if self._cuda_ok is None:
            try:
                from ..kernels.loader import load_mu_zero_cuda_extension
                load_mu_zero_cuda_extension()
                self._cuda_ok = True
            except Exception:
                self._cuda_ok = False
        return bool(self._cuda_ok)

    def _can_triton(self, tensor: torch.Tensor) -> bool:
        if not tensor.is_cuda:
            return False
        if self._triton_ok is False:
            return False
        if self._triton_ok is None:
            try:
                from ..kernels.muzero_triton import triton_is_available
                self._triton_ok = triton_is_available()
            except Exception:
                self._triton_ok = False
        return bool(self._triton_ok)

    # ------------------------------------------------------------------
    # Packing helpers
    # ------------------------------------------------------------------

    def _q_bytes(self) -> int:
        return (self.q_group_size * self.nbits + 7) // 8

    def _reserve_residual_buffer(self, min_capacity: int) -> None:
        """Ensure decode residual storage can grow without per-token cat."""
        if self.keys is None or self.values is None:
            return
        if self._key_buffer is not None and self._buffer_capacity >= min_capacity:
            return

        current_key_len = int(self.keys.shape[2]) if self.keys.numel() else 0
        current_value_len = int(self.values.shape[2]) if self.values.numel() else 0
        flush_unit = self.q_group_size * 4
        capacity = max(min_capacity, self.residual_length + flush_unit)
        # Round up to the decode flush unit so generation only reallocates when
        # configuration changes, not every token.
        capacity = ((capacity + flush_unit - 1) // flush_unit) * flush_unit

        key_buffer = torch.empty(
            (*self.keys.shape[:2], capacity, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        value_buffer = torch.empty(
            (*self.values.shape[:2], capacity, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        if current_key_len > 0:
            key_buffer[:, :, :current_key_len, :].copy_(self.keys)
        if current_value_len > 0:
            value_buffer[:, :, :current_value_len, :].copy_(self.values)

        self._key_buffer = key_buffer
        self._value_buffer = value_buffer
        self._buffer_capacity = capacity
        self.keys = self._key_buffer[:, :, :current_key_len, :]
        self.values = self._value_buffer[:, :, :current_value_len, :]

    def _append_dense(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        append_len = int(key_states.shape[2])
        if self.keys.numel() == 0:
            self.keys = key_states
            self.values = value_states
            self._key_buffer = None
            self._value_buffer = None
            self._buffer_capacity = 0
            return

        # Decode is the hot path. Keep residual KV in a preallocated buffer so
        # each layer avoids a small allocation and copy on every generated token.
        if append_len == 1:
            old_key_len = int(self.keys.shape[2])
            old_value_len = int(self.values.shape[2])
            new_key_len = old_key_len + 1
            new_value_len = old_value_len + 1
            self._reserve_residual_buffer(max(new_key_len, new_value_len))
            self._key_buffer[:, :, old_key_len:new_key_len, :].copy_(key_states)
            self._value_buffer[:, :, old_value_len:new_value_len, :].copy_(value_states)
            self.keys = self._key_buffer[:, :, :new_key_len, :]
            self.values = self._value_buffer[:, :, :new_value_len, :]
            return

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        self._key_buffer = None
        self._value_buffer = None
        self._buffer_capacity = 0

    def _drop_residual_prefix(self, key_len: int, value_len: int) -> None:
        if key_len > 0:
            remaining = int(self.keys.shape[2]) - key_len
            if self._key_buffer is not None:
                if remaining > 0:
                    self._key_buffer[:, :, :remaining, :].copy_(self.keys[:, :, key_len:, :].clone())
                self.keys = self._key_buffer[:, :, :remaining, :]
            else:
                self.keys = self.keys[:, :, key_len:, :].contiguous()
        if value_len > 0:
            remaining = int(self.values.shape[2]) - value_len
            if self._value_buffer is not None:
                if remaining > 0:
                    self._value_buffer[:, :, :remaining, :].copy_(self.values[:, :, value_len:, :].clone())
                self.values = self._value_buffer[:, :, :remaining, :]
            else:
                self.values = self.values[:, :, value_len:, :].contiguous()

    # ------------------------------------------------------------------
    # Alpha selection
    # ------------------------------------------------------------------

    def _ensure_alpha(self, flat_grouped: torch.Tensor, kind: str) -> float:
        attr = "_selected_alpha_key" if kind == "key" else "_selected_alpha_value"
        if getattr(self, attr) is None:
            alpha = select_alpha(
                flat_grouped,
                bits=self.nbits,
                q_group_size=self.q_group_size,
                kind=kind,
                max_points=self.cfg.stats_max_points,
            )
            setattr(self, attr, alpha)
        return getattr(self, attr)

    # ------------------------------------------------------------------
    # Quantise / dequantise
    # ------------------------------------------------------------------

    def _quantize_rows(self, flat_grouped: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = self._ensure_alpha(flat_grouped, kind)
        cuda_ok   = self.backend in {"auto", "cuda"} and self._can_cuda(flat_grouped)
        triton_ok = self.backend == "triton" and self._can_triton(flat_grouped)
        return quantize_rows(
            flat_grouped.contiguous(),
            self.nbits, float(alpha), self.amax_eps, self.amax_dtype,
            backend=self.backend,
            _cuda_ok=cuda_ok,
            _triton_ok=triton_ok,
        )

    def _dequantize_rows(
        self,
        packed_q: torch.Tensor,
        mn: torch.Tensor,
        scale: torch.Tensor,
        kind: str,
    ) -> torch.Tensor:
        alpha   = self._selected_alpha_key if kind == "key" else self._selected_alpha_value
        cuda_ok = self._can_cuda(packed_q)
        return dequantize_rows(
            packed_q, mn, scale,
            self.nbits, float(alpha), self.q_group_size, self.dtype,
            amax_eps=self.amax_eps, _cuda_ok=cuda_ok,
        )

    # ------------------------------------------------------------------
    # Key block quantise / dequantise
    # ------------------------------------------------------------------

    def _quantize_key_block(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, nh, sl, hd = tensor.shape
        nsg = sl // self.q_group_size
        grouped = tensor.reshape(bs, nh, nsg, self.q_group_size, hd)
        grouped = grouped.permute(0, 1, 2, 4, 3).contiguous()
        flat = grouped.reshape(-1, self.q_group_size)
        pq, mn, sc = self._quantize_rows(flat, "key")
        qb = self._q_bytes()
        return (
            pq.reshape(bs, nh, nsg, hd, qb),
            mn.reshape(bs, nh, nsg, hd, 1),
            sc.reshape(bs, nh, nsg, hd, 1),
        )

    def _quantize_value_block(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, nh, sl, hd = tensor.shape
        if self.v_head_dim_padded != hd:
            tensor = F.pad(tensor, (0, self.v_head_dim_padded - hd), value=0)
        grouped = tensor.reshape(bs, nh, sl, self.v_num_groups, self.q_group_size)
        flat = grouped.reshape(-1, self.q_group_size)
        pq, mn, sc = self._quantize_rows(flat, "value")
        qb = self._q_bytes()
        return (
            pq.reshape(bs, nh, sl, self.v_num_groups, qb),
            mn.reshape(bs, nh, sl, self.v_num_groups, 1),
            sc.reshape(bs, nh, sl, self.v_num_groups, 1),
        )

    def _dequantize_key_prefix(self) -> torch.Tensor:
        if self._packed_key_q is None:
            return torch.empty((self.keys.shape[0], self.keys.shape[1], 0, self.k_head_dim),
                               dtype=self.dtype, device=self.device)
        bs, nh, nsg, hd, _ = self._packed_key_q.shape
        flat = self._packed_key_q.reshape(-1, self._packed_key_q.shape[-1])
        recon = self._dequantize_rows(flat, self._key_mn.reshape(-1, 1), self._key_scale.reshape(-1, 1), "key")
        recon = recon.reshape(bs, nh, nsg, hd, self.q_group_size)
        return recon.permute(0, 1, 2, 4, 3).reshape(bs, nh, nsg * self.q_group_size, hd)

    def _dequantize_value_prefix(self) -> torch.Tensor:
        if self._packed_value_q is None:
            return torch.empty((self.values.shape[0], self.values.shape[1], 0, self.v_head_dim),
                               dtype=self.dtype, device=self.device)
        qt = self._packed_value_q.shape[2]
        flat = self._packed_value_q[:, :, :qt, ...].reshape(-1, self._packed_value_q.shape[-1])
        recon = self._dequantize_rows(flat, self._value_mn[:, :, :qt, ...].reshape(-1, 1),
                                       self._value_scale[:, :, :qt, ...].reshape(-1, 1), "value")
        bs, nh = self._packed_value_q.shape[:2]
        return recon.reshape(bs, nh, qt, self.v_num_groups * self.q_group_size)[..., :self.v_head_dim]

    def _dequantize_key_range(self, start_group: int, end_group: int) -> torch.Tensor:
        if self._packed_key_q is None or start_group >= end_group:
            return torch.empty((self.keys.shape[0], self.keys.shape[1], 0, self.k_head_dim),
                               dtype=self.dtype, device=self.device)
        pq   = self._packed_key_q[:, :, start_group:end_group, ...].contiguous()
        mn   = self._key_mn[:, :, start_group:end_group, ...].contiguous()
        sc   = self._key_scale[:, :, start_group:end_group, ...].contiguous()
        bs, nh, nsg, hd, _ = pq.shape
        recon = self._dequantize_rows(pq.reshape(-1, pq.shape[-1]), mn.reshape(-1, 1), sc.reshape(-1, 1), "key")
        recon = recon.reshape(bs, nh, nsg, hd, self.q_group_size)
        return recon.permute(0, 1, 2, 4, 3).reshape(bs, nh, nsg * self.q_group_size, hd)

    def _dequantize_value_range(self, start_token: int, end_token: int) -> torch.Tensor:
        if self._packed_value_q is None or start_token >= end_token:
            return torch.empty((self.values.shape[0], self.values.shape[1], 0, self.v_head_dim),
                               dtype=self.dtype, device=self.device)
        pq  = self._packed_value_q[:, :, start_token:end_token, ...].contiguous()
        mn  = self._value_mn[:, :, start_token:end_token, ...].contiguous()
        sc  = self._value_scale[:, :, start_token:end_token, ...].contiguous()
        recon = self._dequantize_rows(pq.reshape(-1, pq.shape[-1]), mn.reshape(-1, 1), sc.reshape(-1, 1), "value")
        bs, nh, qt = pq.shape[:3]
        return recon.reshape(bs, nh, qt, self.v_num_groups * self.q_group_size)[..., :self.v_head_dim]

    # ------------------------------------------------------------------
    # Append helpers
    # ------------------------------------------------------------------

    def _append_key_quantized(self, pq, mn, sc) -> None:
        if self._packed_key_q is None:
            self._packed_key_q, self._key_mn, self._key_scale = pq, mn, sc
        else:
            self._packed_key_q = torch.cat([self._packed_key_q, pq], dim=2)
            self._key_mn       = torch.cat([self._key_mn,   mn], dim=2)
            self._key_scale    = torch.cat([self._key_scale, sc], dim=2)
        self._key_group_count += pq.shape[2]

    def _append_value_quantized(self, pq, mn, sc) -> None:
        if self._packed_value_q is None:
            self._packed_value_q, self._value_mn, self._value_scale = pq, mn, sc
        else:
            self._packed_value_q = torch.cat([self._packed_value_q, pq], dim=2)
            self._value_mn       = torch.cat([self._value_mn,   mn], dim=2)
            self._value_scale    = torch.cat([self._value_scale, sc], dim=2)
        self._value_storage_len = self._packed_value_q.shape[2]

    def compress_residual_prefix(self, *, decode_flush: bool = False) -> None:
        """Compress the dense residual prefix, preserving the configured tail.

        This is used after dense prefill so the prefill attention can consume
        normal full-precision KV, then the persistent cache is converted to the
        same quantized-prefix/residual-tail layout used by decode.
        """
        if not self.is_initialized or self.keys is None or self.values is None:
            return
        key_excess = max(self.keys.shape[2] - self.residual_length, 0)
        key_flush_unit = self.q_group_size
        if decode_flush and not (self.cfg.logq_compatible_layout or self.cfg.decode_flush_by_group):
            key_flush_unit = self.q_group_size * 4
        key_flush_len = (key_excess // key_flush_unit) * key_flush_unit
        if key_flush_len > 0:
            key_block = self.keys[:, :, :key_flush_len, :].contiguous()
            pq, mn, sc = self._quantize_key_block(key_block)
            self._append_key_quantized(pq, mn, sc)
            self._drop_residual_prefix(key_flush_len, 0)

        if self.cfg.logq_compatible_layout:
            # Legacy LogQ flushes values independently by token count, so the
            # quantized value prefix can be longer than the quantized key prefix
            # when (seq_len - residual_length) is not divisible by q_group_size.
            val_flush_len = max(self.values.shape[2] - self.residual_length, 0)
        else:
            # MuZero's default keeps compressed K/V prefix boundaries aligned.
            val_flush_len = min(key_flush_len, self.values.shape[2])
        if val_flush_len > 0:
            val_block = self.values[:, :, :val_flush_len, :].contiguous()
            pq, mn, sc = self._quantize_value_block(val_block)
            self._append_value_quantized(pq, mn, sc)
            self._drop_residual_prefix(0, val_flush_len)

    # ------------------------------------------------------------------
    # Primary update (called every generation step)
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append key/value states, flush prefix to quantised storage, return KV for attention.

        Phase-3 optimisation: materialize_full_kv defaults to False.
        During decoding the caller (attention hook) handles the quantised
        prefix via fused kernels — we only return the residual tail here.
        Set materialize_full_kv=True explicitly for prefill-only paths or
        beam search where the full KV tensor is required.
        """
        # Default to the safe HF-compatible path. Patched decode forwards pass
        # materialize_full_kv=False explicitly to enable fused-prefix attention.
        materialize_full_kv = True if cache_kwargs is None else cache_kwargs.get("materialize_full_kv", True)
        defer_quantize = False if cache_kwargs is None else cache_kwargs.get("defer_quantize", False)

        is_first = not self.is_initialized
        if is_first:
            self.lazy_initialization(key_states, value_states)

        # Append to dense residual buffers. Decode uses preallocated storage to
        # avoid per-token torch.cat overhead in every hooked attention layer.
        self._append_dense(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]

        raw_keys   = self.keys
        raw_values = self.values

        if not defer_quantize:
            # During decode, batch flushes avoid tiny quantization launches in
            # the timed hot path. During prefill, callers can defer this until
            # after dense attention to avoid dense+quantized cache overlap.
            self.compress_residual_prefix(decode_flush=key_states.shape[-2] == 1)

        # First update: return raw (before quantisation) for prefill attention
        if is_first:
            return raw_keys, raw_values

        # Decode hot path: return only the residual; the fused kernel handles the prefix.
        if not materialize_full_kv:
            return self.keys, self.values

        if self._packed_key_q is None and self._packed_value_q is None:
            return self.keys, self.values

        # Full materialise path (beam search, explicit request)
        return (
            torch.cat([self._dequantize_key_prefix(), self.keys], dim=2),
            torch.cat([self._dequantize_value_prefix(), self.values], dim=2),
        )

    # ------------------------------------------------------------------
    # Attention state (passed to the custom attention hook)
    # ------------------------------------------------------------------

    def get_attention_state(self) -> dict[str, Any] | None:
        if not self.is_initialized:
            return None
        return {
            "backend": "mu_zero",
            "layer":   self,
            "key_prefix_length":   0 if self._packed_key_q is None else self._packed_key_q.shape[2] * self.q_group_size,
            "value_prefix_length": 0 if self._packed_value_q is None else self._packed_value_q.shape[2],
        }

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int | torch.Tensor) -> tuple[int, int]:
        if isinstance(query_length, torch.Tensor):
            query_length = int(query_length.numel())
        return self.get_seq_length() + query_length, 0

    def get_max_cache_shape(self) -> int:
        return -1

    # ------------------------------------------------------------------
    # Beam-search / cache management
    # ------------------------------------------------------------------

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = max(0, self.get_seq_length() + max_length)
        if max_length >= self.get_seq_length():
            return
        full_k = torch.cat([self._dequantize_key_prefix(), self.keys], dim=2)[..., :max_length, :]
        full_v = torch.cat([self._dequantize_value_prefix(), self.values], dim=2)[..., :max_length, :]
        self.reset()
        if max_length > 0:
            self.lazy_initialization(full_k[:, :, :1, :], full_v[:, :, :1, :])
            self.keys   = full_k
            self.values = full_v
            self.cumulative_length = max_length

    def batch_repeat_interleave(self, repeats: int) -> None:
        for attr in ["keys", "values", "_packed_key_q", "_key_mn", "_key_scale",
                     "_packed_value_q", "_value_mn", "_value_scale"]:
            t = getattr(self, attr, None)
            if t is not None:
                setattr(self, attr, t.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for attr in ["keys", "values", "_packed_key_q", "_key_mn", "_key_scale",
                     "_packed_value_q", "_value_mn", "_value_scale"]:
            t = getattr(self, attr, None)
            if t is not None:
                setattr(self, attr, t[indices, ...])

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx.to(self.device))

    def offload(self) -> None:
        if not self.is_initialized:
            return
        for attr in ["keys", "values", "_packed_key_q", "_key_mn", "_key_scale",
                     "_packed_value_q", "_value_mn", "_value_scale"]:
            t = getattr(self, attr, None)
            if t is not None:
                setattr(self, attr, t.to("cpu", non_blocking=True))

    def prefetch(self) -> None:
        if not self.is_initialized:
            return
        for attr in ["keys", "values", "_packed_key_q", "_key_mn", "_key_scale",
                     "_packed_value_q", "_value_mn", "_value_scale"]:
            t = getattr(self, attr, None)
            if t is not None and t.device != self.device:
                setattr(self, attr, t.to(self.device, non_blocking=True))

    def reset(self) -> None:
        self.is_initialized    = False
        self.cumulative_length = 0
        self._cuda_ok          = None
        self._triton_ok        = None
        self._key_group_count  = 0
        self._value_storage_len = 0
        self._packed_key_q     = None
        self._key_mn           = None
        self._key_scale        = None
        self._packed_value_q   = None
        self._value_mn         = None
        self._value_scale      = None
        # Preserve pre-calibrated alpha from config; only clear runtime-selected ones
        self._selected_alpha_key   = self.cfg.alpha_key
        self._selected_alpha_value = self.cfg.alpha_value
        self.keys   = None
        self.values = None
        self._key_buffer = None
        self._value_buffer = None
        self._buffer_capacity = 0

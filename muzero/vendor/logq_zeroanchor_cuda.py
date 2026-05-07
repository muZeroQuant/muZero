import os
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


_ZEROANCHOR_DECODE_STATS = defaultdict(int)
_ZEROANCHOR_PROFILE_ENABLED = os.environ.get("LOGQ_ZEROANCHOR_PROFILE", "0") == "1"


def reset_zeroanchor_decode_stats() -> None:
    _ZEROANCHOR_DECODE_STATS.clear()


def get_zeroanchor_decode_stats() -> dict[str, int]:
    return dict(_ZEROANCHOR_DECODE_STATS)


def _start_zeroanchor_timer(device: torch.device | None) -> tuple[object, object] | None:
    if not _ZEROANCHOR_PROFILE_ENABLED:
        return None
    if device is not None and device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_event, end_event
    return time.perf_counter(), None


def _finish_zeroanchor_timer(timer_state: tuple[object, object] | None, stat_prefix: str) -> None:
    if timer_state is None:
        return
    start_state, end_state = timer_state
    if end_state is not None:
        end_state.record()
        end_state.synchronize()
        elapsed_ms = float(start_state.elapsed_time(end_state))
    else:
        elapsed_ms = float((time.perf_counter() - start_state) * 1000.0)
    _ZEROANCHOR_DECODE_STATS[f"{stat_prefix}_ms"] += elapsed_ms
    _ZEROANCHOR_DECODE_STATS[f"{stat_prefix}_calls"] += 1


_SRC_DIR = Path(__file__).resolve().parent / "csrc"



@lru_cache(maxsize=1)
def load_logq_zeroanchor_cuda_extension():
    from torch.utils.cpp_extension import load

    if os.environ.get("TORCH_CUDA_ARCH_LIST") is None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    return load(
        name="muzero_logq_zeroanchor_cuda_ext_v11",
        sources=[
            str(_SRC_DIR / "logq_zeroanchor_bindings.cpp"),
            str(_SRC_DIR / "logq_zeroanchor_kernels.cu"),
        ],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def logq_zeroanchor_quantize_cuda(grouped, packed_q, mn, scale, bits, alpha, eps):
    ext = load_logq_zeroanchor_cuda_extension()
    return ext.quantize(grouped, packed_q, mn, scale, bits, alpha, eps)


def logq_zeroanchor_dequantize_cuda(packed_q, mn, scale, output, bits, alpha):
    ext = load_logq_zeroanchor_cuda_extension()
    return ext.dequantize(packed_q, mn, scale, output, bits, alpha)


def logq_zeroanchor_qk_cuda(query, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, scaling):
    ext = load_logq_zeroanchor_cuda_extension()
    return ext.qk(query, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, scaling)


def logq_zeroanchor_av_cuda(attn_weights, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, group_size):
    ext = load_logq_zeroanchor_cuda_extension()
    return ext.av(attn_weights, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, group_size)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def zeroanchor_decode_attention_forward(
    module,
    query: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    kv_cache_state: dict,
    *,
    key_chunk_groups: int = 32,
    value_chunk_tokens: int = 256,
):
    layer = kv_cache_state.get("layer")
    if layer is None:
        _ZEROANCHOR_DECODE_STATS["missing_layer"] += 1
        return None
    if query.shape[-2] != 1:
        _ZEROANCHOR_DECODE_STATS["non_decode_query"] += 1
        return None
    _ZEROANCHOR_DECODE_STATS["calls"] += 1
    overall_timer = _start_zeroanchor_timer(query.device)

    key_prefix_length = int(kv_cache_state.get("key_prefix_length", 0))
    value_prefix_length = int(kv_cache_state.get("value_prefix_length", 0))
    can_use_fused_kernels = (
        query.is_cuda
        and query.dtype in {torch.float16, torch.bfloat16}
        and query.shape[0] > 0
        and layer._selected_alpha_key is not None
        and layer._selected_alpha_value is not None
    )
    if not can_use_fused_kernels:
        _ZEROANCHOR_DECODE_STATS["initial_non_fused"] += 1

    logits_parts = []
    key_cursor = 0
    if key_prefix_length > 0:
        if can_use_fused_kernels:
            try:
                qk_timer = _start_zeroanchor_timer(query.device)
                prefix_logits = torch.empty(
                    (query.shape[0], query.shape[1], key_prefix_length), dtype=query.dtype, device=query.device
                )
                logq_zeroanchor_qk_cuda(
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
                _finish_zeroanchor_timer(qk_timer, "fused_qk")
                prefix_logits = prefix_logits.unsqueeze(2)
                if attention_mask is not None:
                    prefix_logits = prefix_logits + attention_mask[..., :key_prefix_length]
                logits_parts.append(prefix_logits)
                key_cursor += key_prefix_length
            except Exception:
                can_use_fused_kernels = False
                _ZEROANCHOR_DECODE_STATS["fused_qk_fallback"] += 1
            else:
                _ZEROANCHOR_DECODE_STATS["fused_qk_success"] += 1

        if not can_use_fused_kernels:
            _ZEROANCHOR_DECODE_STATS["key_dequant_fallback"] += 1
            total_groups = key_prefix_length // layer.q_group_size
            for start_group in range(0, total_groups, key_chunk_groups):
                end_group = min(start_group + key_chunk_groups, total_groups)
                dequant_timer = _start_zeroanchor_timer(query.device)
                key_chunk = layer._dequantize_key_range(start_group, end_group)
                _finish_zeroanchor_timer(dequant_timer, "fallback_key_dequant")
                key_chunk = _repeat_kv(key_chunk, module.num_key_value_groups)
                matmul_timer = _start_zeroanchor_timer(query.device)
                chunk_logits = torch.matmul(query, key_chunk.transpose(2, 3)) * scaling
                _finish_zeroanchor_timer(matmul_timer, "fallback_qk_matmul")
                chunk_length = key_chunk.shape[-2]
                if attention_mask is not None:
                    chunk_logits = chunk_logits + attention_mask[..., key_cursor : key_cursor + chunk_length]
                logits_parts.append(chunk_logits)
                key_cursor += chunk_length

    dense_key_states = _repeat_kv(layer.keys, module.num_key_value_groups)
    dense_qk_timer = _start_zeroanchor_timer(query.device)
    dense_logits = torch.matmul(query, dense_key_states.transpose(2, 3)) * scaling
    _finish_zeroanchor_timer(dense_qk_timer, "dense_qk")
    if attention_mask is not None:
        dense_logits = dense_logits + attention_mask[..., key_cursor : key_cursor + dense_key_states.shape[-2]]
    logits_parts.append(dense_logits)

    softmax_timer = _start_zeroanchor_timer(query.device)
    attn_weights = F.softmax(torch.cat(logits_parts, dim=-1), dim=-1, dtype=torch.float32).to(query.dtype)
    _finish_zeroanchor_timer(softmax_timer, "softmax")

    attn_output = None
    value_cursor = 0
    if value_prefix_length > 0:
        if can_use_fused_kernels:
            try:
                av_timer = _start_zeroanchor_timer(query.device)
                prefix_output = torch.empty(
                    (query.shape[0], query.shape[1], layer.v_head_dim), dtype=query.dtype, device=query.device
                )
                logq_zeroanchor_av_cuda(
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
                _finish_zeroanchor_timer(av_timer, "fused_av")
                attn_output = prefix_output.unsqueeze(2)
                value_cursor += value_prefix_length
            except Exception:
                can_use_fused_kernels = False
                attn_output = None
                value_cursor = 0
                _ZEROANCHOR_DECODE_STATS["fused_av_fallback"] += 1
            else:
                _ZEROANCHOR_DECODE_STATS["fused_av_success"] += 1

        if not can_use_fused_kernels:
            _ZEROANCHOR_DECODE_STATS["value_dequant_fallback"] += 1
            for start_token in range(0, value_prefix_length, value_chunk_tokens):
                end_token = min(start_token + value_chunk_tokens, value_prefix_length)
                dequant_timer = _start_zeroanchor_timer(query.device)
                value_chunk = layer._dequantize_value_range(start_token, end_token)
                _finish_zeroanchor_timer(dequant_timer, "fallback_value_dequant")
                value_chunk = _repeat_kv(value_chunk, module.num_key_value_groups)
                weight_chunk = attn_weights[..., value_cursor : value_cursor + value_chunk.shape[-2]]
                matmul_timer = _start_zeroanchor_timer(query.device)
                chunk_output = torch.matmul(weight_chunk, value_chunk)
                _finish_zeroanchor_timer(matmul_timer, "fallback_av_matmul")
                attn_output = chunk_output if attn_output is None else attn_output + chunk_output
                value_cursor += value_chunk.shape[-2]

    dense_value_states = _repeat_kv(layer.values, module.num_key_value_groups)
    dense_weight_chunk = attn_weights[..., value_cursor : value_cursor + dense_value_states.shape[-2]]
    dense_av_timer = _start_zeroanchor_timer(query.device)
    dense_output = torch.matmul(dense_weight_chunk, dense_value_states)
    _finish_zeroanchor_timer(dense_av_timer, "dense_av")
    attn_output = dense_output if attn_output is None else attn_output + dense_output
    _ZEROANCHOR_DECODE_STATS["returns"] += 1
    _finish_zeroanchor_timer(overall_timer, "decode_total")
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def zeroanchor_decode_attention_forward_opt(
    module,
    query: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    kv_cache_state: dict,
    *,
    key_chunk_groups: int = 32,
    value_chunk_tokens: int = 256,
):
    layer = kv_cache_state.get("layer")
    if layer is None:
        return None
    key_prefix_length = int(kv_cache_state.get("key_prefix_length", 0))
    use_dense_fallback = query.shape[-2] == 1 and key_prefix_length >= 2048 and getattr(module, "head_dim", 0) >= 128
    if not use_dense_fallback:
        return zeroanchor_decode_attention_forward(
            module,
            query,
            attention_mask,
            scaling,
            kv_cache_state,
            key_chunk_groups=key_chunk_groups,
            value_chunk_tokens=value_chunk_tokens,
        )

    logits_parts = []
    key_cursor = 0
    if key_prefix_length > 0:
        total_groups = key_prefix_length // layer.q_group_size
        for start_group in range(0, total_groups, key_chunk_groups):
            end_group = min(start_group + key_chunk_groups, total_groups)
            key_chunk = layer._dequantize_key_range(start_group, end_group)
            key_chunk = _repeat_kv(key_chunk, module.num_key_value_groups)
            chunk_logits = torch.matmul(query, key_chunk.transpose(2, 3)) * scaling
            chunk_length = key_chunk.shape[-2]
            if attention_mask is not None:
                chunk_logits = chunk_logits + attention_mask[..., key_cursor : key_cursor + chunk_length]
            logits_parts.append(chunk_logits)
            key_cursor += chunk_length

    dense_key_states = _repeat_kv(layer.keys, module.num_key_value_groups)
    dense_logits = torch.matmul(query, dense_key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        dense_logits = dense_logits + attention_mask[..., key_cursor : key_cursor + dense_key_states.shape[-2]]
    logits_parts.append(dense_logits)
    attn_weights = F.softmax(torch.cat(logits_parts, dim=-1), dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = None
    value_cursor = 0
    value_prefix_length = int(kv_cache_state.get("value_prefix_length", 0))
    for start_token in range(0, value_prefix_length, value_chunk_tokens):
        end_token = min(start_token + value_chunk_tokens, value_prefix_length)
        value_chunk = layer._dequantize_value_range(start_token, end_token)
        value_chunk = _repeat_kv(value_chunk, module.num_key_value_groups)
        weight_chunk = attn_weights[..., value_cursor : value_cursor + value_chunk.shape[-2]]
        chunk_output = torch.matmul(weight_chunk, value_chunk)
        attn_output = chunk_output if attn_output is None else attn_output + chunk_output
        value_cursor += value_chunk.shape[-2]

    dense_value_states = _repeat_kv(layer.values, module.num_key_value_groups)
    dense_weight_chunk = attn_weights[..., value_cursor : value_cursor + dense_value_states.shape[-2]]
    dense_output = torch.matmul(dense_weight_chunk, dense_value_states)
    attn_output = dense_output if attn_output is None else attn_output + dense_output
    return attn_output.transpose(1, 2).contiguous(), attn_weights

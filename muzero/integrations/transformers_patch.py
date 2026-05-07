"""HuggingFace transformers integration for μ-Zero.

Usage::

    import muzero
    muzero.patch_transformers()   # register mu_zero_*bit cache implementations

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="bfloat16").cuda()
    out = model.generate(
        **inputs,
        cache_implementation="mu_zero_4bit",
        cache_config={"q_group_size": 64, "residual_length": 128},
        max_new_tokens=512,
    )

How it works
------------
HuggingFace `transformers` builds the per-layer cache object from
`CACHE_CLASS_REGISTRY[cache_implementation]`.  We register a thin wrapper
class for each bit-width that:

1. Wraps `MuZeroCacheLayer` as a `DynamicCache` subclass (so it satisfies
   the `isinstance(past_key_values, DynamicCache)` checks).
2. Overrides `update()` to call MuZeroCacheLayer.update() per layer.
3. Registers a custom attention hook via `get_seq_length()` / `get_attention_state()`.

The attention hook (`mu_zero_decode_attention_forward`) is installed once
during `patch_transformers()` and runs before the model's own sdpa call.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from typing import Any, Optional

import torch

from ..config import MuZeroConfig
from ..cache.layer import MuZeroCacheLayer
from ..cache.attention import mu_zero_decode_attention_forward

try:
    from transformers.cache_utils import Cache as _HFCache
except Exception:  # transformers is an install dependency, but keep import-time failure friendly.
    _HFCache = object


_MU_ZERO_BITS = (1, 2, 3, 4, 7, 8)


# ---------------------------------------------------------------------------
# Multi-layer cache wrapper (matches HF DynamicCache interface)
# ---------------------------------------------------------------------------

class MuZeroCache(_HFCache):
    """Multi-layer μ-Zero cache.

    Indexed by layer index like DynamicCache: cache[layer_idx] returns
    (key, value) via __getitem__ for compatibility, but the real path is
    through update() / get_attention_state().
    """

    def __init__(self, cfg: MuZeroConfig):
        self.cfg = cfg
        self._seen_tokens = 0
        self.defer_prefill_quantization = False
        if _HFCache is object:
            self.layers = []
            self.layer_class_to_replicate = None
            self.offloading = False
        else:
            super().__init__(layer_class_to_replicate=lambda: MuZeroCacheLayer(self.cfg))

    @property
    def _layers(self) -> dict[int, MuZeroCacheLayer]:
        return {idx: layer for idx, layer in enumerate(self.layers)}

    def _ensure_layer(self, layer_idx: int) -> MuZeroCacheLayer:
        while len(self.layers) <= layer_idx:
            self.layers.append(MuZeroCacheLayer(self.cfg))
        return self.layers[layer_idx]

    # ------------------------------------------------------------------
    # DynamicCache-compatible interface
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        layer = self._ensure_layer(layer_idx)
        if cache_kwargs is None:
            cache_kwargs = {"materialize_full_kv": True}
        elif "materialize_full_kv" not in cache_kwargs:
            cache_kwargs = {**cache_kwargs, "materialize_full_kv": True}
        return layer.update(key_states, value_states, cache_kwargs)

    def finalize_prefill_quantization(self) -> None:
        for layer in self.layers:
            layer.compress_residual_prefix(decode_flush=False)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length() if layer_idx < len(self.layers) else 0

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_attention_state(self, layer_idx: int) -> dict[str, Any] | None:
        return self.layers[layer_idx].get_attention_state() if layer_idx < len(self.layers) else None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self.layers:
            layer.batch_select_indices(indices)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self._seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, layer_idx: int):
        # Compatibility with code that does cache[layer_idx] -> (k, v)
        if layer_idx >= len(self.layers):
            return [], []
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            return [], []
        k = torch.cat([layer._dequantize_key_prefix(), layer.keys], dim=2) if layer._packed_key_q is not None else layer.keys
        v = torch.cat([layer._dequantize_value_prefix(), layer.values], dim=2) if layer._packed_value_q is not None else layer.values
        return [k], [v]


# ---------------------------------------------------------------------------
# Per-bits wrapper factories
# ---------------------------------------------------------------------------

def _make_mu_zero_cache_class(bits: int):
    """Return a MuZeroCache subclass with fixed bit-width, callable with HF kwargs."""

    class _BitsCache(MuZeroCache):
        _bits = bits

        def __init__(self, **kwargs):
            # Accept HF-style kwargs: q_group_size, residual_length, amax_dtype, …
            if "quant_backend" in kwargs and "backend" not in kwargs:
                kwargs = {**kwargs, "backend": kwargs["quant_backend"]}
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in MuZeroConfig.__dataclass_fields__}
            cfg_kwargs["bits"] = bits
            super().__init__(MuZeroConfig(**cfg_kwargs))

    _BitsCache.__name__ = f"MuZero{bits}bitCache"
    _BitsCache.__qualname__ = f"MuZero{bits}bitCache"
    return _BitsCache


# ---------------------------------------------------------------------------
# patch_transformers()
# ---------------------------------------------------------------------------

_PATCHED = False


def patch_transformers() -> None:
    """Register μ-Zero cache classes in the HuggingFace transformers registry.

    After calling this, pass ``cache_implementation="mu_zero_4bit"``
    (or 1/2/3/7/8bit) to ``model.generate()``.

    Also installs the fused-kernel attention hook via the transformers
    attention-forward hook mechanism if available.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        import transformers.cache_utils as cu
        registry = getattr(cu, "CACHE_CLASS_REGISTRY", None)
        if registry is None:
            # Older transformers: attach directly
            registry = {}
            cu.CACHE_CLASS_REGISTRY = registry
    except ImportError:
        raise ImportError("transformers is not installed.  pip install transformers>=4.40")

    for bits in [1, 2, 3, 4, 7, 8]:
        registry[f"mu_zero_{bits}bit"] = _make_mu_zero_cache_class(bits)

    # Also extend the generation-config allow-list so generate() accepts mu_zero_*bit.
    _patch_generation_config_validation()

    # Make legacy zero-anchor modules available without a patched Transformers checkout.
    _install_legacy_zeroanchor_aliases()

    # Build MuZeroCache from generate(cache_implementation="mu_zero_*bit").
    _patch_generation_cache_preparation()

    # Install attention fast-path hooks for Llama/Qwen3 eager decode.
    _install_attention_hook()

    _PATCHED = True
    print(f"[μ-Zero] Registered mu_zero_{{1,2,3,4,7,8}}bit in transformers CACHE_CLASS_REGISTRY.")


def _install_legacy_zeroanchor_aliases() -> None:
    """Expose repo-local legacy LogQ modules under the names used by old patches."""
    try:
        from ..kernels.legacy_logq import cuda as logq_zeroanchor_cuda
        from ..kernels.legacy_logq import triton as logq_zeroanchor_triton

        sys.modules.setdefault("transformers.integrations.logq_zeroanchor_cuda", logq_zeroanchor_cuda)
        sys.modules.setdefault("transformers.integrations.logq_zeroanchor_triton", logq_zeroanchor_triton)
    except Exception as exc:
        import warnings

        warnings.warn(f"[μ-Zero] Could not install legacy zero-anchor module aliases: {exc}")


def _patch_generation_config_validation() -> None:
    """Extend the generation-config cache_implementation allow-list to include mu_zero_*bit."""
    try:
        import transformers.generation.configuration_utils as gcc
        mu_keys = tuple(f"mu_zero_{b}bit" for b in (1, 2, 3, 4, 7, 8))

        # Patch the module-level tuple so GenerationConfig.validate() accepts mu_zero_*bit.
        if hasattr(gcc, "ALL_CACHE_IMPLEMENTATIONS"):
            existing = gcc.ALL_CACHE_IMPLEMENTATIONS
            if mu_keys[0] not in existing:
                gcc.ALL_CACHE_IMPLEMENTATIONS = existing + mu_keys

        # Also patch DYNAMIC_CACHE_IMPLEMENTATIONS if present (some HF versions use it).
        if hasattr(gcc, "DYNAMIC_CACHE_IMPLEMENTATIONS"):
            existing = gcc.DYNAMIC_CACHE_IMPLEMENTATIONS
            if mu_keys[0] not in existing:
                gcc.DYNAMIC_CACHE_IMPLEMENTATIONS = existing + mu_keys

        # Wire in the cache class registry so _get_cache() can instantiate our classes.
        _patch_get_cache(mu_keys)

    except Exception as e:
        import warnings
        warnings.warn(f"[μ-Zero] Could not extend generation config allow-list: {e}")


def _parse_mu_zero_cache_impl(cache_implementation: str | None) -> int | None:
    if not isinstance(cache_implementation, str):
        return None
    for bits in _MU_ZERO_BITS:
        if cache_implementation == f"mu_zero_{bits}bit":
            return bits
    return None


def _parse_legacy_zeroanchor_backend(cache_implementation: str | None, cache_config: dict[str, Any]) -> int | None:
    if cache_implementation != "quantized":
        return None
    backend = cache_config.get("backend")
    if not isinstance(backend, str):
        return None
    for bits in _MU_ZERO_BITS:
        if backend in {f"logq{bits}_zeroanchor_stat_cuda", f"logq{bits}_zeroanchor_stat_cuda_opt"}:
            return bits
    return None


def _patch_generation_cache_preparation() -> None:
    try:
        import transformers.generation.utils as gu
    except Exception:
        return

    mixin = getattr(gu, "GenerationMixin", None)
    if mixin is None or hasattr(mixin, "_muzero_original_prepare_cache_for_generation"):
        return
    original = getattr(mixin, "_prepare_cache_for_generation", None)
    if original is None:
        return

    def _patched_prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs,
        generation_mode,
        batch_size,
        max_cache_length,
    ):
        cache_implementation = getattr(generation_config, "cache_implementation", None)
        cache_config = dict(getattr(generation_config, "cache_config", None) or {})
        bits = _parse_mu_zero_cache_impl(cache_implementation)
        legacy_bits = _parse_legacy_zeroanchor_backend(cache_implementation, cache_config)
        if bits is None and legacy_bits is None:
            return original(self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length)
        if getattr(self.config, "is_encoder_decoder", False):
            raise ValueError("mu_zero cache currently supports decoder-only models")
        bits = bits if bits is not None else legacy_bits
        cache_config.pop("config", None)
        if legacy_bits is not None:
            cache_config.pop("backend", None)
        if "quant_backend" in cache_config and "backend" not in cache_config:
            cache_config["backend"] = cache_config.pop("quant_backend")
        cache_config["bits"] = bits
        cache = MuZeroCache(MuZeroConfig.from_dict(cache_config))
        cache.defer_prefill_quantization = True
        model_kwargs["past_key_values"] = cache
        return None

    mixin._muzero_original_prepare_cache_for_generation = original
    mixin._prepare_cache_for_generation = _patched_prepare_cache_for_generation


def _patch_get_cache(mu_keys: tuple) -> None:
    """Patch transformers._get_cache (or equivalent) to handle mu_zero_*bit strings."""
    try:
        import transformers.cache_utils as cu

        original_get_cache = getattr(cu, "_get_cache", None)
        if original_get_cache is None:
            return

        registry = getattr(cu, "CACHE_CLASS_REGISTRY", {})

        def _patched_get_cache(cache_implementation: str, model, *args, **kwargs):
            if cache_implementation in registry:
                # Our custom cache: instantiate from registry with cache_config kwargs
                CacheClass = registry[cache_implementation]
                cache_config = kwargs.pop("cache_config", {}) or {}
                return CacheClass(**cache_config)
            return original_get_cache(cache_implementation, model, *args, **kwargs)

        cu._get_cache = _patched_get_cache
    except Exception:
        pass


def _install_attention_hook() -> None:
    """Wire mu_zero_decode_attention_forward into transformers' attention hook system.

    transformers >= 4.46 exposes `_attn_implementation_autoset` and per-module
    attention hooks.  We register a pre-forward hook on all attention modules
    that have a cache supporting mu_zero.
    """
    for module_name in (
        "transformers.models.llama.modeling_llama",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.qwen3.modular_qwen3",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        _patch_eager_attention(module)
        if module_name.endswith("modeling_llama"):
            _patch_llama_forward_if_needed(module)
        else:
            _patch_qwen3_forward_if_needed(module)


def _patch_eager_attention(module) -> None:
    original = getattr(module, "eager_attention_forward", None)
    if original is None or getattr(original, "_muzero_patched", False):
        return

    def _muzero_eager_attention_forward(
        attn_module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        kv_cache_state = kwargs.get("kv_cache_state")
        if (
            kv_cache_state is not None
            and kv_cache_state.get("backend") == "mu_zero"
            and not attn_module.training
            and dropout == 0.0
        ):
            fused_result = mu_zero_decode_attention_forward(attn_module, query, attention_mask, scaling, kv_cache_state)
            if fused_result is not None:
                return fused_result
        return original(attn_module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

    _muzero_eager_attention_forward._muzero_patched = True
    _muzero_eager_attention_forward._muzero_original = original
    module.eager_attention_forward = _muzero_eager_attention_forward


def _source_has_mu_zero_backend(forward) -> bool:
    try:
        source = inspect.getsource(forward)
    except Exception:
        return False
    return "mu_zero" in source and "kv_cache_state" in source and "materialize_full_kv" in source


def _is_mu_zero_cache(cache: Any) -> bool:
    return isinstance(cache, MuZeroCache)


def _patch_llama_forward_if_needed(module) -> None:
    cls = getattr(module, "LlamaAttention", None)
    if cls is None or getattr(cls.forward, "_muzero_patched", False) or _source_has_mu_zero_backend(cls.forward):
        return
    original = cls.forward

    def _muzero_llama_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        if position_embeddings is None:
            return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = module.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        use_mu_zero_decode = False

        if past_key_values is not None:
            if (
                _is_mu_zero_cache(past_key_values)
                and bool(getattr(past_key_values, "defer_prefill_quantization", False))
                and query_states.shape[-2] == 1
                and not self.training
            ):
                past_key_values.finalize_prefill_quantization()
                past_key_values.defer_prefill_quantization = False
            use_mu_zero_decode = (
                _is_mu_zero_cache(past_key_values)
                and query_states.shape[-2] == 1
                and not self.training
                and os.environ.get("MUZERO_DISABLE_FUSED_DECODE", "0") != "1"
            )
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "materialize_full_kv": not use_mu_zero_decode,
                "defer_quantize": bool(getattr(past_key_values, "defer_prefill_quantization", False) and query_states.shape[-2] > 1),
            }
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if use_mu_zero_decode:
                kwargs["kv_cache_state"] = past_key_values.get_attention_state(self.layer_idx)

        if use_mu_zero_decode:
            attention_interface = module.eager_attention_forward
        else:
            attention_interface = module.ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, module.eager_attention_forward
            )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    _muzero_llama_forward._muzero_patched = True
    _muzero_llama_forward._muzero_original = original
    cls.forward = _muzero_llama_forward


def _patch_qwen3_forward_if_needed(module) -> None:
    cls = getattr(module, "Qwen3Attention", None)
    if cls is None or getattr(cls.forward, "_muzero_patched", False) or _source_has_mu_zero_backend(cls.forward):
        return
    original = cls.forward

    def _muzero_qwen3_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = module.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        use_mu_zero_decode = False

        if past_key_values is not None:
            if (
                _is_mu_zero_cache(past_key_values)
                and bool(getattr(past_key_values, "defer_prefill_quantization", False))
                and query_states.shape[-2] == 1
                and not self.training
            ):
                past_key_values.finalize_prefill_quantization()
                past_key_values.defer_prefill_quantization = False
            use_mu_zero_decode = (
                _is_mu_zero_cache(past_key_values)
                and query_states.shape[-2] == 1
                and not self.training
                and os.environ.get("MUZERO_DISABLE_FUSED_DECODE", "0") != "1"
            )
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "materialize_full_kv": not use_mu_zero_decode,
                "defer_quantize": bool(getattr(past_key_values, "defer_prefill_quantization", False) and query_states.shape[-2] > 1),
            }
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if use_mu_zero_decode:
                kwargs["kv_cache_state"] = past_key_values.get_attention_state(self.layer_idx)

        if use_mu_zero_decode:
            attention_interface = module.eager_attention_forward
        else:
            attention_interface = module.ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, module.eager_attention_forward
            )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    _muzero_qwen3_forward._muzero_patched = True
    _muzero_qwen3_forward._muzero_original = original
    cls.forward = _muzero_qwen3_forward

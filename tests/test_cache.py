"""Tests for MuZeroCacheLayer — shapes, flush logic, fused vs fallback agreement."""

import pytest
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from muzero.config import MuZeroConfig
from muzero.cache.layer import MuZeroCacheLayer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16


def _make_layer(**kwargs) -> MuZeroCacheLayer:
    defaults = dict(bits=4, q_group_size=64, residual_length=128)
    defaults.update(kwargs)
    cfg = MuZeroConfig(**defaults)
    return MuZeroCacheLayer(cfg)


def _kv(bs, nh, sl, hd) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(bs, nh, sl, hd, device=DEVICE, dtype=DTYPE)
    v = torch.randn(bs, nh, sl, hd, device=DEVICE, dtype=DTYPE)
    return k, v


# ------------------------------------------------------------------
# Shape tests
# ------------------------------------------------------------------

def test_prefill_shapes():
    layer = _make_layer()
    k, v = _kv(1, 8, 512, 128)
    rk, rv = layer.update(k, v, {"materialize_full_kv": True})
    assert rk.shape == k.shape
    assert rv.shape == v.shape


def test_flush_triggers():
    """After feeding > residual_length tokens, quantised prefix should exist."""
    layer = _make_layer(residual_length=128, q_group_size=64)
    k, v  = _kv(1, 8, 512, 128)  # 512 > 128 residual → flush
    layer.update(k, v)
    assert layer._packed_key_q is not None, "Key prefix should be quantised after flush"
    assert layer._packed_value_q is not None


def test_no_flush_below_residual():
    layer = _make_layer(residual_length=128, q_group_size=64)
    k, v  = _kv(1, 8, 64, 128)   # 64 < 128 → no flush
    layer.update(k, v)
    assert layer._packed_key_q is None


def test_cumulative_length():
    layer = _make_layer()
    k, v  = _kv(1, 8, 256, 128)
    layer.update(k, v)
    k2, v2 = _kv(1, 8, 1, 128)
    layer.update(k2, v2)
    assert layer.get_seq_length() == 257


def test_default_materializes_full_kv():
    """Default update() stays safe for unpatched attention paths."""
    layer = _make_layer(residual_length=128, q_group_size=64)
    k, v  = _kv(1, 8, 512, 128)
    layer.update(k, v)
    # Now do a decode step (1 token)
    k2, v2 = _kv(1, 8, 1, 128)
    rk, rv = layer.update(k2, v2)
    assert rk.shape[-2] == 513, f"Expected full KV, got seq_len={rk.shape[-2]}"


def test_explicit_residual_only_decode_path():
    """Patched model forwards request residual-only tensors for fused decode."""
    layer = _make_layer(residual_length=128, q_group_size=64)
    k, v  = _kv(1, 8, 512, 128)
    layer.update(k, v)
    k2, v2 = _kv(1, 8, 1, 128)
    rk, rv = layer.update(k2, v2, {"materialize_full_kv": False})
    assert rk.shape[-2] <= 128 + 1, f"Expected residual only, got seq_len={rk.shape[-2]}"


def test_decode_flush_by_group_keeps_aligned_prefixes():
    layer = _make_layer(residual_length=128, q_group_size=64, decode_flush_by_group=True)
    k, v = _kv(1, 8, 128, 128)
    layer.update(k, v)
    assert layer._packed_key_q is None

    for _ in range(64):
        k1, v1 = _kv(1, 8, 1, 128)
        layer.update(k1, v1, {"materialize_full_kv": False})

    assert layer._packed_key_q is not None
    assert layer._packed_key_q.shape[2] == 1
    assert layer._packed_value_q.shape[2] == 64
    assert layer.keys.shape[2] == 128
    assert layer.values.shape[2] == 128
def test_materialize_full_kv():
    layer = _make_layer(residual_length=128, q_group_size=64)
    k, v  = _kv(1, 8, 512, 128)
    layer.update(k, v, {"materialize_full_kv": True})
    k2, v2 = _kv(1, 8, 1, 128)
    rk, rv = layer.update(k2, v2, {"materialize_full_kv": True})
    assert rk.shape[-2] == 513, f"Expected full KV, got {rk.shape[-2]}"


# ------------------------------------------------------------------
# Correctness: full materialise matches dequant+residual
# ------------------------------------------------------------------

def test_materialized_kv_correctness():
    """materialize_full_kv path must reproduce the same keys as dequant+residual."""
    layer = _make_layer(residual_length=128, q_group_size=64)
    k, v  = _kv(1, 8, 512, 128)
    # First update: quantises the prefix (512 > residual_length=128)
    layer.update(k, v)
    assert layer._packed_key_q is not None, "Prefix should be quantised after flush"

    # Second update (one decode step) with materialize=True: should return full KV
    k2, v2 = _kv(1, 8, 1, 128)
    rk1, rv1 = layer.update(k2, v2, {"materialize_full_kv": True})

    # Reference: manually dequantise prefix + concat residual
    layer2 = _make_layer(residual_length=128, q_group_size=64)
    layer2.update(k, v)
    layer2.update(k2, v2)
    prefix_k = layer2._dequantize_key_prefix()
    full_k   = torch.cat([prefix_k, layer2.keys], dim=2)

    # They should agree within half-precision rounding
    assert full_k.shape == rk1.shape, f"Shape mismatch: {full_k.shape} vs {rk1.shape}"
    diff = (full_k.float() - rk1.float()).abs().max().item()
    assert diff < 0.05, f"Max diff {diff:.4f} too large"


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------

def test_reset_clears_state():
    layer = _make_layer()
    k, v = _kv(1, 8, 512, 128)
    layer.update(k, v)
    layer.reset()
    assert not layer.is_initialized
    assert layer._packed_key_q is None
    assert layer.cumulative_length == 0


# ------------------------------------------------------------------
# Different bit widths
# ------------------------------------------------------------------

@pytest.mark.parametrize("bits", [2, 4, 8])
def test_bits_variants(bits):
    cfg   = MuZeroConfig(bits=bits, q_group_size=64, residual_length=64)
    layer = MuZeroCacheLayer(cfg)
    k, v  = _kv(1, 4, 256, 64)
    rk, rv = layer.update(k, v, {"materialize_full_kv": True})
    assert rk.shape == k.shape

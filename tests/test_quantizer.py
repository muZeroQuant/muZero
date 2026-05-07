"""Tests for muzero.quantizer — round-trip MSE and alpha selection."""

import pytest
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from muzero.quantizer import select_alpha, quantize_rows, dequantize_rows, roundtrip
from muzero.config import MuZeroConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_roundtrip_mse(bits):
    """Round-trip quantisation MSE should be below a per-bit threshold."""
    torch.manual_seed(42)
    flat = torch.randn(256, 64, device=DEVICE, dtype=torch.float32)
    alpha = 2.0
    amax_dtype = torch.float16

    pq, mn, sc = quantize_rows(flat, bits, alpha, 1e-8, amax_dtype)
    recon = dequantize_rows(pq, mn, sc, bits, alpha, 64, torch.float32)

    mse = ((flat - recon) ** 2).mean().item()
    # Thresholds calibrated for standard-normal inputs (worst case for log-companders)
    # Real KV cache tensors are more concentrated → lower actual MSE.
    threshold = {2: 0.35, 4: 0.015, 8: 1e-4}[bits]
    assert mse < threshold, f"bits={bits}: mse={mse:.6f} > threshold={threshold}"


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_packed_shape(bits):
    flat = torch.randn(128, 64, device=DEVICE, dtype=torch.float16)
    pq, mn, sc = quantize_rows(flat, bits, 2.0, 1e-8, torch.float16)
    q_bytes = (64 * bits + 7) // 8
    assert pq.shape == (128, q_bytes)
    assert mn.shape == (128, 1)
    assert sc.shape == (128, 1)


def test_select_alpha_4bit_is_constant():
    """For 4-bit, alpha selection must return 2.0 without any GPU→CPU sync."""
    t = torch.randn(4, 8, 64, 128, device=DEVICE)
    for kind in ["key", "value"]:
        a = select_alpha(t, bits=4, q_group_size=64, kind=kind)
        assert a == 2.0, f"Expected alpha=2.0 for 4-bit {kind}, got {a}"


@pytest.mark.parametrize("bits,kind", [(1, "key"), (2, "value"), (8, "key")])
def test_select_alpha_returns_float(bits, kind):
    t = torch.randn(2, 4, 32, 64, device=DEVICE)
    a = select_alpha(t, bits=bits, q_group_size=32, kind=kind)
    assert isinstance(a, float) and a > 0


def test_roundtrip_helper():
    """roundtrip() should return tensors with same shape and dtype."""
    k = torch.randn(1, 8, 256, 128, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 8, 256, 128, device=DEVICE, dtype=torch.float16)
    rk, rv = roundtrip(k, v, bits=4, q_group_size=64)
    assert rk.shape == k.shape
    assert rv.shape == v.shape
    assert rk.dtype == k.dtype

    cos_k = torch.nn.functional.cosine_similarity(k.reshape(1, -1), rk.reshape(1, -1)).item()
    cos_v = torch.nn.functional.cosine_similarity(v.reshape(1, -1), rv.reshape(1, -1)).item()
    assert cos_k > 0.99, f"Key cosine similarity {cos_k:.4f} too low"
    assert cos_v > 0.99, f"Value cosine similarity {cos_v:.4f} too low"


def test_edge_constant_tensor():
    """Constant tensors should not produce NaN or inf."""
    flat = torch.ones(64, 64, device=DEVICE, dtype=torch.float32)
    pq, mn, sc = quantize_rows(flat, 4, 2.0, 1e-8, torch.float16)
    recon = dequantize_rows(pq, mn, sc, 4, 2.0, 64, torch.float32)
    assert not torch.isnan(recon).any()
    assert not torch.isinf(recon).any()


def test_edge_zero_tensor():
    flat = torch.zeros(64, 64, device=DEVICE, dtype=torch.float32)
    pq, mn, sc = quantize_rows(flat, 4, 2.0, 1e-8, torch.float16)
    recon = dequantize_rows(pq, mn, sc, 4, 2.0, 64, torch.float32)
    assert not torch.isnan(recon).any()

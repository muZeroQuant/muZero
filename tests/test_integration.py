"""Integration tests: patch_transformers() + generate() on a tiny dummy model."""

import pytest
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import muzero


def _make_tiny_model():
    """Build a minimal 2-layer LlamaForCausalLM for CPU testing."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model


def test_patch_transformers_registers_keys():
    """patch_transformers() should register mu_zero_*bit keys."""
    muzero.patch_transformers()
    from transformers.cache_utils import CACHE_CLASS_REGISTRY
    for bits in [2, 4, 8]:
        key = f"mu_zero_{bits}bit"
        assert key in CACHE_CLASS_REGISTRY, f"{key} not registered"


def test_patch_idempotent():
    """Calling patch_transformers() twice should not error."""
    muzero.patch_transformers()
    muzero.patch_transformers()


def test_mu_zero_cache_instantiation():
    from transformers.cache_utils import CACHE_CLASS_REGISTRY
    muzero.patch_transformers()
    CacheClass = CACHE_CLASS_REGISTRY["mu_zero_4bit"]
    cache = CacheClass(q_group_size=64, residual_length=128)
    assert isinstance(cache, muzero.MuZeroCache)


def test_generate_accepts_mu_zero_cache_cpu():
    muzero.patch_transformers()
    model = _make_tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        do_sample=False,
        cache_implementation="mu_zero_4bit",
        cache_config={"q_group_size": 64, "residual_length": 64, "backend": "torch"},
    )
    assert out.shape[1] == input_ids.shape[1] + 2


def test_generate_accepts_legacy_zeroanchor_backend_cpu():
    muzero.patch_transformers()
    model = _make_tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        do_sample=False,
        cache_implementation="quantized",
        cache_config={"backend": "logq4_zeroanchor_stat_cuda", "q_group_size": 64, "residual_length": 64},
    )
    assert out.shape[1] == input_ids.shape[1] + 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for attention kernels")
def test_generate_with_mu_zero_cache():
    """MuZeroCache can be built and called per-layer without errors."""
    muzero.patch_transformers()

    # Test the cache directly (avoids generate() validation complexity)
    cfg   = muzero.MuZeroConfig(bits=4, q_group_size=64, residual_length=64)
    cache = muzero.MuZeroCache(cfg)

    bs, nh, hd = 1, 4, 64
    device = "cuda"
    dtype  = torch.float16

    # Simulate 3 prefill steps + 2 decode steps
    for step in range(5):
        sl = 32 if step == 0 else 1
        k = torch.randn(bs, nh, sl, hd, device=device, dtype=dtype)
        v = torch.randn(bs, nh, sl, hd, device=device, dtype=dtype)
        materialize = step == 0
        rk, rv = cache.update(k, v, layer_idx=0, cache_kwargs={"materialize_full_kv": materialize})
        assert rk is not None
        assert rv is not None

    assert cache.get_seq_length(0) == 36


def test_config_roundtrip():
    cfg = muzero.MuZeroConfig(bits=4, q_group_size=64, residual_length=128)
    assert cfg.bits == 4
    d = {"bits": 2, "q_group_size": 32, "residual_length": 64}
    cfg2 = muzero.MuZeroConfig.from_dict(d)
    assert cfg2.bits == 2


def test_config_validation():
    with pytest.raises(ValueError):
        muzero.MuZeroConfig(bits=5)   # 5 not in {1,2,3,4,7,8}
    with pytest.raises(ValueError):
        muzero.MuZeroConfig(amax_dtype="float8")

"""μ-Zero: piecewise log-companding KV-cache quantisation with zero-anchor.

Quick start::

    import muzero
    from transformers import AutoModelForCausalLM, AutoTokenizer

    muzero.patch_transformers()
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="bfloat16").cuda()
    tok   = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    inp   = tok("Hello, μ-Zero!", return_tensors="pt").to("cuda")
    out   = model.generate(**inp, cache_implementation="mu_zero_4bit", max_new_tokens=256)
    print(tok.decode(out[0]))
"""

from .config import MuZeroConfig
from .quantizer import select_alpha, quantize_rows, dequantize_rows, roundtrip
from .cache.layer import MuZeroCacheLayer
from .cache.attention import get_mu_zero_decode_stats, mu_zero_decode_attention_forward, reset_mu_zero_decode_stats
from .integrations.transformers_patch import MuZeroCache, patch_transformers

__all__ = [
    "MuZeroConfig",
    "MuZeroCacheLayer",
    "MuZeroCache",
    "select_alpha",
    "quantize_rows",
    "dequantize_rows",
    "roundtrip",
    "mu_zero_decode_attention_forward",
    "reset_mu_zero_decode_stats",
    "get_mu_zero_decode_stats",
    "patch_transformers",
]

__version__ = "0.1.0"

#!/usr/bin/env python
"""μ-Zero quick-start: swap bf16 DynamicCache for μ-Zero 4-bit in one line."""

import torch
import muzero
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Register μ-Zero cache implementations in transformers
muzero.patch_transformers()

MODEL = "/root/autodl-tmp/ann/Qwen3-8B"  # or Llama-3.1-8B-Instruct
tok   = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
).cuda().eval()

prompt  = "Explain the μ-Zero KV-cache quantisation method in one paragraph:"
inputs  = tok(prompt, return_tensors="pt").to("cuda")

# 2. Generate with μ-Zero 4-bit KV cache (single parameter change)
with torch.no_grad():
    out = model.generate(
        **inputs,
        cache_implementation="mu_zero_4bit",   # ← only change vs baseline
        cache_config={
            "q_group_size":    64,
            "residual_length": 128,
        },
        max_new_tokens=256,
        do_sample=False,
    )

print(tok.decode(out[0], skip_special_tokens=True))

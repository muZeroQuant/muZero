#!/usr/bin/env python
"""μ-Zero quality benchmark: perplexity and task accuracy vs bf16 baseline.

Usage::

    python benchmarks/bench_quality.py --config benchmarks/configs/qwen3_8b.yaml --tasks ppl
    python benchmarks/bench_quality.py --config benchmarks/configs/llama31_8b_instruct.yaml --tasks ppl,gsm8k

Tasks:
  ppl    — WikiText-2 perplexity (stride-512 sliding window)
  gsm8k  — GSM8K 0-shot chain-of-thought accuracy (requires lm_eval)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
import muzero
from eval_utils import ensure_model_path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tasks",  default="ppl", help="Comma-separated: ppl,gsm8k")
    p.add_argument("--ppl_max_samples", type=int, default=200,
                   help="Max WikiText-2 samples for perplexity (set 0 for full test set)")
    return p.parse_args()


def _load_model(model_cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name_or_path = ensure_model_path(model_cfg["name_or_path"])
    dtype        = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    device       = model_cfg.get("device", "cuda:0")
    print(f"[model] loading `{name_or_path}`", flush=True)
    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    return model, tok, device


# ---------------------------------------------------------------------------
# Perplexity (sliding window)
# ---------------------------------------------------------------------------

def compute_ppl(
    model,
    tok,
    device: str,
    cache_kwargs: dict | None = None,
    max_samples: int = 200,
    stride: int = 512,
    max_length: int = 1024,
) -> float:
    """WikiText-2 perplexity using a sliding-window approach."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; skipping ppl. pip install datasets")
        return float("nan")

    print("[dataset] loading `wikitext/wikitext-2-raw-v1` split=test", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encodings = tok(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    total_nll = 0.0
    total_tokens = 0
    n_processed = 0

    for begin in range(0, input_ids.shape[1] - max_length, stride):
        end     = min(begin + max_length, input_ids.shape[1])
        target_len = end - (begin + stride)
        if target_len <= 0:
            continue

        chunk = input_ids[:, begin:end]
        with torch.no_grad():
            out = model(chunk)
        logits = out.logits
        # NLL for the target region (last target_len tokens)
        shift_logits = logits[:, -(target_len + 1):-1, :].contiguous()
        shift_labels = chunk[:, -target_len:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll    += loss.item()
        total_tokens += target_len

        n_processed += 1
        if max_samples > 0 and n_processed >= max_samples:
            break

    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------------
# GSM8K via lm_eval
# ---------------------------------------------------------------------------

def run_gsm8k(model_path: str, dtype_str: str, device: str,
              cache_implementation: str | None = None,
              cache_config: dict | None = None) -> float:
    try:
        import lm_eval
    except ImportError:
        print("lm_eval not installed; skipping gsm8k. pip install lm_eval")
        return float("nan")

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    model_kwargs: dict[str, Any] = dict(
        pretrained=model_path,
        dtype=dtype_str,
        device=device,
    )
    if cache_implementation:
        model_kwargs["cache_implementation"] = cache_implementation
    if cache_config:
        model_kwargs["cache_config"] = cache_config

    lm = HFLM(**model_kwargs)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        num_fewshot=0,
        batch_size=1,
    )
    return results["results"]["gsm8k"]["exact_match,get-answer"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = _parse_args()
    tasks   = [t.strip() for t in args.tasks.split(",")]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    muzero.patch_transformers()

    model_cfg  = cfg["model"]
    mu_cfg     = cfg.get("mu_zero", {})
    method_cfg = {k: v for k, v in mu_cfg.items() if k not in {"bits", "also_2bit"}}

    methods = [
        ("bf16",         None,             None),
        ("mu_zero_4bit", "mu_zero_4bit",   method_cfg),
    ]

    print(f"\n{'Method':20s}", end="")
    if "ppl" in tasks:
        print(f"  {'PPL':>8}", end="")
    if "gsm8k" in tasks:
        print(f"  {'GSM8K':>8}", end="")
    print()
    print("-" * 50)

    ppl_results: dict[str, float]   = {}
    gsm_results: dict[str, float]   = {}

    for label, cache_impl, cache_conf in methods:
        model, tok, device = _load_model(model_cfg)
        gen_kwargs: dict[str, Any] = {}
        if cache_impl:
            gen_kwargs["cache_implementation"] = cache_impl
        if cache_conf:
            gen_kwargs["cache_config"] = cache_conf

        print(f"{label:20s}", end="", flush=True)

        if "ppl" in tasks:
            ppl = compute_ppl(model, tok, device, max_samples=args.ppl_max_samples)
            ppl_results[label] = ppl
            print(f"  {ppl:8.3f}", end="", flush=True)

        if "gsm8k" in tasks:
            acc = run_gsm8k(
                ensure_model_path(model_cfg["name_or_path"]),
                model_cfg.get("dtype", "bfloat16"),
                device,
                cache_implementation=cache_impl,
                cache_config=cache_conf,
            )
            gsm_results[label] = acc
            print(f"  {acc*100:7.2f}%", end="", flush=True)

        print()
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()

    # Deltas vs bf16
    print("\n=== Deltas vs bf16 ===")
    for label, _, _ in methods:
        if label == "bf16":
            continue
        if "ppl" in tasks and "bf16" in ppl_results and label in ppl_results:
            dppl = ppl_results[label] - ppl_results["bf16"]
            print(f"  {label}  Δppl={dppl:+.3f}")
        if "gsm8k" in tasks and "bf16" in gsm_results and label in gsm_results:
            dacc = (gsm_results[label] - gsm_results["bf16"]) * 100
            print(f"  {label}  ΔGSM8K={dacc:+.2f}%")


if __name__ == "__main__":
    main()

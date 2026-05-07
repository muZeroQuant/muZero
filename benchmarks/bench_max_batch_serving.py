#!/usr/bin/env python
"""Max-batch serving benchmark.

This benchmark fixes prompt/output lengths, increases batch size until OOM,
and reports aggregate generated-token throughput and peak memory.  It is meant
to expose the serving-capacity benefit of KV-cache compression, complementing
single-request decode latency benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
import muzero
from eval_utils import ensure_dataset_path, ensure_model_path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default="gsm8k", choices=("gsm8k", "humaneval", "longbench_v2", "synthetic"))
    p.add_argument("--prompt-len", type=int, default=161)
    p.add_argument("--gen-len", type=int, default=338)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--batch-sizes", default=None, help="Comma-separated sizes; disables binary OOM search")
    p.add_argument("--methods", default="bf16,mu_zero_4bit_r32,mu_zero_4bit_r128")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--output", default=None)
    return p.parse_args()


def _load_model(model_cfg: dict) -> tuple[Any, Any, str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name_or_path = ensure_model_path(model_cfg["name_or_path"])
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    device = model_cfg.get("device", "cuda:0")
    print(f"[model] loading `{name_or_path}`", flush=True)
    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    return model, tok, device


def _load_texts(dataset: str, limit: int = 2048) -> list[str]:
    if dataset == "synthetic":
        return ["The quick brown fox studies key value cache compression for efficient language model serving."]
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for non-synthetic prompts") from exc

    if dataset == "gsm8k":
        dataset_name = ensure_dataset_path("gsm8k")
        print(f"[dataset] loading `{dataset_name}` split=test", flush=True)
        ds = load_dataset(dataset_name, "main", split="test", download_mode="reuse_dataset_if_exists")
        return [f"Question: {row['question']}\nAnswer:" for row in ds.select(range(min(limit, len(ds))))]
    if dataset == "humaneval":
        dataset_name = ensure_dataset_path("openai_humaneval")
        print(f"[dataset] loading `{dataset_name}` split=test", flush=True)
        ds = load_dataset(dataset_name, split="test", download_mode="reuse_dataset_if_exists")
        return [row["prompt"] for row in ds.select(range(min(limit, len(ds))))]
    print("[dataset] loading `zai-org/LongBench-v2` split=train", flush=True)
    ds = load_dataset("zai-org/LongBench-v2", split="train", download_mode="reuse_dataset_if_exists")
    rows = ds.select(range(min(limit, len(ds))))
    return [
        (
            f"Context:\n{row['context']}\n\nQuestion: {row['question']}\n"
            f"A. {row['choice_A']}\nB. {row['choice_B']}\nC. {row['choice_C']}\nD. {row['choice_D']}\nAnswer:"
        )
        for row in rows
    ]


def _fit_ids(tok, text: str, prompt_len: int) -> list[int]:
    ids = tok.encode(text, add_special_tokens=True)
    if not ids:
        ids = tok.encode("Benchmark prompt", add_special_tokens=True)
    if len(ids) >= prompt_len:
        return ids[:prompt_len]
    filler = tok.encode(" We continue the benchmark prompt with neutral text about language model serving.", add_special_tokens=False)
    if not filler:
        filler = ids
    need = prompt_len - len(ids)
    repeats = (need + len(filler) - 1) // len(filler)
    return ids + (filler * repeats)[:need]


def _make_batch(tok, texts: list[str], batch_size: int, prompt_len: int, device: str) -> dict[str, torch.Tensor]:
    rows = [_fit_ids(tok, texts[i % len(texts)], prompt_len) for i in range(batch_size)]
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _method_kwargs(label: str, mu_cfg: dict) -> dict[str, Any]:
    if label == "bf16":
        return {}
    match = re.fullmatch(r"mu_zero_(\d+)bit_r(\d+)", label)
    if match is None:
        raise ValueError(f"unknown method {label}")
    bits = int(match.group(1))
    residual_length = int(match.group(2))
    cache_config = {**mu_cfg, "residual_length": residual_length}
    cache_config.pop("bits", None)
    cache_config.pop("also_2bit", None)
    return {"cache_implementation": f"mu_zero_{bits}bit", "cache_config": cache_config}


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@torch.inference_mode()
def _run_once(model, inputs: dict[str, torch.Tensor], gen_len: int, kwargs: dict[str, Any]) -> tuple[float, float, int]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=gen_len,
        do_sample=False,
        temperature=None,
        top_p=None,
        **kwargs,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    generated = max(0, out.shape[1] - inputs["input_ids"].shape[1]) * inputs["input_ids"].shape[0]
    return elapsed, peak_mb, generated


def _try_batch(model, tok, texts, device: str, method: str, mu_cfg: dict, batch_size: int, prompt_len: int, gen_len: int, warmup: int, runs: int) -> dict[str, Any] | None:
    kwargs = _method_kwargs(method, mu_cfg)
    inputs = _make_batch(tok, texts, batch_size, prompt_len, device)
    try:
        for _ in range(warmup):
            _run_once(model, inputs, min(gen_len, 8), kwargs)
            _cleanup()
        latencies = []
        peaks = []
        generated_counts = []
        for _ in range(runs):
            elapsed, peak_mb, generated = _run_once(model, inputs, gen_len, kwargs)
            latencies.append(elapsed)
            peaks.append(peak_mb)
            generated_counts.append(generated)
            _cleanup()
        avg_latency = sum(latencies) / len(latencies)
        generated = min(generated_counts)
        return {
            "method": method,
            "batch_size": batch_size,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "generated_tokens": generated,
            "avg_latency_s": round(avg_latency, 4),
            "output_tok_s": round(generated / avg_latency, 2),
            "requests_s": round(batch_size / avg_latency, 4),
            "peak_mem_mb": round(max(peaks), 1),
        }
    except RuntimeError as exc:
        if _is_oom(exc):
            _cleanup()
            return None
        raise


def _search_method(model, tok, texts, device: str, method: str, mu_cfg: dict, prompt_len: int, gen_len: int, max_batch: int, batch_sizes: list[int] | None, warmup: int, runs: int) -> list[dict[str, Any]]:
    rows = []
    sizes = batch_sizes if batch_sizes is not None else []
    if not sizes:
        size = 1
        while size <= max_batch:
            sizes.append(size)
            size *= 2
        if sizes[-1] != max_batch:
            sizes.append(max_batch)

    for bs in sizes:
        print(f"  trying {method:16s} batch={bs}", flush=True)
        row = _try_batch(model, tok, texts, device, method, mu_cfg, bs, prompt_len, gen_len, warmup, runs)
        if row is None:
            print(f"    OOM at batch={bs}", flush=True)
            break
        rows.append(row)
        print(
            f"    ok tok/s={row['output_tok_s']:.1f} req/s={row['requests_s']:.3f} peak={row['peak_mem_mb']:.0f}MB",
            flush=True,
        )
    return rows


def main() -> None:
    args = _parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    muzero.patch_transformers()
    model, tok, device = _load_model(cfg["model"])
    texts = _load_texts(args.dataset)
    mu_cfg = cfg.get("mu_zero", {})
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()] if args.batch_sizes else None

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    all_rows = []
    print(
        f"Dataset={args.dataset} prompt_len={args.prompt_len} gen_len={args.gen_len} max_batch={args.max_batch}",
        flush=True,
    )
    for method in methods:
        rows = _search_method(
            model, tok, texts, device, method, mu_cfg, args.prompt_len, args.gen_len,
            args.max_batch, batch_sizes, args.warmup, args.runs,
        )
        all_rows.extend(rows)

    print("\n=== Max Successful Batch Per Method ===")
    print(f"{'method':20s} {'max_bs':>6} {'tok/s':>10} {'req/s':>8} {'peak_MB':>9}")
    print("-" * 62)
    for method in methods:
        rows = [r for r in all_rows if r["method"] == method]
        if not rows:
            print(f"{method:20s} {'OOM':>6}")
            continue
        row = max(rows, key=lambda r: r["batch_size"])
        print(f"{method:20s} {row['batch_size']:6d} {row['output_tok_s']:10.1f} {row['requests_s']:8.3f} {row['peak_mem_mb']:9.1f}")

    if args.output and all_rows:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

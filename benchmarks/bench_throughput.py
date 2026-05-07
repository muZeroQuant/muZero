#!/usr/bin/env python
"""μ-Zero throughput benchmark: tokens/sec vs bf16 baseline.

Usage::

    python benchmarks/bench_throughput.py --config benchmarks/configs/qwen3_8b.yaml
    python benchmarks/bench_throughput.py --config benchmarks/configs/llama31_8b_instruct.yaml

Measures decode-phase tokens/sec and peak GPU memory across multiple context
lengths (prefill sizes), comparing:
  • bf16     — standard full-precision DynamicCache (baseline)
  • mu_zero_4bit — μ-Zero 4-bit KV cache
  • mu_zero_2bit — μ-Zero 2-bit KV cache (optional)
"""

from __future__ import annotations

import argparse
import gc
import inspect
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
import muzero
from eval_utils import ensure_model_path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML benchmark config file")
    p.add_argument("--output", default=None, help="Save results to this CSV path")
    p.add_argument("--warmup", type=int, default=2, help="Warm-up runs discarded")
    p.add_argument("--runs",   type=int, default=5, help="Timed runs averaged")
    p.add_argument("--contexts", default=None, help="Comma-separated context lengths overriding the config")
    p.add_argument(
        "--methods",
        default="bf16,mu_zero_4bit",
        help=(
            "Comma-separated methods: bf16,bf16_static,mu_zero_4bit,mu_zero_2bit,"
            "mu_zero_4bit_old_decode_flush,mu_zero_2bit_old_decode_flush,"
            "mu_zero_4bit_logq_compatible_layout,mu_zero_2bit_logq_compatible_layout,"
            "logq{bits}_zeroanchor_stat_cuda"
        ),
    )
    p.add_argument("--batch-size", type=int, default=1, help="Number of identical no-padding prompts per benchmark batch")
    p.add_argument("--new-tokens", type=int, default=None, help="Override config new_tokens")
    p.add_argument(
        "--mode",
        choices=("decode", "generate"),
        default="decode",
        help="decode excludes prefill from timing; generate measures end-to-end model.generate().",
    )
    p.add_argument(
        "--decode-attention-mask",
        choices=("full", "static", "none"),
        default="full",
        help="Decode-mode attention mask policy. 'static' reuses one full-size mask; 'none' is valid for this no-padding benchmark prompt batch.",
    )
    p.add_argument(
        "--empty-cache-before-decode",
        action="store_true",
        help="Release PyTorch cached prefill blocks before decode timing/memory measurement.",
    )
    p.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help="If >0, build the prompt KV cache in sequence chunks to reduce prefill peak memory.",
    )
    p.add_argument(
        "--residual-length",
        type=int,
        default=None,
        help="Override mu_zero.residual_length from the config for MuZero methods.",
    )
    return p.parse_args()


def _load_model(model_cfg: dict) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name_or_path = model_cfg["name_or_path"]
    dtype        = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    device       = model_cfg.get("device", "cuda:0")
    name_or_path = ensure_model_path(name_or_path)
    print(f"Loading {name_or_path} …", flush=True)
    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    return model, tok


def _make_prompt(tok, target_tokens: int, device: str, batch_size: int = 1) -> dict:
    """Build a prompt with exactly target_tokens tokens when possible."""
    prefix = tok.encode("Benchmark prompt:\n", add_special_tokens=True)
    base = tok.encode(
        "The quick brown fox jumped over the lazy dog while the model keeps a long key value cache. ",
        add_special_tokens=False,
    )
    if not base:
        raise ValueError("Tokenizer produced no tokens for benchmark prompt text")
    ids = prefix[:target_tokens]
    remaining = target_tokens - len(ids)
    if remaining > 0:
        repeats = (remaining + len(base) - 1) // len(base)
        ids = ids + (base * repeats)[:remaining]
    input_ids = torch.tensor([ids] * batch_size, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _supports_logits_to_keep(model) -> bool:
    return "logits_to_keep" in set(inspect.signature(model.forward).parameters.keys())


def _make_cache(model, method_name: str, cache_config: dict, max_cache_len: int | None = None):
    if method_name == "bf16":
        from transformers import DynamicCache

        return DynamicCache(config=model.config.get_text_config(decoder=True))
    if method_name == "bf16_static":
        from transformers import StaticCache

        if max_cache_len is None:
            raise ValueError("max_cache_len is required for bf16_static")
        return StaticCache(config=model.config.get_text_config(decoder=True), max_cache_len=max_cache_len)
    if method_name.startswith("mu_zero_"):
        base_name = _base_mu_zero_method_name(method_name)
        bits = int(base_name.removeprefix("mu_zero_").removesuffix("bit"))
        cfg = muzero.MuZeroConfig.from_dict({**cache_config, "bits": bits})
        return muzero.MuZeroCache(cfg)
    legacy_bits = _parse_logq_zeroanchor_bits(method_name)
    if legacy_bits is not None:
        cfg = {k: v for k, v in cache_config.items() if k != "backend"}
        return muzero.MuZeroCache(muzero.MuZeroConfig.from_dict({**cfg, "bits": legacy_bits}))
    raise ValueError(method_name)


def _parse_logq_zeroanchor_bits(method_name: str) -> int | None:
    match = re.fullmatch(r"logq(\d+)_zeroanchor_stat_cuda", method_name)
    return int(match.group(1)) if match else None


def _base_mu_zero_method_name(method_name: str) -> str:
    for suffix in ("_logq_compatible_layout", "_logq_compatible", "_group_flush", "_old_decode_flush"):
        if method_name.endswith(suffix):
            return method_name[: -len(suffix)]
    return method_name


def _uses_logq_compatible_layout(method_name: str) -> bool:
    return method_name.endswith(("_logq_compatible_layout", "_logq_compatible"))


def _uses_group_flush(method_name: str) -> bool:
    return method_name.endswith("_group_flush")


def _uses_old_decode_flush(method_name: str) -> bool:
    return method_name.endswith("_old_decode_flush")


def _uses_muzero_cache(method_name: str) -> bool:
    return method_name.startswith("mu_zero_") or _parse_logq_zeroanchor_bits(method_name) is not None


def _generation_cache_kwargs(method_name: str, cache_config: dict) -> dict[str, Any]:
    legacy_bits = _parse_logq_zeroanchor_bits(method_name)
    if legacy_bits is not None:
        return {
            "cache_implementation": "quantized",
            "cache_config": {**cache_config, "backend": method_name},
        }
    return {
        "cache_implementation": _base_mu_zero_method_name(method_name),
        "cache_config": cache_config,
    }


def _forward_kwargs(model) -> dict[str, Any]:
    return {"logits_to_keep": 1} if _supports_logits_to_keep(model) else {}


def _tensor_mb(tensor: torch.Tensor | None) -> float:
    return 0.0 if tensor is None else tensor.numel() * tensor.element_size() / 1024**2


def _muzero_cache_stats(cache: Any) -> dict[str, float]:
    if not hasattr(cache, "layers"):
        return {
            "cache_packed_mb": 0.0,
            "cache_meta_mb": 0.0,
            "cache_residual_mb": 0.0,
            "cache_total_mb": 0.0,
        }
    packed_mb = 0.0
    meta_mb = 0.0
    residual_mb = 0.0
    for layer in cache.layers:
        packed_mb += _tensor_mb(getattr(layer, "_packed_key_q", None))
        packed_mb += _tensor_mb(getattr(layer, "_packed_value_q", None))
        for attr in ["_key_mn", "_key_scale", "_value_mn", "_value_scale"]:
            meta_mb += _tensor_mb(getattr(layer, attr, None))
        key_buffer = getattr(layer, "_key_buffer", None)
        value_buffer = getattr(layer, "_value_buffer", None)
        residual_mb += _tensor_mb(key_buffer if key_buffer is not None else getattr(layer, "keys", None))
        residual_mb += _tensor_mb(value_buffer if value_buffer is not None else getattr(layer, "values", None))
    return {
        "cache_packed_mb": packed_mb,
        "cache_meta_mb": meta_mb,
        "cache_residual_mb": residual_mb,
        "cache_total_mb": packed_mb + meta_mb + residual_mb,
    }


def _cache_position_kwargs(method_name: str, start: int, end: int, device: torch.device) -> dict[str, torch.Tensor]:
    if method_name != "bf16_static":
        return {}
    return {"cache_position": torch.arange(start, end, device=device, dtype=torch.long)}


@torch.no_grad()
def _prefill_cache(
    model,
    inputs: dict[str, torch.Tensor],
    method_name: str,
    cache_config: dict,
    use_attention_mask: bool,
    prefill_chunk_size: int,
    prefill_extra_tokens: int,
):
    input_ids = inputs["input_ids"]
    cache = _make_cache(model, method_name, cache_config, max_cache_len=input_ids.shape[1] + prefill_extra_tokens)
    defer_mu_zero_prefill = (
        _uses_muzero_cache(method_name)
        and (prefill_chunk_size <= 0 or prefill_chunk_size >= input_ids.shape[1])
    )
    if defer_mu_zero_prefill:
        cache.defer_prefill_quantization = True
    attention_mask_full = inputs["attention_mask"]
    if prefill_chunk_size <= 0 or prefill_chunk_size >= input_ids.shape[1]:
        attention_mask = attention_mask_full if use_attention_mask else None
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            **_cache_position_kwargs(method_name, 0, input_ids.shape[1], input_ids.device),
            **_forward_kwargs(model),
        )
    else:
        outputs = None
        for start in range(0, input_ids.shape[1], prefill_chunk_size):
            end = min(start + prefill_chunk_size, input_ids.shape[1])
            attention_mask = attention_mask_full[:, :end] if use_attention_mask else None
            outputs = model(
                input_ids=input_ids[:, start:end],
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                **_cache_position_kwargs(method_name, start, end, input_ids.device),
                **_forward_kwargs(model),
            )
    if outputs is None:
        raise RuntimeError("Prefill produced no outputs")
    if defer_mu_zero_prefill:
        cache.defer_prefill_quantization = False
        cache.finalize_prefill_quantization()
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return cache, next_token


@torch.no_grad()
def _decode_only(
    model,
    inputs: dict[str, torch.Tensor],
    method_name: str,
    cache_config: dict,
    new_tokens: int,
    decode_attention_mask: str,
    empty_cache_before_decode: bool,
    prefill_chunk_size: int,
) -> tuple[int, float, int, int, dict[str, float]]:
    use_attention_mask = decode_attention_mask != "none"
    cache, next_token = _prefill_cache(
        model,
        inputs,
        method_name,
        cache_config,
        use_attention_mask,
        prefill_chunk_size,
        prefill_extra_tokens=new_tokens,
    )
    torch.cuda.synchronize()
    if empty_cache_before_decode:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    attention_len = inputs["input_ids"].shape[1]
    static_attention_mask = None
    if decode_attention_mask == "static":
        static_attention_mask = torch.ones(
            (inputs["input_ids"].shape[0], attention_len + new_tokens),
            dtype=torch.long,
            device=inputs["input_ids"].device,
        )
    fused_before = muzero.get_mu_zero_decode_stats() if _uses_muzero_cache(method_name) else {}
    t0 = time.perf_counter()
    for _ in range(new_tokens):
        attention_len += 1
        attention_mask = None
        if decode_attention_mask == "full":
            attention_mask = torch.ones((inputs["input_ids"].shape[0], attention_len), dtype=torch.long, device=inputs["input_ids"].device)
        elif decode_attention_mask == "static":
            attention_mask = static_attention_mask[:, :attention_len]
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            **_cache_position_kwargs(method_name, attention_len - 1, attention_len, next_token.device),
            **_forward_kwargs(model),
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    fused_after = muzero.get_mu_zero_decode_stats() if _uses_muzero_cache(method_name) else {}
    fused_qk_calls = int(fused_after.get("fused_qk_success", 0) - fused_before.get("fused_qk_success", 0))
    fused_prefix_calls = int(
        fused_after.get("fused_prefix_decode_success", 0) - fused_before.get("fused_prefix_decode_success", 0)
    )
    cache_stats = _muzero_cache_stats(cache) if _uses_muzero_cache(method_name) else _muzero_cache_stats(None)
    return new_tokens * inputs["input_ids"].shape[0], elapsed, fused_qk_calls, fused_prefix_calls, cache_stats


def _benchmark_method(
    model,
    tok,
    method_name: str,
    cache_config: dict,
    context_lengths: list[int],
    new_tokens: int,
    warmup: int,
    runs: int,
    device: str,
    mode: str,
    decode_attention_mask: str,
    batch_size: int,
    empty_cache_before_decode: bool,
    prefill_chunk_size: int,
) -> list[dict]:
    results = []
    use_attention_mask = decode_attention_mask != "none"
    for ctx_len in context_lengths:
        inputs = _make_prompt(tok, ctx_len, device, batch_size=batch_size)
        actual_ctx = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        if method_name not in {"bf16", "bf16_static"}:
            gen_kwargs.update(_generation_cache_kwargs(method_name, cache_config))

        # Warm-up
        for _ in range(warmup):
            if mode == "decode":
                _decode_only(
                    model,
                    inputs,
                    method_name,
                    cache_config,
                    new_tokens,
                    decode_attention_mask,
                    empty_cache_before_decode,
                    prefill_chunk_size,
                )
            else:
                with torch.no_grad():
                    model.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # Timed runs
        latencies = []
        new_counts = []
        fused_qk_calls = []
        fused_prefix_calls = []
        peak_values = []
        peak_reserved_values = []
        cache_stats_values = []
        for _ in range(runs):
            if mode == "decode":
                actual_new, elapsed, fused_qk, fused_prefix, cache_stats = _decode_only(
                    model,
                    inputs,
                    method_name,
                    cache_config,
                    new_tokens,
                    decode_attention_mask,
                    empty_cache_before_decode,
                    prefill_chunk_size,
                )
                latencies.append(elapsed)
                new_counts.append(actual_new)
                fused_qk_calls.append(fused_qk)
                fused_prefix_calls.append(fused_prefix)
                cache_stats_values.append(cache_stats)
            else:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)
                new_counts.append((out.shape[1] - actual_ctx) * inputs["input_ids"].shape[0])
                fused_qk_calls.append(0)
                fused_prefix_calls.append(0)
                cache_stats_values.append(_muzero_cache_stats(None))
            peak_values.append(torch.cuda.max_memory_allocated() / 1024**2)
            peak_reserved_values.append(torch.cuda.max_memory_reserved() / 1024**2)

        peak_mb  = max(peak_values)
        peak_reserved_mb = max(peak_reserved_values)
        actual_new = min(new_counts)
        avg_lat  = sum(latencies) / len(latencies)
        tps      = actual_new / avg_lat
        avg_fused_qk = sum(fused_qk_calls) / max(len(fused_qk_calls), 1)
        avg_fused_prefix = sum(fused_prefix_calls) / max(len(fused_prefix_calls), 1)
        cache_stats = {
            key: max(stats.get(key, 0.0) for stats in cache_stats_values)
            for key in ["cache_packed_mb", "cache_meta_mb", "cache_residual_mb", "cache_total_mb"]
        }

        results.append({
            "method":       method_name,
            "context_len":  ctx_len,
            "actual_context_len": actual_ctx,
            "batch_size": inputs["input_ids"].shape[0],
            "mode":         mode,
            "decode_attention_mask": decode_attention_mask if mode == "decode" else "generate",
            "prefill_chunk_size": prefill_chunk_size,
            "generated_tokens": actual_new,
            "avg_latency_s": round(avg_lat, 4),
            "tokens_per_sec": round(tps, 2),
            "peak_mem_mb":  round(peak_mb, 1),
            "peak_reserved_mb": round(peak_reserved_mb, 1),
            "cache_packed_mb": round(cache_stats["cache_packed_mb"], 1),
            "cache_meta_mb": round(cache_stats["cache_meta_mb"], 1),
            "cache_residual_mb": round(cache_stats["cache_residual_mb"], 1),
            "cache_total_mb": round(cache_stats["cache_total_mb"], 1),
            "fused_qk_calls": round(avg_fused_qk, 1),
            "fused_prefix_decode_calls": round(avg_fused_prefix, 1),
        })
        print(
            f"  {method_name:20s} ctx={ctx_len:5d} actual={actual_ctx:5d}  "
            f"batch={batch_size:4d}  {tps:7.1f} tok/s  peak={peak_mb:.0f} MB  "
            f"reserved={peak_reserved_mb:.0f} MB  fused_qk/run={avg_fused_qk:.0f}  "
            f"fused_prefix/run={avg_fused_prefix:.0f}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    muzero.patch_transformers()

    model, tok = _load_model(cfg["model"])
    device     = cfg["model"].get("device", "cuda:0")

    mu_cfg  = cfg.get("mu_zero", {})
    if args.residual_length is not None:
        mu_cfg = {**mu_cfg, "residual_length": args.residual_length}
    ctx_lens = cfg.get("context_lengths", [512, 1024, 2048, 4096, 8192])
    if args.contexts:
        ctx_lens = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    new_tok  = args.new_tokens if args.new_tokens is not None else cfg.get("new_tokens", 128)

    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed_methods = {
        "bf16",
        "bf16_static",
        "mu_zero_4bit",
        "mu_zero_2bit",
        "mu_zero_4bit_group_flush",
        "mu_zero_2bit_group_flush",
        "mu_zero_4bit_old_decode_flush",
        "mu_zero_2bit_old_decode_flush",
        "mu_zero_4bit_logq_compatible_layout",
        "mu_zero_2bit_logq_compatible_layout",
        "mu_zero_4bit_logq_compatible",
        "mu_zero_2bit_logq_compatible",
    }
    unknown_methods = sorted(
        method for method in requested_methods
        if method not in allowed_methods and _parse_logq_zeroanchor_bits(method) is None
    )
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    methods = []
    for method_name in requested_methods:
        if method_name in {"bf16", "bf16_static"}:
            method_cfg = {}
        else:
            method_cfg = {k: v for k, v in mu_cfg.items() if k != "bits"}
            if _uses_logq_compatible_layout(method_name):
                method_cfg = {**method_cfg, "logq_compatible_layout": True}
            elif _uses_old_decode_flush(method_name):
                method_cfg = {**method_cfg, "logq_compatible_layout": False, "decode_flush_by_group": False}
            elif _uses_group_flush(method_name):
                method_cfg = {**method_cfg, "logq_compatible_layout": False, "decode_flush_by_group": True}
            else:
                method_cfg = {**method_cfg, "logq_compatible_layout": False, "decode_flush_by_group": True}
        methods.append((method_name, method_cfg))

    all_results = []
    if args.mode == "generate" and args.decode_attention_mask != "full":
        raise ValueError("--decode-attention-mask only applies to --mode decode")
    if args.mode == "generate" and any(method_name == "bf16_static" for method_name, _ in methods):
        raise ValueError("bf16_static is only implemented for --mode decode in this benchmark")
    print(f"\nMode: {args.mode}")
    print(f"Batch size: {args.batch_size}")
    if args.mode == "decode":
        print(f"Decode attention mask: {args.decode_attention_mask}")
        if args.prefill_chunk_size > 0:
            print(f"Prefill chunk size: {args.prefill_chunk_size}")
    print(f"{'Method':20s} {'ctx':>5} {'actual':>6} {'batch':>6} {'tok/s':>7}  peak_MB  reserved_MB  fused_qk/run  fused_prefix/run")
    print("-" * 104)
    for method_name, method_cfg in methods:
        rows = _benchmark_method(
            model, tok,
            method_name, method_cfg,
            ctx_lens, new_tok,
            args.warmup, args.runs,
            device,
            args.mode,
            args.decode_attention_mask,
            args.batch_size,
            args.empty_cache_before_decode,
            args.prefill_chunk_size,
        )
        all_results.extend(rows)

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Method':20s} {'ctx':>6} {'actual':>6} {'batch':>6} {'tok/s':>8} {'peak_MB':>9} {'reserved_MB':>11} {'qk':>8} {'prefix':>8}")
    print("-" * 86)
    for r in all_results:
        print(
            f"{r['method']:20s} {r['context_len']:6d} {r['actual_context_len']:6d} {r['batch_size']:6d} "
            f"{r['tokens_per_sec']:8.1f} {r['peak_mem_mb']:9.1f} {r['peak_reserved_mb']:11.1f} "
            f"{r['fused_qk_calls']:8.0f} {r['fused_prefix_decode_calls']:8.0f}"
        )

    # Speedup vs bf16. Prefer the standard DynamicCache baseline when both
    # BF16 variants are present; bf16_static is useful only as an opt-in probe.
    bf16_map = {r["context_len"]: r["tokens_per_sec"] for r in all_results if r["method"] == "bf16"}
    for r in all_results:
        if r["method"] == "bf16_static" and r["context_len"] not in bf16_map:
            bf16_map[r["context_len"]] = r["tokens_per_sec"]
    if bf16_map:
        print("\n=== Speedup vs bf16 ===")
        for r in all_results:
            if r["method"] in {"bf16", "bf16_static"}:
                continue
            baseline = bf16_map.get(r["context_len"])
            if baseline:
                speedup = r["tokens_per_sec"] / baseline
                print(f"  {r['method']} ctx={r['context_len']:5d}  ×{speedup:.2f}")

    # Save CSV
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

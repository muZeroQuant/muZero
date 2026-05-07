#!/usr/bin/env python
"""Collect MuZero scaling benchmark results.

Each method/point is executed in a separate subprocess so BF16 OOMs do not
prevent MuZero 4-bit/2-bit measurements from being collected at larger sizes.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_METHODS = ["bf16", "mu_zero_4bit", "mu_zero_2bit"]
ALLOWED_METHODS = [
    *DEFAULT_METHODS,
    "bf16_static",
    "mu_zero_4bit_group_flush",
    "mu_zero_2bit_group_flush",
    "mu_zero_4bit_old_decode_flush",
    "mu_zero_2bit_old_decode_flush",
    "mu_zero_4bit_logq_compatible_layout",
    "mu_zero_2bit_logq_compatible_layout",
    "mu_zero_4bit_logq_compatible",
    "mu_zero_2bit_logq_compatible",
]


def is_logq_zeroanchor_method(method: str) -> bool:
    return re.fullmatch(r"logq\d+_zeroanchor_stat_cuda", method) is not None


def is_muzero_method(method: str) -> bool:
    return method.startswith("mu_zero_") or is_logq_zeroanchor_method(method)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/configs/qwen3_8b.yaml")
    parser.add_argument("--output", default="runs/muzero_scaling/results.csv")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--mode", choices=("decode", "generate"), default="decode")
    parser.add_argument("--decode-attention-mask", choices=("full", "static", "none"), default="static")
    parser.add_argument(
        "--empty-cache-before-decode",
        action="store_true",
        help="Forward --empty-cache-before-decode to bench_throughput.py.",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help="Forward --prefill-chunk-size to bench_throughput.py.",
    )
    parser.add_argument(
        "--residual-length",
        type=int,
        default=None,
        help="Override MuZero residual_length for all MuZero methods.",
    )
    parser.add_argument(
        "--cuda-alloc-conf",
        default="expandable_segments:True",
        help=(
            "Set PYTORCH_CUDA_ALLOC_CONF for benchmark subprocesses. "
            "The default reduces DynamicCache reservation/fragmentation during decode; "
            "pass an empty string to leave the environment unchanged."
        ),
    )
    parser.add_argument(
        "--fused-prefix-decode",
        action="store_true",
        help=(
            "Set MUZERO_FUSED_PREFIX_DECODE=1 for subprocesses. "
            "This enables the single-kernel MuZero 4-bit prefix attention path when --decode-attention-mask none is used."
        ),
    )
    parser.add_argument(
        "--prefix-chunk-groups",
        type=int,
        default=2,
        help="Set MUZERO_PREFIX_CHUNK_GROUPS for fused prefix decode chunking.",
    )

    parser.add_argument("--batch-context", type=int, default=8192)
    parser.add_argument("--batch-new-tokens", type=int, default=256)
    parser.add_argument("--batch-sizes", default="48,56,64,66,68,72,80,88,96,104,112,120")

    parser.add_argument("--sequence-axis", choices=("context", "new_tokens"), default="context")
    parser.add_argument("--sequence-batch-size", type=int, default=32)
    parser.add_argument("--sequence-contexts", default="2048,4096,8192,12288,16384,24576,32768")
    parser.add_argument("--sequence-context", type=int, default=8192)
    parser.add_argument("--sequence-new-tokens", type=int, default=256)
    parser.add_argument("--sequence-new-token-values", default="128,256,512,768,1024")
    return parser.parse_args()


def bench_command(
    *,
    config: str,
    method: str,
    context_len: int,
    new_tokens: int,
    batch_size: int,
    warmup: int,
    runs: int,
    mode: str,
    decode_attention_mask: str,
    empty_cache_before_decode: bool,
    prefill_chunk_size: int,
    residual_length: int | None,
    raw_output: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "benchmarks/bench_throughput.py",
        "--config",
        config,
        "--methods",
        method,
        "--contexts",
        str(context_len),
        "--new-tokens",
        str(new_tokens),
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--mode",
        mode,
        "--decode-attention-mask",
        decode_attention_mask,
        "--output",
        str(raw_output),
    ]
    if empty_cache_before_decode:
        cmd.append("--empty-cache-before-decode")
    if prefill_chunk_size > 0:
        cmd.extend(["--prefill-chunk-size", str(prefill_chunk_size)])
    if residual_length is not None and is_muzero_method(method):
        cmd.extend(["--residual-length", str(residual_length)])
    return cmd


def read_single_result(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one result row in {path}, found {len(rows)}")
    return rows[0]


def failure_kind(text: str) -> str:
    lowered = text.lower()
    if "out of memory" in lowered or "cuda error: out of memory" in lowered or "torch.outofmemoryerror" in lowered:
        return "oom"
    return "error"


def run_point(
    *,
    repo_root: Path,
    config: str,
    method: str,
    experiment: str,
    sweep_value: int,
    context_len: int,
    new_tokens: int,
    batch_size: int,
    warmup: int,
    runs: int,
    mode: str,
    decode_attention_mask: str,
    empty_cache_before_decode: bool,
    prefill_chunk_size: int,
    residual_length: int | None,
    cuda_alloc_conf: str,
    fused_prefix_decode: bool,
    prefix_chunk_groups: int,
    raw_dir: Path,
) -> dict[str, object]:
    stem = f"{experiment}_{sweep_value}_{method}_ctx{context_len}_nt{new_tokens}_bs{batch_size}"
    raw_csv = raw_dir / f"{stem}.csv"
    raw_log = raw_dir / f"{stem}.log"
    cmd = bench_command(
        config=config,
        method=method,
        context_len=context_len,
        new_tokens=new_tokens,
        batch_size=batch_size,
        warmup=warmup,
        runs=runs,
        mode=mode,
        decode_attention_mask=decode_attention_mask,
        empty_cache_before_decode=empty_cache_before_decode,
        prefill_chunk_size=prefill_chunk_size,
        residual_length=residual_length,
        raw_output=raw_csv,
    )
    print(f"[collect] {experiment}={sweep_value} method={method} ctx={context_len} new={new_tokens} batch={batch_size}", flush=True)
    env = os.environ.copy()
    if cuda_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf
    if fused_prefix_decode:
        env["MUZERO_FUSED_PREFIX_DECODE"] = "1"
        env["MUZERO_PREFIX_CHUNK_GROUPS"] = str(prefix_chunk_groups)
    proc = subprocess.run(cmd, cwd=repo_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    raw_log.write_text(proc.stdout, encoding="utf-8")

    base = {
        "experiment": experiment,
        "sweep_value": sweep_value,
        "method": method,
        "context_len": context_len,
        "new_tokens": new_tokens,
        "batch_size": batch_size,
        "mode": mode,
        "decode_attention_mask": decode_attention_mask,
        "empty_cache_before_decode": int(empty_cache_before_decode),
        "prefill_chunk_size": prefill_chunk_size,
        "residual_length": "" if residual_length is None or not is_muzero_method(method) else residual_length,
        "cuda_alloc_conf": cuda_alloc_conf,
        "fused_prefix_decode": int(fused_prefix_decode),
        "prefix_chunk_groups": "" if not fused_prefix_decode else prefix_chunk_groups,
        "command": " ".join(cmd),
        "log_path": str(raw_log),
    }
    if proc.returncode != 0 or not raw_csv.exists():
        return {
            **base,
            "status": failure_kind(proc.stdout),
            "actual_context_len": "",
            "generated_tokens": "",
            "avg_latency_s": "",
            "tokens_per_sec": "",
            "peak_mem_mb": "",
            "peak_reserved_mb": "",
            "cache_packed_mb": "",
            "cache_meta_mb": "",
            "cache_residual_mb": "",
            "cache_total_mb": "",
            "fused_qk_calls": "",
            "fused_prefix_decode_calls": "",
        }

    row = read_single_result(raw_csv)
    return {
        **base,
        "status": "ok",
        "actual_context_len": row.get("actual_context_len", ""),
        "generated_tokens": row.get("generated_tokens", ""),
        "avg_latency_s": row.get("avg_latency_s", ""),
        "tokens_per_sec": row.get("tokens_per_sec", ""),
        "peak_mem_mb": row.get("peak_mem_mb", ""),
        "peak_reserved_mb": row.get("peak_reserved_mb", ""),
        "cache_packed_mb": row.get("cache_packed_mb", ""),
        "cache_meta_mb": row.get("cache_meta_mb", ""),
        "cache_residual_mb": row.get("cache_residual_mb", ""),
        "cache_total_mb": row.get("cache_total_mb", ""),
        "fused_qk_calls": row.get("fused_qk_calls", ""),
        "fused_prefix_decode_calls": row.get("fused_prefix_decode_calls", ""),
    }


def add_speedups(rows: list[dict[str, object]]) -> None:
    bf16_dynamic = {
        (row["experiment"], row["sweep_value"]): float(row["tokens_per_sec"])
        for row in rows
        if row["status"] == "ok" and row["method"] == "bf16" and row["tokens_per_sec"] != ""
    }
    bf16_static = {
        (row["experiment"], row["sweep_value"]): float(row["tokens_per_sec"])
        for row in rows
        if row["status"] == "ok" and row["method"] == "bf16_static" and row["tokens_per_sec"] != ""
    }
    for row in rows:
        key = (row["experiment"], row["sweep_value"])
        baseline = bf16_dynamic.get(key) or bf16_static.get(key)
        if row["status"] == "ok" and row["method"] not in {"bf16", "bf16_static"} and baseline:
            row["speedup_vs_bf16"] = f"{float(row['tokens_per_sec']) / baseline:.4f}"
        else:
            row["speedup_vs_bf16"] = ""


SUMMARY_FIELDNAMES = [
    "experiment",
    "sweep_value",
    "method",
    "status",
    "peak_reserved_mb",
    "peak_mem_mb",
    "tokens_per_sec",
    "speedup_vs_bf16",
    "context_len",
    "new_tokens",
    "batch_size",
    "cache_total_mb",
    "avg_latency_s",
    "generated_tokens",
    "log_path",
]


DETAIL_FIELDNAMES = [
        "experiment",
        "sweep_value",
        "method",
        "status",
        "batch_size",
        "context_len",
        "new_tokens",
        "tokens_per_sec",
        "speedup_vs_bf16",
        "peak_mem_mb",
        "peak_reserved_mb",
        "cache_total_mb",
        "cache_packed_mb",
        "cache_meta_mb",
        "cache_residual_mb",
        "actual_context_len",
        "generated_tokens",
        "avg_latency_s",
        "fused_qk_calls",
        "fused_prefix_decode_calls",
        "residual_length",
        "prefill_chunk_size",
        "decode_attention_mask",
        "empty_cache_before_decode",
        "cuda_alloc_conf",
        "fused_prefix_decode",
        "prefix_chunk_groups",
        "mode",
        "command",
        "log_path",
]


def details_path_for(path: Path) -> Path:
    return path.with_name(f"{path.stem}.details{path.suffix}")


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    write_csv_rows(path, rows, SUMMARY_FIELDNAMES)
    write_csv_rows(details_path_for(path), rows, DETAIL_FIELDNAMES)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    raw_dir = output.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(method for method in methods if method not in ALLOWED_METHODS and not is_logq_zeroanchor_method(method))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    rows: list[dict[str, object]] = []
    for batch_size in parse_int_list(args.batch_sizes):
        for method in methods:
            rows.append(
                run_point(
                    repo_root=repo_root,
                    config=args.config,
                    method=method,
                    experiment="batch",
                    sweep_value=batch_size,
                    context_len=args.batch_context,
                    new_tokens=args.batch_new_tokens,
                    batch_size=batch_size,
                    warmup=args.warmup,
                    runs=args.runs,
                    mode=args.mode,
                    decode_attention_mask=args.decode_attention_mask,
                    empty_cache_before_decode=args.empty_cache_before_decode,
                    prefill_chunk_size=args.prefill_chunk_size,
                    residual_length=args.residual_length,
                    cuda_alloc_conf=args.cuda_alloc_conf,
                    fused_prefix_decode=args.fused_prefix_decode,
                    prefix_chunk_groups=args.prefix_chunk_groups,
                    raw_dir=raw_dir,
                )
            )
            add_speedups(rows)
            write_rows(output, rows)

    if args.sequence_axis == "context":
        sequence_points = [(ctx, ctx, args.sequence_new_tokens) for ctx in parse_int_list(args.sequence_contexts)]
        experiment = "context"
    else:
        sequence_points = [(nt, args.sequence_context, nt) for nt in parse_int_list(args.sequence_new_token_values)]
        experiment = "new_tokens"

    for sweep_value, context_len, new_tokens in sequence_points:
        for method in methods:
            rows.append(
                run_point(
                    repo_root=repo_root,
                    config=args.config,
                    method=method,
                    experiment=experiment,
                    sweep_value=sweep_value,
                    context_len=context_len,
                    new_tokens=new_tokens,
                    batch_size=args.sequence_batch_size,
                    warmup=args.warmup,
                    runs=args.runs,
                    mode=args.mode,
                    decode_attention_mask=args.decode_attention_mask,
                    empty_cache_before_decode=args.empty_cache_before_decode,
                    prefill_chunk_size=args.prefill_chunk_size,
                    residual_length=args.residual_length,
                    cuda_alloc_conf=args.cuda_alloc_conf,
                    fused_prefix_decode=args.fused_prefix_decode,
                    prefix_chunk_groups=args.prefix_chunk_groups,
                    raw_dir=raw_dir,
                )
            )
            add_speedups(rows)
            write_rows(output, rows)

    print(f"[collect] wrote {len(rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()

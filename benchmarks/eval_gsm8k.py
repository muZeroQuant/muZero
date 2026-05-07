#!/usr/bin/env python
"""Run GSM8K CoT evaluation with MuZero KV-cache quantization."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import muzero

from eval_utils import (
    build_method_cache_kwargs,
    apply_yaml_config,
    exact_gsm8k_match,
    extract_gsm8k_answer,
    ensure_dataset_path,
    generate_text,
    load_model_and_tokenizer,
    mean,
    resolve_repo_path,
    set_seed,
    write_csv,
    write_json,
    write_jsonl,
    write_run_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--method", default=None, help="bf16, mu_zero_{bits}bit, or logq{bits}_zeroanchor_stat_cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--amax-dtype", default="float16")
    parser.add_argument("--stats-max-points", type=int, default=65536)
    parser.add_argument("--logq-compatible-layout", action="store_true")
    parser.add_argument("--decode-flush-by-group", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dense-dequant", action="store_true", help="disable MuZero fused decode and use materialized dequantized K/V attention")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset-name", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-shot", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None, help="omit for full split")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="runs/gsm8k_muzero")
    args = parser.parse_args()
    apply_yaml_config(args, parser, sys.argv, ("model", "method", "data", "run", "eval"))
    if not args.model:
        parser.error("--model is required unless provided by --config")
    if args.method is None:
        args.method = f"mu_zero_{args.bits}bit"
    return args


def build_fewshot_prompt(train_rows, question: str, n_shot: int) -> str:
    examples = []
    for row in train_rows.select(range(min(n_shot, len(train_rows)))):
        examples.append(f"Question: {row['question']}\nAnswer: {row['answer']}")
    examples.append(f"Question: {question}\nAnswer: Let's think step by step.")
    return "\n\n".join(examples)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.dense_dequant:
        os.environ["MUZERO_DISABLE_FUSED_DECODE"] = "1"
    if args.method.startswith("mu_zero_"):
        muzero.patch_transformers()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("GSM8K evaluation requires `datasets`. Install with `pip install datasets`.") from exc

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device)
    dataset_name_or_path = ensure_dataset_path(args.dataset)
    print(f"[dataset] loading `{dataset_name_or_path}` split={args.split}", flush=True)
    train_rows = load_dataset(dataset_name_or_path, args.dataset_name, split="train")
    eval_rows = load_dataset(dataset_name_or_path, args.dataset_name, split=args.split)
    if args.max_samples is not None:
        eval_rows = eval_rows.select(range(min(args.max_samples, len(eval_rows))))

    generation_kwargs = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        **build_method_cache_kwargs(
            args.method,
            bits=args.bits,
            q_group_size=args.q_group_size,
            residual_length=args.residual_length,
            backend=args.backend,
            amax_dtype=args.amax_dtype,
            stats_max_points=args.stats_max_points,
            logq_compatible_layout=args.logq_compatible_layout,
            decode_flush_by_group=args.decode_flush_by_group,
        ),
    }

    output_dir = resolve_repo_path(args.output_dir)
    write_run_config(output_dir / "run_config.json", args, generation_kwargs=generation_kwargs)

    rows = []
    for index, sample in enumerate(eval_rows):
        prompt = build_fewshot_prompt(train_rows, sample["question"], args.n_shot)
        text, input_tokens, output_tokens, elapsed, peak_memory = generate_text(
            model,
            tokenizer,
            prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            generation_kwargs=copy.deepcopy(generation_kwargs),
        )
        row = {
            "sample_index": index,
            "question": sample["question"],
            "reference": sample["answer"],
            "prediction": text,
            "prediction_answer": extract_gsm8k_answer(text),
            "reference_answer": extract_gsm8k_answer(sample["answer"]),
            "exact_match": exact_gsm8k_match(text, sample["answer"]),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": output_tokens / max(elapsed, 1e-8),
            "peak_memory_bytes": peak_memory,
        }
        rows.append(row)
        print(
            f"[gsm8k] {index + 1}/{len(eval_rows)} exact={row['exact_match']:.0f} "
            f"running_em={mean([r['exact_match'] for r in rows]):.4f} "
            f"pred={row['prediction_answer']} ref={row['reference_answer']}",
            flush=True,
        )

    summary = {
        "task": "gsm8k",
        "model": args.model,
        "method": args.method,
        "num_samples": len(rows),
        "exact_match": mean([row["exact_match"] for row in rows]),
        "avg_tokens_per_second": mean([row["tokens_per_second"] for row in rows]),
        "avg_peak_memory_bytes": mean([row["peak_memory_bytes"] for row in rows]),
    }
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_csv(output_dir / "results.csv", rows)
    write_json(output_dir / "summary.json", summary)
    print(summary)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

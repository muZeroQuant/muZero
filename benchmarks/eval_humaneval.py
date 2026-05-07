#!/usr/bin/env python
"""Run HumanEval pass@k evaluation with MuZero KV-cache quantization."""

from __future__ import annotations

import argparse
import copy
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import muzero

from eval_utils import (
    build_method_cache_kwargs,
    apply_yaml_config,
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


DEFAULT_STOP_STRINGS = ["\nclass ", "\ndef ", "\nif __name__ == '__main__':", '\nif __name__ == "__main__":', "\nprint(", "\n#"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--method", default=None, help="mu_zero_{bits}bit or logq{bits}_zeroanchor_stat_cuda")
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
    parser.add_argument("--dataset", default="openai/openai_humaneval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None, help="omit for full split")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-samples-per-task", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--pass-k", default="1,10")
    parser.add_argument("--execution-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="runs/humaneval_muzero")
    args = parser.parse_args()
    apply_yaml_config(args, parser, sys.argv, ("model", "method", "data", "run", "eval"))
    if not args.model:
        parser.error("--model is required unless provided by --config")
    if args.method is None:
        args.method = f"mu_zero_{args.bits}bit"
    return args


def trim_completion(completion: str, stop_strings: list[str] | None = None) -> str:
    cutoff = len(completion)
    for stop in stop_strings or DEFAULT_STOP_STRINGS:
        index = completion.find(stop)
        if index != -1:
            cutoff = min(cutoff, index)
    return completion[:cutoff].rstrip()


def evaluate_completion(prompt: str, completion: str, test_code: str, entry_point: str, timeout: float) -> dict[str, Any]:
    program = "\n".join([prompt.rstrip(), completion.rstrip(), "", test_code.strip(), "", f"check({entry_point})", ""])
    with tempfile.TemporaryDirectory(prefix="muzero-humaneval-") as tmp_dir:
        script_path = Path(tmp_dir) / "candidate.py"
        script_path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                capture_output=True,
                cwd=tmp_dir,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"passed": 0.0, "status": "timed out", "message": f"Timed out after {timeout:.1f}s."}
    message = (completed.stderr or completed.stdout or "").strip()
    if len(message) > 500:
        message = message[:497] + "..."
    if completed.returncode == 0:
        return {"passed": 1.0, "status": "passed", "message": message}
    return {"passed": 0.0, "status": "failed", "message": message or f"Exited with code {completed.returncode}."}


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples < k:
        return float("nan")
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / value for value in range(num_samples - num_correct + 1, num_samples + 1))


def main() -> None:
    args = parse_args()
    pass_k = sorted({int(value) for value in args.pass_k.split(",") if value.strip()})
    if args.method.startswith("mu_zero_"):
        muzero.patch_transformers()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("HumanEval evaluation requires `datasets`. Install with `pip install datasets`.") from exc

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device)
    dataset_name_or_path = ensure_dataset_path(args.dataset)
    print(f"[dataset] loading `{dataset_name_or_path}` split={args.split}", flush=True)
    tasks = load_dataset(dataset_name_or_path, split=args.split)
    if args.max_samples is not None:
        tasks = tasks.select(range(min(args.max_samples, len(tasks))))

    generation_kwargs = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
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
    write_run_config(output_dir / "run_config.json", args, generation_kwargs=generation_kwargs, pass_k=pass_k)

    rows = []
    task_rows = []
    for task_index, task in enumerate(tasks):
        passed = []
        for completion_id in range(args.n_samples_per_task):
            seed = args.seed + task_index * args.n_samples_per_task + completion_id
            set_seed(seed)
            text, input_tokens, output_tokens, elapsed, peak_memory = generate_text(
                model,
                tokenizer,
                task["prompt"],
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                max_prompt_tokens=args.max_prompt_tokens,
                generation_kwargs=copy.deepcopy(generation_kwargs),
            )
            completion = trim_completion(text)
            result = evaluate_completion(task["prompt"], completion, task["test"], task["entry_point"], args.execution_timeout)
            passed.append(int(result["passed"]))
            row = {
                "task_id": task.get("task_id", ""),
                "entry_point": task.get("entry_point", ""),
                "completion_id": completion_id,
                "seed": seed,
                "completion": completion,
                "passed": float(result["passed"]),
                "execution_status": result["status"],
                "execution_message": result["message"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_seconds": elapsed,
                "tokens_per_second": output_tokens / max(elapsed, 1e-8),
                "peak_memory_bytes": peak_memory,
            }
            rows.append(row)
            print(
                f"[humaneval] task={task_index + 1}/{len(tasks)} completion={completion_id + 1}/{args.n_samples_per_task} "
                f"passed={row['passed']:.0f} running_pass@1={mean([r['passed'] for r in rows]):.4f}",
                flush=True,
            )
        task_row = {
            "task_id": task.get("task_id", ""),
            "num_samples": args.n_samples_per_task,
            "num_correct": sum(passed),
        }
        for k in pass_k:
            task_row[f"pass@{k}"] = estimate_pass_at_k(args.n_samples_per_task, sum(passed), k)
        task_rows.append(task_row)

    summary = {
        "task": "humaneval",
        "model": args.model,
        "method": args.method,
        "num_tasks": len(task_rows),
        "n_samples_per_task": args.n_samples_per_task,
        "avg_tokens_per_second": mean([row["tokens_per_second"] for row in rows]),
        "avg_peak_memory_bytes": mean([row["peak_memory_bytes"] for row in rows]),
    }
    for k in pass_k:
        values = [row[f"pass@{k}"] for row in task_rows if not math.isnan(row[f"pass@{k}"])]
        summary[f"pass@{k}"] = mean(values)

    write_jsonl(output_dir / "completions.jsonl", rows)
    write_csv(output_dir / "results.csv", rows)
    write_csv(output_dir / "task_scores.csv", task_rows)
    write_json(output_dir / "summary.json", summary)
    print(summary)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

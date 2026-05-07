#!/usr/bin/env python
"""Run official LongBench evaluation with MuZero KV-cache quantization."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from transformers import GenerationConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
import muzero

from eval_utils import apply_yaml_config, build_method_cache_kwargs, ensure_longbench_archive, load_model_and_tokenizer, mean, resolve_repo_path, set_seed, write_csv, write_json, write_jsonl, write_run_config


LONG_BENCH_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

LONG_BENCH_E_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

NO_CHAT_WRAP_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

DATASET_GROUPS = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "multi_doc_qa": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "few_shot": ["trec", "triviaqa", "samsum", "lsht"],
    "synthetic": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    "code": ["lcc", "repobench-p"],
}


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
    parser.add_argument("--longbench-root", "--root-dir", dest="longbench_root", default="./data/LongBench")
    parser.add_argument("--longbench-code-repo", default="https://github.com/THUDM/LongBench.git")
    parser.add_argument("--dataset-repo", default="THUDM/LongBench")
    parser.add_argument("--data-archive", default=None)
    parser.add_argument("--datasets", default="", help="Comma-separated datasets; empty means all official datasets")
    parser.add_argument("--use-e", action="store_true", help="Evaluate LongBench-E split/tasks")
    parser.add_argument("--model-name", default=None, help="LongBench model name for official max length lookup")
    parser.add_argument(
        "--chat-template",
        default="off",
        choices=["off", "hf", "official", "official_longbench", "qjl", "official_qjl"],
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=0, help="0 means infer from LongBench config/model")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 means use LongBench dataset default")
    parser.add_argument("--max-samples", type=int, default=None, help="omit for full dataset")
    parser.add_argument("--deterministic", action="store_true", help="disable atomic MuZero AV kernels for reproducible greedy decoding")
    parser.add_argument("--dense-dequant", action="store_true", help="disable MuZero fused decode and use materialized dequantized K/V attention")
    parser.add_argument("--log-cuda-memory", action="store_true", help="print per-sample allocated/reserved CUDA memory after cleanup")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="runs/longbench_muzero")
    args = parser.parse_args()
    apply_yaml_config(args, parser, sys.argv, ("model", "method", "longbench", "data", "run", "eval"))
    if isinstance(args.datasets, list):
        args.datasets = ",".join(str(value) for value in args.datasets)
    if not args.model:
        parser.error("--model is required unless provided by --config")
    if args.method is None:
        args.method = f"mu_zero_{args.bits}bit"
    return args


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_longbench_repo(root: Path, repo_url: str) -> None:
    if (root / "LongBench" / "eval.py").exists():
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            f"Missing LongBench/eval.py under {root}, and the directory is not empty. "
            "Clone the official LongBench repo there or set --longbench-root."
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"[longbench] cloning official LongBench repo into `{root}`", flush=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(root)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"Missing LongBench/eval.py under {root}, and automatic clone failed. "
            "Clone https://github.com/THUDM/LongBench.git there or set --longbench-root."
        ) from exc


def load_longbench_helpers(root: Path, repo_url: str) -> tuple[dict[str, Any], Any, Any]:
    ensure_longbench_repo(root, repo_url)
    bench_dir = root / "LongBench"
    if not (bench_dir / "eval.py").exists():
        raise ValueError(f"Missing LongBench/eval.py under {root}. Clone the official LongBench repo or set --longbench-root.")
    sys.path.insert(0, str(bench_dir))
    try:
        eval_module = load_module(bench_dir / "eval.py", "muzero_longbench_eval")
    finally:
        sys.path.remove(str(bench_dir))
    return eval_module.dataset2metric, eval_module.scorer, eval_module.scorer_e


def load_longbench_configs(root: Path) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
    config_dir = root / "LongBench" / "config"
    return (
        json.loads((config_dir / "dataset2prompt.json").read_text(encoding="utf-8")),
        json.loads((config_dir / "dataset2maxlen.json").read_text(encoding="utf-8")),
        json.loads((config_dir / "model2maxlen.json").read_text(encoding="utf-8")),
    )


def resolve_eval_limit(args: argparse.Namespace, key: str) -> Any:
    return getattr(args, key)


def load_longbench_samples(args: argparse.Namespace, dataset: str) -> list[dict[str, Any]]:
    archive_path = ensure_longbench_archive(args.data_archive, args.dataset_repo)
    if archive_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("LongBench evaluation requires `huggingface_hub` or --data-archive.") from exc
        print(f"[dataset] LongBench archive not provided; downloading `data.zip` from `{args.dataset_repo}` if needed.", flush=True)
        archive_path = hf_hub_download(repo_id=args.dataset_repo, repo_type="dataset", filename="data.zip")
    member_name = f"data/{dataset}.jsonl"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as handle:
            return [json.loads(line.decode("utf-8")) for line in handle if line.strip()]


def get_dataset_names(args: argparse.Namespace) -> list[str]:
    if args.datasets.strip():
        return [name.strip() for name in args.datasets.split(",") if name.strip()]
    return list(LONG_BENCH_E_DATASETS if args.use_e else LONG_BENCH_DATASETS)


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    return Path(args.model).name.lower()


def infer_max_prompt_tokens(model, tokenizer, args: argparse.Namespace, model_max_len_map: dict[str, int]) -> int:
    if args.max_prompt_tokens is not None:
        return int(args.max_prompt_tokens)
    official = model_max_len_map.get(resolve_model_name(args))
    if official is not None:
        return int(official)
    tokenizer_max = int(getattr(tokenizer, "model_max_length", 0) or 0)
    text_config = model.config.get_text_config(decoder=True)
    max_positions = getattr(text_config, "max_position_embeddings", None)
    candidates = []
    if 0 < tokenizer_max < 1_000_000:
        candidates.append(tokenizer_max)
    if max_positions is not None:
        candidates.append(int(max_positions))
    if not candidates:
        raise ValueError("Unable to infer max prompt tokens; pass --max-prompt-tokens.")
    return min(candidates)


def truncate_prompt_middle(tokenizer, prompt: str, max_length: int) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_length:
        return prompt
    half = max_length // 2
    return tokenizer.decode(token_ids[:half], skip_special_tokens=True) + tokenizer.decode(token_ids[-half:], skip_special_tokens=True)


def build_official_chat(tokenizer, prompt: str, model_name: str) -> str:
    name = model_name.lower()
    if "chatglm3" in name and hasattr(tokenizer, "build_chat_input"):
        built = tokenizer.build_chat_input(prompt)
        return built if isinstance(built, str) else prompt
    if "chatglm" in name and hasattr(tokenizer, "build_prompt"):
        return str(tokenizer.build_prompt(prompt))
    if "longchat" in name or "vicuna" in name:
        try:
            from fastchat.model import get_conversation_template
        except ImportError:
            return prompt
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return str(conv.get_prompt())
    if "llama2" in name:
        return f"[INST]{prompt}[/INST]"
    if "xgen" in name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        return header + f" ### Human: {prompt}\n###"
    if "internlm" in name:
        return f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def maybe_wrap_prompt(tokenizer, prompt: str, dataset: str, args: argparse.Namespace) -> str:
    chat_template = args.chat_template
    if isinstance(chat_template, bool):
        chat_template = "official" if chat_template else "off"
    chat_template = str(chat_template)
    if chat_template == "off" or dataset in NO_CHAT_WRAP_DATASETS:
        return prompt
    if chat_template in {"official", "official_longbench"}:
        return build_official_chat(tokenizer, prompt, resolve_model_name(args))
    if chat_template in {"qjl", "official_qjl"}:
        return f"[INST]{prompt}[/INST]"
    if hasattr(tokenizer, "apply_chat_template"):
        return str(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))
    return prompt


def group_scores(dataset_scores: dict[str, float]) -> dict[str, float]:
    grouped = {}
    for group, datasets in DATASET_GROUPS.items():
        values = [dataset_scores[name] for name in datasets if name in dataset_scores]
        if values:
            grouped[group] = round(sum(values) / len(values), 2)
    return grouped


def cuda_memory_stats(device: str) -> dict[str, int]:
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device=device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device=device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device=device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device=device)),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.max_prompt_tokens == 0:
        args.max_prompt_tokens = None
    if args.deterministic:
        os.environ["MUZERO_DETERMINISTIC_AV"] = "1"
    if args.dense_dequant:
        os.environ["MUZERO_DISABLE_FUSED_DECODE"] = "1"
    if args.method.startswith("mu_zero_"):
        muzero.patch_transformers()
    root = resolve_repo_path(args.longbench_root).resolve()
    _, scorer, scorer_e = load_longbench_helpers(root, args.longbench_code_repo)
    prompt_map, max_gen_map, model_max_len_map = load_longbench_configs(root)
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device)
    if not hasattr(model, "generation_config") or model.generation_config is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model_max_length = infer_max_prompt_tokens(model, tokenizer, args, model_max_len_map)
    cache_kwargs = build_method_cache_kwargs(
        args.method,
        bits=args.bits,
        q_group_size=args.q_group_size,
        residual_length=args.residual_length,
        backend=args.backend,
        amax_dtype=args.amax_dtype,
        stats_max_points=args.stats_max_points,
        logq_compatible_layout=args.logq_compatible_layout,
        decode_flush_by_group=args.decode_flush_by_group,
    )

    output_dir = resolve_repo_path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    write_run_config(
        output_dir / "run_config.json",
        args,
        cache_kwargs=cache_kwargs,
        model_max_length=model_max_length,
        resolved_datasets=get_dataset_names(args),
    )
    dataset_rows = []
    prediction_rows = []
    dataset_scores: dict[str, float] = {}
    dataset_scores_e: dict[str, dict[str, float]] = {}

    for dataset_name in get_dataset_names(args):
        hf_dataset_name = f"{dataset_name}_e" if args.use_e else dataset_name
        samples = load_longbench_samples(args, hf_dataset_name)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        prompt_format = prompt_map[dataset_name]
        max_gen = int(resolve_eval_limit(args, "max_new_tokens") or max_gen_map[dataset_name])
        prompt_max_length = max(1, model_max_length - max_gen)
        predictions = []
        answers = []
        lengths = []
        all_classes = None
        output_path = predictions_dir / f"{dataset_name}.jsonl"

        print(f"[longbench] dataset={dataset_name} samples={len(samples)} max_new_tokens={max_gen}", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as prediction_handle:
            for sample_index, sample in enumerate(samples):
                print(
                    f"[longbench] sample {sample_index + 1}/{len(samples)} for `{dataset_name}`",
                    flush=True,
                )
                prompt = prompt_format.format(**sample)
                prompt = truncate_prompt_middle(tokenizer, prompt, prompt_max_length)
                prompt = maybe_wrap_prompt(tokenizer, prompt, dataset_name, args)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device=args.device)
                    torch.cuda.synchronize(device=args.device)
                pre_memory = cuda_memory_stats(args.device)
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_gen,
                    "num_beams": 1,
                    "do_sample": False,
                    "temperature": 1.0,
                    "pad_token_id": tokenizer.eos_token_id,
                    "disable_compile": True,
                    **copy.deepcopy(cache_kwargs),
                }
                if dataset_name == "samsum":
                    eos_ids = [tokenizer.eos_token_id]
                    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
                    if newline_tokens:
                        eos_ids.append(int(newline_tokens[-1]))
                    generation_kwargs["eos_token_id"] = eos_ids
                    generation_kwargs["min_length"] = int(inputs["input_ids"].shape[-1]) + 1
                start = time.perf_counter()
                print(
                    f"[longbench] generating sample {sample_index + 1}/{len(samples)} for `{dataset_name}` "
                    f"input_tokens={int(inputs['input_ids'].shape[-1])}",
                    flush=True,
                )
                outputs = model.generate(**inputs, **generation_kwargs)
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.synchronize(device=args.device)
                elapsed = time.perf_counter() - start
                input_length = int(inputs["input_ids"].shape[-1])
                generated_tokens = int(outputs.shape[-1] - input_length)
                text = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
                peak_memory = (
                    int(torch.cuda.max_memory_allocated(device=args.device))
                    if torch.cuda.is_available() and str(args.device).startswith("cuda")
                    else 0
                )
                peak_reserved_memory = (
                    int(torch.cuda.max_memory_reserved(device=args.device))
                    if torch.cuda.is_available() and str(args.device).startswith("cuda")
                    else 0
                )
                record = {
                    "pred": text,
                    "answers": sample["answers"],
                    "all_classes": sample.get("all_classes"),
                    "length": int(sample.get("length", 0)),
                }
                prediction_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                predictions.append(text)
                answers.append(list(sample["answers"]))
                lengths.append(int(sample.get("length", 0)))
                all_classes = sample.get("all_classes")
                print(f"[longbench] {dataset_name} {sample_index + 1}/{len(samples)} tokens={generated_tokens} elapsed={elapsed:.2f}s", flush=True)
                del inputs, outputs, generation_kwargs
                gc.collect()
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device=args.device)
                post_memory = cuda_memory_stats(args.device)
                if args.log_cuda_memory:
                    print(
                        "[longbench] cuda_memory "
                        f"dataset={dataset_name} sample={sample_index + 1} "
                        f"pre_alloc={pre_memory['allocated_bytes']} pre_reserved={pre_memory['reserved_bytes']} "
                        f"peak_alloc={peak_memory} peak_reserved={peak_reserved_memory} "
                        f"post_alloc={post_memory['allocated_bytes']} post_reserved={post_memory['reserved_bytes']}",
                        flush=True,
                    )
                prediction_rows.append(
                    {
                        "dataset": dataset_name,
                        "sample_index": sample_index,
                        "input_tokens": input_length,
                        "output_tokens": generated_tokens,
                        "elapsed_seconds": elapsed,
                        "tokens_per_second": generated_tokens / max(elapsed, 1e-8),
                        "pre_allocated_bytes": pre_memory["allocated_bytes"],
                        "pre_reserved_bytes": pre_memory["reserved_bytes"],
                        "peak_memory_bytes": peak_memory,
                        "peak_reserved_bytes": peak_reserved_memory,
                        "post_allocated_bytes": post_memory["allocated_bytes"],
                        "post_reserved_bytes": post_memory["reserved_bytes"],
                    }
                )

        if args.use_e:
            score_e = scorer_e(dataset_name, predictions, answers, lengths, all_classes)
            dataset_scores_e[dataset_name] = score_e
            dataset_score = float(round(sum(score_e.values()) / max(1, len(score_e)), 2))
        else:
            dataset_score = float(scorer(dataset_name, predictions, answers, all_classes))
        dataset_scores[dataset_name] = dataset_score
        dataset_token_rows = [row for row in prediction_rows if row["dataset"] == dataset_name]
        dataset_rows.append(
            {
                "dataset": dataset_name,
                "score": dataset_score,
                "num_samples": len(predictions),
                "avg_tokens_per_second": mean([row["tokens_per_second"] for row in dataset_token_rows]),
                "avg_output_tokens": mean([row["output_tokens"] for row in dataset_token_rows]),
                "avg_peak_memory_bytes": mean([row["peak_memory_bytes"] for row in dataset_token_rows]),
                "prediction_path": str(output_path),
            }
        )
        print(f"[longbench] finished {dataset_name} score={dataset_score}", flush=True)
        del samples, predictions, answers, lengths
        gc.collect()
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary: dict[str, Any] = {
        "task": "longbench_e" if args.use_e else "longbench",
        "model": args.model,
        "method": args.method,
        "datasets": list(dataset_scores),
        "scores_by_dataset": dataset_scores,
        "group_scores": group_scores(dataset_scores),
        "longbench_score": round(sum(dataset_scores.values()) / max(1, len(dataset_scores)), 2),
        "avg_tokens_per_second": mean([row["tokens_per_second"] for row in prediction_rows]),
        "avg_peak_memory_bytes": mean([row["peak_memory_bytes"] for row in prediction_rows]),
    }
    if args.use_e:
        summary["scores_by_dataset_e"] = dataset_scores_e
    write_csv(output_dir / "scores_by_dataset.csv", dataset_rows)
    write_csv(output_dir / "prediction_stats.csv", prediction_rows)
    write_json(output_dir / "summary.json", summary)
    print(summary)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

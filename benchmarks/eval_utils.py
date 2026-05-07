from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


MODEL_REPO_ALIASES = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama31-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

DATASET_REPO_ALIASES = {
    "gsm8k": "openai/gsm8k",
    "openai_humaneval": "openai/openai_humaneval",
    "humaneval": "openai/openai_humaneval",
    "longbench": "THUDM/LongBench",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _looks_like_local_path(value: str) -> bool:
    return value.startswith(('/', './', '../', '~')) or value.count('/') > 1


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve relative local paths from the muZero repository root."""
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return REPO_ROOT / expanded


def _infer_repo_id(path_or_name: str, aliases: dict[str, str], artifact_kind: str) -> str:
    normalized = Path(path_or_name).name.lower()
    if normalized in aliases:
        return aliases[normalized]
    if '/' in path_or_name and not _looks_like_local_path(path_or_name):
        return path_or_name
    raise ValueError(
        f"Missing local {artifact_kind} path `{path_or_name}` and no repository alias is known for `{normalized}`. "
        f"Use a Hugging Face repo id or add an alias in benchmarks/eval_utils.py."
    )


def ensure_hf_snapshot(path_or_repo: str, *, repo_type: str, aliases: dict[str, str], artifact_kind: str) -> str:
    expanded = resolve_repo_path(path_or_repo) if _looks_like_local_path(path_or_repo) else Path(path_or_repo).expanduser()
    if _looks_like_local_path(path_or_repo) and expanded.exists():
        return str(expanded)
    if _looks_like_local_path(path_or_repo):
        repo_id = _infer_repo_id(path_or_repo, aliases, artifact_kind)
        print(
            f"[{artifact_kind}] local path `{expanded}` is missing; downloading `{repo_id}` to that directory.",
            flush=True,
        )
        from huggingface_hub import snapshot_download

        expanded.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(expanded), local_dir_use_symlinks=False)
        print(f"[{artifact_kind}] download complete: {expanded}", flush=True)
        return str(expanded)
    alias = aliases.get(path_or_repo.lower())
    if alias is not None:
        print(f"[{artifact_kind}] using repository alias `{path_or_repo}` -> `{alias}`", flush=True)
        return alias
    return path_or_repo


def ensure_model_path(model_path: str) -> str:
    return ensure_hf_snapshot(model_path, repo_type="model", aliases=MODEL_REPO_ALIASES, artifact_kind="model")


def ensure_dataset_path(dataset_path: str) -> str:
    return ensure_hf_snapshot(dataset_path, repo_type="dataset", aliases=DATASET_REPO_ALIASES, artifact_kind="dataset")


def ensure_longbench_archive(data_archive: str | None, dataset_repo: str) -> str | None:
    if data_archive is None:
        return None
    archive_path = resolve_repo_path(data_archive)
    if archive_path.exists():
        return str(archive_path)
    print(
        f"[dataset] LongBench archive `{archive_path}` is missing; downloading `data.zip` from `{dataset_repo}`.",
        flush=True,
    )
    from huggingface_hub import hf_hub_download

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        filename="data.zip",
        local_dir=str(archive_path.parent),
        local_dir_use_symlinks=False,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != archive_path:
        downloaded_path.replace(archive_path)
    print(f"[dataset] LongBench archive ready: {archive_path}", flush=True)
    return str(archive_path)


def load_model_and_tokenizer(model_path: str, dtype: str, device: str, trust_remote_code: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = ensure_model_path(model_path)
    print(f"[model] loading `{model_path}`", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(torch, dtype),
        trust_remote_code=trust_remote_code,
    ).to(device).eval()
    return model, tokenizer


def build_cache_kwargs(
    bits: int,
    q_group_size: int,
    residual_length: int,
    backend: str,
    amax_dtype: str,
    logq_compatible_layout: bool = False,
    decode_flush_by_group: bool = True,
) -> dict[str, Any]:
    return {
        "cache_implementation": f"mu_zero_{bits}bit",
        "cache_config": {
            "q_group_size": q_group_size,
            "residual_length": residual_length,
            "backend": backend,
            "amax_dtype": amax_dtype,
            "logq_compatible_layout": logq_compatible_layout,
            "decode_flush_by_group": decode_flush_by_group,
        },
    }


def build_method_cache_kwargs(
    method: str,
    *,
    bits: int,
    q_group_size: int,
    residual_length: int,
    backend: str,
    amax_dtype: str,
    stats_max_points: int = 65536,
    logq_compatible_layout: bool = False,
    decode_flush_by_group: bool = True,
) -> dict[str, Any]:
    if method in {"bf16", "raw"}:
        return {}
    if method.startswith("mu_zero_"):
        return build_cache_kwargs(
            bits,
            q_group_size,
            residual_length,
            backend,
            amax_dtype,
            logq_compatible_layout,
            decode_flush_by_group,
        )
    if re.fullmatch(r"logq\d+_zeroanchor_stat_cuda", method):
        return {
            "cache_implementation": "quantized",
            "cache_config": {
                "backend": method,
                "q_group_size": q_group_size,
                "residual_length": residual_length,
                "amax_dtype": amax_dtype,
                "stats_max_points": stats_max_points,
            },
        }
    raise ValueError(f"Unsupported benchmark method: {method}")


@torch.inference_mode()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
    max_prompt_tokens: int,
    generation_kwargs: dict[str, Any] | None = None,
) -> tuple[str, int, int, float, int]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)
    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        **dict(generation_kwargs or {}),
    )
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device=device)
    elapsed = time.perf_counter() - start
    input_tokens = int(inputs["input_ids"].shape[-1])
    output_tokens = int(outputs.shape[-1] - input_tokens)
    text = tokenizer.decode(outputs[0, input_tokens:], skip_special_tokens=True)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device=device))
        if torch.cuda.is_available() and str(device).startswith("cuda")
        else 0
    )
    return text, input_tokens, output_tokens, elapsed, peak_memory


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_run_config(path: str | Path, args: Any, **extra: Any) -> None:
    env_keys = [
        "MUZERO_DETERMINISTIC_AV",
        "MUZERO_DISABLE_FUSED_DECODE",
        "MUZERO_FUSED_PREFIX_DECODE",
        "MUZERO_FUSED_PREFIX_DECODE_LEGACY",
        "MUZERO_PREFIX_CHUNK_GROUPS",
        "CUDA_VISIBLE_DEVICES",
        "TORCH_CUDA_ARCH_LIST",
    ]
    data = {
        "argv": list(sys.argv),
        "args": _jsonable(vars(args)),
        "env": {key: os.environ[key] for key in env_keys if key in os.environ},
        **{key: _jsonable(value) for key, value in extra.items()},
    }
    write_json(path, data)


def extract_gsm8k_answer(text: str) -> str:
    text = str(text)
    marker_match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if marker_match:
        return marker_match.group(1).replace(",", "")
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else ""


def exact_gsm8k_match(prediction: str, reference: str) -> float:
    return float(extract_gsm8k_answer(prediction) == extract_gsm8k_answer(reference))


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def apply_yaml_config(args, parser, argv: list[str], sections: tuple[str, ...]) -> None:
    config_path = getattr(args, "config", None)
    if not config_path:
        return
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("YAML configs require `pyyaml`. Install with `pip install pyyaml`.") from exc

    with Path(config_path).expanduser().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    cli_keys = {
        token[2:].replace("-", "_")
        for token in argv[1:]
        if token.startswith("--") and token != "--config"
    }
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }

    flattened: dict[str, Any] = {}
    for key, value in config.items():
        if key in sections and isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value

    if "name_or_path" in flattened and "model" not in flattened:
        flattened["model"] = flattened["name_or_path"]
    if "root_dir" in flattened and "longbench_root" not in flattened:
        flattened["longbench_root"] = flattened["root_dir"]
    params = flattened.get("params")
    if isinstance(params, dict):
        for key, value in params.items():
            flattened.setdefault(key, value)
    method_name = str(flattened.get("name", ""))
    if method_name and "method" not in flattened:
        flattened["method"] = method_name
    match = re.fullmatch(r"mu_zero_(\d+)bit", method_name)
    if match and "bits" not in flattened:
        flattened["bits"] = int(match.group(1))
    match = re.fullmatch(r"logq(\d+)_zeroanchor_stat_cuda", method_name)
    if match and "bits" not in flattened:
        flattened["bits"] = int(match.group(1))

    for key, value in flattened.items():
        if not hasattr(args, key) or key in cli_keys:
            continue
        current = getattr(args, key)
        if current == defaults.get(key):
            setattr(args, key, value)

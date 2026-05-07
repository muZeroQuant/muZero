#!/usr/bin/env python
"""Compare generated token IDs across BF16, MuZero, and legacy LogQ paths.

This is a correctness/debugging check, not a quality benchmark. Quantized paths
are approximate, so exact equality with BF16 is not guaranteed. The useful signal
is whether two supposedly equivalent integration paths diverge immediately, fail,
or produce obviously different continuations under greedy decoding.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TURBOQUANT_ROOT = Path("/root/autodl-tmp/turboquant")
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
from eval_utils import ensure_model_path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/autodl-tmp/ann/Qwen3-8B")
    p.add_argument(
        "--methods",
        default="hf_bf16,hf_muzero,hf_legacy_logq",
        help="Comma-separated methods",
    )
    p.add_argument("--prompt", default="Explain why KV cache compression can help long-context decoding.")
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--q-group-size", type=int, default=64)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--output", default=None)
    return p.parse_args()


def _run_method(args, method: str) -> dict:
    code = _method_code(args, method)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        script = f.name
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    extra_pythonpath = [str(REPO_ROOT)]
    if TURBOQUANT_ROOT.exists():
        extra_pythonpath.append(str(TURBOQUANT_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(extra_pythonpath + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        proc = subprocess.run([sys.executable, script], capture_output=True, text=True, env=env, timeout=args.timeout)
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass

    if proc.returncode != 0:
        return {"method": method, "ok": False, "error": proc.stderr[-6000:]}
    for line in reversed(proc.stdout.strip().splitlines()):
        try:
            row = json.loads(line)
            row["ok"] = True
            return row
        except json.JSONDecodeError:
            continue
    return {"method": method, "ok": False, "error": proc.stdout[-6000:]}


def _method_code(args, method: str) -> str:
    return f'''
from __future__ import annotations

import json
import torch

METHOD = {method!r}
MODEL = {args.model!r}
PROMPT = {args.prompt!r}
CONTEXT_LEN = {args.context_len}
NEW_TOKENS = {args.new_tokens}
DTYPE = {args.dtype!r}
BITS = {args.bits}
Q_GROUP_SIZE = {args.q_group_size}
RESIDUAL_LENGTH = {args.residual_length}


def make_prompt_ids(tok):
    prefix = tok.encode(PROMPT, add_special_tokens=True)
    filler = tok.encode(
        " This sentence pads the prompt to a fixed token length for deterministic cache testing.",
        add_special_tokens=False,
    )
    ids = prefix[:CONTEXT_LEN]
    if len(ids) < CONTEXT_LEN:
        ids = ids + (filler * ((CONTEXT_LEN - len(ids) + len(filler) - 1) // len(filler)))[: CONTEXT_LEN - len(ids)]
    return ids


def run_hf(cache_kind):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if cache_kind == "muzero":
        import muzero
        muzero.patch_transformers()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=getattr(torch, DTYPE),
        trust_remote_code=True,
    ).cuda().eval()
    input_ids = torch.tensor([make_prompt_ids(tok)], dtype=torch.long, device="cuda")
    kwargs = dict(max_new_tokens=NEW_TOKENS, do_sample=False, temperature=None, top_p=None)
    if cache_kind == "muzero":
        kwargs["cache_implementation"] = f"mu_zero_{{BITS}}bit"
        kwargs["cache_config"] = {{"q_group_size": Q_GROUP_SIZE, "residual_length": RESIDUAL_LENGTH, "backend": "cuda"}}
    elif cache_kind == "legacy_logq":
        kwargs["cache_implementation"] = "quantized"
        kwargs["cache_config"] = {{
            "backend": f"logq{{BITS}}_zeroanchor_stat_cuda",
            "q_group_size": Q_GROUP_SIZE,
            "residual_length": RESIDUAL_LENGTH,
            "amax_dtype": "float16",
        }}

    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, **kwargs)
    torch.cuda.synchronize()
    gen = out[0, input_ids.shape[1]:].tolist()
    return {{"tokens": gen, "text": tok.decode(gen, skip_special_tokens=False)}}


if METHOD == "hf_bf16":
    result = run_hf("bf16")
elif METHOD == "hf_muzero":
    result = run_hf("muzero")
elif METHOD == "hf_legacy_logq":
    result = run_hf("legacy_logq")
else:
    raise ValueError(METHOD)

result.update({{"method": METHOD, "bits": BITS, "context_len": CONTEXT_LEN, "new_tokens": NEW_TOKENS}})
print(json.dumps(result, ensure_ascii=False))
'''


def _first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def main() -> None:
    args = _parse_args()
    args.model = ensure_model_path(args.model)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    rows = [_run_method(args, method) for method in methods]

    print(f"{'method':22s} {'ok':>4} {'len':>4} {'vs_hf_bf16':>12} {'first_diff':>10} preview")
    print("-" * 96)
    baseline = next((r for r in rows if r.get("method") == "hf_bf16" and r.get("ok")), None)
    baseline_tokens = baseline.get("tokens") if baseline else None
    for row in rows:
        if not row.get("ok"):
            print(f"{row['method']:22s} {'no':>4} {'-':>4} {'-':>12} {'-':>10} {row.get('error', '')[-180:].replace(chr(10), ' ')}")
            continue
        tokens = row["tokens"]
        if baseline_tokens is None:
            match = "n/a"
            diff = None
        else:
            diff = _first_diff(baseline_tokens, tokens)
            match = "exact" if diff is None else "diff"
        preview = row.get("text", "").replace("\n", " ")[:80]
        print(f"{row['method']:22s} {'yes':>4} {len(tokens):4d} {match:>12} {str(diff):>10} {preview}")

    print("\n=== Pairwise Token Equality ===")
    good = [r for r in rows if r.get("ok")]
    for i, left in enumerate(good):
        for right in good[i + 1:]:
            diff = _first_diff(left["tokens"], right["tokens"])
            status = "exact" if diff is None else f"diff@{diff}"
            print(f"{left['method']:22s} vs {right['method']:22s}: {status}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

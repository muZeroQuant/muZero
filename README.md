# muZero KV Cache

muZero is a plug-in KV-cache quantizer for Hugging Face `transformers` causal language models. It registers `mu_zero_*bit` cache implementations that can be used from `model.generate()` with minimal code changes.

The cache keeps the most recent tokens in full precision and quantizes older KV entries with a zero-anchored piecewise log compander. This is intended to reduce long-context KV memory while preserving generation quality.

## Features

- Hugging Face `transformers` integration via `muzero.patch_transformers()`.
- 1, 2, 3, 4, 7, and 8-bit KV-cache quantization.
- Configurable group size, residual full-precision window, metadata dtype, and backend.
- CUDA extension fast paths, optional Triton quantization backend, and PyTorch fallback.
- Benchmark and evaluation scripts for throughput, GSM8K, HumanEval, and LongBench.

## Environment Setup

Python 3.10 or newer is required. Install PyTorch for your CUDA version first, then install this package in editable mode.

```bash
cd /path/to/muZero

# Optional: create an isolated environment.
python -m venv .venv
source .venv/bin/activate

# Install PyTorch matching your platform from https://pytorch.org/get-started/locally/.
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Core runtime dependencies.
pip install -e .

# Optional benchmark/evaluation dependencies.
pip install -e ".[bench]"

# Optional Triton quantization backend.
pip install -e ".[triton]"
```

For a full development environment, install both extras:

```bash
pip install -e ".[bench,triton]"
pip install pytest
```

No separate extension build command is required. CUDA helpers are compiled or loaded on first use by the runtime backend.

## Quickstart

```python
import torch
import muzero
from transformers import AutoModelForCausalLM, AutoTokenizer

muzero.patch_transformers()

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).cuda().eval()

inputs = tokenizer("Explain KV-cache quantization briefly.", return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        cache_implementation="mu_zero_4bit",
        cache_config={"q_group_size": 64, "residual_length": 128},
        max_new_tokens=128,
        do_sample=False,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

See `examples/quickstart.py` for a runnable example.

## Configuration

```python
from muzero import MuZeroConfig

cfg = MuZeroConfig(
    bits=4,
    q_group_size=64,
    residual_length=128,
    amax_dtype="float16",
    backend="auto",  # auto, cuda, triton, or torch
)
```

The default `auto` backend tries the CUDA extension on CUDA tensors and falls back to PyTorch. Use `backend="triton"` only when you explicitly want the optional Triton quantization path.

When `decode_flush_by_group=True` is used, `residual_length` must be a multiple of `q_group_size`.

## Benchmarks

Throughput benchmark:

```bash
python benchmarks/bench_throughput.py --config benchmarks/configs/qwen3_8b.yaml
```

Evaluation examples:

```bash
python benchmarks/eval_gsm8k.py --config benchmarks/configs/eval/gsm8k_qwen3_8b_muzero4.yaml
python benchmarks/eval_longbench.py --config benchmarks/configs/eval/longbench_qwen3_8b_muzero4.yaml
python benchmarks/eval_humaneval.py --model /path/to/model --bits 4 --output-dir runs/humaneval_muzero
```

Update the model paths in `benchmarks/configs/*.yaml` before running benchmarks on a new machine.

## Tests

```bash
python -m pytest tests/ -v
```

Some tests use the CPU fallback. CUDA-specific tests are skipped automatically when CUDA is unavailable.

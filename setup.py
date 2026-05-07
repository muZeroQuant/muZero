from setuptools import setup, find_packages

setup(
    name="muzero-kv",
    version="0.1.0",
    description="μ-Zero: piecewise log-companding KV-cache quantisation with zero-anchor",
    author="μ-Zero Authors",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.40",
    ],
    extras_require={
        "bench": ["datasets", "huggingface_hub", "lm_eval", "pyyaml", "tqdm"],
        "triton": ["triton>=2.1"],
    },
)

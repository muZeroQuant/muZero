from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MuZeroConfig:
    """Configuration for the μ-Zero KV-cache quantizer.

    Attributes:
        bits: Quantization bit-width (1, 2, 3, 4, 7, or 8).
        q_group_size: Number of values per quantization group.
        residual_length: Number of most-recent tokens kept at full precision.
        amax_dtype: Dtype for min/scale metadata ("float16", "bfloat16", "float32").
        stats_max_points: Maximum sample points used when selecting alpha at runtime.
        backend: Kernel backend ("auto", "cuda", "triton", "torch").
        alpha_key: Pre-calibrated alpha for keys.  None = select at first flush.
        alpha_value: Pre-calibrated alpha for values.  None = select at first flush.
        logq_compatible_layout: Match legacy LogQ key/value flush boundaries.
        decode_flush_by_group: In MuZero layout, flush decode residuals every
            q_group_size tokens. This is the default; disabling it restores the
            older throughput-oriented 4-group decode flush batch.
    """

    bits: int = 4
    q_group_size: int = 64
    residual_length: int = 128
    amax_dtype: str = "float16"
    stats_max_points: int = 65536
    backend: str = "auto"
    alpha_key: Optional[float] = None
    alpha_value: Optional[float] = None
    logq_compatible_layout: bool = False
    decode_flush_by_group: bool = True

    def __post_init__(self):
        if self.bits not in {1, 2, 3, 4, 7, 8}:
            raise ValueError(f"`bits` must be one of {{1,2,3,4,7,8}}, got {self.bits}")
        if self.q_group_size <= 0:
            raise ValueError(f"`q_group_size` must be > 0, got {self.q_group_size}")
        if self.amax_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"`amax_dtype` must be one of {{'float16','bfloat16','float32'}}, got {self.amax_dtype}")
        if self.backend not in {"auto", "cuda", "triton", "torch"}:
            raise ValueError(f"`backend` must be one of {{'auto','cuda','triton','torch'}}, got {self.backend}")
        if self.decode_flush_by_group and self.residual_length % self.q_group_size != 0:
            raise ValueError("`residual_length` must be a multiple of `q_group_size` when `decode_flush_by_group=True`")

    @classmethod
    def from_dict(cls, d: dict) -> "MuZeroConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-benchmarks/configs/throughput/qwen3_8b.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-runs/throughput_memory/qwen3_8b}
METHODS=${METHODS:-bf16,mu_zero_2bit,mu_zero_4bit}
MODE=${MODE:-decode}
DECODE_ATTENTION_MASK=${DECODE_ATTENTION_MASK:-static}
RESIDUAL_LENGTH=${RESIDUAL_LENGTH:-0}
PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-128}
WARMUP=${WARMUP:-0}
RUNS=${RUNS:-1}

BATCH_CONTEXT=${BATCH_CONTEXT:-4096}
BATCH_NEW_TOKENS=${BATCH_NEW_TOKENS:-512}
BATCH_SIZES=${BATCH_SIZES:-32,64,96,128,160,224,256,288,320,352,384,400}

SEQUENCE_BATCH_SIZE=${SEQUENCE_BATCH_SIZE:-64}
SEQUENCE_CONTEXTS=${SEQUENCE_CONTEXTS:-2048,4096,8192,16384,20480}
SEQUENCE_NEW_TOKENS=${SEQUENCE_NEW_TOKENS:-512}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${OUTPUT_DIR}"

python benchmarks/scripts/collect_muzero_scaling_results.py \
  --config "${CONFIG}" \
  --output "${OUTPUT_DIR}/scaling.csv" \
  --methods "${METHODS}" \
  --warmup "${WARMUP}" \
  --runs "${RUNS}" \
  --mode "${MODE}" \
  --decode-attention-mask "${DECODE_ATTENTION_MASK}" \
  --empty-cache-before-decode \
  --residual-length "${RESIDUAL_LENGTH}" \
  --batch-context "${BATCH_CONTEXT}" \
  --batch-new-tokens "${BATCH_NEW_TOKENS}" \
  --batch-sizes "${BATCH_SIZES}" \
  --sequence-axis context \
  --sequence-batch-size "${SEQUENCE_BATCH_SIZE}" \
  --sequence-contexts "${SEQUENCE_CONTEXTS}" \
  --sequence-new-tokens "${SEQUENCE_NEW_TOKENS}" \
  --prefill-chunk-size "${PREFILL_CHUNK_SIZE}"

echo "Results written to ${OUTPUT_DIR}/scaling.csv and ${OUTPUT_DIR}/scaling.details.csv"

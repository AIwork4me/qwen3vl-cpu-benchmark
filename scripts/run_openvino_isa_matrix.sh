#!/usr/bin/env bash
# Phase 4 — ISA ablation matrix for the OpenVINO INT8 conditioning bridge.
#
# Runnable-config matrix established by probing (see phase4_isa_matrix.md):
#   ISA ceiling        hint=bf16(default)   hint=f32
#   DEFAULT            brgemm_avx512_bf16   brgemm_avx512_f32
#   AVX512_CORE_BF16   brgemm_avx512_bf16   (same HW ceiling)
#   AVX512_CORE_VNNI   NOT RUNNABLE         brgemm_avx512_f32
#   AVX512_CORE        NOT RUNNABLE         brgemm_avx512_f32
#   AVX2_VNNI          NOT RUNNABLE         brgemm_avx2_f32
#   AVX2               NOT RUNNABLE         NOT RUNNABLE
# INFERENCE_PRECISION_HINT=i8 is rejected by the plugin (bf16/f16/f32 only).
#
# Per config: 3 fresh processes, each 2 warmup + 12 measured iters per prompt
# (P1/P2/P3), threads=16, NUM_STREAMS=1, LATENCY. Strictly sequential.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv-openvino/bin/python
OUT=results/openvino_isa
mkdir -p "$OUT/runlogs"

# name|ONEDNN_MAX_CPU_ISA (empty = unset)|extra args
CONFIGS=(
  "DEFAULT||"
  "BF16CEIL|AVX512_CORE_BF16|"
  "DEFAULT_F32||--inference-precision f32"
  "VNNI512_F32|AVX512_CORE_VNNI|--inference-precision f32"
  "AVX512_F32|AVX512_CORE|--inference-precision f32"
  "AVX2VNNI_F32|AVX2_VNNI|--inference-precision f32"
)
RUNS=3
WARM=2
MEASURE=12

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name isa extra <<< "$cfg"
  for r in $(seq 1 "$RUNS"); do
    tag="isa_${name}_r${r}"
    if [ -f "$OUT/profiling_${tag}_summary.json" ]; then
      echo "[skip] $tag (exists)"; continue
    fi
    log="$OUT/runlogs/${tag}.log"
    echo "[run ] $tag isa=${isa:-<default>} extra='$extra' $(date +%H:%M:%S)"
    if [ -z "$isa" ]; then
      # shellcheck disable=SC2086
      env -u ONEDNN_MAX_CPU_ISA "$PY" scripts/investigate_openvino_runtime.py \
        --tag "$tag" --warm-iters "$WARM" --measure-iters "$MEASURE" \
        --out-dir "$OUT" $extra \
        > "$log" 2>&1 || { echo "FAILED, see $log"; exit 1; }
    else
      # shellcheck disable=SC2086
      env ONEDNN_MAX_CPU_ISA="$isa" "$PY" scripts/investigate_openvino_runtime.py \
        --tag "$tag" --warm-iters "$WARM" --measure-iters "$MEASURE" \
        --out-dir "$OUT" $extra \
        > "$log" 2>&1 || { echo "FAILED, see $log"; exit 1; }
    fi
    grep -m2 "^\[" "$log" | tail -1
  done
done
echo "[done] $(date +%H:%M:%S)"

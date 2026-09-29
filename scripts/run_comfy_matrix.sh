#!/bin/bash
# ComfyUI benchmark matrix — runs sequentially to avoid CPU pollution.
set -x
cd "$(dirname "$0")/.."
PY=.venv-comfy/bin/python
S=scripts/bench_comfy_qwen3vl_cpu.py

# A2 (product conditioning path): 3 fresh processes, default threads (16)
$PY $S --tag run1 --warm-iters 5 --save-npy > logs/comfy_run1.log 2>&1 || echo "RUN1 FAILED"
$PY $S --tag run2 --warm-iters 5 > logs/comfy_run2.log 2>&1 || echo "RUN2 FAILED"
$PY $S --tag run3 --warm-iters 5 > logs/comfy_run3.log 2>&1 || echo "RUN3 FAILED"

# A2 thread sweep (1 fresh process each)
$PY $S --tag t8  --threads 8  --warm-iters 5 > logs/comfy_t8.log 2>&1 || echo "T8 FAILED"
$PY $S --tag t32 --threads 32 --warm-iters 5 > logs/comfy_t32.log 2>&1 || echo "T32 FAILED"

# A1-forced (true int8 GEMM): default + thread sweep, with profiler on default
$PY $S --tag a1_t16 --warm-iters 5 --save-npy --force-quant-mm --profile > logs/comfy_a1_t16.log 2>&1 || echo "A1T16 FAILED"
$PY $S --tag a1_t8  --threads 8  --warm-iters 5 --force-quant-mm > logs/comfy_a1_t8.log 2>&1 || echo "A1T8 FAILED"
$PY $S --tag a1_t32 --threads 32 --warm-iters 5 --force-quant-mm > logs/comfy_a1_t32.log 2>&1 || echo "A1T32 FAILED"

echo "MATRIX DONE"
for f in run1 run2 run3 t8 t32 a1_t16 a1_t8 a1_t32; do
  echo "== $f =="; grep -E "warm mean|FIRST" logs/comfy_$f.log | head -8
done

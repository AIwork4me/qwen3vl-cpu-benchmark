#!/usr/bin/env bash
# E2E experiment suite driver.
# All GPU runs strictly serial; every run = fresh process; driver records
# process-spawn wall time for cold-start accounting (results/e2e/raw/process_walls.csv).
# Usage: bash scripts/run_e2e_suite.sh <stage>
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv-comfy-rocm/bin/python
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
SEED=20260929
mkdir -p results/e2e/raw

run() { # run <tag> <cmd...>  (stdout-> results/e2e/raw/<tag>.log, wall recorded)
  local tag=$1; shift
  local t0 t1
  t0=$(date +%s.%N)
  "$PY" "$@" > "results/e2e/raw/${tag}.log" 2>&1 || { tail -5 "results/e2e/raw/${tag}.log"; exit 1; }
  t1=$(date +%s.%N)
  echo "${tag},${t0},${t1}" >> results/e2e/raw/process_walls.csv
  awk -v a="$t1" -v b="$t0" -v t="$tag" 'BEGIN{printf "  [wall] %s: %.1fs\n", t, a-b}' 
}

# ---- core A/B: G vs H, 3 fresh processes each, 1 cold + 3 warm per process ----
core_ab() {
  for i in 1 2 3; do
    for route in $(if [ $((i % 2)) -eq 1 ]; then echo "gpu ov"; else echo "ov gpu"; fi); do
      run "ab_p${i}_${route}" scripts/e2e_pipeline.py --tag "ab_p${i}_${route}" \
        --group ab --route "$route" --dq 32 --prompts P3 --seeds "$SEED" \
        --steps 20 --iters 4 --save-cond
    done
  done
}

# ---- native CPU-TE context route ----
core_native() {
  run ab_p1_native scripts/e2e_pipeline.py --tag ab_p1_native --group ab --route native \
    --prompts P3 --seeds "$SEED" --steps 20 --iters 4 --save-cond
}

# ---- prompt matrix (Phase 9) ----
prompt_matrix() {
  run pm_gpu scripts/e2e_pipeline.py --tag pm_gpu --group matrix --route gpu \
    --dq 32 --prompts P4,P1,P2,P3,P5 --seeds "$SEED" --steps 20 --iters 2
  run pm_ov scripts/e2e_pipeline.py --tag pm_ov --group matrix --route ov \
    --dq 32 --prompts P4,P1,P2,P3,P5 --seeds "$SEED" --steps 20 --iters 2
  run pm_native scripts/e2e_pipeline.py --tag pm_native --group matrix --route native \
    --prompts P4,P1,P2,P3,P5 --seeds "$SEED" --steps 20 --iters 1
}

# ---- step matrix (Phase 11): 40 steps ----
step_matrix() {
  for route in gpu ov; do
    run "sm40_${route}" scripts/e2e_pipeline.py --tag "sm40_${route}" --group matrix \
      --route "$route" --dq 32 --prompts P3 --seeds "$SEED" --steps 40 --iters 2
  done
}

# ---- resolution matrix (Phase 10): 1328x1328 ----
res_matrix() {
  for route in gpu ov; do
    run "rm1328_${route}" scripts/e2e_pipeline.py --tag "rm1328_${route}" --group matrix \
      --route "$route" --dq 32 --prompts P3 --seeds "$SEED" --steps 20 --iters 2 \
      --width 1328 --height 1328
  done
}

# ---- DiT kernel config note: default kernels (no AOTriton), ov route ----
dit_kernel_note() {
  local t0 t1 tag=cfg_default_kernel
  t0=$(date +%s.%N)
  env -u TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL \
    "$PY" scripts/e2e_pipeline.py --tag "$tag" --group matrix --route ov \
      --dq 32 --prompts P3 --seeds "$SEED" --steps 20 --iters 2 \
      > "results/e2e/raw/${tag}.log" 2>&1
  t1=$(date +%s.%N)
  echo "${tag},${t0},${t1}" >> results/e2e/raw/process_walls.csv
}

# ---- continuous 5-prompt batch (Phase 14) ----
continuous() {
  run cont_gpu scripts/e2e_pipeline.py --tag cont_gpu --group throughput --route gpu \
    --dq 32 --prompts P4,P1,P2,P3,P5 --seeds "$SEED" --steps 20 --iters 1
  run cont_ov scripts/e2e_pipeline.py --tag cont_ov --group throughput --route ov \
    --dq 32 --prompts P4,P1,P2,P3,P5 --seeds "$SEED" --steps 20 --iters 1
  # sequential ov again at 10 images for the overlap comparison pair
  run cont_ov10 scripts/e2e_pipeline.py --tag cont_ov10 --group throughput --route ov \
    --dq 32 --prompts P4,P1,P2,P3,P5 --seeds "$SEED" --steps 20 --iters 2
}

# ---- pipelined overlap (Phase 15): CPU encodes n+1 during GPU DiT of n ----
pipelined() {
  run pipe_ov5 scripts/e2e_pipeline.py --tag pipe_ov5 --group throughput --route ov \
    --dq 32 --prompts P4,P1,P2,P3,P5 --seeds "$SEED" --steps 20 --iters 1 --pipelined
  run pipe_ov10 scripts/e2e_pipeline.py --tag pipe_ov10 --group throughput --route ov \
    --dq 32 --prompts P4,P1,P2,P3,P5 --seeds "$SEED" --steps 20 --iters 2 --pipelined
  # GPU route cannot pipeline (its encoder occupies the GPU) — documented, not run.
}

# ---- CPU/GPU UMA contention (Phase 16): DiT alone vs DiT + CPU encode storm ----
contention() {
  P5TEXT=$(python3 -c "import json;print([p['text'] for p in json.load(open('prompts/prompts.json'))['prompts'] if p['id']=='P5'][0])")
  run contd_ov scripts/e2e_pipeline.py --tag contd_ov --group throughput --route ov \
    --dq 32 --prompts P3 --seeds "$SEED" --steps 20 --iters 3
  run contd_ov_enc scripts/e2e_pipeline.py --tag contd_ov_enc --group throughput --route ov \
    --dq 32 --prompts P3 --seeds "$SEED" --steps 20 --iters 3 --contend-text "$P5TEXT"
}

# ---- DQ group matrix (Phase 17): dq in {0,64,128} x 2 seeds (dq32 = core_ab) ----
dq_matrix() {
  for dq in 0 64 128; do
    run "dq${dq}_s1" scripts/e2e_pipeline.py --tag "dq${dq}_s1" --group dq --route ov \
      --dq "$dq" --prompts P3 --seeds "$SEED" --steps 20 --iters 2 --save-cond
    run "dq${dq}_s2" scripts/e2e_pipeline.py --tag "dq${dq}_s2" --group dq --route ov \
      --dq "$dq" --prompts P3 --seeds 42 --steps 20 --iters 2 --save-cond
  done
}

# ---- OpenVINO compile cache (Phase 26): cold cache vs warm cache ----
ov_cache() {
  rm -rf models/.ov_cache_cold && mkdir -p models/.ov_cache_cold
  run ovcache_cold scripts/e2e_pipeline.py --tag ovcache_cold --group dq --route ov \
    --dq 32 --prompts P1 --seeds 7 --steps 4 --iters 1 --ov-cache-dir models/.ov_cache_cold
  run ovcache_cold2 scripts/e2e_pipeline.py --tag ovcache_cold2 --group dq --route ov \
    --dq 32 --prompts P1 --seeds 7 --steps 4 --iters 1 --ov-cache-dir models/.ov_cache_cold
  run ovcache_warm scripts/e2e_pipeline.py --tag ovcache_warm --group dq --route ov \
    --dq 32 --prompts P1 --seeds 7 --steps 4 --iters 1 --ov-cache-dir models/.ov_cache
}

# ---- quality dataset (Phase 18): 30 prompts x routes ----
quality() {
  run q_gpu scripts/e2e_pipeline.py --tag q_gpu --group quality --route gpu \
    --dq 32 --prompt-file prompts/prompts_quality.json --prompts all \
    --steps 20 --iters 1 --warmup-steps 2
  run q_dq32 scripts/e2e_pipeline.py --tag q_dq32 --group quality --route ov \
    --dq 32 --prompt-file prompts/prompts_quality.json --prompts all \
    --steps 20 --iters 1 --warmup-steps 2
}

quality_dq_subset() {  # DQ64/128 on Q01-Q12 (documented subset reduction)
  for dq in 64 128; do
    run "q_dq${dq}" scripts/e2e_pipeline.py --tag "q_dq${dq}" --group quality \
      --route ov --dq "$dq" --prompt-file prompts/prompts_quality.json \
      --prompts Q01,Q02,Q03,Q04,Q05,Q06,Q07,Q08,Q09,Q10,Q11,Q12 --steps 20 --iters 1 --warmup-steps 2
  done
}

# ---- repeated seeds (Phase 31): 5 representative prompts x 5 seeds x {gpu, dq128} ----
repeated_seeds() {
  QS="Q02,Q05,Q10,Q18,Q23"
  run rs_gpu scripts/e2e_pipeline.py --tag rs_gpu --group quality --route gpu \
    --dq 32 --prompt-file prompts/prompts_quality.json --prompts "$QS" \
    --seeds 11,22,33,44,55 --steps 20 --iters 1 --warmup-steps 2
  run rs_dq128 scripts/e2e_pipeline.py --tag rs_dq128 --group quality --route ov \
    --dq 128 --prompt-file prompts/prompts_quality.json --prompts "$QS" \
    --seeds 11,22,33,44,55 --steps 20 --iters 1 --warmup-steps 2
}

# ---- controlled 20-step kernel head-to-head (AOTriton vs default), back-to-back ----
kernel_note20() {
  run kn20_aot scripts/e2e_pipeline.py --tag kn20_aot --group matrix --route ov \
    --dq 32 --prompts P3 --seeds "$SEED" --steps 20 --iters 2 --warmup-steps 4
  local t0 t1 tag=kn20_def
  t0=$(date +%s.%N)
  env -u TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL \
    "$PY" scripts/e2e_pipeline.py --tag "$tag" --group matrix --route ov \
      --dq 32 --prompts P3 --seeds "$SEED" --steps 20 --iters 2 --warmup-steps 4 \
      > "results/e2e/raw/${tag}.log" 2>&1
  t1=$(date +%s.%N)
  echo "${tag},${t0},${t1}" >> results/e2e/raw/process_walls.csv
}

# ---- determinism validation of --warmup-steps (compare i0 sha vs known steady-state) ----
det_test() {
  run det_test scripts/e2e_pipeline.py --tag det_test --group dq --route ov \
    --dq 32 --prompts P3 --seeds 20260929 --steps 20 --iters 1 --warmup-steps 2 --save-cond
}

case "${1:-}" in
  core_ab|core_native|prompt_matrix|step_matrix|res_matrix|dit_kernel_note|\
  continuous|pipelined|contention|dq_matrix|ov_cache|quality|quality_dq_subset|repeated_seeds|det_test|kernel_note20) "$1" ;;
  *) echo "stages: core_ab core_native prompt_matrix step_matrix res_matrix dit_kernel_note continuous pipelined contention dq_matrix ov_cache quality quality_dq_subset repeated_seeds"; exit 1 ;;
esac
echo "[suite $(date +%H:%M:%S)] stage $1 done"

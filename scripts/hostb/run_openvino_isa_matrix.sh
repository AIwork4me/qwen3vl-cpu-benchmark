#!/usr/bin/env bash
# Frozen HOST-B snapshot — paths assume this file lives at scripts/<name>;
# copy there (or adjust ROOT) before running.
# ISA / dispatch / dynamic-quantization matrix for the OpenVINO Qwen3-VL INT8
# conditioning workload (root-cause investigation, HOST B).
#
# Config set (each: 3 fresh processes, 2 warmup + 10 measured iters, P1/P2/P3,
# 16 threads pinned via taskset 0-15, LATENCY hint => NUM_STREAMS=1):
#
#   default          no env, no hints (oneDNN isa = AVX512_CORE_BF16)
#   isa_bf16         ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16   (expect == default)
#   vnni_f32         ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI + INFERENCE_PRECISION_HINT=f32
#   core_f32         ONEDNN_MAX_CPU_ISA=AVX512_CORE + INFERENCE_PRECISION_HINT=f32
#                    (NOTE: JIT dump proves this ceiling STILL dispatches vpdpbusd
#                     on this build — it is NOT a no-VNNI counterfactual)
#   dq0              DYNAMIC_QUANTIZATION_GROUP_SIZE=0 (bf16 vdpbf16ps path, no VNNI)
#   dq128            DYNAMIC_QUANTIZATION_GROUP_SIZE=128
#
# Non-runnable configs (recorded as dispatch evidence, run once, expect rc!=0):
#   err_avx2         ONEDNN_MAX_CPU_ISA=AVX2
#   err_avx2_vnni    ONEDNN_MAX_CPU_ISA=AVX2_VNNI
#   err_core         ONEDNN_MAX_CPU_ISA=AVX512_CORE (no hint)
#   err_vnni         ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI (no hint)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv-openvino/bin/python"
OUT="$ROOT/results/openvino_isa"
mkdir -p "$OUT"

PROCS=${PROCS:-3}
WARM=${WARM:-2}
MEAS=${MEAS:-10}

MATRIX_CSV="$OUT/isa_matrix.csv"
echo "config,proc,prompt,warm_p50_s,warm_mean_s,warm_std_s,warm_min_s,warm_max_s,cv,read_s,compile_s,rss_gib,cos_hostA,rc" > "$MATRIX_CSV"

run_one () {  # $1=config $2=proc $3=env-prefix $4=extra-args
  local CFG=$1 P=$2 ENVPFX=$3 EXTRA=$4
  local JSON="$OUT/investigate_${CFG}_p${P}.json"
  local OUT_LOG="$OUT/${CFG}_p${P}.stdout"
  env $ENVPFX taskset -c 0-15 "$PY" "$ROOT/scripts/investigate_openvino_runtime.py" \
      --mode bench --tag "${CFG}_p${P}" --threads 16 --out-dir "$OUT" \
      --warm-iters "$WARM" --measure-iters "$MEAS" $EXTRA > "$OUT_LOG" 2>&1
  local RC=$?
  "$PY" - "$JSON" "$CFG" "$P" "$MATRIX_CSV" "$RC" <<'EOF'
import json, sys, math, statistics, os
jpath, cfg, p, csv_path, rc = sys.argv[1:6]
if not os.path.exists(jpath):
    with open(csv_path, "a") as f:
        f.write(f"{cfg},{p},FAILED,,,,,,,,,,,,{rc}\n")
    sys.exit(0)
d = json.load(open(jpath)); m = d["meta"]
for pid, r in d["prompts"].items():
    w = r["warm_s"]
    mean = sum(w)/len(w); std = math.sqrt(sum((x-mean)**2 for x in w)/len(w))
    row = [cfg, p, pid, f"{statistics.median(w):.5f}", f"{mean:.5f}", f"{std:.5f}",
           f"{min(w):.5f}", f"{max(w):.5f}", f"{std/mean:.4f}",
           f"{m['model_read_s']:.3f}", f"{m['model_compile_s']:.3f}",
           f"{m.get('rss_gib_after_compile',0):.2f}",
           f"{r.get('cosine_vs_hostA_npy','')}", rc]
    with open(csv_path, "a") as f: f.write(",".join(str(x) for x in row)+"\n")
EOF
  echo "[matrix] $CFG proc $P rc=$RC"
}

for P in $(seq 1 "$PROCS"); do
  run_one default       "$P" ""                                      ""
  run_one isa_bf16      "$P" "ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16"   ""
  run_one vnni_f32      "$P" "ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI"   "--precision-hint f32"
  run_one core_f32      "$P" "ONEDNN_MAX_CPU_ISA=AVX512_CORE"        "--precision-hint f32"
  run_one dq0           "$P" ""                                      "--dq-group-size 0"
  run_one dq128         "$P" ""                                      "--dq-group-size 128"
done

# non-runnable dispatch evidence (single attempt each)
run_one err_avx2       1 "ONEDNN_MAX_CPU_ISA=AVX2"                   ""
run_one err_avx2_vnni  1 "ONEDNN_MAX_CPU_ISA=AVX2_VNNI"              ""
run_one err_core       1 "ONEDNN_MAX_CPU_ISA=AVX512_CORE"            ""
run_one err_vnni       1 "ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI"       ""

echo "[matrix] complete -> $MATRIX_CSV"

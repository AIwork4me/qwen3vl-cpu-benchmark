# Phase 10 — Controlled OpenVINO FP16 comparison (weight-compression isolation)

Control model: `OpenVINO/Qwen3-VL-8B-Instruct-fp16-ov` (ModelScope; strictly matching
export pipeline/architecture — same `openvino_language_model.xml` structure, same 36
layers, hidden 4096, vocab 151936; downloaded 2026-09-29, ~17 GB). Local:
`models/qwen3vl-openvino-fp16/` (gitignored). Download log: `logs/dl_fp16_ov.log`.

Protocol: identical to the ISA matrix — `scripts/investigate_openvino_runtime.py
--model-dir models/qwen3vl-openvino-fp16 --tag fp16_r<N>`, 3 fresh processes, 2 warmup +
12 measured iters, P1/P2/P3, 16 threads, NUM_STREAMS=1, LATENCY, `--save-hidden`.
Raw: `profiling_fp16_r*_*.{json,csv}`, `runtime_nodes_fp16_r*.json`, hidden npys.

## Result (median-of-3 p50, s; run spread <=1.6% by (max-min)/min)

| model | P1 | P2 | P3 | cosine vs BF16 ref |
|---|---:|---:|---:|---|
| OV INT8-WC (u8 weights, dq+VNNI path) | 0.1854 | 0.3770 | 0.7064 | 0.9972–0.9985 |
| OV FP16 (f16 weights) | 0.2609 | 0.6289 | 1.0241 | 0.99982–0.99992 |
| **INT8-WC speedup** | **1.41x** | **1.67x** | **1.45x** | |

## Dispatch of the FP16 control (runtime model)

253 FullyConnected nodes -> `brgemm_avx512_bf16`, **rtPrecision = bf16** (vs u8 for the
INT8 model), outputs bf16. Weights are f16 on disk and converted to bf16 at load (OV's
default inference precision); there is no u8 storage and no dynamic-quantization engine —
per the Phase 6 mechanism this is the vdpbf16ps (BF16 dot) path, i.e. the same engine the
INT8 model falls back to when DYNAMIC_QUANTIZATION_GROUP_SIZE=0.

## What this isolates

Everything is held constant (OpenVINO build, graph, bridge, threads, prompts, host) except
the weight storage + GEMM engine: FP16 storage + BF16 dot vs INT8 storage + dynamic
quantization + VNNI int8 dot. The measured **1.41–1.67x** is therefore the combined
contribution of INT8 weight compression + the VNNI dynamic-quantization compute path on
this workload — the "weight-compression factor" in the root-cause decomposition.

Consistency cross-checks:
- FP16 P1 run-to-run p50s 0.2610/0.2609/0.2609 (spread 0.04%) — the control is extremely
  stable, so the 1.41–1.67x deltas are far outside noise.
- The INT8 advantage tracks token count (max at P2/P3 where MAC work is larger), consistent
  with the engine difference, not load-time effects.
- Cosines: FP16 0.99982-0.99992 (storage precision ≈ reference), INT8-WC 0.9972+ — the accuracy
  cost of the INT8 path is visible but small, same order as ComfyUI's own int8 convrot.

Note: this control became possible because a strictly matching fp16-ov artifact exists on
ModelScope; no semantic substitution was made.

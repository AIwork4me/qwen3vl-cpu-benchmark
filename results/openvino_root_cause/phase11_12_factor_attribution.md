# Phase 11–12 — Root-cause factor attribution and VNNI contribution quantification

All numbers are medians of fresh-process runs from the verified phase artifacts
(ISA matrix, dq matrix, fp16 control, committed ComfyUI data). Confidence labels per task
rules: CONFIRMED / STRONG EVIDENCE / LIKELY / NOT ESTABLISHED / DISPROVEN.

## Phase 11 — Factor table

| # | Factor | Evidence | Counterfactual | Measured effect | Confidence |
|---|---|---|---|---|---|
| 1 | INT8 weight compression (u8 weights resident, ~2x smaller than FP16) | rtPrecision=u8 in every runnable config (Phase 2/4/5); file sizes (Phase 9); Phase 10 control | OV FP16 same stack/graph | 1.41x / 1.67x / 1.45x (P1/P2/P3) combined with factor 2 | CONFIRMED |
| 2 | Dynamic activation quantization (group 32, u8) feeding int8 MACs | Phase 5 latency+cosine response; Phase 6 vpbroadcastd group scales + vpdpbusd loops | DQ=0 -> kernels become pure bf16-dot (0 VNNI in JIT) | +23% / +42% / +43% latency when disabled (i.e. factor 1.23–1.43x) | CONFIRMED |
| 3 | AVX-512 vector width / register file | Phase 4 matrix | 256-bit AVX2_VNNI ceiling (keeps int8 engine) | 1.43x / 1.38x / 1.55x | CONFIRMED |
| 4 | AVX512_VNNI int8 dot engine (vpdpbusd) | Phase 6 JIT: 12 brgemm kernels x 208 vpdpbusd (zmm) in default; Level B dispatch 91–92% of time in those kernels | (a) DQ=0 switches engine to vdpbf16ps: 1.23–1.43x slower; (b) FP16 control (bf16 engine end-to-end): 1.41–1.67x slower incl. storage | engine-level increment over next-best engine (BF16 dot): 1.23–1.43x | CONFIRMED (execution); Level D sample proof not obtained |
| 5 | 16-core threading (TBB, NUM_STREAMS=1) | committed thread sweep (results/comparison/summary.csv: OV t8 vs default P1 0.2005 vs 0.1401 standalone; t32 flat) | threads=8 | ~1.4x vs 8 threads; no gain beyond 16 | STRONG EVIDENCE (committed repo data, not re-measured this round) |
| 6 | OpenVINO graph/runtime: fused dequant-in-kernel, no materialized bf16 copy, precompiled pipeline | Phase 9 traffic accounting (8.5 GB vs 34.7 GB per encode); ComfyUI A2 dequant counters | ComfyUI A2 -> A1 (kill dequant, keep torch) | dequant elimination alone: 12.98 -> 7.32 s (1.77x, committed) | CONFIRMED |
| 7 | ComfyUI FP compute path loss (never uses int8 engine) | committed forensics (torch._int_mm=0, dequant 252/encode) | A1 forced int8 | A2 12.98 s vs A1 7.32 s (P1) | CONFIRMED (prior round) |
| 8 | Memory-traffic reduction (implied streams) | Phase 9 implied 46.3 GB/s (OV) vs 2.7 GB/s (A2) at 4.1x fewer bytes | — | qualitative; DRAM bandwidth NOT measured (perf blocked) | STRONG EVIDENCE (implied, assumption-labeled) |
| 9 | GPU-side fp32-cast loss (Radeon route) | committed experiment C forensics (252/252 linears fp32xfp32) | OV CPU bridge | 0.60/1.23/1.83 s vs 0.19/0.38/0.69 s = 2.7–3.3x | CONFIRMED (prior round) |

Note on factor 4's counterfactual (b): the ISA-ceiling comparison (AVX512_CORE vs
AVX512_CORE_VNNI) is NOT a valid VNNI isolation — Phase 6 proved both ceilings emit and run
vpdpbusd (32/126 binaries differ only in codegen shape). The honest VNNI counterfactual is
the engine switch (DQ 32->0: int8-dot vs bf16-dot in the same nodes).

## Phase 12 — VNNI contribution, quantified

| comparison | P1 | P2 | P3 | meaning |
|---|---:|---:|---:|---|
| DEFAULT (DQ=32, VNNI int8) vs DQ=0 (bf16 dot, same everything) | 1.23x | 1.42x | 1.43x | **VNNI engine increment over BF16 engine** |
| VNNI512 ceiling vs AVX512 ceiling (both f32 hint; ratio = AVX512_F32 / VNNI512_F32 latency) | 0.92x | 1.00x | 0.93x | NOT a VNNI isolation — both ceilings execute vpdpbusd (Phase 6); the deltas reflect codegen differences only (VNNI512 ceiling is not faster) |
| OV INT8-WC (dq+VNNI) vs OV FP16 (bf16 engine + 2x weights) | 1.41x | 1.67x | 1.45x | compression + engine combined |
| ComfyUI A2 (dequant+FP, no int8) vs OV INT8 bridge | 70x | 36x | 22x | full product-path gap; P1 = 12.978/0.1854 (dq_default median). With the committed bridge mean 0.1871 this is the repo's headline '69x' — same comparison, both baselines quoted |
| ComfyUI A1 (torch int8) vs OV INT8 bridge | 40x | 20x | 12x | torch-eager int8 vs OV brgemm pipeline |

**VNNI's specific contribution, stated honestly:** within the OpenVINO stack, VNNI
(vpdpbusd) buys **1.23–1.43x** over the BF16-dot engine on the same compressed weights
(and enables the whole u8-activation dynamic-quantization scheme). It is a real, measured,
instruction-level-verified enabler — but it is NOT the headline "69x". The big factors are
the elimination of ComfyUI's per-encode dequant + the inference-engine GEMM pipeline
(factors 6+7+1+3 multiplied).

## Q1 verdict against the task's case template

**Case A (definitively executed), with one documented evidence gap:**

- Level B: the 253 weight-compressed FC nodes (91–92% of node time) are dispatched to
  brgemm kernels with u8 weights — the kernels that the JIT dump proves contain vpdpbusd.
  (Note: the OV primitive NAME "brgemm_avx512_bf16" is an ISA-family label and does NOT
  mean bf16 MACs — Phase 6 shows the MAC dtype is decided by the DQ knob.)
- Level C: 2,496 vpdpbusd (zmm) instructions in 12 brgemm JIT kernels of the default run,
  in MAC loops with per-group scale broadcasts; 0 VNNI when DQ=0.
- Level E: toggling the knob that adds/removes exactly these VNNI kernels changes latency
  23–43% AND changes output numerics monotonically with group size — the loops demonstrably
  execute.
- Level D (perf samples on the JIT symbols): NOT OBTAINED — perf_event_paranoid=4 and no
  functional perf for kernel 6.17.0-1032-oem (phase7_perf_findings.md). No system security
  settings were changed to force it.

Therefore: **AVX512_VNNI is confirmed executed by the OpenVINO CPU path for this workload
on Ryzen AI Max+ PRO 395 (Levels B+C+E); direct executed-instruction sampling (Level D) was
not obtainable in this environment.** "CPU supports VNNI" was never used as evidence.

## The 22–69x decomposition (P1 extreme: 12.98 s -> 0.185 s)

1. ComfyUI A2 -> A1 (stop dequantizing 252 layers to bf16 every encode; still torch eager):
   12.98 -> 7.32 s (1.77x, committed counters).
2. A1 -> OV INT8 bridge (0.185 s): the remaining ~40x is the OpenVINO pipeline on the same
   u8 weights: precompiled graph, cached weight repacking, fused dynamic quantization,
   VNNI/AVX-512 brgemm, 16-thread scheduling. Within this stack the measurable engine
   factors are: AVX-512 width 1.38–1.55x, VNNI-vs-BF16 1.23–1.43x, int8-storage+dq vs
   fp16 storage 1.41–1.67x (overlapping, not independent multipliers).
No single "because VNNI" story survives; the dominant term is architectural (fused,
never-materialized-dequant inference engine) with VNNI as a verified but secondary
multiplier.

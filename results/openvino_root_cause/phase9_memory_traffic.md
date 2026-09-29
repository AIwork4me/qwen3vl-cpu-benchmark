# Phase 9 — Weight compression / memory-traffic root cause

Data: `phase9_memory_traffic.json` (computed from model files, IR config, and committed
run counters; script logic embedded in the json generation — sizes via os.path.getsize,
FLOP/bytes via H=4096, I=12288, V=151936, L=36). perf stat NOT available
(perf_event_paranoid=4, no matching linux-tools — Phase 7 file), so no measured
cycles/cache counters; all GB/s below are IMPLIED (bytes/latency) under the stated
one-touch streaming assumption, not DRAM measurements.

## Weight footprint (files)

| artifact | bytes | per-linear-element |
|---|---:|---|
| OV INT8-WC language-model bin | 7,573,989,065 | u8 + per-channel scales |
| OV INT8-WC embeddings bin | 622,633,732 | f16 embed table |
| ComfyUI int8 convrot safetensors | 9,350,798,360 | i8 + convrot params |
| ComfyUI bf16 safetensors | 17,534,334,616 | bf16 |

Per-encode int8 linear weight bytes (36 layers x {qkvo 4x4096x4096, gate+up 2x4096x12288,
down 12288x4096} + lm_head 4096x151936) = **8,474,066,944 B ≈ 7.89 GiB**.

## Traffic per encode: OpenVINO bridge vs ComfyUI A2 product path

| | OpenVINO INT8 bridge | ComfyUI A2 (per its own counters) |
|---|---:|---:|
| weight bytes touched | 8.47 GB (u8 only; scales negligible) | 6.95 GB int8 read + 13.89 GB dequant write + 13.89 GB dequant read-back = **34.7 GB** |
| separate dequant pass | none (fused into brgemm copy-B/kernel) | 252 layers dequantized EVERY encode |
| implied stream @ P1 | 46.3 GB/s over 0.183 s | 2.7 GB/s over 12.98 s |
| implied stream @ P3 | 12.6 GB/s over 0.675 s | 2.3 GB/s over 15.43 s |
| traffic ratio | 1x | **4.1x** |

## Compute-side implied utilization

- P1: 661 GFLOP in 0.183 s -> 3.6 TFLOP/s; P3: 3,135 GFLOP in 0.675 s -> 4.7 TFLOP/s.
- Theoretical int8 envelope with the stated assumptions (16 Zen5 cores @ ~3.1 GHz, 2
  512-bit VNNI pipes, 64 int8 MACs per vpdpbusd zmm, 2 ops/MAC) is ~12.7 TOPS — measured
  utilization is ~28-37% of that envelope; the workload is not compute-saturated at these
  token counts (conclusion robust to the exact envelope: 2-4x more pipes/Hz would still
  leave utilization well under half); it is also not near DRAM peak (LPDDR5X-class UMA >> 46 GB/s).
- Conclusion: at 39–185 computed tokens (25–171 kept) the FC work is overhead/latency/weight-stream-bound, not
  MAC-throughput-bound. This is why bf16-vs-f32 compute (Phase 4) barely moved the needle
  and why bigger dq groups (Phase 5) helped: both reduce per-call overheads, not MAC time.

## Why ComfyUI A2 is 22–69x slower — quantified decomposition

ComfyUI A2 per encode: (a) 4.1x more bytes touched, (b) the dequant itself runs as a
torch elementwise kernel at ~1 GB/s effective (13.9 GB written at 2.3–2.7 GB/s implied
total traffic) — it is dequant-CPU-bound, not GEMM-bound, (c) only after dequant does a
bf16/fp GEMM run, re-reading the 13.9 GB it just wrote, (d) no int8 dot engine is ever
used (torch._int_mm = 0). OpenVINO: reads u8 once, dequantizes inside the brgemm
pipeline, computes on the VNNI int8 engine (Phase 6), never materializes a 13.9 GB bf16
copy of the weights.

## RSS trade-off

OV bridge peak RSS 14.57–15.1 GiB (u8 weights 8.2 decimal GB = 7.6 GiB + runtime scratch/allocator pools)
vs ComfyUI A2 7.59 GiB. OpenVINO trades ~2x RAM for the 22–69x latency — on this 94 GiB
UMA part that is 16% of RAM. (Numbers from committed 20 Hz sampling; unchanged.)

# Phase 9 — weight compression / memory-traffic accounting (arithmetic only)

`perf stat` is unavailable on HOST B (see phase7_8_perf_unavailable.md), so
this file derives effective traffic from measured latencies + exact byte
counts. No DRAM counters are claimed.

## Byte counts (exact, from the IR / safetensors headers)

- OV INT8 model, language-model bin: **7,573,989,065 B** (SHA256-matched to
  HOST A artifact). IR census: 506 U8 constants = 7.570 GB (weights), 506
  Convert→f16 + Multiply (per-channel dequant), 253 FP16 consts = 3 MB.
- OV FP16 control, language-model bin: **15,136,811,461 B** (= 1.999× the
  int8 bin).
- lm_head weight: 4096 × 151,936 = 622,329,856 rows·cols → 622 MB as u8
  = **8.2% of all weight bytes**, spent computing logits the conditioning
  bridge does not consume (matches profiling: lm_head ≈ 12.8 ms of 185 ms).
- ComfyUI A2 per-encode dequant traffic (HOST A, committed evidence):
  13,891,534,848 B dequantized **per encode** (252 layers), then fp32 GEMM
  reads ~2× the bf16 weight bytes again as fp32.

## Effective weight-stream rates (bytes / warm p50)

HOST B (this investigation; 16 cores pinned, 3-proc medians):

| path | weights streamed | P1 p50 | eff. GB/s | P3 p50 | eff. GB/s |
|---|---:|---:|---:|---:|---:|
| int8 + DQ(32) VNNI (default) | 7.57 GB | 0.192 s | 39.4 | 0.931 s | 8.1 (see note) |
| int8 + DQ=128 VNNI | 7.57 GB | 0.157 s | 48.2 | 0.739 s | 10.2 |
| int8, DQ=0 (bf16 decompressed) | ~15.1 GB | 0.263 s | 57.4 | 1.307 s | 11.6 |
| FP16 model (no compression) | 15.14 GB | 0.232 s | 65.2 | 1.149 s | 13.2 |

HOST A (committed repo numbers, Zen 5): 7.57 GB / 0.187 s ≈ **40.5 GB/s** (P1).

Note: P3 effective "GB/s" is not pure weight streaming — per-token attention,
RoPE and KV-cache traffic grow with sequence length and the same 7.57 GB is
amortized over more compute; the P1 column (25 tokens, most weight-bound
regime) is the cleanest bandwidth proxy. The P1 effective rate saturating
around 39–48 GB/s on 16 pinned Zen 4 cores — versus 57–65 GB/s when twice the
bytes stream — indicates the DQ brgemm path is not purely DRAM-limited on
HOST B; kernel/quantization overheads (the src-quantization prologue) also
cost time. On HOST A the near-identical 40.5 GB/s suggests the same regime.

## Interpretation

1. The **dominant** memory-traffic saving versus the ComfyUI product path is
   structural: OpenVINO streams u8 weights ONCE per encode (7.57 GB) with
   dequantization fused into the JIT kernel; ComfyUI A2 materializes 13.9 GB
   of bf16 dequantized weights per encode and then reads them again as fp32
   GEMM inputs — i.e. > 3× the bytes, plus unfused aten::mm.
2. Versus a hypothetical *efficient* bf16/fp16 CPU path, the int8 storage
   contributes exactly the 2× byte ratio; measured: FP16-OV control is
   1.21× (P1) / 1.47× (P2) / 1.23× (P3) slower than the int8 DQ-VNNI default.
3. RSS: int8 ≈ 14.8 GiB in every DQ config (≈ 2× the u8 bin: original blob +
   blocked-layout packed copy); FP16 model ≈ 28.7 GiB. HOST A measured the
   same ~14.6–15.1 GiB for the int8 model, consistent with HOST B's replica.

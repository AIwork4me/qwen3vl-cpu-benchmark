# oneDNN verbose findings on the OpenVINO 2026.4.0 vendored build (HOST B)

All logs under results/openvino_isa/. Baseline facts established from live runs:

## What works

- `ONEDNN_VERBOSE=profile_exec` (captured: onednn_profile_exec_partial.stdout):
  - header: `oneDNN v3.13.0 (commit 1289c3b65dd6a119a5ed12a816517d9c3a21d81b)`
  - `cpu,runtime:threadpool,nthr:16` — threadpool honors OV INFERENCE_NUM_THREADS=16
  - `cpu,isa:Intel AVX-512 with Intel DL Boost and bfloat16 support`
    (oneDNN isa = AVX512_CORE_BF16, which subsumes AVX512_CORE_VNNI)
  - exec lines before a mid-inference abort (see below):
    - `reorder,jit:uni, src:u8::blocked:ab dst:u8::blocked:AB4b32a4b, 4096x4096, 1.68`
      → int8 weights are laid out into a 4-byte-B-blocked brgemm packing layout
      (VNNI-dword-friendly), i.e. weights are NOT decompressed to bf16/fp32 here.
    - `reorder,jit_direct_copy:uni, src:f32 dst:bf16, 1x39x4096`
      → an inference-time activation tensor (P1: 39 tokens) converted f32→bf16.
    - several `reorder src:u8 dst:s32` small constant folds.
- On a tiny FP32 MatMul model (de-risk probe, not this workload), full verbose
  works and shows `inner_product, brgemm:avx512_core_bf16, src:bf16 wei:bf16 dst:f32`
  — demonstrating the plugin auto-selects bf16 brgemm on this CPU when allowed.

## What does NOT work on the real workload (vendored-build limitations, honestly recorded)

- `ONEDNN_VERBOSE=1` and `=2`: compile of openvino_language_model fails at
  `Node Constant_57349_u8_i32_Broadcast_123803 of type Reorder could not execute
  a primitive` (captured in onednn_default.log / verbose_default.stdout).
  No such failure without verbose.
- `ONEDNN_VERBOSE=profile_exec`: passes compile, emits the lines above, then the
  process aborts silently (rc=1, no traceback) during the first main-model
  inference, before any matmul/brgemm exec line is emitted.
- `ONEDNN_VERBOSE=action|create|forward|oneline`: run succeeds but zero verbose
  lines are produced (values not accepted by this build → verbose off).
- `DNNL_VERBOSE=1` (legacy name): same compile failure as =1.

## Consequence for the evidence chain

Per-primitive oneDNN verbose for the *entire real workload* is NOT obtainable on
this build/host. Dispatch evidence therefore rests on:
  (B) OpenVINO runtime profiling execTypes per node (results/openvino_root_cause/profiling_*.csv)
  (C) ONEDNN_JIT_DUMP machine-code disassembly (results/openvino_isa/jit_*)
  (E) ONEDNN_MAX_CPU_ISA ablation with per-ISA exec types + latency (results/openvino_isa/isa_matrix.*)
plus the partial verbose lines above (weights u8 blocked reorder + f32→bf16 activation).

# Phase 3 — oneDNN verbose findings

## What was captured (raw: `onednn_verbose_exec_crash.log`, parsed: `onednn_default_summary.json`)

Running the exact bridge workload with `ONEDNN_VERBOSE=all` (or `1`, `2`, `profile_exec`)
emits the oneDNN verbose header and *starts* logging primitive creation, then the OpenVINO
compile **crashes** at a compile-time constant-fold reorder. Header (verbatim, Level B for
runtime identity/ISA):

```
onednn_verbose,v1,info,oneDNN v3.13.0 (commit 1289c3b65dd6a119a5ed12a816517d9c3a21d81b)
onednn_verbose,v1,info,cpu,runtime:threadpool,nthr:16
onednn_verbose,v1,info,cpu,isa:Intel AVX-512 with Intel DL Boost and bfloat16 support
onednn_verbose,v1,info,gpu,runtime:none
```

- **oneDNN = v3.13.0**, statically embedded in `libopenvino_intel_cpu_plugin.so` (Phase 1).
- CPU threading: **threadpool runtime, 16 threads** (the threadpool is backed by the bundled
  oneTBB 2021.13.1 — Phase 1).
- Runtime ISA string: "Intel AVX-512 with Intel DL Boost and bfloat16 support" — oneDNN's name
  for **avx512_core_bf16-class hardware** (AVX-512 + VNNI (DL Boost) + AVX512_BF16).
  This is the *process-level ISA capability*, not a per-primitive dispatch decision.

Creation lines captured before the crash (80 total):
- 79x `create:dispatch,reorder,unsupported datatype` (dispatch rejections logged from
  `src/cpu/x64/matmul/brgemm_matmul_reorders.cpp:292` — evidence the embedded oneDNN *is* the
  compute backend being asked for brgemm reorders), then
- 1x `create:cache_miss,cpu,reorder,jit_direct_copy:uni,undef,src:u8::blocked:a::f0
  dst:s32::blocked:a::f0` — the primitive whose execution then fails.

## The verbose-mode crash (documented limitation)

Every verbose mode that enables execution-time logging (`all`, `1`, `2`, `profile_exec`)
makes `core.compile_model()` fail with:

```
Node Constant_57349_u8_i32_Broadcast_123803 of type Reorder
could not execute a primitive
```

The node is a compile-time constant fold (u8 const -> broadcast -> i32). With NO verbose env
the same compile succeeds and the model runs (Phase 0/2). The failure reproduces under
`--dyn-quant-group-size 0`, `ONEDNN_MAX_CPU_ISA=AVX2` and `AVX512_CORE_BF16` — it is
dtype-driven, not ISA-driven. Verbose modes without exec logging (`create`,
`create:cache_miss`, `create:dispatch`) produce no output at all in this embedded build.
Setting `ONEDNN_VERBOSE` only after `compile_model()` produces no exec lines either (oneDNN
fixes the logging decision at primitive creation).

Conclusion: per-primitive oneDNN verbose dispatch/exec lines are **NOT obtainable** for this
model + OpenVINO 2026.4.0 + oneDNN 3.13.0 build on this host. Per task rules this is NOT
interpreted as "OpenVINO does not use oneDNN" — the dispatch rejections above plus Phase 2's
OV exec types and Phases 6/7 (JIT dump, perf) carry that question instead.

## Why there is no `onednn_default.log` exec content

`onednn_default.log` in this directory is the stdout of a successful
`ONEDNN_VERBOSE=create` run (no verbose lines emitted — kept as evidence of that behavior).
The strongest partial verbose evidence is `onednn_verbose_exec_crash.log` (header + creation
lines + crash). Parsing: `scripts/analyze_onednn_verbose.py <log> --out <json>`.

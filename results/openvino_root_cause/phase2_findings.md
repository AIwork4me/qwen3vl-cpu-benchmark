# Phase 2 — OpenVINO runtime model + profiling findings

Script: `scripts/investigate_openvino_runtime.py --tag default` (workload identical to the
committed bridge: same add_outputs patch — `__module.model.language_model.layers.35/aten::add/Add_1`,
same T2I tokenization, same feeds; compiled with `ov.properties.enable_profiling()`).

Artifacts: `profiling_default_P{1,2,3}.csv`, `profiling_default_summary.json`,
`runtime_nodes_default.json` (2721 nodes), `runtime_model_default.xml` (1.8 MB serialized).

## Effective runtime properties (from compiled model)

- PERF_COUNT=YES, INFERENCE_NUM_THREADS=16, NUM_STREAMS=1, PERFORMANCE_HINT=LATENCY
- **INFERENCE_PRECISION_HINT = bfloat16 (default)** — CPU plugin's default inference precision on this part
- **DYNAMIC_QUANTIZATION_GROUP_SIZE = 32 (default, RW property confirmed present)**
- SCHEDULING_CORE_TYPE=ANY_CORE(0), ENABLE_HYPER_THREADING=False, TBB_PARTITIONer=STATIC(1)

## Headline dispatch result (Level B evidence — runtime dispatch)

All 253 dense-projection nodes (36 layers x 7 linears + lm_head) are executed as:

```
node_type   = FullyConnectedCompressed
exec_type   = brgemm_avx512_bf16        <-- NO "vnni" in the primitive name
rtPrecision = u8                        (weights runtime precision: int8 storage kept)
outputs     = bf16, layout abc (blocked)
```

No `*_vnni` exec_type appears anywhere in the 1294 profiled rows
(1037–1051 of which are Status.EXECUTED per prompt; the rest NOT_RUN — no vnni either way). The only non-bf16 GEMM is
rotary_emb's small `brgemm_avx512_f32` MatMul (25 us).

Graph-wide: RMS=jit_avx512_bf16, Eltwise=jit_avx512_bf16/f32, RoPE=ref_any_bf16,
SDPA=undef_bf16, Reorders=jit_uni_bf16 (f32->bf16 activations) + jit_uni_f32 — the activation
pipeline runs in bf16, NOT u8: there are no activation-quantize nodes in the executed graph.

## Time distribution (get_profiling_info, sum of node times)

| exec_type | P1 share | P2 | P3 |
|---|---:|---:|---:|
| brgemm_avx512_bf16 (253 FC nodes) | 91.35% | 92.40% | 92.20% |
| everything else (145 RMS, 77 eltwise f32, 73 reorder bf16, 72 RoPE, 35 SDPA, ...) | 8.65% | 7.60% | 7.80% |

Node-time totals: P1 0.179 s / P2 0.270 s / P3 0.401 s vs wall p50 0.182/0.372/0.684 s.
(P3 node sum < wall: profiling reports per-node execution on 16 threads; overlap/parallel
execution makes node sums a dispatch census, not a serial decomposition.)

## What this establishes and what it does NOT

ESTABLISHED (Level B):
- The hot path (91–92% of time) is a oneDNN BRGEMM dispatched at **avx512_bf16** ISA with
  **u8 weight precision and bf16 activations/outputs** — consistent with oneDNN brgemm
  weight-decompression (u8 -> bf16 inside the copy-B repack), NOT with a u8xs8 int8 GEMM.
- The workload does NOT run W8A8 integer matmul; activations are converted f32->bf16, never quantized to u8.
- Weight compression (u8 storage) is preserved at runtime — weights are not dequantized once
  at load (that would show f32/bf16 rtPrecision + 2x weight bytes).

NOT YET ESTABLISHED:
- Which machine instructions the brgemm_avx512_bf16 JIT kernel emits (vdpbf16ps vs vpdpbusd) → Phase 6 JIT dump.
- oneDNN-level confirmation of src/wei/dst dtypes and ISA string → Phase 3 verbose.
- Whether AVX512_VNNI is used anywhere in the hot path → Phases 3/6/7.
- Causal contribution of BF16/AVX-512 vs weight compression → Phase 4 ISA ablation.

## Interpretation notes (hypotheses to test, not conclusions)

- `runtimePrecision=u8` + `brgemm_avx512_bf16` matches OpenVINO's weight-compressed FC
  (FullyConnectedCompressed) path: u8 weights are repacked+dequantized per call by a oneDNN
  BrgemmCopyB kernel (the plugin string `/oneDNN:BrgemmCopyBKernel` exists) and the MACs run
  in bf16 on the AVX512_BF16 unit.
- The 12.3 ms P1 top node is `__module.lm_head` (N=151936): 6.88% of total node time.
- DYNAMIC_QUANTIZATION_GROUP_SIZE=32 is active by default, but no u8 activation quantization
  is visible in this graph; its actual effect is tested in Phase 5.

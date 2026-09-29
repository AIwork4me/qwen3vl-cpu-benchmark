# Phase 5 — Dynamic quantization ablation

Protocol: `scripts/investigate_openvino_runtime.py --tag dq_<g>_r<N> --dyn-quant-group-size <g>
--warm-iters 2 --measure-iters 12 --save-hidden` (fresh process per run). g=0/64/128: 3/1/1
runs (g=32 default measured with 3 fresh runs as `dq_default_r*`). Same bridge workload, 16
threads, NUM_STREAMS=1, LATENCY. Outputs: `dynamic_quant_matrix.csv/json`, per-run
`profiling_dq_*_*.csv`, `runtime_nodes_dq_*.json`, hidden npys + `hidden_cosine_summary.json`.

## Headline (median-of-runs p50, s; cosines vs BF16 reference)

| DYNAMIC_QUANTIZATION_GROUP_SIZE | P1 | P2 | P3 | cosine P1/P2/P3 |
|---|---:|---:|---:|---|
| **0 (disabled)** | 0.2274 | 0.5371 | 1.0101 | 0.99764 / 0.99838 / 0.99879 (best) |
| 32 (default) | 0.1854 | 0.3770 | 0.7064 | 0.99724 / 0.99806 / 0.99845 |
| 64 | 0.1672 | 0.3414 | 0.6037 | 0.99687 / 0.99780 / 0.99821 |
| 128 | 0.1590 | 0.3359 | 0.5559 | 0.99592 / 0.99690 / 0.99707 |

- Disabling dynamic quantization costs **+23% (P1) / +42% (P2) / +43% (P3)**; dq_g0 p50s
  across 3 runs are tight (P2 0.5344/0.5384/0.5371, spread 0.7%).
- Coarser groups speed it up further (g128: -11% P2, **-21% P3**) at monotonically decreasing
  cosine — a textbook quantization speed/accuracy trade-off curve. This proves the knob
  changes NUMERICS inside the FullyConnectedCompressed nodes, not just scheduling.

## Task-question answers

1. **Is dynamic quantization enabled by default?** YES — counterfactually: setting it to 0
   slows every FC node uniformly (per-node median 0.57 -> 0.83 ms, lm_head 19.8 -> 27.9 ms on
   P2; brgemm total 0.252 -> 0.355 s) and changes output values.
2. **Does MatMul exec type change when disabled?** NO — runtime model still reports 253 x
   `brgemm_avx512_bf16` / rtPrecision u8 / outputs bf16. The parameter acts INSIDE the node.
3. **Do VNNI kernels disappear when disabled?** No `vnni`-named primitive exists in ANY dq
   config (dispatch tables identical across g=0/32/64/128, verified from runtime_nodes files).
   What differs is the internal kernel behavior — resolved at instruction level in Phase 6.
4. **Latency when disabled:** see table (+23% to +43%).
5. **Cosine changes:** yes — g0 is *more* accurate (no dynamic requantization), coarser groups
   progressively less accurate. All variants remain >= 0.9959.

## Interpretation (hypothesis, tested in Phase 6)

The monotonic speed-vs-accuracy response to the group size, uniform across all 253 FC nodes,
with unchanged u8 weight storage and unchanged primitive name, is the signature of **runtime
(dynamic) requantization of activations** in group-wise fashion inside the weight-compressed
FC kernel: g=0 disables it (weights dequantized to bf16, pure bf16 compute — slower);
g=32 (default) quantizes in groups of 32 (fast); larger groups amortize scale handling
(faster, coarser). WHICH instruction family performs the MACs (bf16 dot vs int8 dot) is not
determinable from OpenVINO metadata alone — Phase 6 disassembles the JIT kernels for g=0
vs g=32 to settle whether the default path executes VNNI (vpdpbusd) or BF16 (vdpbf16ps)
instructions.

## Side finding

g=128 is 21% faster than the shipped default at P3 with cosine still 0.9971 — a free
latency/accuracy trade worth knowing for product tuning (out of scope to change defaults
here; recorded as evidence).

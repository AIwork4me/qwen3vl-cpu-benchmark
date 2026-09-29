# Phase 4 — ISA ablation matrix (core experiment)

Runner: `scripts/run_openvino_isa_matrix.sh` — 6 runnable configs x 3 fresh processes,
each 2 warmup + 12 measured inferences of P1/P2/P3 (identical bridge workload), 16 threads,
NUM_STREAMS=1, LATENCY hint, strictly sequential. Raw per-run: `profiling_isa_<cfg>_r<N>.*`
+ `runlogs/*.log`. Aggregate: `isa_matrix.csv` / `isa_matrix.json`.

## ISA ceiling runnability (probed first; one fresh process each)

| ONEDNN_MAX_CPU_ISA | verbose isa-string | INFERENCE_PRECISION_HINT | runnable? | FC primitive |
|---|---|---|---|---|
| (unset=DEFAULT) | AVX-512 + DL Boost + bf16 | bf16 (auto) | YES | brgemm_avx512_bf16 |
| AVX512_CORE_BF16 | same | bf16 | YES | brgemm_avx512_bf16 |
| AVX512_CORE_VNNI | AVX-512 + DL Boost | f32 forced | YES | brgemm_avx512_f32 |
| AVX512_CORE | AVX-512 + BW/VL/DQ | f32 forced | YES | brgemm_avx512_f32 |
| AVX2_VNNI | AVX2 + DL Boost | f32 forced | YES | brgemm_avx2_f32 |
| AVX2 | AVX2 | f32 | **NO** — "FullyConnectedCompressed ... could not create a primitive descriptor for the inner product forward propagation primitive" (runlogs/isa_AVX2_NOT_RUNNABLE.log) |
| AVX2 / AVX2_VNNI / AVX512_CORE / AVX512_CORE_VNNI | — | bf16 | **NO** — same primitive-descriptor failure | — |
| any | — | i8 | **NO** — plugin rejects: INFERENCE_PRECISION_HINT supports bf16/f16/f32 only |

Env-var effectiveness: every accepted ceiling changes the oneDNN verbose `cpu,isa:` header
string, AND the runtime-model dispatch table changes accordingly (below) — ONEDNN_MAX_CPU_ISA
is confirmed applied, not ignored. Invalid values silently fall back to the full ISA.

## Dispatch verification per config (253 FullyConnected nodes; from runtime_nodes_isa_*)

| config | primitiveType | weights rtPrecision | outputs |
|---|---|---|---|
| DEFAULT | brgemm_avx512_bf16 | u8 | bf16 |
| BF16CEIL | brgemm_avx512_bf16 | u8 | bf16 |
| DEFAULT_F32 | brgemm_avx512_f32 | u8 | f32 |
| VNNI512_F32 | brgemm_avx512_f32 | u8 | f32 |
| AVX512_F32 | brgemm_avx512_f32 | u8 | f32 |
| AVX2VNNI_F32 | brgemm_avx2_f32 | u8 | f32 |

In EVERY runnable config the weights stay u8 (int8 storage preserved, dequant-on-the-fly),
and no primitive name ever contains "vnni" — including the VNNI-capable ceilings. **No
VNNI-dispatched kernel exists anywhere in this workload's ISA matrix.**

## Latency results (median-of-3-fresh-process p50, seconds)

| config | P1 (25 tok) | P2 (67) | P3 (171) |
|---|---:|---:|---:|
| DEFAULT (bf16@512b) | 0.1765 | 0.3763 | 0.7070 |
| BF16CEIL (sanity, same HW ceiling) | 0.1868 | 0.3767 | 0.7150 |
| DEFAULT_F32 (f32@512b, same ISA) | 0.1829 | 0.3766 | 0.7285 |
| VNNI512_F32 (f32, VNNI ceiling) | 0.1843 | 0.3763 | 0.7275 |
| AVX512_F32 (f32, no-VNNI ceiling) | 0.1701 | 0.3758 | 0.6740 |
| AVX2VNNI_F32 (f32@256b) | 0.2426 | 0.5183 | 1.0448 |

Run-to-run p50 spread per config-prompt: 0.5–9.5% (P1/P3 noisier; P2 tightest, 0.5–2.4%).
Within-run CV: 0.8–3.8%.

## Pairwise counterfactuals (latency ratio = a/b; >1 means b is faster)

| comparison | P1 | P2 | P3 | reading |
|---|---:|---:|---:|---|
| BF16 vs F32 compute, same ISA (DEFAULT_F32→DEFAULT) | 1.036x | **1.001x** | 1.030x | BF16 dot-product adds ~0–4% at most (P2, the tightest dataset, says 0.1%); differences are within P1/P3 run noise |
| VNNI ceiling vs no-VNNI ceiling, both f32 (AVX512_F32→VNNI512_F32) | 0.923x | **0.999x** | 0.926x | bit-identical dispatch tables; P2 shows exact parity (0.999x, tightest data). P1/P3 show a CONSISTENT ~7.5–8% gap (non-overlapping run ranges: P1 0.1678–0.1716 vs 0.1835–0.1861) — real but unexplained by the dispatch table (possible ceiling-dependent JIT codegen, e.g. ymm/zmm preference inside oneDNN; compared at binary level in Phase 6). Either way the VNNI ceiling is never FASTER — no VNNI benefit in any direction |
| AVX-512 vs AVX2 width, f32 (AVX2VNNI_F32→AVX512_F32) | 1.426x | **1.379x** | 1.550x | 512-bit registers give a real, repeatable ~1.4–1.55x over 256-bit |
| BF16CEIL→DEFAULT (sanity; ratio = DEFAULT latency / BF16CEIL latency) | 1.058x | 1.001x | 1.011x | same HW ceiling → parity within noise (P1 outlier run 0.1924 inflated spread) |

## Conclusions (within the limits of this matrix)

1. **[SUPERSEDED by Phases 5+6]** As written at Phase-4 time (primitive-name evidence
   only): "AVX512_VNNI contributes nothing measurable — never dispatched (no vnni primitive
   in any config), VNNI-capable ceiling changes latency -0.1% (P2) to noise (P1/P3)".
   Phases 5+6 overturned the 'never dispatched' reading: no *primitive name* contains vnni,
   but the dispatched brgemm kernels DO execute vpdpbusd (VNNI) via dynamic quantization,
   and the DQ engine switch measures VNNI's increment at 1.23–1.43x. The ceiling-comparison
   sentence stands as written (it is NOT a valid VNNI isolation — see phase6 conclusion 4).
2. **AVX-512 width (512-bit registers + more of them) contributes ~1.4–1.55x** vs the
   256-bit AVX2 ceiling — the only sizeable SIMD-ISA factor.
3. **BF16 compute vs F32 compute (same AVX-512 ISA, same u8 weight traffic): ~0–4%** —
   the workload is not compute-precision-bound; the memory-side (int8 weight footprint)
   dominates. (Quantified in Phase 9/10.)
4. The INT8 weight-compressed FC path **requires at least AVX2_VNNI-class hardware** to run
   at all in this build (AVX2 alone: primitive-descriptor failure; the preserved failure log `isa_AVX2_NOT_RUNNABLE.log` is the AVX2+bf16-hint probe, the AVX2+f32 probe showed the identical message), and uses bf16 kernels
   only when avx512_core_bf16 is available; VNNI capability alone does not select an int8
   dot-product kernel for this model.
5. ISA env is honored: dispatch table + verbose isa-string change with every ceiling.

Caveats:
- P1/P3 carry occasional outlier runs (DEFAULT P1 r1 = 0.1924 vs 0.1756/0.1765) — medians
  and the P2 column are the reliable comparisons; no comparison in this report relies on a
  <5% difference alone.
- "AVX2" pure (no VNNI) is not runnable, so the width comparison uses AVX2_VNNI (256-bit
  f32 kernels) as the 256-bit baseline.

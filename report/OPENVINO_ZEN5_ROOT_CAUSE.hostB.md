# Why OpenVINO is Fast on Ryzen AI Max+ PRO 395 — Root-Cause Investigation of the Qwen3-VL INT8 Conditioning Workload

> **两主机合并说明（2026-09-29）**：本报告为 HOST B 的原样保留版本；与其冲突路径的
> 工件已加 `_hostB` 后缀（如 `results/openvino_isa/isa_matrix_hostB.csv`），其脚本保存在
> `scripts/hostb/`（文中 Reproduce 命令里的 `scripts/...` 在合并后的树上对应 `scripts/hostb/...`）。
> HOST A（Ryzen AI Max+ PRO 395 / Zen 5，原 benchmark 主机）的独立结论见
> [OPENVINO_ZEN5_ROOT_CAUSE.md](OPENVINO_ZEN5_ROOT_CAUSE.md)——两台机器独立得出一致判定。

Date: 2026-09-29 · Investigation branch: `investigate/openvino-zen5-vnni-root-cause`

> **Host scope (read this first).** The original benchmark (README.md,
> report/RESULTS.md) was measured on **HOST A** = AMD Ryzen AI Max+ PRO 395
> (Zen 5, 16C/32T, 94 GiB LPDDR5X UMA). This root-cause investigation was
> executed on **HOST B** = 2× AMD EPYC 9334 (Zen 4, 64C/128T, 503 GiB DDR5),
> because it is the machine the investigation ran on. It is a valid replica
> for the *mechanistic* questions: same OpenVINO wheel version
> (2026.4.0-22959-99c81491cc3), byte-identical model artifacts (SHA256 match,
> `results/openvino_root_cause/host_b_model_sha256.txt`), and the same
> oneDNN-relevant ISA set (AVX512_CORE + AVX512_CORE_VNNI + AVX512_CORE_BF16
> present, AMX absent). The replica was verified numerically: hidden states
> cosine **1.0000001** vs the committed HOST-A artifacts for P1/P2/P3.
> Absolute latencies are host-specific; every table below states its host.

## TL;DR

1. **Yes — the OpenVINO CPU hot path for this workload is an integer-dot
   (VNNI) kernel, not a float kernel.** All 253 fully-connected layers execute
   as `FullyConnectedCompressed` nodes whose JIT kernels contain the
   AVX-512 VNNI instruction `vpdpbusd` (208 instances × 12 kernel variants),
   fed u8 weights and *dynamically quantized* activations
   (`DYNAMIC_QUANTIZATION_GROUP_SIZE=32` by default; dedicated
   `brgemm_src_quantization_kernel_t` JIT kernels are present).
2. **The execType string `brgemm_avx512_bf16` is about the *activation
   dtype entering the node* (bf16), not the compute**: bf16 activations are
   quantized to 8-bit in-flight and multiplied against u8 weights with
   `vpdpbusd`; the epilogue converts s32 → f32 → bf16 (`vcvtneps2bf16`).
3. **Counterfactual proof**: setting `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`
   removes every `vpdpbusd` from the JIT dump (the node then runs pure-bf16
   `vdpbf16ps` kernels) and slows the workload **1.37–1.47×**
   (P1 0.1922→0.2632 s, P2 0.4375→0.6425 s, P3 0.9311→1.3067 s; 3 fresh
   processes × 10 iters, per-proc CV ≤ 3%).
4. **AVX-512 is a hard requirement** (AVX2/AVX2_VNNI/AVX512_CORE ceilings →
   `FullyConnectedCompressed` cannot create a primitive at all), but the
   **isolated "VNNI vs no-VNNI" increment could NOT be measured** on this
   build: `ONEDNN_MAX_CPU_ISA=AVX512_CORE` still JITs `vpdpbusd` kernels
   (the ceiling only gates bf16). The honest VNNI-size estimate is the DQ=0
   counterfactual, which bundles u8-weight streaming with VNNI compute:
   **1.37–1.47×**.
5. **The INT8 weights are never fully decompressed at runtime** (default
   path): they are reordered once at load into a blocked u8 layout
   (`AB4b32a4b`) and dequantization is fused into the JIT kernel. An FP16
   model of the same architecture (same export pipeline) runs **1.21–1.47×
   slower** and needs **1.999× the weight bytes, 1.94× RSS**.
6. The default group size 32 is **not latency-optimal** — `DQ=128` is
   **1.16–1.26× faster** than the shipped default (P1 0.157 s on HOST B), at
   cosine 0.998 vs the default's numerics (DQ=64: ~1.2× at cosine 0.999 —
   the closer-numerics trade-off; HOST A measurements agree in direction).
7. The 22–69× vs the ComfyUI CPU product path is dominated by *structural*
   factors (no per-encode 13.9 GB dequant, no fp32 GEMM, persistent packed
   weights, fused JIT, 1.26 s load) — **not** by "VNNI magic". Even ComfyUI's
   own forced-int8 path (A1, true `torch._int_mm`, HOST A) is still 12–39×
   slower than OpenVINO.
8. What was **not** obtained: perf/PMU hotspot proof (Level D) — `perf` is
   unavailable in this container (`perf_event_paranoid=4`, no binary). The
   execution claim rests on dispatch + JIT machine code + causal ablation
   (Levels B, C, E).

## What was known before this investigation

From the committed repo (HOST A):

- OpenVINO 2026.4.0, CPU device asserted, LATENCY hint, 16 threads, 1 stream.
- Bridge latencies P1/P2/P3 = 0.187/0.377/0.687 s; load 1.26 s; RSS 14.57–15.1 GiB.
- Model = NNCF INT8_ASYM weight compression, group_size=-1 (per-channel), **not W8A8**.
- README said: *True INT8 GEMM: UNKNOWN¹ — OpenVINO's internal oneDNN kernel
  choice was not dumped.*

This investigation replaces that UNKNOWN.

## Hardware capability (Level A — necessary, not sufficient)

| Host | CPU | ISA flags relevant here |
|---|---|---|
| A (original) | Ryzen AI Max+ PRO 395 (Zen 5) | avx2, avx512f/vl/bw/dq, **avx512_vnni**, avx_vnni, avx512_bf16, no AMX |
| B (this investigation) | 2× EPYC 9334 (Zen 4) | avx2, avx512f/vl/bw/dq, **avx512_vnni**, avx512_bf16, no AMX (`environment/host_b_system.json`) |

oneDNN verbose header on HOST B: `cpu,isa:Intel AVX-512 with Intel DL Boost and
bfloat16 support` (= oneDNN `AVX512_CORE_BF16`, which subsumes `AVX512_CORE_VNNI`).

## OpenVINO CPU backend (Phase 1)

- `libopenvino_intel_cpu_plugin.so` — **no dynamic `libdnnl.so` dependency**
  (ldd/readelf): oneDNN is **statically vendored** (192 `thirdparty/onednn`
  debug paths). Version via verbose header: **oneDNN v3.13.0**
  (commit 1289c3b6…). Threading: bundled **oneTBB** (`libtbb.so.12`),
  verbose `cpu,runtime:threadpool,nthr:16` — the threadpool honors
  `INFERENCE_NUM_THREADS=16`.
- Full record + SHA256: `results/openvino_root_cause/backend_linkage.txt`.

## Runtime execution graph (Phase 2)

Profiling (`ov.properties.enable_profiling`, per-node `get_profiling_info`,
`results/openvino_root_cause/profiling_P1_baseline_p*.csv`,
`results/openvino_isa/profiling_P*_default_p*.csv`):

- 1294 profiled nodes; **253 nodes with execType `brgemm_avx512_bf16`
  (92.4–93.7% of total time)** + 1 `brgemm_avx512_f32` (lm_head tail variant
  in some runs). Zero int8-named exec types — the OV execType taxonomy does
  not expose the inner compute flavor; that required JIT evidence below.
- lm_head alone: 12.8 ms of 185 ms (P1) — 8.2% of weight bytes computing
  logits the conditioning bridge discards.
- IR weight census: **506 U8 constants = 7.570 GB** + 506 `Convert→f16` +
  `Multiply` (per-channel scale). Weights are u8 *in the graph*; the runtime
  keeps them u8 (default path) and packs them once (blocked reorder) at load.
- Supported properties include `DYNAMIC_QUANTIZATION_GROUP_SIZE` (default
  **32**), `INFERENCE_PRECISION_HINT` (default bfloat16), KV-cache u8,
  `TBB_PARTITIONER=STATIC`.

## oneDNN dispatch evidence (Phase 3)

`ONEDNN_VERBOSE` is only partially usable on this vendored build
(`results/openvino_isa/onednn_verbose_findings.md`): `1`/`2` crash at compile
(a Reorder u8→i32 constant fold), `profile_exec` aborts mid-inference after
emitting the load-time **u8 weight reorder `ab → AB4b32a4b` (4096×4096)** and
an inference-time **f32→bf16 activation reorder (1×39×4096)**. Those two
lines + the JIT dump below are the primitive-level dispatch evidence.

## Is AVX512_VNNI actually executed? (Q1 — the core answer)

**On HOST B: dispatched and JIT-compiled, with causal latency effect —
YES (machine-code level); direct PMU hotspot proof NOT obtained (no perf).**

Evidence chain:

- **Level B (dispatch)**: 253 FC nodes run as `FullyConnectedCompressed` /
  `brgemm_avx512_bf16`; `DYNAMIC_QUANTIZATION_GROUP_SIZE=32` default;
  6 JIT'd `brgemm_src_quantization_kernel_t` instances in the default dump.
- **Level C (JIT machine code)**: 12 `jit_brgemm_kernel_t` binaries ×
  **208 `vpdpbusd`** each in the default run
  (`results/openvino_isa/jit_vnni_hits.txt/json`); zero in the DQ=0 and FP16
  runs; 12 × 208–304 in both f32-precision-hint runs.
- **Level D (hotspot)**: NOT AVAILABLE — perf absent (documented, with
  reproduction command, in `results/openvino_isa/phase7_8_perf_unavailable.md`).
- **Level E (counterfactual)**: DQ=0 → VNNI kernels vanish + 1.37–1.47×
  slower; AVX2-family ceilings → primitive-creation failure; VNNI ceiling →
  FC compiles (VNNI is what its int8 kernels need) and the model then fails
  in the *attention* bf16 brgemm ("could only be used above avx512_bf16").

**On HOST A: not directly measured — STRONG EVIDENCE of the same path**
(same wheel, byte-identical IR+weights, same ISA decision inputs, cross-host
hidden-state cosine 1.0000001, matching RSS ≈ 2× int8 bin). Per the task's
conclusion template this is **Case A for HOST B (dispatch + JIT + causal
ablation; perf samples missing)** and a transfer argument for HOST A.

## JIT instruction evidence (Phase 6)

| Run (all HOST B, P1 workload) | JIT binaries | `vpdpbusd` | `vdpbf16ps` | src-quant kernels |
|---|---:|---:|---:|---:|
| default (DQ=32) | 137 | **2496** (12×208) | present (attention/aux) | **6** |
| DQ=0 | 141 | **0** | 18 files | **0** |
| AVX512_CORE_VNNI + f32 hint | 126 | 2496 (12×208) | 0 | 6 |
| AVX512_CORE + f32 hint | 126 | 3648 (12×304) | 0 | 6 |
| FP16 model (no compression) | 132 | **0** | 18 files | 0 |

VNNI-hit excerpts with ±10 instruction context: `jit_vnni_hits.txt`.
Raw binaries kept locally (SHA256 in `jit_instruction_summary.txt`);
only summaries/hits/hashes are committed (660 KB of binaries excluded).

## perf hotspot evidence (Phase 7/8)

Not obtainable: no `perf` binary, `perf_event_paranoid=4`, no CAP_PERFMON.
Exact errors + a ready-to-run command for perf-enabled hosts:
`results/openvino_isa/phase7_8_perf_unavailable.md`. No PMU event census was
possible, so no VNNI retired-instruction counting claim is made.

## ISA ablation (Phase 4)

`results/openvino_isa/isa_matrix_hostB.csv（注：未加 _hostB 后缀的 isa_matrix.csv 现为 HOST A 数据）` — 3 fresh processes each, 2 warm + 10
measured iterations, P1/P2/P3, 16 pinned cores, LATENCY/1-stream. Warm p50
median across processes (HOST B, seconds):

| config | P1 | P2 | P3 |
|---|---:|---:|---:|
| default (isa=AVX512_CORE_BF16, DQ=32) | 0.1922 | 0.4375 | 0.9311 |
| ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 | 0.1869 | 0.4181 | 0.8821 |
| AVX512_CORE_VNNI + f32 hint | 0.1971 | 0.4466 | 0.9618 |
| AVX512_CORE + f32 hint | 0.1773 | 0.4422 | 0.9039 |
| DQ=0 (bf16-decompressed kernels) | 0.2632 | 0.6425 | 1.3067 |
| DQ=128 | **0.1565** | **0.3786** | **0.7393** |
| AVX2 / AVX2_VNNI / AVX512_CORE (no hint) | **cannot run** — `FullyConnectedCompressed` primitive creation fails | | |
| AVX512_CORE_VNNI (no hint) | compiles FC, dies at attention bf16 brgemm | | |

- **Comparison A (AVX2 vs AVX-512): not measurable as a latency delta — the
  workload has no AVX2 fallback on this build.** AVX-512 is a prerequisite,
  not a tunable.
- **Comparison B (VNNI increment): the ceiling does NOT remove VNNI on this
  build** (AVX512_CORE ceiling still JITs vpdpbusd — discovered via JIT dump,
  exactly the trap the investigation plan warned about). The valid
  counterfactual is DQ=0 → **1.37× (P1) / 1.47× (P2) / 1.40× (P3)** slower,
  bundling u8-vs-bf16 weight streaming with VNNI compute.
- **Comparison C (DEFAULT vs AVX512_CORE_BF16 ceiling): 0.95–0.97× (≤5%,
  borderline vs 1–3% CV) — no higher path beyond bf16-ISA is in use.**
- `core_f32` vs `vnni_f32` differ ~10% on P1 — both dispatch VNNI kernels;
  this is kernel-flavor variance (208 vs 304 vpdpbusd unroll), not an ISA
  effect. Not used as evidence.

## Dynamic quantization ablation (Phase 5)

`DYNAMIC_QUANTIZATION_GROUP_SIZE` is supported, writable, and the override is
reflected by `get_property` (0/32/64/128 all verified applied).

| group size | P1 | P2 | P3 | cos vs HOST-A ref (P1/P2/P3) | VNNI kernels in JIT |
|---|---:|---:|---:|---|---|
| 0 (disabled) | 0.2632 | 0.6425 | 1.3067 | 0.99911 / 0.99928 / 0.99939 | none (vdpbf16ps only) |
| 32 (default) | 0.1922 | 0.4375 | 0.9311 | **1.0000001 / 1.0000001 / 1.0** | 12×208 vpdpbusd |
| 64 | 0.1658 | 0.3942 | 0.8691 | 0.99897 / 0.99916 / 0.99925 | (not dumped) |
| 128 | 0.1565 | 0.3786 | 0.7393 | 0.99807 / 0.99791 / 0.99818 | (not dumped) |

(Cosines measured in a follow-up run with `--compare-npy`
(`results/openvino_root_cause/investigate_dqcos_*.json`); the original
ablation runs did not capture cosine — a gap found by self-audit and closed
by the follow-up data. The HOST-A (Zen 5) main report measured the same
trade-off against the BF16 reference: DQ=128 cos 0.9959–0.9971 vs DQ=32
0.9972–0.9985 — direction identical on both hosts.)

Answers to the task's five questions:
1. Default **does** enable dynamic activation quantization (property=32 +
   src-quant JIT kernels + counterfactual behavior).
2. Disabling does not change the execType label (OV taxonomy limitation) but
   changes the *kernel*: bf16 `vdpbf16ps` brgemm on decompressed weights.
3. Disabling removes every VNNI kernel from the dump. **Yes, they vanish.**
4. Latency cost of disabling: **+37% (P1) / +47% (P2) / +40% (P3)**.
5. Output cosine vs the HOST-A (DQ=32) reference is **exactly 1.0000001 only
   at DQ=32** — group size is numerics-relevant, not a pure speed knob:
   DQ=64 → 0.9990–0.9994, DQ=128 → 0.9979–0.9982, DQ=0 (bf16 compute) →
   0.9991–0.9994. These sit in the same band as the repo's accepted variants
   (OV-bridge 0.9972–0.9985, ComfyUI A1 0.9978–0.9986, A2 0.9989–0.9993 vs
   BF16 reference), so DQ=128's accuracy cost is comparable to switching
   between already-accepted paths — but it is **not** numerically equivalent,
   and DQ=64 offers a better speed/accuracy trade-off than DQ=128 if closer (e2e round: DQ=32 recommended — see report/QWEN_IMAGE_E2E_HYBRID.md)
   numerics are required.

## Weight-compression effect (Phase 9 + 10)

Controlled FP16 model (`OpenVINO/Qwen3-VL-8B-Instruct-fp16-ov`, same export
pipeline; language bin exactly 15,136,811,461 B = 1.999× the int8 bin;
SHA256 in `results/openvino_root_cause/fp16_control_sha256.txt`; bridge
cosine vs HOST-A int8 artifacts 0.9971 — within the repo's known
int8-vs-BF16 range, so semantics verified):

| | INT8 WC (default) | FP16 model | ratio |
|---|---:|---:|---:|
| P1/P2/P3 p50 (s) | 0.1922 / 0.4375 / 0.9311 | 0.2324 / 0.6450 / 1.1494 | **1.21 / 1.47 / 1.23×** |
| weight bytes | 7.57 GB (u8) | 15.14 GB | 1.999× |
| RSS | 14.8 GiB | 28.7 GiB | 1.94× |
| JIT vpdpbusd | 2496 | 0 | — |

Effective weight-stream rate at P1: int8-DQ ≈ 39–48 GB/s, 2-byte-weight paths
≈ 57–65 GB/s (HOST B); HOST A default ≈ 40.5 GB/s. Full accounting:
`results/openvino_isa/memory_traffic_accounting.md`.

## Threading / runtime contribution

HOST A committed sweep: 16 threads optimal (8 slower, 32 flat); LATENCY hint
⇒ NUM_STREAMS=1. HOST B replica: same 16-thread setting via
`INFERENCE_NUM_THREADS` + `taskset 0-15`; oneDNN threadpool reports nthr:16.
TBB static partitioner recorded. No further thread search was re-run (out of
investigation scope; committed HOST-A data stands).

## Root-cause decomposition (Phase 11)

Confidence vocabulary per task: CONFIRMED / STRONG EVIDENCE / LIKELY /
NOT ESTABLISHED / DISPROVEN.

| Factor | Evidence | Counterfactual | Measured effect | Confidence |
|---|---|---|---|---|
| INT8 weight compression (u8 resident, 2× fewer bytes) | IR census 506×U8=7.57 GB; RSS 14.8 vs 28.7 GiB | FP16-OV control | 1.21×/1.47×/1.23× (P1/P2/P3) | CONFIRMED |
| Dynamic activation quantization (bf16/f32→u8 in-flight) | DQ property=32; 6 src-quant JIT kernels | DQ=0 | 1.37×/1.47×/1.40×; VNNI kernels vanish | CONFIRMED |
| Integer MatMul actually executed | vpdpbusd in 12 JIT kernels (machine code) | DQ=0 removes them + slows | same as above | CONFIRMED at dispatch+machine-code level; PMU hotspot NOT ESTABLISHED (no perf) |
| AVX-512 (vs AVX2) | primitive-creation failure at AVX2/AVX2_VNNI ceilings | ceiling runs | prerequisite; magnitude not measurable (no fallback) | CONFIRMED (requirement), magnitude NOT ESTABLISHED |
| AVX512_VNNI increment over avx512 bf16-dot | JIT: DQ kernels are VNNI; ceiling cannot disable VNNI on this build | DQ=0 (bundles bytes+VNNI) | 1.37–1.47× bundled; pure ISA increment NOT ESTABLISHED | STRONG EVIDENCE (bundled) |
| DQ group size tuning (32→64/128) | matrix + dqcos runs | DQ sweep | 1.16–1.26× faster; cosine 0.998 (DQ=128) / 0.999 (DQ=64) vs shipped numerics | CONFIRMED (speed); accuracy cost CONFIRMED small but non-zero |
| Threading (16c, 1 stream, LATENCY) | HOST A sweep + HOST B replica | 8/32 threads (HOST A) | 8 slower, 32 flat | CONFIRMED (HOST A) |
| OpenVINO graph/runtime (persistent packed weights, fused dequant, no per-call Python) | load 1.26 s; one-time blocked reorder; JIT fusion; vs A1-forced (true int8 GEMM) still 12–39× slower (HOST A) | A1 vs OV | 39×/20×/12× (HOST A: 7.32/7.46/8.36 s vs 0.187/0.377/0.687 s) | STRONG EVIDENCE |
| Avoided per-encode full dequant (13.9 GB/encode, 252 layers) | HOST A runtime counters | OV keeps u8 + fused dequant | major part of the 22–69× | CONFIRMED |
| ComfyUI FP32 compute path (A2) | HOST A profiler/counters (aten::mm fp32, _int_mm=0) | — | removed by OV path | CONFIRMED (HOST A forensics) |
| Memory-traffic reduction overall | byte accounting both runtimes | arithmetic (Phase 9 file) | >3× fewer bytes than A2 per encode | CONFIRMED (arithmetic) |
| BF16 compute precision (elementwise/attention) | jit_avx512_bf16 nodes; f32→bf16 reorder line | — | aux path only (≈8% time) | CONFIRMED |

## What explains the 22–69× vs the ComfyUI CPU result (HOST A numbers)

Ordered by leverage (multiplicative attribution across different runtimes is
not strictly decomposable; each row cites its counterfactual):

1. **Eliminating per-encode full dequant + fp32 GEMM (the A2 product path)**:
   A2 does 252 layer-dequants (13.89 GB materialized per encode) and then
   fp32 GEMM (2× more weight bytes again) — HOST A counters. OpenVINO streams
   7.57 GB of u8 once with dequant fused in-kernel. This is the dominant
   structural gap. (A1-forced — true `torch._int_mm` int8 GEMM, zero dequant
   — still leaves 7.32/7.46/8.36 s, i.e. **12–39×** from OpenVINO's runtime
   alone: persistent pre-packed weights, fused epilogues, no per-call
   re-packing, TBB static scheduling.)
2. **INT8 storage → half the bytes** of any 16-bit path: 1.21–1.47× (FP16
   control, HOST B).
3. **Dynamic-quantization VNNI integer compute**: 1.37–1.47× over the
   bf16-decompressed counterfactual (HOST B).
4. **Load behavior**: 1.26 s read+compile vs 0.30 s mmap-assign whose real
   page-in lands as a 12.11 s first encode (HOST A).
5. Threading at the sweet spot (16) — necessary but not differentiating.

## What explains the 2.7–3.3× vs the current ComfyUI Radeon GPU product path (HOST A)

The GPU route (committed Experiment C forensics) is **BF16 storage → cast →
fp32 GEMM** on all 252 linears: 2× the weight bytes of bf16, no bf16 tensor
cores, 16.6 GiB resident in GTT on a UMA part sharing the same LPDDR5X, GPU
85% busy. The OpenVINO CPU path streams 1-byte weights with VNNI integer
dots and leaves the GPU entirely free. Note the scope: this is about
**ComfyUI's current conditioning path**, not about what a bf16-compute GPU
path could theoretically do.

## What we can claim

- On HOST B (directly measured): the 253 fully-connected layers of this exact
  model artifact run through JIT kernels containing `vpdpbusd`, with
  dynamically quantized activations, u8 weights resident, and this path is
  1.37–1.47× faster than the bf16-decompressed alternative and 1.21–1.47×
  faster than an FP16-weights model; DQ=128 adds another 1.16–1.26× at
  (Superseded 2026-09-30 by the e2e round: at image level DQ=128/64 measurably reduce
  SSIM/CLIP similarity for 0.11–0.16 s per ~153 s image saved; recommended default is
  **DQ=32** — see report/QWEN_IMAGE_E2E_HYBRID.md, Q7 answer. This item stands as the
  encoder-level record.)
cosine 0.998 (DQ=64 ≈1.2× at 0.999).
- On HOST A (transfer): same wheel + byte-identical model + same ISA decision
  inputs + cosine-1.0 replica ⇒ the same dispatch decision; STRONG EVIDENCE,
  not a direct measurement.
- The previously "UNKNOWN" README cell can now be stated precisely (see
  README update): *OpenVINO keeps the INT8 weights and executes VNNI integer
  dot-product kernels (machine-code-verified on the investigation host) with
  in-flight activation quantization — it is weight-only INT8 storage + dynamic
  activation quantization, still not a statically-quantized W8A8 model.*

## What we cannot claim

- That PMU samples land in the VNNI kernels (no perf on the investigation
  host) — Level D missing.
- A pure "VNNI-only" speedup number on this build (the ISA ceiling cannot
  disable VNNI; DQ=0 bundles byte-width and ISA effects).
- Any AVX2-vs-AVX512 latency ratio for this model (no AVX2 fallback exists).
- That HOST A's absolute VNNI contribution equals HOST B's (different
  memory subsystem; the bundled 1.37–1.47× is HOST B's).
- That the GPU is inherently slower than this CPU (only ComfyUI's current
  fp32-compute conditioning path was measured, on HOST A).

## Reproduce (HOST B, from repo root)

```bash
.venv-openvino/bin/pip install openvino==2026.4.0 numpy transformers
# model: modelscope → models/qwen3vl-openvino-int8 (SHA256 must match environment/model_sha256.txt)

# baseline + profiling + runtime model + properties
taskset -c 0-15 .venv-openvino/bin/python scripts/investigate_openvino_runtime.py \
  --mode bench --tag baseline --threads 16 --warm-iters 2 --measure-iters 10 --compare-npy

# ISA / DQ matrix (3 fresh procs per config; ~25 min)
bash scripts/run_openvino_isa_matrix.sh

# dynamic quantization single runs
taskset -c 0-15 .venv-openvino/bin/python scripts/investigate_openvino_runtime.py \
  --mode bench --tag dq_0 --dq-group-size 0 --threads 16   # etc. 32/64/128

# JIT dump + VNNI instruction scan
mkdir -p results/openvino_isa/jit_dump && cd results/openvino_isa/jit_dump
ONEDNN_JIT_DUMP=1 taskset -c 0-15 ../../../../.venv-openvino/bin/python \
  ../../../scripts/investigate_openvino_runtime.py --mode bench --tag jitdump \
  --threads 16 --warm-iters 1 --measure-iters 1 --prompts P1
cd ../../../.. && .venv-openvino/bin/python scripts/analyze_jit_vnni.py \
  results/openvino_isa/jit_dump --out-dir results/openvino_isa

# FP16 control (model at /root/models/qwen3vl-openvino-fp16, outside repo)
taskset -c 0-15 .venv-openvino/bin/python scripts/investigate_openvino_runtime.py \
  --mode bench --tag fp16 --threads 16 --model-dir /root/models/qwen3vl-openvino-fp16

# aggregate tables
.venv-openvino/bin/python scripts/aggregate_root_cause.py
.venv-openvino/bin/python scripts/analyze_onednn_verbose.py results/openvino_isa/onednn_profile_exec_partial.stdout
```

## Raw evidence index

| File | Content |
|---|---|
| `environment/root_cause_host_b.md`, `environment/host_b_system.json` | HOST B baseline + host-divergence scoping |
| `results/openvino_root_cause/host_b_model_sha256.txt` | model artifact hashes (match HOST A) |
| `results/openvino_root_cause/backend_linkage.txt` | plugin/oneDNN/TBB linkage + hashes |
| `results/openvino_root_cause/investigate_baseline_p*.json`, `profiling_P*_*.csv`, `runtime_model_*.xml` | baseline, per-node profiling, runtime graphs |
| `results/openvino_isa/isa_matrix_hostB.csv（注：未加 _hostB 后缀的 isa_matrix.csv 现为 HOST A 数据）` (+ `investigate_*_p*.json`, `*_p*.stdout`) | ISA/DQ matrix raw runs |
| `results/openvino_isa/jit_dump/` (local), `jit_vnni_hits*.txt/json`, `jit_instruction_summary*.txt` | JIT binaries + instruction census |
| `results/openvino_isa/onednn_verbose_findings.md`, `onednn_*.log/stdout` | verbose capabilities + partial dispatch lines |
| `results/openvino_isa/phase7_8_perf_unavailable.md` | perf unavailability record |
| `results/openvino_isa/memory_traffic_accounting.md` | byte/GB/s arithmetic |
| `results/openvino_root_cause/fp16_control_sha256.txt`, `investigate_fp16_p*.json`, `results/openvino_isa/profiling_P1_fp16_p*.csv` | FP16 control |
| `results/openvino_root_cause/root_cause_factor_table.md`, `vnni_contribution.json` | machine-generated summaries |
| `environment/root_cause_checkpoints.md` | per-phase verification record |

## Social-media-safe conclusions

### Confirmed facts (safe to state verbatim)

- OpenVINO 2026.4 on this AMD AVX-512 CPU runs the Qwen3-VL INT8
  weight-compressed conditioning workload with **INT8 weights resident in
  memory** (7.57 GB, no per-encode dequant), at 0.19–0.69 s per encode on
  Ryzen AI Max+ 395 (P1/P3), 22–69× faster than the ComfyUI CPU product path
  and 2.7–3.3× faster than ComfyUI's current GPU conditioning route.
- In the OpenVINO runtime, the 253 linear layers execute **JIT kernels that
  contain the AVX-512 VNNI instruction `vpdpbusd`** (verified by
  disassembly), with **activations dynamically quantized in-flight**
  (default group size 32) — the "INT8 model" behaves as weight-only INT8
  storage + dynamic activation quantization, and it is *not* a W8A8
  statically-quantized model.
- Disabling dynamic quantization makes the workload 1.37–1.47× slower and
  removes every VNNI instruction from the generated kernels (causal,
  3-process repeated).
- An FP16-weights build of the same model is 1.21–1.47× slower and uses
  1.999× the weight bytes (RSS 1.94×).
- Raising the dynamic-quantization group size to 128 gives a further
  1.16–1.26× speedup at cosine 0.998 vs the shipped-default numerics
  (same band as the repo's other accepted int8 variants; DQ=64 is the
  closer-numerics option at ~1.2×) — a speed/accuracy tuning trade-off
  found by this investigation, confirmed in the same direction by the
  HOST-A (Zen 5) measurements in the main report.

### Strong evidence (needs the qualifier)

- "The same VNNI integer path runs on the Ryzen AI Max+ 395 original
  benchmark host" — same OpenVINO build, byte-identical model, same ISA
  capability set, and a numerically identical replica (cosine 1.0000001);
  stated as strong evidence, not as a direct measurement on that host.
- "VNNI contributes ≈1.4×" — this is the *bundled* effect of the
  u8-weight+VNNI path vs the bf16-decompressed path on the investigation
  host; a VNNI-only number was not isolatable on this OpenVINO build.

### Do not claim

- "VNNI was proven with perf counters" — no perf/PMU access on the
  investigation host.
- "VNNI alone explains 69×" — false; the 22–69× is dominated by eliminating
  ComfyUI's per-encode full dequant + fp32 GEMM; the measured DQ-VNNI factor
  is ~1.4× and INT8 bytes ~1.2–1.5× on top of an efficient runtime.
- "The CPU is faster than the GPU" — only ComfyUI's current fp32-compute
  conditioning path on the Radeon 8060S was measured; not a general GPU claim.
- "OpenVINO runs W8A8" — it does not; activations are bf16/f32 inputs
  quantized in-flight.
- Any AVX2-vs-AVX512 speedup figure for this model — it cannot run without
  AVX-512 on this build.

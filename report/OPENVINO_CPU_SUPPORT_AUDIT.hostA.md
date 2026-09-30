# OpenVINO 2026.4.0 CPU Plugin — Full Support Audit on HOST A (AMD Ryzen AI Max+ PRO 395 / Zen 5)

Date: 2026-09-30 · Wheel: `openvino==2026.4.0-22959-99c81491cc3-releases/2026/4`
(byte-identical string to both hosts) · Backend: statically vendored **oneDNN v3.13.0**
+ oneTBB (`results/openvino_root_cause/backend_linkage.txt`).

This is the HOST A counterpart of the HOST B audit
([OPENVINO_CPU_SUPPORT_AUDIT.md](OPENVINO_CPU_SUPPORT_AUDIT.md), PR #3) — same script
(`scripts/audit_openvino_cpu_plugin.py --out-dir results/openvino_cpu_support_hostA`),
same protocol, run on the machine that carried the original benchmark, the root-cause
investigation and the end-to-end round. It closes the gap the HOST B report noted:
"ISA-ceiling behavior … and the silent-ignore behavior … were not re-measured on HOST A."

CPU: `AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S`, kernel 6.17.0-1032-oem.
`/proc/cpuinfo` flags relevant here: **avx512_bf16 ✓, avx512_vnni ✓, avx_vnni ✓,
avx512_fp16 ✗ (absent), amx ✗ (absent)**.

## TL;DR support matrix (all measured on HOST A)

| Area | Status | Key evidence |
|---|---|---|
| Device discovery (CPU) | ✅ | `FULL_DEVICE_NAME = AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S`, exec on CPU asserted |
| FP32 compute | ✅ strict-correct | `gemm_mlas_f32`, `jit_avx512_f32`, `brgconv_avx512_f32` (f32 hint strict: op-smoke ≈9.5e-6, tiny-FC 1.9e-5) |
| BF16 auto-conversion (default) | ✅ | f32 matmul/conv/sdpa default to `brgemm_avx512_bf16` / `brgconv_avx512_bf16`; rel-err ≈0.32% @ \|ref\|≈43 — bf16 rounding, same magnitude as HOST B |
| BF16 via hint | ✅ | identical to default |
| FP16 via hint | ⚠ accepted, **no FP16 kernel** | legal hint; result **bit-identical to the bf16 path** (max_abs_err 0.13666439056396484 — the exact HOST B value); `OPTIMIZATION_CAPABILITIES = ['BF16','WINOGRAD','FP32','INT8','BIN','EXPORT_IMPORT']` has no FP16 entry; consistent with no `avx512_fp16` cpuinfo flag on this Zen 5 SKU |
| i8 via `INFERENCE_PRECISION_HINT` | ❌ rejected config | `Wrong value i8 … Supported values: bf16, f16, f32, undefined` (same exception as HOST B) |
| INT8 weight-compressed + dynamic quant | ✅ | real workload capstone: **253/253 FC = `brgemm_avx512_bf16` rtPrecision=u8**; DQ=32 effective |
| f64 graph | ⚠ **silent downgrade** | compiles & runs, output dtype f64, math in bf16 (`brgemm_avx512_bf16` exec types; rel-err 0.283% ≈ bf16 rounding) — same build quirk as HOST B |
| INT8 in `OPTIMIZATION_CAPABILITIES` | ✅ declared | same capability list as HOST B |
| Threading | ✅ | explicit 1/4/16 honored; device default (LATENCY, tiny model) = **16 threads** (= 16 physical cores); `TBB_PARTITIONER=STATIC`, `ENABLE_CPU_PINNING=True`, compiled `ENABLE_HYPER_THREADING=False` |
| Streams | ✅ | `RANGE_FOR_STREAMS (1,32)` — HOST B reported (1,128); LATENCY ⇒ NUM_STREAMS=1 |
| KV-cache quant knobs | ✅ exposed | u8 key/value cache precisions, group sizes 0 (device properties census) |
| i8 activations as graph input | ✅ | i8-in matmul compiles, numerically correct; exec types incl. `jit_avx512_i8` + `brgemm_avx512_bf16` (explicit Convert; not the DQ path) — same as HOST B |

**Verdict: the OpenVINO CPU plugin fully supports this CPU's available acceleration
(AVX-512 + AVX512_VNNI + AVX512_BF16) and engages it by default on the real workload;
the two features this silicon lacks (AVX512_FP16, AMX) are absent from capabilities and
degrade exactly as on HOST B (f16→bf16 aliasing, AMX ceilings→BF16 level).**

## ISA ceiling matrix — row-by-row agreement with HOST B

`ONEDNN_MAX_CPU_ISA` × {default, f32-hint}, fresh subprocess per value
(`results/openvino_cpu_support_hostA/isa_ceiling_matrix.csv`; HOST B diff done live):

| Ceiling | HOST A default | HOST A f32-hint | HOST B agreement |
|---|---|---|---|
| SSE41 / AVX / AVX2 / AVX2_VNNI | ❌ primitive-descriptor failure | ✅ runs | identical (both hosts) |
| AVX512_CORE / AVX512_CORE_VNNI | ✅ `jit_gemm_bf16*` fallback | ✅ | identical incl. fallback kernel |
| AVX512_CORE_BF16 / _FP16 / _AMX / _AMX_INT8 / _AMX_BF16 | ✅ `brgemm_avx512_bf16` | ✅ | identical |
| BEST / ALL | ✅ == default | ✅ | identical |
| **NOT_A_REAL_ISA** (invalid) | ✅ == default — **silently ignored** | ✅ | identical |
| (unset) | ✅ `brgemm_avx512_bf16` | ✅ | identical |

oneDNN identifies this CPU (verbose header, non-FC matmul) as **"Intel AVX-512 with
Intel DL Boost and bfloat16 support"** — the same isa-family label HOST B gets
(`AVX512_CORE_BF16`); the label is an ISA-family string, not a vendor claim.

Every conclusion the HOST B audit drew about ISA-ceiling semantics transfers to HOST A
by measurement, not assumption: AVX-512 is a practical floor for the default auto-bf16
path; AVX2-class ceilings need an explicit f32 hint; invalid ceiling values fail
silently; AMX ceilings degrade gracefully on non-AMX silicon.

## Precision semantics (tiny FC, vs f64 numpy ref)

| INFERENCE_PRECISION_HINT | compile | max_abs_err (\|ref\|≈42.9) | kernel |
|---|---|---|---|
| f32 | ✅ | **1.9e-5** (strict) | `gemm_mlas_f32` |
| bf16 | ✅ | 0.13666439056396484 (0.32%) | `brgemm_avx512_bf16` |
| f16 | ✅ | 0.13666439056396484 — **bit-identical to bf16** | `brgemm_avx512_bf16` |
| i8 | ❌ | — | same rejection string as HOST B |
| (default) | ✅ | 0.13666… (bf16 path) | `brgemm_avx512_bf16` |

Tiny-FC latency inversion reproduces here too
(`tiny_fc_latency_f32_vs_bf16.txt`): f32 0.0259 ms vs bf16 0.0289 ms median at
1×256×256 — activation-reorder overhead beats the bf16 math advantage at toy sizes;
bf16/int8 pays off at real model sizes (root-cause report).

## Operator smoke (default vs f32-hint, numpy-referenced)

Identical kernel families and pass/fail pattern to HOST B on every op:
matmul → `brgemm_avx512_bf16` / `gemm_mlas_f32`; conv2d 3×3 → `brgconv_avx512_bf16` /
`brgconv_avx512_f32`; sdpa-lite → 2× `brgemm_avx512_bf16` + `jit_avx512_bf16` /
2× `gemm_mlas_f32` + `jit_avx512_f32`; elementwise/rmsnorm/gather stay strict-f32
`jit_avx512_f32` / `ref_i64` in both modes. Full table:
`results/openvino_cpu_support_hostA/op_smoke_results.csv`.

## Real-workload capstone (Qwen3-VL INT8, fresh run this audit)

`.venv-openvino/bin/python scripts/investigate_openvino_runtime.py --tag
audit_confirm_hosta --threads 16 --warm-iters 1 --measure-iters 3 --prompts P1
--save-hidden` (NO taskset — see topology gotcha):

- effective: `INFERENCE_NUM_THREADS=16`, `NUM_STREAMS=1`, `DQ=32`,
  `execution_devices=['CPU']`; compile 1.17 s (warm cache; profiling_audit_confirm_hosta_summary.json compile_s=1.1669)
- **P1 warm p50 = 0.1837 s** (round-1 on this host: 0.1854 s; HOST B: 0.1876 s)
- 253/253 weight-compressed FC nodes dispatch `brgemm_avx512_bf16` rtPrecision=u8
  (the VNNI int8 path of the root-cause report) + 73 `jit_uni_bf16` reorders
- hidden-state checks (`results/openvino_root_cause/audit_confirm_hosta.stdout` — a
  curated summary note whose every number is independently checkable against the
  committed summary/runtime JSONs and npys):
  **cos(HOST A today, HOST B audit artifact) = 1.0000000** and **bit-identical to
  HOST A's round-1 artifact** (`hidden_P1_dq_default_r1.npy`) — weeks of intervening
  experiments, hundreds of runs, same wheel → deterministic reproduction and
  cross-host numerical agreement both hold.

### Topology gotcha (portability note)

Copying HOST B's `taskset -c 0-15` verbatim to HOST A is wrong: on this host cpu0-15
are **8 physical cores × 2 SMT siblings** (`thread_siblings_list 0-1, 2-3, …`), so the
mask clamps `INFERENCE_NUM_THREADS` to 8 (measured: P1 p50 0.209 s). HOST B's 0-15
were 16 physical cores. Pin-by-core-count, not by CPU index, when moving between hosts.

## HOST A vs HOST B — differences found

| Item | HOST A (Zen 5) | HOST B (Zen 4) |
|---|---|---|
| Default threads (LATENCY, tiny) | 16 | 32 |
| `RANGE_FOR_STREAMS` | (1, 32) | (1, 128) |
| FP16 capability/kernel | absent (`avx512_fp16` not in cpuinfo; f16→bf16 alias) | absent (same aliasing) |
| f16-hint max_abs_err | 0.13666439056396484 | 0.13666439056396484 (identical) |
| Real P1 p50 (16 thr) | **0.1837 s** | 0.1876 s |
| AVX2-boundary / silent-ignore / AMX-degrade / f64-downgrade | all reproduce | all reproduce |

Everything else in the support matrix matches row-for-row.

## Reproduce

```bash
.venv-openvino/bin/python scripts/audit_openvino_cpu_plugin.py \
  --out-dir results/openvino_cpu_support_hostA          # ~4 min
.venv-openvino/bin/python scripts/investigate_openvino_runtime.py \
  --tag audit_confirm_hosta --threads 16 --warm-iters 1 --measure-iters 3 \
  --prompts P1 --save-hidden
```

## Verification

Independent read-only subagent verification (2026-09-30): **HOSTA-AUDIT: PASS** —
9/9 checks, including recomputation of both hidden-state cosines from the npys
(cross-host arrays bit-identical, max abs diff 0.0), a programmatic 15/15-row ISA
matrix agreement check against the HOST B CSV, the 253/253 u8-FC node recount, and
confirmation that HOST B artifacts are untouched.

## Evidence index

- `results/openvino_cpu_support_hostA/` — `cpu_support_audit.json` (all 5 sections),
  `isa_ceiling_matrix.csv`, `op_smoke_results.csv`, `device_properties.txt`,
  `tiny_fc_latency_f32_vs_bf16.txt`
- `results/openvino_root_cause/audit_confirm_hosta.stdout`,
  `hidden_P1_audit_confirm_hosta.npy`, `profiling_audit_confirm_hosta_summary.json`,
  `runtime_nodes_audit_confirm_hosta.json`
- Depth evidence per ISA/DQ/VNNI on this host: `report/OPENVINO_ZEN5_ROOT_CAUSE.md`
- HOST B counterpart: `report/OPENVINO_CPU_SUPPORT_AUDIT.md` +
  `results/openvino_cpu_support/`

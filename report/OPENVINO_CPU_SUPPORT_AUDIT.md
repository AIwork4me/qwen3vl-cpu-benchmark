# OpenVINO 2026.4.0 CPU Plugin — Full Support Audit on HOST B (2× AMD EPYC 9334 / Zen 4)

Date: 2026-09-29 · Wheel: `openvino==2026.4.0-22959-99c81491cc3-releases/2026/4`
(PIPy, identical string to both investigation hosts) · Backend: statically
vendored **oneDNN v3.13.0** + oneTBB (`results/openvino_root_cause/backend_linkage.txt`).

Scope: the CPU plugin only (a GPU device is visible; out of scope). All claims
below are measured on HOST B by `scripts/audit_openvino_cpu_plugin.py` plus
one real-workload confirmation run; raw data in `results/openvino_cpu_support/`
and `results/openvino_root_cause/audit_*`.

## TL;DR support matrix

| Area | Status | Key evidence |
|---|---|---|
| Device discovery (CPU) | ✅ | `FULL_DEVICE_NAME = AMD EPYC 9334 32-Core Processor`, exec on CPU asserted |
| FP32 compute | ✅ strict-correct | `gemm_mlas_f32`, `jit_avx512_f32`, `brgconv_avx512_f32` (op smoke, f32 hint) |
| BF16 auto-conversion (default) | ✅ | f32 graphs' matmul/conv/attention default to `brgemm_avx512_bf16` / `brgconv_avx512_bf16`; rel-err ≈0.3% @ \|ref\|≈43 — expected bf16 rounding |
| BF16 via hint | ✅ | identical to default |
| FP16 via hint | ⚠ accepted, **no FP16 kernel** | legal hint value; result bit-identical to bf16 path (max_abs_err 0.13666… = bf16); `OPTIMIZATION_CAPABILITIES` has no FP16 entry (Zen 4: no AVX512_FP16) |
| i8 via `INFERENCE_PRECISION_HINT` | ❌ rejected config | `Wrong value i8 … Supported values: bf16, f16, f32, undefined` — int8 compute enters via weight compression + dynamic quantization instead (proven in root-cause report) |
| INT8 weight-compressed + dynamic quant | ✅ | real workload: 253/253 FC = `brgemm_avx512_bf16` w/ VNNI kernels; DQ group 0/32/64/128 all effective |
| f64 graph | ⚠ **silent downgrade** | f64 model compiles & runs, output dtype f64, but math executes in bf16 (`brgemm_avx512_bf16` in exec types; rel-err 0.28% ≈ bf16 rounding) — f64 compute is NOT honored |
| INT8 in `OPTIMIZATION_CAPABILITIES` | ✅ declared | `['BF16','WINOGRAD','FP32','INT8','BIN','EXPORT_IMPORT']` (device property) |
| Threading | ✅ | explicit 1/4/16 honored; device default (LATENCY, tiny model) = **32 threads**; `TBB_PARTITIONER=STATIC`, `ENABLE_CPU_PINNING=True`; compiled default sets `ENABLE_HYPER_THREADING=False` (device-level True) |
| Streams | ✅ | `RANGE_FOR_STREAMS (1,128)`, LATENCY ⇒ NUM_STREAMS=1 |
| KV-cache quant knobs | ✅ exposed | `KV_CACHE/KEY_CACHE/VALUE_CACHE_PRECISION = u8`, group sizes 0 |
| i8 activations as graph input | ✅ | i8-in matmul model compiles, numerically correct; exec types incl. `jit_avx512_i8` + `brgemm_avx512_bf16` (explicit Convert present; not the DQ path) |

## ISA support & `ONEDNN_MAX_CPU_ISA` behavior (14 values + unset, fresh process each)

Probe: tiny FC-style matmul (f32 in/out), two modes per ceiling — default
(bf16 auto) and f32-hint; exec types from OV profiling
(`results/openvino_cpu_support/isa_ceiling_matrix.csv`):

| Ceiling | default mode | f32-hint mode |
|---|---|---|
| SSE41 / AVX / AVX2 / AVX2_VNNI | ❌ "could not create a primitive descriptor" (auto-bf16 path has no fallback this low) | ✅ runs |
| AVX512_CORE / AVX512_CORE_VNNI | ✅ runs, exec types incl. `jit_gemm_bf16*` (non-brgemm fallback engaged; see note) | ✅ runs |
| AVX512_CORE_BF16 / _FP16 / _AMX / _AMX_INT8 / _AMX_BF16 | ✅ == default (`brgemm_avx512_bf16`) | ✅ |
| BEST / ALL | ✅ == default | ✅ |
| **NOT_A_REAL_ISA** (invalid) | ✅ == default — **invalid values are silently ignored** | ✅ |
| (unset) | ✅ `brgemm_avx512_bf16` | ✅ |

Findings:
1. oneDNN identifies this CPU as **"Intel AVX-512 with Intel DL Boost and
   bfloat16 support"** (= isa `AVX512_CORE_BF16`; header via verbose on an
   f32 non-FC matmul).
2. **AVX-512 (≥AVX512_CORE) is a practical requirement for the default
   (auto-bf16) math path; AVX2-class ceilings need `INFERENCE_PRECISION_HINT=f32`
   to run at all.** For the real weight-compressed model even AVX512_CORE is
   not enough (root-cause: `FullyConnectedCompressed` needs ≥VNNI for its int8
   kernels, attention needs BF16).
3. **Invalid `ONEDNN_MAX_CPU_ISA` values are silently ignored** (no warning
   line, dispatch identical to unset). Anyone using ISA ceilings for
   ablations must verify the ceiling actually changed dispatch (as this repo
   does via exec types / JIT dumps) — a wrong value looks exactly like "no
   effect".
4. AMX ceilings on a non-AMX CPU degrade gracefully to the BF16 level.
5. Under AVX512_CORE(_VNNI) ceilings the tiny uncompressed FC falls back to a
   `jit_gemm_bf16*`-labeled kernel and still runs — the compressed
   (`FullyConnectedCompressed`) path has no such fallback (root-cause
   evidence). The execType label vs actual ISA requirement of that fallback
   kernel was not dissected further (out of audit scope).

## Precision semantics (tiny FC, `--compare` vs f64 numpy ref)

| INFERENCE_PRECISION_HINT | compile | max_abs_err (\|ref\|≈42.9) | kernel (exec types) |
|---|---|---|---|
| f32 | ✅ | **1.9e-5** (strict) | `gemm_mlas_f32` |
| bf16 | ✅ | 0.137 (=0.32% scale) | `brgemm_avx512_bf16` |
| f16 | ✅ | 0.13666439056396484 — **bit-identical to bf16** | `brgemm_avx512_bf16` |
| i8 | ❌ | — | exception: legal values are bf16/f16/f32/undefined |
| (default) | ✅ | 0.13666… (= bf16 path) | `brgemm_avx512_bf16` |

⇒ Default policy on this CPU: **f32 inputs are auto-converted to bf16 for
matmul/conv/attention math**; f16 hint aliases onto the bf16 path; exact f32
compute requires an explicit `f32` hint. Note the tiny-FC latency inversion
(`tiny_fc_latency_f32_vs_bf16.txt`: f32 0.036 ms vs bf16 0.043 ms median) —
at 1 token × 256×256 the bf16 path pays activation-reorder overhead that
exceeds its math advantage; bf16 pays off at real model sizes (root-cause
report).

## Operator smoke (default vs f32-hint, numpy-referenced)

| op | default exec | f32-hint exec | numeric |
|---|---|---|---|
| matmul (FC-style) | `brgemm_avx512_bf16` | `gemm_mlas_f32` | both ✅ (bf16 within tolerance; f32 strict) |
| conv2d 3×3 | `brgconv_avx512_bf16` | `brgconv_avx512_f32` | both ✅ |
| sdpa-lite (qk·softmax·av) | `brgemm_avx512_bf16` ×2 + `jit_avx512_bf16` | `gemm_mlas_f32` ×2 + `jit_avx512_f32` | both ✅ |
| add·mul elementwise | `jit_avx512_f32` (no bf16 conversion) | same | strict ✅ |
| rmsnorm chain | `jit_avx512_f32` | same | strict ✅ |
| gather | `ref_i64` + `jit_avx512_f32` | same | strict ✅ |

Note the deliberate asymmetry: only throughput math (matmul/conv) is
auto-bf16'd; elementwise/gather stay f32 regardless of hint.

## Real-workload confirmation (this audit, fresh runs)

- default (16 threads pinned, LATENCY): P1 p50 **0.1876 s**, all FC nodes
  `brgemm_avx512_bf16`, hidden-state cosine vs committed HOST-A artifact
  **1.0000001** (`results/openvino_root_cause/audit_confirm.stdout`,
  `hidden_P1_audit_confirm.npy`).
- f16 hint on the real model: not runnable **through the repo script**
  (`investigate_openvino_runtime.py` maps only f32/bf16/i8 → `KeyError f16`);
  the plugin itself accepts f16 (tiny-model row above). Recorded as a script
  limitation, not a plugin limitation.

## Known plugin/build quirks recorded during the audit

1. `ONEDNN_VERBOSE=1` is unusable on bf16 inner-product paths in this build:
   real LM → compile-time crash (Reorder u8→i32 constant fold); tiny FC →
   execution-time crash ("could not execute a primitive"); f32 paths emit
   nothing at all for FC models (verbose silently inert); only non-FC
   matmul/reorder emit lines. (Crash behaviors documented in
   `results/openvino_isa/onednn_verbose_findings.md` + the cited historical
   logs; re-confirmed here only for the "non-FC matmul emits verbose lines"
   half, which this audit uses to capture the isa header.)
2. Invalid ISA ceiling values are silently ignored (see finding 3 above).
3. `DEVICE_ARCHITECTURE=intel64` / `DEVICE_TYPE=INTEGRATED` are generic
   labels on this AMD host — cosmetic, but worth knowing when scripting
   against device properties.
4. **f64 graphs are silently downgraded to bf16 math** (output still f64) —
   anyone expecting double-precision results from OV CPU will get bf16-grade
   accuracy without any warning.

## Reproduce

```bash
.venv-openvino/bin/python scripts/audit_openvino_cpu_plugin.py          # ~4 min
taskset -c 0-15 .venv-openvino/bin/python scripts/investigate_openvino_runtime.py \
  --tag audit_confirm --threads 16 --warm-iters 1 --measure-iters 3 --prompts P1 --save-hidden
```

## Evidence index

- `results/openvino_cpu_support/cpu_support_audit.json` — all five sections
- `results/openvino_cpu_support/isa_ceiling_matrix.csv`,
  `op_smoke_results.csv`, `device_properties.txt`
- `results/openvino_root_cause/audit_confirm.stdout`,
  `hidden_P1_audit_confirm.npy`, `audit_f16hint.stdout` (script-key error)
- Backend linkage / oneDNN version: `results/openvino_root_cause/backend_linkage.txt`
- Real-model ISA/DQ/VNNI depth evidence: `report/OPENVINO_ZEN5_ROOT_CAUSE.md`
  (HOST A) and `.hostB.md` (HOST B)

Host scoping: all direct measurements above are HOST B (Zen 4). HOST A
(Zen 5 / Ryzen AI Max+ 395) shares the wheel, the ISA family, and the
dispatch-relevant CPU capabilities; its own 16-phase run is in the main
report. ISA-ceiling behavior at the AVX2 boundary and the silent-ignore
behavior are properties of the build and are expected to transfer, but were
not re-measured on HOST A.

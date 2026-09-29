# Phase 6 — JIT binary instruction evidence (Level C)

Capture: `JITDUMPDIR=. ONEDNN_JIT_DUMP=1` around a single-P1 run of the bridge workload
(run.log per config dir records the run's properties/digest; the exact commands are in the Reproduce block). Dumps: one `.bin` per JIT-compiled
kernel, filename = kernel class (`dnnl_dump_cpu_<kernel_class>.<id>.bin`).
Analysis: `scripts/analyze_jit_vnni.py` (objdump -D -b binary; VNNI = vpdpbusd/vpdpbusds/
vpdpwssd/vpdpwssds, BF16 = vdpbf16ps; per-hit +/-10 instruction context in
`jit_vnni_hits.txt`; SHA256 manifest in `jit_dump_manifest.json`).
Binaries are kept locally only (gitignored); manifests + summaries + hit excerpts are committed.

## Verdict-grade result (Q1)

**The DEFAULT (product) path's brgemm JIT kernels contain AVX-512 VNNI machine instructions.**

```
dnnl_dump_cpu_jit_brgemm_kernel_t.100.bin @ 0x557: vpdpbusd %zmm0,%zmm4,%zmm30
   0x53c: vmovups (%r10),%zmm4          <- u8 activation / s8 weight block load
   0x542: vmovups 0x40(%r10),%zmm3
   0x549: vpbroadcastd (%r11),%zmm0     <- dynamic-quantization GROUP SCALE broadcast (dword)
   0x54f: prefetcht0 0x400(%r10)
```

All VNNI hits across the default dump are pure `vpdpbusd` (no vpdpbusds/vpdpwssd/vpdpwssds variants;
2,496 instruction instances, echoed with context in jit_vnni_hits.txt) (u8 x s8 -> int32
accumulate, zmm 512-bit) — the classic int8 dot product, with per-group scale broadcasts:
this is the **dynamic activation quantization (groups of 32) feeding VNNI MACs**, in the
kernels that serve the 253 FullyConnectedCompressed nodes (Phase 2: brgemm = 91-92% of time).

## Instruction census per config (brgemm kernels only carry dot-product instructions)

| config | bins | brgemm kernels w/ VNNI | total vpdpbusd | brgemm kernels w/ vdpbf16ps | total vdpbf16ps | FMA |
|---|---:|---:|---:|---:|---:|---:|
| **default (dq=32, bf16 hint)** | 137 | **12** (208 instrs each) | 2,496 | 6 | 988 | 892 |
| dq0 (dynamic quantization OFF) | 141 | **0** | 0 | 18 (28-448 instrs; 12x144) | 2,716 | 680 |
| dq128 (coarse groups) | 137 | 12 (**832** instrs each — 4x unroll) | 9,984 | 6 | 988 | 892 |
| AVX512_CORE ceiling + f32 hint | 126 | 12 (**304** each) | 3,648 | 0 | 0 | 1,156 |
| AVX512_CORE_VNNI ceiling + f32 hint | 126 | 12 (**208** each) | 2,496 | 0 | 0 | 892 |
| AVX2_VNNI ceiling + f32 hint | 127 | 12 x 96 (256-bit `{vex} vpdpbusd %ymm`) | 1,152 (VNNI_256) | 0 | 0 | 885 |

## Mechanism conclusions

1. **Q1 = YES for the default path**: AVX-512 VNNI (vpdpbusd) is present in the JIT machine
   code of the kernels that carry 91–92% of the workload's time. Combined with Level E
   counterfactual (Phase 5: disabling dynamic quantization removes ALL VNNI kernels and costs
   +23–43%), the default path's use of VNNI is established at Level B+C+E. Direct executed-
   sample proof (perf) was not obtainable — see phase7_perf_findings.md (perf_event_paranoid=4
   + no functional perf for kernel 6.17.0-1032-oem).
2. **The compute engine is selected by DYNAMIC_QUANTIZATION_GROUP_SIZE, not by the OV
   primitive name**: same "brgemm_avx512_bf16" primitive name, but g=32 -> int8 VNNI MACs;
   g=0 -> pure bf16 dot (vdpbf16ps) on dequantized weights. "brgemm_avx512_bf16" is an ISA
   label of the kernel family, NOT a statement that MACs run in bf16.
3. **Weights never leave u8** in any VNNI path: vpdpbusd consumes the u8/s8 data directly
   (u8 activations x s8 weights), with the model's stored per-channel scales applied via
   broadcast. The BF16 files in default (6 kernels, 988 vdpbf16ps) serve the remaining
   GEMM shapes (dq0 has 18 bf16 kernels = all shapes when dynamic quantization is off).
4. **The Phase-4 "8% VNNI-ceiling gap" mechanism found**: AVX512_CORE vs AVX512_CORE_VNNI
   ceilings produce DIFFERENT JIT code for the same primitive (304 vs 208 vpdpbusd per
   kernel; 32/126 binaries differ by SHA256; different sizes 9056–20544 vs 5920–14208) —
   same instruction family, different codegen/unrolling. Notably the *non-VNNI* ceiling
   still emits (and evidently executes — no SIGILL, faster wall time) vpdpbusd: the
   oneDNN compressed-GEMM path does not strip VNNI at the avx512_core ceiling.
5. AVX2_VNNI ceiling + f32 hint: the int8 dot engine is RETAINED at 256-bit — 12 kernels
   with 96 `{vex} vpdpbusd %ymm` each (1,152 total). The AVX2VNNI_F32 slowdown (1.38-1.55x)
   is therefore attributable to vector width/register-file, not to a lost dot engine.
   (Correction: an earlier version of this file reported 0 VNNI for this config due to a
   parser bug — objdump prints 256-bit VEX forms as "{vex} vpdpbusd"; fixed in
   analyze_jit_vnni.py and regenerated.)

## Reproduce

```bash
cd results/openvino_isa/jit_dump/<cfg>
JITDUMPDIR=. ONEDNN_JIT_DUMP=1 \
  <repo>/.venv-openvino/bin/python <repo>/scripts/investigate_openvino_runtime.py \
  --tag jitdump_<cfg> --prompts P1 --warm-iters 1 --measure-iters 2 [<extra cfg args>]
<repo>/.venv-openvino/bin/python <repo>/scripts/analyze_jit_vnni.py . --out <cfg>_analysis
```

Per-config extra args: default = none; dq0 = `--dyn-quant-group-size 0`;
dq128 = `--dyn-quant-group-size 128`; avx512f32 = `ONEDNN_MAX_CPU_ISA=AVX512_CORE
--inference-precision f32`; vnni512f32 = `ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI
--inference-precision f32`; avx2vnnif32 = `ONEDNN_MAX_CPU_ISA=AVX2_VNNI --inference-precision f32`.

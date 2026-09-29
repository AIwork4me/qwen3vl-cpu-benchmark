# Qwen3-VL Text Encoder on Ryzen AI Max+ 395: ComfyUI INT8 ConvRot vs OpenVINO INT8

A/B benchmark on **AMD Ryzen AI Max+ PRO 395** (Strix Halo, 16C/32T, AVX-512 + AVX512_VNNI + AVX512_BF16, 94 GiB UMA), measured 2026-09-29. All numbers are real measurements (20 Hz psutil sampling + `perf_counter_ns`), fully reproducible from this repo.

**TL;DR**

1. **ComfyUI's `qwen3vl_8b_int8_convrot` on CPU does NOT run INT8 GEMM.** It is **INT8 storage → full dequant every forward (252 layers, 13.9 GB/encode) → FP32 compute** ("A2"). Proven by zero `torch._int_mm` calls, byte-exact dequant accounting, profiler, and the source chain.
2. **True INT8 CPU GEMM works on this chip** (`torch._int_mm` / oneDNN-VNNI). Forcing ComfyUI's own int8 kernels (A1 variant) gives **1.77×** speedup — the capability exists, the conditioning path just disables it.
3. **OpenVINO INT8 weight-compressed wins by an order of magnitude**: with a zero-recompute `add_outputs` bridge that exactly matches Qwen-Image 2.1 conditioning semantics (cosine ≥ 0.997 vs BF16 reference), it is **22–69× faster** than ComfyUI's product path and loads ~10× faster.
4. **Putting the BF16 encoder on the Radeon 8060S GPU does NOT help** — measured: 0.60/1.23/1.83 s (P1/P2/P3). ComfyUI's conditioning path forces **fp32 GEMM on GPU too** (all 252 linears measured `fp32×fp32`), so the OpenVINO INT8 CPU bridge is still **2.7–3.3× faster** than the GPU route, while the GPU route parks 16.6 GiB in GTT and keeps the GPU 85% busy — competing with the DiT.
5. **Recommendation for `CPU → Qwen3-VL conditioning → Radeon GPU → Qwen-Image 2.1 DiT`:** run the encoder with **OpenVINO INT8 + the `add_outputs` bridge** (scripts included). Cost: ~2× peak RAM (15 vs 7.6 GiB, still only 16% of 94 GiB), zero GPU occupancy.

## Decision table

| You are… | Use | Why |
|---|---|---|
| Building a Qwen-Image 2.1 pipeline on Ryzen AI Max+ 395 / Strix Halo | **OpenVINO INT8 + bridge** (scripts/bench_openvino_bridge.py) | 0.19–0.69 s conditioning, 1.26 s load, semantics verified, GPU left free for the DiT |
| "Just put the encoder on the GPU" | Don't — measured 0.60–1.83 s (BF16→fp32 compute) and 85% GPU busy | OV INT8 CPU bridge is 2.7–3.3× faster and uses 0% GPU |
| Locked into the ComfyUI ecosystem | ComfyUI product path works, but 22–69× slower; consider upstream patch to enable int8 path (A1) | A1 measured: 1.77×, cos 0.9978, kernels already ship |
| RAM-constrained (UMA shared with GPU) | ComfyUI route uses less peak RAM | 7.6 vs 15.1 GiB |
| Wanting TRUE int8 on CPU in ComfyUI | Flip `comfy_force_cast_weights` + `use_quantized_matmul` | See A1 evidence below; needs upstream acceptance |

## Headline numbers (warm encode, best config = 16 threads, median of 3 fresh processes)

| Metric | ComfyUI INT8 ConvRot (product, A2) | ComfyUI forced-int8 (A1) | ComfyUI BF16 on Radeon GPU | OpenVINO INT8 (bridge) |
|---|---:|---:|---:|---:|
| Warm P1 (25 tok) | 12.98 s | 7.32 s | 0.601 s | **0.187 s** |
| Warm P2 (67 tok) | 13.73 s | 7.46 s | 1.228 s | **0.377 s** |
| Warm P3 (171 tok) | 15.43 s | 8.36 s | 1.834 s | **0.687 s** |
| Model load | 0.30 s (mmap assign; real page-in lands in first encode 12.11 s) | same | 4.48 s (disk→GTT) | **1.26 s** (read+compile) |
| First usable encode (cold) | 12.11 s | ~7.5 s | 0.60 s | **0.18 s** |
| Memory | 7.59 GiB CPU RSS | 7.35 GiB | 16.63 GiB GPU (GTT, resident) | 14.57 GiB CPU RSS |
| Compute unit busy | CPU ~1318% (16 thr) | CPU | **GPU 85%** (busy mean) | CPU ~1552–1572% warm-interval⁵ (GPU free) |
| Model disk | 9.351 GB | same weights | 17.534 GB | 8.810 GB |
| True INT8 GEMM | **NO** | **YES** | n/a (BF16 weights, fp32 GEMM) | **YES — dynamic-quantization int8 dot (AVX512_VNNI `vpdpbusd`)**² |
| Cosine vs BF16 reference | 0.9989–0.9993 | 0.9978–0.9986 | **1.000000** | 0.9972–0.9985 |

¹² superscripts resolved by the root-cause investigation (below). **History preserved:**
before that investigation this cell read `UNKNOWN¹` ("OpenVINO's internal oneDNN kernel
choice was not dumped"). After it (evidence Levels B+C+E, no perf sampling available):
the 253 weight-compressed FC nodes (91–92% of node time) dispatch to oneDNN brgemm JIT
kernels whose machine code contains `vpdpbusd` (AVX-512 VNNI, u8×s8→int32) MAC loops with
dynamic-quantization group scales — the model is INT8_ASYM weight-compressed **plus
runtime dynamic activation quantization (group 32)**, i.e. NOT static W8A8, but the MACs
do run int8 on VNNI. Setting `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` switches the same nodes
to pure BF16 dot (`vdpbf16ps`) and costs +23–43% latency. Full chain:
`report/OPENVINO_ZEN5_ROOT_CAUSE.md`.

⁵ Warm-encode interval process CPU (`proc_cpu_pct`, 20 Hz): 1552–1572% ≈ 15.5–15.7 of 16
cores (results/comparison/summary.csv openvino rows). The earlier "~1332%" figure was the
whole-run mean including model-load/idle phases (`openvino_default.json
meta.full_run.proc_cpu_mean_pct` = 1331.9) — both trace to raw data, different windows.

Thread sweep (both runtimes): **16 (default) is fastest**; 8 is clearly slower; 32 gives no gain.

## The INT8 truth (evidence chain)

**Product path = A2 `INT8_WEIGHT_STORAGE_FP_COMPUTE`:**

- Runtime counters (identical across 8 runs × 3 prompts): `torch._int_mm` = **0**; `QuantizedTensor.dequantize` = **252 layers per encode**; dequantized bytes = **13,891,534,848 B/encode**, which reconciles *exactly* to `(int8 linear payload 8,190,427,136 − embed 622,329,856 − lm_head 622,329,856) × 2 (bf16)`.
- torch profiler: A1 run shows `aten::_int_mm` ×252 and `comfy_kitchen::int8_linear` ×252; A2 runs show none (FP `aten::mm`).
- Source chain (ComfyUI 0.37.0): `comfy/sd1_clip.py:114` hard-codes `full_precision_mm=True` for text encoders → `comfy/model_patcher.py:1019` `set_model_compute_dtype()` sets `comfy_force_cast_weights=True` on every module → `comfy/ops.py:1448` `_use_quantized` requires `not comfy_force_cast_weights` → always False → `comfy/ops.py:437` dequantizes and runs FP `F.linear`.
- Output is **fp32** (`sd1_clip.py:279` forces fp32 forward); device evidence: `load_device/offload_device=cpu` in ComfyUI's own log, all weights and outputs on cpu (hard-asserted), comfy-kitchen backends `cuda=false, hip=false`.

**A1 forced variant** (NOT the product path): wraps encode in ComfyUI's own `comfy.ops.use_quantized_matmul` (the context manager generate() uses) plus flipping `comfy_force_cast_weights` on 575 modules → `aten::_int_mm` ×252/encode, dequant ×0, warm P1 12.98→7.32 s (**1.77×**), cos 0.9978. This shows the CPU int8 kernel path is fully functional — it is switched off for conditioning by design.

## OpenVINO conditioning bridge (how apples are compared to apples)

`OpenVINO/Qwen3-VL-8B-Instruct-int8-ov` exports a generation-oriented IR whose only output is `logits`. But the graph contains the final RMSNorm, so:

- `core.read_model()` → `model.add_outputs(<layer-35 residual node>)` exposes the **layer-36 hidden state *before* final RMSNorm** — exactly ComfyUI's semantics (`layer_idx=-1`, `layer_norm_hidden_state=False`). Zero recompute, no re-export.
- Tokenization replicates ComfyUI's T2I template (system turn + user turn, no think block) using the same tokenizer files → 39/81/185 tokens → trim from the 2nd `<|im_start|>` → **[1, 25/67/171, 4096]**, shape-identical to ComfyUI's cond.
- Validation: cosine vs a BF16 reference (same template, same path) = 0.9972–0.9985 across all prompts. A misaligned tokenization or wrong layer could not reach 0.997.

Standalone OpenVINO numbers with the **official chat template** (different semantics) are in `report/RESULTS.md §2` and marked NOT DIRECTLY COMPARABLE for conditioning use.

## Accuracy (vs BF16 reference, same template/path)

| Variant | P1 | P2 | P3 | shape_equal |
|---|---:|---:|---:|---|
| ComfyUI A2 (fp32 compute) | 0.99893 | 0.99921 | 0.99926 | ✓ |
| ComfyUI A1 (true int8 GEMM) | 0.99783 | 0.99845 | 0.99859 | ✓ |
| OpenVINO INT8 bridge | 0.99724 | 0.99806 | 0.99845 | ✓ |
| ComfyUI BF16 on Radeon GPU (fp32 compute) | 1.000000 | 1.000000 | 1.000000 | ✓ |

## BF16 on the Radeon 8060S GPU (Experiment C)

Same ComfyUI product path, weights on the HIP device (torch 2.12.0+rocm7.14.0, Radeon 8060S, UMA/GTT):

- Warm encode: **P1 0.601 s, P2 1.228 s, P3 1.834 s** (median of 3 fresh processes), GPU busy **85%** (max 100%), peak GPU allocation **16.63 GiB**.
- **Compute-dtype forensics**: all 252 linear layers measured running **fp32 × fp32 GEMM** on GPU (forward-pre-hook + `F.linear` wrapper). The conditioning path forces fp32 (`sd1_clip.py:279`) on every device — so "BF16 on GPU" is *BF16 storage → cast → fp32 compute*, never bf16 tensor-core math. That is why it loses to the int8 CPU bridge.
- Precision: cosine **1.000000** vs the CPU BF16 reference (RMSE 6–8e-5, GEMM summation-order noise only) — which also cross-validates that the CPU int8 path really computes in fp32.
- Takeaway: the GPU route parks 16.6 GiB in GTT, keeps the iGPU at 85% busy, and is still 2.7–3.3× slower than the OpenVINO INT8 CPU bridge. Conditioning belongs on the CPU; the GPU belongs to the DiT.

## Reproduce

```bash
# models via ModelScope (HF unreachable on this network)
modelscope download --model Comfy-Org/Qwen-Image-2.1 text_encoders/qwen3vl_8b_int8_convrot.safetensors --local_dir models/qwen3vl-comfy-int8
modelscope download --model Comfy-Org/Qwen-Image-2.1 text_encoders/qwen3vl_8b_bf16.safetensors --local_dir models/qwen-image-2.1
modelscope download --model OpenVINO/Qwen3-VL-8B-Instruct-int8-ov --local_dir models/qwen3vl-openvino-int8

# Experiment A: ComfyUI product path + A1-forced + thread sweep (needs .venv-comfy: torch 2.9.1+cpu, -r ComfyUI/requirements.txt)
.venv-comfy/bin/python scripts/bench_comfy_qwen3vl_cpu.py --tag run1 --warm-iters 5 --save-npy
.venv-comfy/bin/python scripts/bench_comfy_qwen3vl_cpu.py --tag a1_t16 --warm-iters 5 --force-quant-mm --profile

# Experiment B: OpenVINO standalone + conditioning bridge (needs .venv-openvino: openvino>=2026.1, tokenizers, jinja2, transformers)
.venv-openvino/bin/python scripts/bench_openvino_qwen3vl_cpu.py --tag default --warm-iters 5
.venv-openvino/bin/python scripts/bench_openvino_bridge.py --tag bridge --warm-iters 5

# Experiment C: BF16 on Radeon GPU (needs .venv-comfy-rocm: torch 2.12.0+rocm7.14.0, -r ComfyUI/requirements.txt)
.venv-comfy-rocm/bin/python scripts/bench_comfy_bf16_gpu.py --tag gpu_run1 --warm-iters 5 --save-npy

# accuracy + aggregation
.venv-comfy/bin/python scripts/bench_bf16_reference.py
python3 scripts/compare_embeddings.py && python3 scripts/aggregate_summary.py
python3 scripts/classify_comfy_path.py run1 run2 run3 t8 t32 a1_t16 a1_t8 a1_t32

# root-cause investigation (see report/OPENVINO_ZEN5_ROOT_CAUSE.md §Reproduce)
bash scripts/run_openvino_isa_matrix.sh && .venv-openvino/bin/python scripts/aggregate_root_cause.py
```

## End-to-End Qwen-Image 2.1 (follow-up round)

> Does moving Qwen3-VL conditioning to the Ryzen CPU improve the **complete** generation
> pipeline (prompt → encoder → DiT → VAE → PNG) on this APU?

Measured on the full pipeline (real DiT 14.23 GB + VAE, real images, 20 Hz resource
sampling; full report: **[report/QWEN_IMAGE_E2E_HYBRID.md](report/QWEN_IMAGE_E2E_HYBRID.md)**):

| | GPU-heavy (product path) | Hybrid (OV INT8 CPU encoder) |
|---|---:|---:|
| Warm E2E image (P3, 1024², 20 st) | 154.2 s | **151.3 s** |
| Cold time-to-first-image | 176.1 s | **168.8 s** |
| Text-encoder (warm, pos+neg) | 2.62 s | **0.99 s** |
| DiT latency | 149.1 s | 148.0 s (equal within noise) |
| GPU (GTT) residency | 51.9 GiB | **35.4 GiB (−16.5)** |
| Physical UMA consumed | 52.7 GiB | **43.5 GiB (−9.2)** |
| Same-seed image output | reference | CLIP similarity 0.993 |

Verdict: the pipeline is DiT-dominated, so the warm win is small (~2%) — the real value
is **16.5 GiB of GPU memory freed, 9 GiB less physical RAM, faster cold start, and a GPU
that stays fully available to the DiT during conditioning**. Two honest negatives:
sustained CPU-encoding during GPU-DiT is 2.05× slower (shared LPDDR5X bandwidth), and
pipelining hides the encoder but does not multiply throughput. Recommended on this
machine: Hybrid with DQ 32/64 (see report for the DQ trade-off and quality dataset).

## Root-cause investigation

Follow-up (same day, branch `investigate/openvino-zen5-vnni-root-cause`): a 16-phase,
independently-verified investigation of **what OpenVINO actually executes on the Zen 5 CPU**.
Full report: **[report/OPENVINO_ZEN5_ROOT_CAUSE.md](report/OPENVINO_ZEN5_ROOT_CAUSE.md)** — every phase
cross-checked by read-only subagents (`environment/root_cause_checkpoints.md`, all PASS).

**Two-host cross-validation**: the same protocol was run independently on a second machine
(HOST B = 2× EPYC 9334 / Zen 4, byte-identical model + OpenVINO wheel, hidden-state cosine
1.0000001 vs HOST A) — its report: [report/OPENVINO_ZEN5_ROOT_CAUSE.hostB.md](report/OPENVINO_ZEN5_ROOT_CAUSE.hostB.md).
Both hosts independently reached the same verdict:

- **AVX512_VNNI is executed** (runtime dispatch + JIT disassembly + knob counterfactual;
  perf-level sampling unavailable on both hosts). The 253 weight-compressed FC nodes run
  `vpdpbusd` (u8×s8→int32) MAC loops fed by dynamic activation quantization (group 32) —
  HOST B additionally identified the dedicated `brgemm_src_quantization_kernel_t` JIT
  kernels and the `vcvtneps2bf16` s32→bf16 epilogue.
- **But VNNI is not the headline**: same-stack counterfactuals measure VNNI-vs-BF16 engine
  **1.23–1.43×** (HOST A) / **1.37–1.47×** (HOST B); INT8-WC-vs-FP16 **1.41–1.67×** (A) /
  **1.21–1.47×** (B); AVX-512 width **1.38–1.55×** (A; a hard requirement on B's build for
  the compressed node). The 22–69× is dominated by never materializing a dequantized
  weight copy (8.5 GB vs 34.7 GB traffic per encode) inside a precompiled inference
  pipeline. HOST B also showed the u8 weights are repacked **once at load** into a blocked
  layout (`AB4b32a4b`) and dequantization is fused into the JIT kernel.
- The compute engine is chosen by `DYNAMIC_QUANTIZATION_GROUP_SIZE` (32 = int8/VNNI path,
  0 = BF16-dot path) — the OpenVINO primitive name `brgemm_avx512_bf16` labels the
  activation dtype/ISA family, not the MAC dtype. Bonus finding: group size 128 is another
  **11–21% faster** (HOST A) / **1.16–1.26×** (HOST B), cosine ≥ 0.9959.
- Blocked evidence recorded honestly: oneDNN per-primitive verbose crashes on this build
  (compile-time constant-fold reorder, dtype-driven); perf/PMU blocked by
  `perf_event_paranoid=4` + missing kernel-matched tools (both hosts); no system settings
  were changed. `ONEDNN_MAX_CPU_ISA=AVX512_CORE` does **not** strip VNNI from the
  compressed path on this build, so a "VNNI vs no-VNNI" ceiling ablation does not exist.

### CPU-utilization figure clarification (~1332% vs ~1550–1570%)

Both numbers are real, different aggregation windows: `~1332%` is the **whole-process**
`proc_cpu_mean_pct` of the standalone `default` run (`openvino_default.json` `full_run`,
start→stop including model load and idle gaps; the bridge run's whole-process figure is
1312%). During the **warm-encode windows** the process averages **1509–1572%**
(≈15.1–15.7 of 16 cores; per-prompt `interval_stats.proc_cpu_mean_pct`, also the
`cpu_util_mean_pct` column of `results/comparison/summary.csv`). No historical number was
changed; the headline table quotes the warm-interval figure with footnote 5.

## Verification

Three independent read-only subagent checkpoints (setup/downloads → ComfyUI route → final report fidelity) all **PASS**; every headline number in this README can be re-derived from the committed JSON/CSV. Record: `environment/checkpoints.md`. Classification: `results/comfy_cpu/runtime_path.md`.

## Honest caveats

- OS page cache was **not** dropped (no root); "cold" = fresh process, model file may be page-cached. Warm numbers are unaffected.
- ComfyUI `model_load_s=0.30 s` is the safetensors mmap-assign; the real disk page-in shows up inside the first encode (12.11 s).
- ~~OpenVINO internal GEMM dtype not dumped → marked UNKNOWN, deliberately **not** claimed as W8A8.~~ **Resolved by the root-cause investigation** (same day): dynamic-quantization int8 dot on AVX512_VNNI confirmed at dispatch+JIT+counterfactual level; NOT static W8A8 (activations are quantized at runtime, group 32). perf-sampling-level proof unavailable (host restrictions). See `report/OPENVINO_ZEN5_ROOT_CAUSE.md`.
- Root-cause investigation caveats: no measured DRAM bandwidth/cache counters (perf blocked; GB/s figures are byte-accounting derivations); oneDNN per-primitive verbose unavailable on this build (documented crash); GPU comparison covers only ComfyUI's forced-fp32 product path — a bf16-compute GPU path was not measured.
- End-to-end image-generation A/B was out of scope for this round (conditioning semantics verified; the 33 GB Qwen-Image-2.1 checkpoint was not pulled).

## Repo layout

```
environment/   system baseline, model inventory + SHA256, checkpoint records
scripts/       monitor + 6 benchmark/analysis scripts (see Reproduce)
prompts/       unified P1/P2/P3
models/        local ModelScope checkouts (gitignored, ~36 GB)
results/       raw 20Hz sampling CSVs, per-run JSON/CSV, npy embeddings, comparison tables
                + openvino_isa/ & openvino_root_cause/ (root-cause investigation artifacts)
report/        RESULTS.md (full, 中文) + qwen_image_conditioning_semantics.md (中文) + OPENVINO_ZEN5_ROOT_CAUSE.md (root-cause, 中文)
ComfyUI/       dedicated checkout used by the benchmark (gitignored)
```

Detailed reports are in Chinese (`report/`); all machine-readable evidence is language-neutral JSON/CSV.

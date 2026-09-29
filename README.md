# Qwen3-VL Text Encoder on Ryzen AI Max+ 395: ComfyUI INT8 ConvRot vs OpenVINO INT8

A/B benchmark on **AMD Ryzen AI Max+ PRO 395** (Strix Halo, 16C/32T, AVX-512 + AVX512_VNNI + AVX512_BF16, 94 GiB UMA), measured 2026-09-29. All numbers are real measurements (20 Hz psutil sampling + `perf_counter_ns`), fully reproducible from this repo.

**TL;DR**

1. **ComfyUI's `qwen3vl_8b_int8_convrot` on CPU does NOT run INT8 GEMM.** It is **INT8 storage → full dequant every forward (252 layers, 13.9 GB/encode) → FP32 compute** ("A2"). Proven by zero `torch._int_mm` calls, byte-exact dequant accounting, profiler, and the source chain.
2. **True INT8 CPU GEMM works on this chip** (`torch._int_mm` / oneDNN-VNNI). Forcing ComfyUI's own int8 kernels (A1 variant) gives **1.77×** speedup — the capability exists, the conditioning path just disables it.
3. **OpenVINO INT8 weight-compressed wins by an order of magnitude**: with a zero-recompute `add_outputs` bridge that exactly matches Qwen-Image 2.1 conditioning semantics (cosine ≥ 0.997 vs BF16 reference), it is **22–69× faster** than ComfyUI's product path and loads ~10× faster.
4. **Recommendation for `CPU → Qwen3-VL conditioning → Radeon GPU → Qwen-Image 2.1 DiT`:** run the encoder with **OpenVINO INT8 + the `add_outputs` bridge** (scripts included). Cost: ~2× peak RAM (15 vs 7.6 GiB, still only 16% of 94 GiB).

## Decision table

| You are… | Use | Why |
|---|---|---|
| Building a Qwen-Image 2.1 pipeline on Ryzen AI Max+ 395 / Strix Halo | **OpenVINO INT8 + bridge** (scripts/bench_openvino_bridge.py) | 0.19–0.69 s conditioning, 1.26 s load, semantics verified |
| Locked into the ComfyUI ecosystem | ComfyUI product path works, but 22–69× slower; consider upstream patch to enable int8 path (A1) | A1 measured: 1.77×, cos 0.9978, kernels already ship |
| RAM-constrained (UMA shared with GPU) | ComfyUI route uses less peak RAM | 7.6 vs 15.1 GiB |
| Wanting TRUE int8 on CPU in ComfyUI | Flip `comfy_force_cast_weights` + `use_quantized_matmul` | See A1 evidence below; needs upstream acceptance |

## Headline numbers (warm encode, best config = 16 threads, median of 3 fresh processes)

| Metric | ComfyUI INT8 ConvRot (product, A2) | ComfyUI forced-int8 (A1) | OpenVINO INT8 (bridge) |
|---|---:|---:|---:|
| Warm P1 (25 tok) | 12.98 s | 7.32 s | **0.187 s** |
| Warm P2 (67 tok) | 13.73 s | 7.46 s | **0.377 s** |
| Warm P3 (171 tok) | 15.43 s | 8.36 s | **0.687 s** |
| Model load | 0.30 s (mmap assign; real page-in lands in first encode 12.11 s) | same | **1.26 s** (read+compile) |
| First usable encode (cold) | 12.11 s | ~7.5 s | **0.18 s** |
| Peak RAM | 7.59 GiB | 7.35 GiB | 14.57 GiB |
| Model disk | 9.351 GB | same weights | 8.810 GB |
| True INT8 GEMM | **NO** | **YES** | UNKNOWN¹ |
| Cosine vs BF16 reference | 0.9989–0.9993 | 0.9978–0.9986 | 0.9972–0.9985 |

¹ OpenVINO's internal oneDNN kernel choice was not dumped; what is proven: int8 weights resident, CPU execution asserted at runtime, accuracy consistent with int8 weight-only compute.

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

# accuracy + aggregation
.venv-comfy/bin/python scripts/bench_bf16_reference.py
python3 scripts/compare_embeddings.py && python3 scripts/aggregate_summary.py
python3 scripts/classify_comfy_path.py run1 run2 run3 t8 t32 a1_t16 a1_t8 a1_t32
```

## Verification

Three independent read-only subagent checkpoints (setup/downloads → ComfyUI route → final report fidelity) all **PASS**; every headline number in this README can be re-derived from the committed JSON/CSV. Record: `environment/checkpoints.md`. Classification: `results/comfy_cpu/runtime_path.md`.

## Honest caveats

- OS page cache was **not** dropped (no root); "cold" = fresh process, model file may be page-cached. Warm numbers are unaffected.
- ComfyUI `model_load_s=0.30 s` is the safetensors mmap-assign; the real disk page-in shows up inside the first encode (12.11 s).
- OpenVINO internal GEMM dtype not dumped → marked UNKNOWN, deliberately **not** claimed as W8A8.
- End-to-end image-generation A/B was out of scope for this round (conditioning semantics verified; the 33 GB Qwen-Image-2.1 checkpoint was not pulled).

## Repo layout

```
environment/   system baseline, model inventory + SHA256, checkpoint records
scripts/       monitor + 6 benchmark/analysis scripts (see Reproduce)
prompts/       unified P1/P2/P3
models/        local ModelScope checkouts (gitignored, ~36 GB)
results/       raw 20Hz sampling CSVs, per-run JSON/CSV, npy embeddings, comparison tables
report/        RESULTS.md (full, 中文) + qwen_image_conditioning_semantics.md (中文)
ComfyUI/       dedicated checkout used by the benchmark (gitignored)
```

Detailed reports are in Chinese (`report/`); all machine-readable evidence is language-neutral JSON/CSV.

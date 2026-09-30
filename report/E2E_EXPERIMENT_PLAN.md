# E2E Experiment Plan — Qwen-Image 2.1 Hybrid (CPU Encoder + GPU DiT) on Ryzen AI Max+ PRO 395

> Phase 0 deliverable. Written 2026-09-29 after full repo audit (README.md, report/RESULTS.md,
> report/OPENVINO_ZEN5_ROOT_CAUSE.md, report/qwen_image_conditioning_semantics.md,
> environment/system.md, environment/model_inventory.md, results/comparison/*.csv,
> scripts/bench_openvino_bridge.py, scripts/bench_comfy_bf16_gpu.py, HANDOFF.md, and the
> three untracked e2e scripts from the previous session).
> Branch: `experiment/qwen-image-e2e-hybrid` (from main `3ee1388`).
>
> **状态（2026-09-30，PR #4 已合并）**：本计划已全部执行完毕 — Gate 1–7 + Final
> 20-point audit 全部 PASS，用户盲评 PASS（`environment/e2e_checkpoints.md`）；最终
> 结论见 `report/QWEN_IMAGE_E2E_HYBRID.md`。与本文预案的差异：质量数据集实际为
> 30 prompts × {gpu, dq32} 全量 + 12-prompt dq64/128 子集；repeated seeds 实际
> 5×5 × 2 路线（gpu vs dq128，25 对）；KNOWN 表中 "DQ=128 free 11–21%" 的产品化
> 建议在 e2e 尺度被推翻（最终推荐 DQ 32）。以下原文未改动。

## North-star question

On this APU (Zen 5 CPU + Radeon 8060S + 94 GiB UMA), does moving the Qwen3-VL text encoder
of the **complete Qwen-Image 2.1 pipeline** to the CPU (OpenVINO INT8) — leaving the GPU
exclusively for DiT + VAE — make the whole generation pipeline faster / lighter / better
suited to local AI than running everything on the GPU?

All conclusions must come from real runs (real models, prompts, latencies, utilization,
memory, output PNGs). No estimates, no mocks, no extrapolation from the encoder-only rounds.

## KNOWN (established by merged work, reused as input)

| Fact | Source |
|---|---|
| HOST A = Ryzen AI Max+ PRO 395, Zen 5, 16C/32T, AVX-512 + VNNI + BF16, Radeon 8060S, 94 GiB UMA | environment/system.md |
| ComfyUI int8-convrot TE product path = INT8 storage → full dequant → FP compute; P1 warm 12.98 s | results/comparison/summary.csv |
| ComfyUI BF16 TE on Radeon GPU (product path, forced-fp32 GEMM): 0.601/1.228/1.834 s (P1/P2/P3), GPU busy 85%, 16.63 GiB GTT, cos 1.000000 | results/comparison/summary.csv, report/RESULTS.md |
| OpenVINO INT8 bridge (add_outputs pre-norm layer-36, T2I template, trim from 2nd `<\|im_start\|>`): 0.187/0.377/0.687 s, cos 0.9972–0.9985, GPU idle | results/comparison/summary.csv |
| VNNI executes (vpdpbusd); 22–69× is architectural (no dequant materialization), VNNI only 1.23–1.47×; DQ=128 free 11–21% faster, cos ≥0.9959 | report/OPENVINO_ZEN5_ROOT_CAUSE.md |
| Conditioning semantics fully mapped (T2I template, layer_idx=-1, no final RMSNorm, fp32, trim) | report/qwen_image_conditioning_semantics.md |
| `encode_from_tokens_scheduled` returns `[[tensor, {"pooled_output": pooled}]]` for t2i (no hooks); the Qwen-Image DiT consumes only the tensor + attention_mask/reference_latents keys, so OV injection `[[tensor.float(), {}]]` is functionally identical (Gate-1 review correction; identity compares the tensor) | ComfyUI sd.py:340-350/420-426, Gate-1 audit |
| Models local: OV int8 (8.8 GB), comfy int8 TE (9.35 GB), BF16 TE (17.5 GB); DiT bf16 14.23 GB + VAE 0.676 GB downloaded from ModelScope `Comfy-Org/Qwen-Image-2.1` (sizes match declarations) | environment/model_inventory.md + logs/dl_qwen_image_dit.log |
| venvs: `.venv-comfy-rocm` (torch 2.12.0+rocm7.14.0), `.venv-openvino` (OV 2026.4.0), `.venv-comfy` (torch 2.9.1+cpu) | repo |
| GPU sysfs available: card1 gpu_busy_percent, mem_info_gtt_used, mem_info_vis_vram_used; hwmon k10temp + amdgpu; powercap RAPL present | probed this session |

## UNKNOWN / TO VERIFY (this round must measure)

1. **End-to-end latency** of the full pipeline (T0–T8) per route — never measured (no DiT before).
2. Whether the GPU TE route slows the **DiT stage** (GTT residency 16.6 GiB, load/offload churn) — Q3/Q24.
3. GPU memory/GTT freed by the hybrid across load+generation — Q22.
4. Warm single-image and continuous multi-prompt throughput; images/min — Q2/Q14.
5. CPU/GPU overlap potential (encode next prompt during DiT) and UMA memory-bandwidth contention — Q5/Q15/Q16.
6. Real image output equivalence/quality per route and per DQ group (32/64/128, plus 0 as control) — Q6/Q17/Q18–21.
7. Prompt-length scaling end-to-end (10→300+ tokens) — Q8.
8. Cold start (fresh process, first image) incl. OpenVINO compile cache — Q12/Q26.
9. Thermal/power behavior over long runs — Q36/Q37.
10. What default ComfyUI actually does with each TE file on this machine (device placement, offload) — to record honestly, not assume (Q/G baseline fairness).

## Architectures under test

**G — GPU-heavy (baseline, ComfyUI product path):**
`Text Encoder (BF16 weights on Radeon via ComfyUI default device selection) → DiT (Radeon) → VAE (Radeon)`.
Runs both (a) through the real ComfyUI server with an official-style workflow (Phase 2 product
baseline, workflow JSON archived) and (b) through the controlled harness using the same
`comfy.*` modules and default devices (for stage timing). If ComfyUI offloads any module by
default, it is recorded as-is. A probe also records where ComfyUI puts the int8-convrot TE
file (for documentation; it is not the G baseline).

**H — Hybrid (experiment):**
`OpenVINO Qwen3-VL INT8 (Zen 5 CPU, DQ∈{0,32,64,128}) → conditioning bridge → DiT (Radeon) → VAE (Radeon)`.
Implemented in-harness (Option A, least invasive): OV runs in-process in the ROCm venv
(openvino is torch-independent) producing the exact `[[tensor, {}]]` conditioning structure;
DiT/VAE/sampler/seed identical to G. Pipelined variant (Phase 15) computes P(n+1) conditioning
on CPU threads while the GPU runs DiT(n).

**C — reference point (not a headline route):** ComfyUI int8-convrot TE on CPU (product path A2),
the route the repo already characterizes — included in stage tables for context.

## Controlled variables (fixed for all quality comparisons)

prompt set (P1–P5 + quality dataset), negative `" "` (prevent_empty_text), seed per image
(default 20260929 for matrices; seed sets for repetition), euler/simple, 20 steps (40 in step
matrix), CFG 2.5, 1024×1024 (1328×1328 in resolution matrix), same DiT bf16 file, same VAE
bf16 file, same latent (`prepare_noise(latent, seed)` after `fix_empty_latent_channels`).
**Only the conditioning route (and DQ group) varies.**

## Timing & monitoring spec (Phases 5–7)

Per image: T0 model-init start, T1 tokenizer start/end, T2 text-encoder start/end,
T3 conditioning transfer/preparation, T4 DiT first-step start, T5 DiT final step,
T6 VAE start, T7 image ready (headline boundary), T8 PNG saved (reported separately).
GPU-side boundaries wrapped with `torch.cuda.synchronize()` (both sides). Monitor thread at
20 Hz: process CPU%/RSS/threads/freq, system CPU%/MemAvailable; GPU busy%, GTT used, VRAM
used, amdgpu temp/power (hwmon), k10temp, RAPL package energy. Every run writes a
timestamped CSV to `results/e2e/raw/`. Unavailable metrics are recorded as `unavailable`,
never estimated. (Gate-2 amendments: actual monitor rate is 18.0–18.2 Hz — within the
10–20 Hz spec; `rapl_energy_uj` is root-only on this host (mode 0400) → CPU package
energy recorded as unavailable; `dit_steps` callback timestamps are host-paced launch
diagnostics, not per-step GPU ground truth (T4/T5 totals are properly synchronized);
route-level `encode_s` includes tokenization for the OV route but not for G — footnoted
wherever encode is compared.)

## Statistics (Phase 32)

Headline latencies = **median of fresh-process medians** over ≥3 fresh processes; each
process = 1 cold + ≥3 warm generations (reduced from 5 if per-image time proves too long —
spec Rule 2 allows 3×3 with disclosure; the actual choice is recorded per experiment).
Bootstrap 95% CI where sample size permits (≥5 samples). Mean/std/min/max/CV reported for
warm runs.

## Phase mapping & audit gates

Spec phases are executed in grouped milestones, each ending with an independent read-only
subagent audit (PASS/FAIL recorded in `environment/e2e_checkpoints.md`):

| Gate | Spec phases | Content |
|---|---|---|
| 1 | 0 | This plan reviewed (design audit) |
| 2 | 1–7, 29–30 | models verified (SHA256), harness + smoke, ComfyUI server baseline, conditioning identity (shape/cos/RMSE/relL2/maxabs), timing sync, monitor |
| 3 | 8, 12–13, 32 | core A/B G vs H (cold+warm, 3 fresh procs), stats |
| 4 | 9–11, 14–17, 31 | prompt/resolution/step matrices, continuous, pipelined, contention, DQ matrix + repeated seeds |
| 5 | 18–21 | quality dataset (30+ prompts × routes), SSIM/PSNR (+LPIPS if obtainable), CLIP/SigLIP score, blind A/B/C/D package |
| 6 | 22–28, 33–39 | memory/GTT/UMA/residency/cache/isolation analyses, primary tables, scorecard, failure cases, thermal, power, recommendation |
| 7 | 40–42 | ComfyUI custom node + benchmark with it + one-click scripts (only after main conclusions PASS) |
| Final | 43–46 + 20-point audit | report/QWEN_IMAGE_E2E_HYBRID.md, README section, social-media-safe conclusions, final audit, commit/PR |

## Scale calibration policy (decided by smoke measurement, before Gate 3)

A single 1024²×20-step image's measured cost sets the dataset sizes: target ≥100 images
overall and ≥30 per headline route (G, H-DQ32) if per-image time ≤ ~90 s; otherwise apply the
documented reduction (30 prompts → 12–15, DQ64/128 on subset, repeated seeds 5×5 → 3×3) and
disclose in the final report. Generation parameters are never silently changed; OOM is a
result, not a reason to lower resolution.

## Results structure (Phase 43)

```
results/e2e/
  gpu_baseline/   ComfyUI server product runs (workflow JSON, logs, meta)
  ab/             core A/B per-process JSONs
  matrix/         prompt/resolution/step matrices
  throughput/     continuous + pipelined + contention
  dq/             DQ group matrix incl. real images
  quality/        dataset, metrics, blind package
  raw/            20 Hz monitor CSVs per run
  images/         PNGs + sidecar JSON (prompt/seed/route/dq/res/steps/cfg/latencies/hashes)
  cond/           conditioning npy + identity metrics
  summary/        auto-generated CSV/MD tables (Phase 33) — no hand-typed numbers
```

## Gate-1 review amendments (audit: PASS, findings applied)

- Old handoff script `bench_e2e_qwen_image21.py` called `mm.unload_all_models()` inside
  the warm loop — that would evict the TE from GTT between images and destroy the
  residency effect Q3 measures. Superseded by `scripts/e2e_pipeline.py` (no unload
  between warm images; fresh process per route for headline numbers).
- G protocol pinned: pure-default `load_clip` (verified: TE lands GPU-resident),
  vram_state + devices + weight dtype recorded, never unload during warm runs.
- In-process OV controls: `openvino==2026.4.0` installed into `.venv-comfy-rocm`
  (same wheel version as `.venv-openvino`); DiT smoke re-run in that venv after
  install; headline encode timing = in-process `encode_s` (pos+neg), cross-checked
  against fresh-process `prep_e2e_cond.py` protocol numbers.
- DiT kernel env: all main runs use `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
  (ComfyUI's own startup recommendation on AMD; identical for G and H; default vs
  AOTriton DiT delta measured and documented). GPU power/thermal state is firmware-
  managed (no writable power cap, dpm=auto, no root) — recorded, not tuned.
- Pre-declared unavailable: perf PMU (paranoid=4), oneDNN per-primitive verbose,
  DRAM bandwidth counters, page-cache drop for "cold" (= fresh-process cold only),
  possibly amdgpu power on some SKUs (hwmon7 power1_average IS available here).
- Prompt ladder: P4=15 / P1=25 / P2=67 / P3=171 / P5=299 kept tokens (measured).
- Route-order policy: one route per fresh process; core A/B order counterbalanced
  across processes (proc1: G,H; proc2: H,G; proc3: G,H by wall-clock pairing).
- Original task spec archived verbatim: `report/E2E_TASK_SPEC.md`.

## Non-goals / guardrails

- No changes to merged historical results (read-only).
- No `reset --hard`/force-push; work only on the experiment branch.
- No synthetic numbers in README/report — every number traceable to JSON/CSV/PNG sidecars.
- 69×/22–69× claims stay encoder-scoped; e2e claims only from e2e data.
- Headline "pipeline is more reasonable" requires ≥3 of the 5 evidence conditions in the spec.

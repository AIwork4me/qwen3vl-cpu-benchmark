# Validate OpenVINO CPU conditioning + Radeon GPU DiT for Qwen-Image 2.1

## Hypothesis

On the Ryzen AI Max+ PRO 395 (Zen 5 CPU + Radeon 8060S + 94 GiB UMA), moving the
Qwen3-VL text encoder of the **complete** Qwen-Image 2.1 pipeline to the CPU (OpenVINO
INT8) — leaving the GPU exclusively for DiT + VAE — makes the whole pipeline better
suited to local AI than running everything on the GPU. The prior rounds proved the
encoder-stage win (22–69× vs CPU product path, 2.7–3.3× vs GPU route); this round tests
whether it survives contact with a 149-second DiT.

## Architecture A — GPU-heavy (G, ComfyUI product baseline)

BF16 Qwen3-VL TE via `comfy.sd.load_clip` pure defaults (verified GPU-resident, 16.7 GiB
staged) + DiT 14.23 GB + VAE on the Radeon. Also run through the real ComfyUI server
(official node set, workflow JSONs archived): P1/P2/P3 = 150.1/140.6/141.6 s.

## Architecture B — Hybrid (H)

OpenVINO INT8 Qwen3-VL on the Zen 5 CPU in-process (ComfyUI's own tokenizer, add_outputs
pre-norm tap, `[[tensor, {}]]` injection), DQ ∈ {0,32,64,128}; identical DiT/VAE/sampler/
seed/latent. Productized as a custom node and benchmarked through the real server.

## Exact hardware/software

AMD Ryzen AI Max+ PRO 395 (16C/32T Zen 5, AVX-512+VNNI+BF16), Radeon 8060S (gfx1151,
GTT 100 GiB, firmware power), 94 GiB LPDDR5X UMA, Ubuntu 24.04.4, ComfyUI 0.37.0,
torch 2.12.0+rocm7.14.0, OpenVINO 2026.4.0, AOTriton enabled (identical for both
routes; default-vs-AOTriton = 1.39× DiT, measured). Model SHA256s in
environment/model_sha256.txt.

## E2E latency (P3, 1024², 20 steps, median of 3 fresh-process medians)

- Cold time-to-first-image: **G 176.11 s → H 168.82 s (−4.1%)**
- Warm image: **G 154.17 s → H 151.25 s (−1.9%)** (harness, force-loaded DiT); through
  the real server: **140.6–141.6 s vs 140.9–141.1 s — statistically identical**
- Encoder stage: G 2.618 s → **H 0.993 s** (P3 pos+neg)

## Stage-level latency

Encoder is the only stage that differs: DiT 149.05 vs 147.95 s (statistically
indistinguishable, ±3–5 s thermal drift documented), VAE 2.43 vs 2.25 s. Bonus config
finding: ComfyUI's default DynamicVRAM weight path is ~7% faster than force-full-load
on this APU (probe run included).

## GPU memory

- GTT residency: **G 51.86 GiB → H 35.40 GiB (−16.46 GiB)** — the TE's footprint freed
- Physical UMA consumed (MemAvailable): **52.7 → 43.5 GiB (−9.2 GiB)**; RSS 32.0 → 30.1
- At 1328², G peaks at 65.07 GiB of 100 GiB GTT — H's headroom matters more as work grows

## CPU/GPU utilization

DiT windows: GPU busy 99.4–99.7% on both routes (GPU-bound). Encode windows: G keeps
the GPU 83–90% busy; H does the same work on 15.5 CPU cores with the GPU idle. CPU
package energy (RAPL) unavailable (root-only) — recorded, not estimated.

## Pipeline-overlap result

- One-prompt-ahead pipelining hides the encoder (post-DiT wait ≈ 0.0002 s), penalty-free
  over 5 images; **the 10-image prototype degrades** (DiT → 172.5 s, GTT −5.4 GiB
  accumulation) — disclosed as an implementation defect, root cause not fully diagnosed.
- Sustained concurrent CPU-encode + GPU-DiT (storm test): **DiT 2.11× slower** (299.6 vs
  142.0 s mean) — the UMA shared-bandwidth ceiling. Sequential hybrid is the
  recommended steady-state mode.

## DQ group result

Encode 1.44 / 0.993 / 0.884 / 0.834 s (0/32/64/128); conditioning cosine 0.9988 / 0.9985
/ 0.9982 / 0.9971; image similarity degrades monotonically while prompt adherence is
noise-level everywhere. **DQ 32 recommended** — 64/128 save ≤0.16 s per ~153 s image.

## Image-quality result

30 prompts (12 categories incl. Chinese, text rendering, counting), per-prompt fixed
seed, warmup-deterministic generation, DQ32 vs GPU route: SSIM med 0.9299, CLIP
image-similarity med **0.9926**, CLIP prompt-adherence Δ med **−0.0006** (range
−0.022…+0.011). Text-rendering prompts score equal or better on the hybrid. Same-route
nondeterminism floor measured (SSIM ≥ 0.993) — divergences are conditioning-driven.
Repeated seeds 5 prompts × 5 seeds (DQ128): SSIM med 0.925, no seed collapse. Blind
A/B(/C/D) contact-sheet package committed for human review; no subjective scores invented.

## Failure cases

Encode-storm contention (2.11×), pipelining >5 images (prototype defect), very short
prompts (0.6% win), DiT-heavy operating points (gain dilution), torch-ROCm-build CPU TE
slowness (97.8–99.5 s vs 15.4 s torch-cpu), thermal warm-up drift vs flat steady state.

## Reproduction

`bash scripts/run_e2e_suite.sh <stage>` (core_ab / continuous / pipelined / contention /
prompt_matrix / step_matrix / res_matrix / dq_matrix / ov_cache / det_test /
kernel_note20 / quality / quality_dq_subset / repeated_seeds) +
`scripts/run_comfy_server_baseline.py` + `scripts/run_custom_node_benchmark.py` +
`python3 scripts/aggregate_e2e.py` / `make_primary_table.py` /
`analyze_e2e_resources.py` / `.venv-comfy/bin/python scripts/evaluate_quality.py`.
Models via ModelScope (commands in environment/model_inventory.md).

## Raw evidence

271 real images (265 harness + 6 server); every headline number regenerable from
committed JSON/CSV (results/e2e/summary/, results/e2e/{ab,matrix,throughput,dq,quality},
20 Hz monitor CSVs for core runs, per-PNG sidecars, blind package). Evidence tiering
documented in the report's raw-evidence index. Gates 1–6 audited by independent
read-only subagents (one FAIL→fixed→re-audit PASS cycle); final 20-point audit result
in environment/e2e_checkpoints.md.

## Limitations

Fresh-process cold only (page cache not droppable, no root); no bf16-tensor-core GPU
encoder path measured (G is ComfyUI's forced-fp32 product path); LPIPS weights
unreachable offline (SSIM/PSNR/CLIP used); perf PMU and RAPL energy unavailable
(host restrictions); pipelining root-cause undiagnosed; single machine (host B
cross-validation belongs to the prior encoder-only round).

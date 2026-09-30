# E2E Experiment Checkpoints (subagent audit record)

Protocol: each gate ends with an independent read-only subagent audit
(PASS/FAIL + findings). Findings are fixed before the next gate starts.

## Gate 1 — experiment design review (Phase 0)

- **Verdict: PASS** (2026-09-29, agent general-purpose, read-only).
- Reviewed `report/E2E_EXPERIMENT_PLAN.md` against repo + ComfyUI 0.37.0 source.
- Findings & resolutions:
  1. HIGH: old handoff script `bench_e2e_qwen_image21.py:192` unloads all models
     inside the warm loop (14 GB reload inside timed window; evicts TE from GTT).
     → Superseded by `scripts/e2e_pipeline.py` (no unload between warm images).
  2. MED: plan claimed `encode_from_tokens_scheduled` returns `[[tensor, {}]]`;
     actual no-hook return is `[[tensor, {"pooled_output": pooled}]]` (sd.py:340).
     Functionally equivalent for this DiT; plan text corrected; identity compares
     the tensor.
  3. MED: P4/P5 prompts did not exist. → Authored + tokenized:
     P4=15 / P1=25 / P2=67 / P3=171 / P5=299 kept tokens.
  4. MED: task spec not in repo. → Archived `report/E2E_TASK_SPEC.md`.
  5. MED: 20 Hz monitor not wired into old script. → New harness integrates
     `scripts/e2e_monitor.py` (CPU+GPU+GTT+temp+power+RAPL, 20 Hz, marks-aligned).
  6. MED: pin G device protocol. → Pure-default `load_clip`; vram_state/device/
     dtype audit recorded; verified TE lands GPU-resident by default on this box.
  7. LOW: add `keep_vision=True` to tokenize (exactness). → Applied.
  8. LOW: OV output shape assert. → Applied ([1, n_tok, 4096]).
  9. LOW: in-process OV controls. → openvino 2026.4.0 installed in .venv-comfy-rocm
     (same version as .venv-openvino); DiT smoke re-run post-install OK; headline
     encode timing = in-process encode_s; fresh-process cross-check kept.
  10. LOW: route-order policy + governor recording. → Counterbalanced fresh
      processes; governor/EPP now in run meta.
- Additional decisions recorded at Gate 1→2 boundary:
  - AOTriton (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`) enabled for ALL main
    runs (ComfyUI's own startup recommendation; identical across G/H; measured
    default-vs-AOTriton DiT delta to be documented from smoke data).
  - GPU power state: firmware-managed, no writable cap, dpm=auto, no root —
    honest recording, no tuning attempted.

## Gate 2 — models + harness + baseline + conditioning identity (Phases 1–7, 29–30)

- **Verdict: PASS** (2026-09-29, read-only agent; independent SHA256 recompute,
  identity recompute from npys, monitor-window recompute from CSVs+marks).
- Key verified facts: models byte-exact; server baseline P1/P2/P3 success
  (150.1/140.6/141.6 s prompt-exec, TE+DiT+VAE all cuda:0); harness sync placement
  correct at all 4 boundaries; within-route i0/i1 PNGs bit-identical (determinism);
  OV-DQ32-vs-GPU P1 pos identity cos 0.9972436 == merged fresh-process value to
  7 decimals; monitor 18.0–18.2 Hz; DiT-window GPU busy independently recomputed
  (96.66%, n=745).
- Findings & resolutions:
  1. MED sidecar total_gen_s null (write order) → harness reordered + backfill
     (scripts/backfill_sidecars.py); verified on subsequent runs.
  2. MED negative-prompt identity unreported → analyze_e2e_cond.py extended;
     measured: OV vs GPU neg cos 0.9710 (relL2 0.239, 9 tokens; CFG consumes it) —
     disclosed in report + quality section.
  3. MED AOTriton attribution via tag names → dit_kernel_note stage queued
     (recorded aotriton field in all new runs).
  4. LOW stale dit_preload on warm rows → only-first-image now.
  5. LOW dit_steps host-paced → documented footnote (plan).
  6. LOW T7→T8 gap unattributed → postprocess_s stage added.
  7. LOW encode_s definitional asymmetry → footnote (plan).
  8. LOW RAPL unreadable (0400) → CPU package energy = unavailable; plan amended.
  9. INFO k10temp hits 99–100 °C during CPU encode bursts (powersave/balance_performance)
     → thermal watch item for Phase 36.
  10. INFO server wall_s 2 s polling quantization → use server "Prompt executed in".
- Also at this gate: quality evaluator CLIP ViT-L/14 downloaded from ModelScope
  (AI-ModelScope/clip-vit-large-patch14, 1.71 GB, snapshot ready); scikit-image
  installed into .venv-comfy for SSIM/PSNR (CPU venv, no GPU timing impact).

## Gate 3+4 — core A/B, matrices, throughput/overlap/contention, DQ/cache (Phases 8–17, 26, 32)

- **Verdict: FAIL → all findings fixed → RE-AUDIT PASS** (2026-09-30; all 10 checks verified, cold TTFI recomputed exactly 176.11, two sub-ms cosmetic cells corrected post-re-audit).
- Auditor independently reproduced headline stats, cold TTFI, all matrix numbers, one
  DQ cosine from npys (0.997068 exact), determinism sha, kernel correction.
- Findings & resolutions:
  1. HIGH: CLIP img-sim 0.993 published in TL;DR/scorecard/social/README with no
     backing artifact (quality runs still executing) → removed everywhere; will be
     re-added only from results/e2e/quality/quality_metrics.csv.
  2. HIGH: report claimed pipe_ov10 "confirms" no penalty — pipe_ov10 actually
     degrades (DiT 148.9→172.5 s, E2E +8.7%, GTT −5.4 GiB monotonic; not thermal;
     final no-encode image still slow → accumulated allocator state) → overlap
     section rewritten; pipelining downgraded to "prototype defect"; added to
     failure cases.
  3. MED: cont_gpu encode mean corrected 1.89→2.29 s.
  4. MED: cold-start compile narrative corrected to measured values (A/B processes
     4.4–4.7 s; cold dir 3.45 s; fully warm 0.69 s).
  5. MED: cold TTFI mixed definitions → aggregate now uses uniform epoch-anchored
     reconstruction (fixed a double-count bug found while patching: runs WITH
     first_image_ready_epoch were getting +off added); gpu 176.11 / ov 168.82 /
     native 255.03.
  6. LOW: dq64 root-cause cell 0.99780→0.99821 (was P2 value, now P3 comparator).
  7. LOW: native range 97.8–99.5 s.
  8. LOW: postprocess 0.003–0.007 s; encode-window GPU busy 83–90%; te staging 5.8–10.3 s.
  9. LOW: DQ32 0.993-vs-1.02 s divergence footnoted; contention ratio unified to 2.11×
     (DiT-mean based, conservative).
  10. LOW: ab_p1_ov sidecars total_gen_s null → backfill re-run (129 sidecars).

## Gate 5+6 — quality dataset, repeated seeds, custom node, analyses (Phases 18–39, 40–41)

- **Status: complete 2026-09-30; final audit pending.** Quality dataset 30 prompts ×
  {gpu, dq32} + 12-prompt {dq64, dq128} subset (documented reduction), warmup-
  deterministic generation; SSIM/PSNR/CLIP (chunked ViT-L/14) + blind package; quality
  acceptance PASS at the pre-declared gate. Repeated seeds 5×5 (gpu vs dq128) after
  fixing a per-prompt-seed override bug (--force-seeds). Custom node benchmarked
  through the real server (warm parity 141 s both routes; symlink depth + realpath
  fixes). Same-route nondeterminism floor measured (SSIM ≥ 0.993) — DQ divergences are
  conditioning-driven. Dynamic-vs-full weight residency config note measured
  (dynload_probe: dynamic ~7% faster; A/B unaffected). Thermal: no runaway over
  30-image runs. Aggregate hardened (non-run JSONs skipped); route labels fixed for
  dq64/128. Totals: 271 real images (265 harness + 6 server); headline routes
  gpu=88, ov_dq32=107, ov_dq128=41 images.

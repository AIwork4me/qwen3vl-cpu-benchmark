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

(pending)

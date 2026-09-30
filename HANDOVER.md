# HANDOVER — OpenVINO Zen 5 VNNI Root-Cause Investigation

Written: 2026-09-29 (session stop). Author: HOST-B investigation session.
Read this fully before touching the branch/repo.

## 1. Stop-time exact state

### Remote (github.com/AIwork4me/qwen3vl-cpu-benchmark)

- **PR #1 is MERGED into `main`** (state: closed/merged, merge head `2b7b786`).
- `main` head lineage: `b1037bb` (upstream baseline) → `8a6a790` (**HOST A
  direct measurement, Ryzen AI Max+ PRO 395 / Zen 5** — a full 16-phase
  independent investigation with 14 subagent verifications, committed
  2026-09-29T11:51Z by "bench") → merge `2b7b786` = `8a6a790` + `9c3fb79`.
- Merge renamed my HOST-B report to
  `report/OPENVINO_ZEN5_ROOT_CAUSE.hostB.md`; the canonical report on main
  (`report/OPENVINO_ZEN5_ROOT_CAUSE.md`, ~12.7 KB) is the HOST-A version.
- Merge-message highlights: both investigations independently reached the
  same Q1 verdict (AVX512_VNNI executed, Levels B+C+E; 253 compressed FC
  nodes = 91–92% of node time).

### Local (this machine, /workspace/qwen3vl-cpu-benchmark)

- Branch: `investigate/openvino-zen5-vnni-root-cause`, HEAD = `0da6849`
  (`9c3fb79` + one fix commit). **`0da6849` was NEVER pushed** (egress proxy
  intermittently returns CONNECT 503 to git-over-HTTPS; multiple retries
  failed; later pushes were rejected as non-fast-forward because remote had
  moved to `2b7b786`).
- Working tree clean (only this handover file is new).

## 2. What `0da6849` contains (the unpushed asset)

Commit message: "fix: DQ group size is numerics-relevant — close self-audited
cosine gap". Contents:

- New raw data: `results/openvino_root_cause/investigate_dqcos_{0,32,64,128}.json`
  (+ stdout, profiling CSVs) — HOST-B runs with `--compare-npy`:
  | DQ | cos vs HOST-A DQ=32 npy (P1/P2/P3) | P1 p50 |
  |---|---|---:|
  | 32 (default) | **1.0000001 / 1.0000001 / 1.0** (bit-faithful) | 0.194 s |
  | 0 (bf16 kernels) | 0.99911 / 0.99928 / 0.99939 | 0.262 s |
  | 64 | 0.99897 / 0.99916 / 0.99925 | 0.159 s |
  | 128 | 0.99807 / 0.99791 / 0.99818 | 0.145 s |
- Text corrections in README.md / report / checkpoints removing my earlier
  wrong claim that DQ changes leave cosine "unchanged".

### Relationship to what is already merged

The merged HOST-A report measured the same trade-off on Zen 5 against the
**BF16 reference** (DQ=32: 0.9972–0.9985; DQ=128: 0.9959–0.9971) and already
states "教科书式速度/精度权衡" — so **the correction is already covered on
main by HOST-A data**. The two datasets agree in direction and are **not
contradictory**: different references (HOST-B cosines are vs the HOST-A
DQ=32 outputs, hence 1.0000001 at default). `0da6849` therefore no longer
"fixes" main; its remaining value is **independent second-host confirmation**
of the DQ speed/accuracy trade-off.

## 3. Disposal of `0da6849` — RESOLVED by branch `fix/hostb-dq-cosine`

This follow-up branch (based on main `3ee1388`) carries:
- the dqcos data files from `0da6849` (`investigate_dqcos_*`, `dqcos_*.stdout`,
  `profiling_P*_dqcos_*.csv`, `runtime_model_summary_dqcos_*.json`)
- the corrected Phase 5 table/answer-5, TL;DR #6, factor-table row and
  social-media bullet in `report/OPENVINO_ZEN5_ROOT_CAUSE.hostB.md`
  (main's merged copy still had the stale "cosine unchanged" wording)
- a post-merge correction record appended to `environment/root_cause_checkpoints.md`
- this handover file (updated)

Note: main's HOST-A report already documents the same trade-off with Zen 5
data (DQ=128 cos 0.9959–0.9971 vs BF16 ref) — the two hosts agree in
direction; the HOST-B numbers are a second-host confirmation, not a
contradiction (different cosine reference: HOST-B compares against the
HOST-A DQ=32 outputs, hence 1.0000001 at default).

## 4. Open items, in priority order

1. ~~Dispose of `0da6849`~~ DONE via `fix/hostb-dq-cosine` (this branch).
2. ~~Check HOST-A Level D~~ CHECKED: not obtained on HOST A either (`results/openvino_isa/phase7_perf_findings.md`: paranoid=4 + perf/kernel 6.17.0-1032-oem mismatch, no sudo) — Level D remains the only open evidence level, requires a perf-enabled machine or one-time sudo on the Ryzen host (commands ready in phase7 files).
3. ~~DQ productization decision data~~ AVAILABLE on both hosts; the *decision* itself (default DQ for `bench_openvino_bridge.py`) is still open — recommend adding `--dq-group-size` with default None (historical behavior) and documenting DQ=64/128 trade-off.
4. Longer term: ComfyUI upstream conversation about the conditioning-path
   force-cast (A1 evidence in README), end-to-end image sanity test (33 GB
   DiT not yet pulled).

(Original numbering preserved below for reference; items 2–4 were renumbered above.)
X. **Check whether HOST-A run obtained Level D (perf hotspot) evidence** —
   the merged commit message truncates before saying. Look for
   `results/openvino_isa/perf*/` or perf sections in main's checkpoints file.
   If absent, Level D remains the only unproven evidence level (commands
   ready in `results/openvino_isa/phase7_8_perf_unavailable.md` on the HOST-B
   tree; run on any perf-enabled machine).
3. **Productization decision: DQ group size.** Two hosts now agree:
   non-default groups are faster (HOST A: DQ=128 P1 0.159 s vs 0.185 s;
   HOST B: 0.145–0.157 s vs 0.192–0.194 s) at a small cosine cost
   (≈−0.001..−0.002 vs default numerics; DQ=64 is the closer-numerics
   option). Decide default for `scripts/bench_openvino_bridge.py` follow-up
   (add `--dq-group-size`; do NOT silently change historical defaults).
4. Longer term: ComfyUI upstream conversation about the conditioning-path
   force-cast (A1 evidence in README), end-to-end image sanity test (33 GB
   DiT not yet pulled).

## 5. Findings summary (both hosts, final)

- Q1 (VNNI executed?): **YES at Levels B+C+E on both hosts independently**
  (dispatch `FullyConnectedCompressed`/DQ kernels; JIT machine code with
  `vpdpbusd` 12×208 on HOST B; causal DQ ablation 1.37–1.47×). Level D
  (perf) not obtained on HOST B (container); check §4.2 for HOST A.
- Compute path: weight-only u8 INT8 + in-flight dynamic activation
  quantization (group 32 default) + VNNI integer dots; **not W8A8**.
- execType `brgemm_avx512_bf16` = activation dtype, not compute dtype.
- AVX-512 is a hard requirement (AVX2-family ceilings cannot run the model);
  `ONEDNN_MAX_CPU_ISA=AVX512_CORE` does NOT disable VNNI on this build.
- 22–69× vs ComfyUI CPU is structural (no per-encode 13.9 GB dequant, no
  fp32 GEMM, packed resident weights, fused JIT, 1.26 s load); VNNI ≈1.4×
  and INT8 bytes ≈1.2–1.5× are real but secondary.
- 2.7–3.3× vs GPU: scoped to ComfyUI's current fp32-compute conditioning
  path only.
- README "True INT8 GEMM: UNKNOWN" resolved with evidence-backed wording.

## 6. Environment facts for whoever continues here (HOST B machine)

- Repo: `/workspace/qwen3vl-cpu-benchmark`; venv `.venv-openvino/`
  (openvino==2026.4.0, transformers 5.17.0).
- Models: `models/qwen3vl-openvino-int8/` (SHA256 match to committed
  hashes); FP16 control at `/root/models/qwen3vl-openvino-fp16/` (**outside
  the repo**, hashes in `results/openvino_root_cause/fp16_control_sha256.txt`).
- Benchmark runs must be `taskset -c 0-15` (16 physical cores, socket 0);
  LATENCY hint + `INFERENCE_NUM_THREADS=16` replicates the original setup.
- Network: pypi/ModelScope/GitHub-API work; git-over-HTTPS is intermittently
  blocked (CONNECT 503) — retry loops or use the API; **never force-push**.
- perf/PMU unavailable here (`perf_event_paranoid=4`, no binary, no
  CAP_PERFMON) — documented with exact errors.

## 7. Evidence index (HOST-B side)

Everything lives under `results/openvino_isa/` and
`results/openvino_root_cause/`; per-phase PASS records in
`environment/root_cause_checkpoints.md`; host scoping in
`environment/root_cause_host_b.md`; machine-generated summaries:
`root_cause_factor_table.md`, `vnni_contribution.json`. Verification trail
this session: Phase 0/1 PASS → Phase 2–6 FAIL(round1)→fixed→PASS →
Phase 7–10 PASS → final 3-persona audit PASS → DQ-cosine correction PASS
(subagent ses_f125369f). The last subagent-verified change is exactly the
unpushed `0da6849`.

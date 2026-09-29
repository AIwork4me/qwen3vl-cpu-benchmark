# Root-cause investigation — independent verification checkpoints (HOST A: Ryzen AI Max+ PRO 395, Zen 5)

> Two-host merge note: this branch contains the same investigation run independently on HOST A (this file) and
> HOST B (EPYC 9334 / Zen 4, record appended below; report: report/OPENVINO_ZEN5_ROOT_CAUSE.hostB.md;
> scripts: scripts/hostb/). Both reached the same verdict independently.

Format per task Rule 2. Verifiers are fresh read-only subagents with no shared state;
each received only the phase claim, file list, and instruction to recompute numbers.

## Phase 0 — baseline reproduction
Verifier: read-only subagent (agent_df49f403), recomputed p50s from raw warm arrays, diffed script vs HEAD, cross-checked historical JSON/CSV/npy
Verdict: PASS
Evidence checked: results/openvino_root_cause/openvino_rc_p0_baseline.{json,csv}, results/openvino_cpu/openvino_bridge.json, results/comparison/summary.csv, results/comparison/embedding_accuracy.csv, scripts/bench_openvino_bridge.py (unmodified vs HEAD), results/openvino_cpu/openvino_properties.json
Concerns: (1) md cited "3.90 GHz under encode" — that figure belongs to ComfyUI CPU runs, not the OV bridge; (2) P3 p50 delta vs historical p50 is -3.19% (vs -1.80% against historical mean, the metric actually quoted); (3) "no thermal throttle" only indirectly supported by sustained ~3.1 GHz clocks.
Resolution: (1) fixed in phase0_baseline.md (now cites ~3.1–3.2 GHz from this run's own resources CSV); (2) verifier judged it within run-to-run variability (historical P3 spread 6.8%); comparison is explicitly framed against historical means; (3) md now labels run-time temperature as unavailable rather than claiming no throttle.

## Phase 1 — CPU backend linkage
Verifier: read-only subagent (agent_00807cb8), re-ran ldd/readelf/nm/strings/grep -aob/sha256sum, checked /proc/self/maps device->plugin proof
Verdict: PASS
Evidence checked: results/openvino_root_cause/backend_linkage.txt (wheel lib listing, ldd, readelf NEEDED, oneDNN static-embed strings, ISA suffix table, JIT dump template, SHA256 of 3 libs, TBB 2021.13.1, no OpenMP, no libdnnl anywhere)
Concerns: "(13 refs)" for VERBOSE counts prefixed error-message mentions rather than suffix-only refs — cosmetic annotation nuance.
Resolution: kept as is with this note; substantive claims (knob compiled in, suffix-only storage + runtime prefix) verified at offsets VERBOSE@57439004, JIT_DUMP@57442977, MAX_CPU_ISA@57443503. File does not overclaim oneDNN version (deferred to Phase 3) nor execution (deferred to later phases).

## Phase 2 — runtime model + profiling
Verifier: read-only subagent (agent_8db75fd4), recomputed % shares from CSVs, counted node tuples from runtime_nodes JSON, AST-diffed workload replication vs committed bridge script
Verdict: PASS
Evidence checked: profiling_default_P{1,2,3}.csv (1294 rows each, 91.35/92.40/92.20% brgemm_avx512_bf16, zero vnni exec_types), runtime_nodes_default.json (253 FullyConnected all brgemm_avx512_bf16/u8/bf16-out, rotary_emb lone f32 brgemm), profiling_default_summary.json (properties + warm p50s), script workload fidelity vs bench_openvino_bridge.py (byte-identical template/feeds)
Concerns: (1) "1294 executed nodes" wording — 1037-1051 rows are EXECUTED, rest NOT_RUN; (2) lm_head share is 6.88% of total node time (7.53% only of bf16-GEMM subtotal); (3) 35/36 SDPA are undef_bf16, one avx512_bf16.
Resolution: (1)+(2) fixed in phase2_findings.md; (3) immaterial simplification, noted here. Instruction-level VNNI questions correctly deferred to Phases 3/6/7.

## Phase 3 — oneDNN verbose
Verifier: read-only subagent (agent_ee17a470), raw-log recount + live sequential reproduction of crash modes (all, 1, AVX2-capped, BF16-capped, dq0) and silent mode (create)
Verdict: PASS
Evidence checked: results/openvino_isa/onednn_verbose_exec_crash.log (header oneDNN v3.13.0 + threadpool nthr:16 + isa string; 79 dispatch rejections + 1 cache_miss; crash node), onednn_default.log (silent create-mode run), onednn_default_summary.json vs raw, md overclaim scope
Concerns: (1) parser miscounted template line as exec line; (2) filename onednn_default.log actually a create-mode run (disclosed in md); (3) AVX2-capped run shows 225 dispatch rejections vs 79 (expected, different dispatch path).
Resolution: (1) parser fixed (template now parsed separately), summary regenerated; (2) kept, disclosed; (3) consistent with ISA-dependent dispatch, no md claim affected.

## Phase 4 — ISA ablation matrix
Verifier: read-only subagent (agent_d3f16eb2), recomputed all medians/ratios from isa_matrix.csv, verified dispatch tables across 18 runtime_nodes files, checked env records + failure logs
Verdict: PASS
Evidence checked: isa_matrix.csv/json (54 rows, medians + 4 comparison ratios exact), 18x profiling_isa_*_summary.json (12 iters, env per config), 18x runtime_nodes_isa_*.json (253 FC, u8, zero vnni primitive; AVX512_F32 vs VNNI512_F32 dispatch bit-identical; DEFAULT vs BF16CEIL bit-identical), isa_AVX2_NOT_RUNNABLE.log, runner script protocol
Concerns: (1) md called the P1/P3 VNNI-ceiling ~7.5-8% gap "noise" — it is consistent and non-overlapping, real-but-unexplained with identical dispatch; (2) failure log is the AVX2+bf16 probe, cited on the AVX2+f32 row; (3) minor stat-range wording + inverted sanity-row ratio label; (4) runnability probes not all preserved as logs.
Resolution: (1) md reworded to "consistent gap, unexplained, JIT codegen comparison deferred to Phase 6; VNNI never faster in any case"; (2) attribution fixed in md; (3) fixed; (4) noted — probe failures reproduce deterministically from the documented commands; dispatch tables are the preserved evidence.

## Phase 5 — dynamic quantization ablation
Verifier: read-only subagent (agent_d4c1444f), recomputed medians/percentages, recomputed all 18 cosines in float64 from npys, diffed dispatch tables, recomputed per-node stats from CSVs
Verdict: PASS
Evidence checked: dynamic_quant_matrix.csv, profiling_dq_*_summary.json (env knob values 0/32/64/128, 12 iters), runtime_nodes_dq_*.json (byte-identical dispatch, 253 FC, 0 vnni/quant names), hidden_cosine_summary.json + 18 npys vs bf16 refs (monotonic degradation), profiling CSVs per-node stats, md hypothesis framing
Concerns: (1) "-9% P2" side note actually -10.9%; (2) FullyConnectedCompressed (profile node_type) vs FullyConnected (rt layerType) naming nuance; (3) cosine captured for r1 only of default/g64/g128 (g0 runs bit-identical = deterministic).
Resolution: (1) fixed to -11%; (2) same nodes, both labels valid, noted; (3) consistent with presentation; determinism itself is evidence (same inputs -> same outputs across processes).

## Phase 6 — JIT binary instruction evidence
Verifier: read-only subagent round 1 (agent_2b5bbba2) — Verdict: FAIL (parser missed {vex}-prefixed 256-bit vpdpbusd -> avx2vnnif32 wrongly reported 0 VNNI; "64/126 SHA differ" actually 32/126; dq0 "144 each" imprecise; dangling Phase 7 ref)
Resolution: analyze_jit_vnni.py fixed (parse_mnem strips {vex}/{evex}, classify splits VNNI_512/%zmm vs VNNI_256/%ymm); all 6 analyses regenerated (avx2vnnif32 now 12 kernels x 96 = 1152 VNNI_256); md rewritten (width-only loss, correction note, 32/126, phase7 ref); phase7_perf_findings.md created.
Verifier: read-only subagent round 2 (agent_8343eab4) — re-derived all manifests byte-identical, raw objdump recount (12x96 ymm vpdpbusd), SHA diff 32/126 recomputed from 794 files, md consistency confirmed.
Verdict (round 2): PASS
Evidence checked: 6x jit_dump_manifest.json + summaries + hits, raw objdump of brgemm kernels (default 12x208 zmm vpdpbusd with vpbroadcastd group scales; dq0 0 VNNI/18 bf16; dq128 12x832; avx512f32 12x304; vnni512f32 12x208; avx2vnnif32 12x96 ymm), .gitignore *.bin exclusion, no Level D claim.
Concerns: classify's %ymm substring test could misclassify a mixed ymm/zmm EVEX form — none exists in any of the 794 dumped binaries (verified).
Resolution: noted as parser limitation; data unaffected.

## Phase 7/8 — perf hotspot + PMU events
Verifier: folded into Phase 6 round-2 subagent (checked phase7_perf_findings.md exists, documents perf_event_paranoid=4, no passwordless sudo, kernel-tools mismatch, no Level D claim, Phase 8 anti-fabrication statement)
Verdict: PASS (as documentation of a blocked measurement; Level D evidence honestly recorded NOT OBTAINED)

## Phase 9 — weight compression / memory traffic
Verifier: read-only subagent (agent_a91bc704), recomputed all arithmetic, stat'ed model files, parsed safetensors header to re-derive 8,190,427,136/622,329,856, checked lm_head compression in profiling CSV, validated model dims from config.json, checked disclaimer framing
Verdict: PASS
Evidence checked: phase9_memory_traffic.{md,json} arithmetic (8,474,066,944 B; 34.7 GB; 4.1x; 661/1373/3135 GFLOP; 46.3/22.8/12.6 GB/s implied; 3.61/3.69/4.65 TFLOP/s), file sizes byte-exact, lm_head in 253 FullyConnectedCompressed, hidden 4096/intermediate 12288/vocab 151936/36 layers, implied-not-measured framing
Concerns: (1) A2 latency provenance = run2 means not run1-3 means (within 1%); (2) int8 payload numbers cited to runtime_path.md but actually from README/RESULTS + safetensors header; (3) "~50 TOPS" envelope overstated (~12.7 TOPS with stated assumptions -> utilization 28-37% not <=10%); (4) "25-171 tokens" wording vs 39-185 computed; (5) 8.2 GiB vs GB.
Resolution: all five fixed in md+json (provenance corrected, envelope recomputed to ~12.7 TOPS with utilization ~28-37%, conclusion "not compute-saturated" unchanged and robust).

## Phase 10 — controlled OV FP16 comparison
Verifier: read-only subagent (agent_e42c5ad0), recomputed medians from 3 fp16 run JSONs, diffed dispatch tables, independently recomputed cosines in float64, byte-compared the two config.json files, checked download provenance and speedup arithmetic
Verdict: PASS
Evidence checked: profiling_fp16_r{1,2,3}_summary.json (model_dir=fp16 checkout, env unchanged, 12 iters, n_tokens 39/81/185), runtime_nodes fp16(bf16) vs dq_default(u8) both 253x brgemm_avx512_bf16, hidden cosines recomputed exact, config.json byte-equal between fp16-ov and int8-ov models, dl_fp16_ov.log provenance, 1.41/1.67/1.45x arithmetic
Concerns: (1) "spread <=1.4%" understated (actual max 1.56% for fp16 P2 under repo formula); (2) cosine upper bound 0.99992 not 0.99990; (3) p50 = upper-median convention (consistent across all runs).
Resolution: (1)+(2) fixed in md; (3) internally consistent convention. 1.41-1.67x correctly framed as COMBINED compression+engine factor.

## Phase 11-12 — factor attribution + VNNI quantification
Verifier: read-only subagent (agent_42a9038f), recomputed every ratio from raw phase artifacts + committed summary.csv + GPU run JSONs, re-checked both jit manifests for the isolation-invalidity claim, audited confidence-label tiering
Verdict: PASS
Evidence checked: all numbers in phase11_12_factor_attribution.md (1.41/1.67/1.45; +23/42/43%; 1.43/1.38/1.55; 1.23/1.42/1.43; 1.77x A2->A1; 70/36/22x; 40/20/12x; 2.7-3.3x GPU), Level B+C+E / no-Level-D verdict wording, decomposition arithmetic, label discipline
Concerns: (1) 70x (dq_default denominator) vs headline 69x (committed bridge mean) mixed; (2) row-2 ratio direction convention unstated; (3) factor 6 CONFIRMED vs factor 5 STRONG for committed data (both disclosed); (4) pre-existing summary.csv GPU rows appear to have warm_mean/warm_p50 transposed vs raw JSONs (old-repo data quirk, not introduced by this investigation).
Resolution: (1)+(2) fixed in md (denominators and direction now explicit); (3) disclosed in cells; (4) left untouched per Rule 3 (existing benchmark records), noted here and flagged for the README-consistency pass.

## Phase 15 — final independent audit
Verifier: read-only subagent (agent_233836f8) with OpenVINO-plugin-maintainer + oneDNN-perf-engineer + AMD-Zen-optimizer persona; 14-point challenge list; spot-recomputed ~20 report numbers and all 15 cosines from npys
Verdict: FINAL VERDICT: PASS (2 pre-merge issues + 5 non-blocking observations)
Issues fixed before commit: (13) "9-21%" DQ-g128 claim corrected to 11-21% in README+report (raw: 10.9-21.3%); (14) truncated RESULTS.md banner sentence completed.
Non-blocking fixed: phase4 stale conclusion 1 marked [SUPERSEDED by Phases 5+6] with explanation; phase10 table cosine range 0.99992; hidden_cosine_summary.json regenerated for all tags; 0-byte runtime_model_default.bin removed; reproducibility path gap fixed (--out-dir added to investigate script; ISA runner now writes results/openvino_isa directly, skip-check path aligned).
Notable upheld points: no Level-A-as-execution reasoning; no W8A8 conflation; activation quantization proven via numerics+machine code; ISA env application verified; no perf-based claims; GPU findings properly scoped; VNNI attribution framed as engine increment with invalid-isolation warning; checkpoint record judged "self-critical and credible" incl. the genuine Phase-6 round-1 FAIL.


===========================================================================
HOST B CHECKPOINTS (2x EPYC 9334, Zen 4) — verbatim from commit 9c3fb79
===========================================================================

# Root-cause investigation checkpoints (HOST B)

Format per task Rule 2. HOST B = the machine this investigation runs on
(2× AMD EPYC 9334, Zen 4); HOST A = the original benchmark host
(Ryzen AI Max+ PRO 395, Zen 5). See environment/root_cause_host_b.md.

---

## Phase 0 — Baseline reproduction on HOST B

Verifier: independent read-only subagent (2026-09-29, session ses_f134e39...)
Verdict: PASS
Evidence checked: git-status integrity of tracked benchmark data; 3 baseline JSONs
(version/exec devices/threads/hint); P1 warm_p50 0.2443/0.1834/0.1862 (median 0.186);
cosine_vs_hostA_npy=1.0000001 every prompt; model SHA256 cross-match; host-divergence
doc; live reproduction of oneDNN v3.13.0 verbose header; hidden_mean vs HOST-A npy
means agree to ~1 ulp → genuine recompute, not file copy.
Concerns: (a) checkpoint text said p50 range "0.183-0.267" — actual max p50 0.2443
(0.267 was proc-1 mean of an earlier aborted partial run); corrected here.
(b) runtime_model_*.bin exports are 0 bytes (OV runtime export omits weights data —
expected; XML + summaries populated). (c) verbose log not yet archived — Phase 3.
Resolution: concerns addressed; corrected number recorded; proceed.

Summary of what was done:
- Repo recreated from upstream tarball HEAD b1037bb (git-over-HTTPS blocked by
  egress proxy; GitHub API used; content committed as snapshot, byte-identical).
- openvino==2026.4.0 installed from PyPI (exact version string match with HOST A:
  `2026.4.0-22959-99c81491cc3-releases/2026/4`).
- Model re-downloaded from ModelScope; **both .bin SHA256 match HOST A committed
  hashes exactly** (results/openvino_root_cause/host_b_model_sha256.txt).
- ComfyUI qwen25_tokenizer files fetched from ComfyUI master tarball (GitHub API).
- Baseline: scripts/investigate_openvino_runtime.py --mode bench, 3 fresh
  processes, taskset 0-15 (16 physical cores socket 0), threads=16,
  PERFORMANCE_HINT=LATENCY (NUM_STREAMS=1 by LATENCY default), warm 2 + 10 iters.
- HOST B warm p50 (median of 3 fresh processes):
  P1 0.186 s (HOST A 0.187) · P2 0.421 s (HOST A 0.377) · P3 0.886 s (HOST A 0.687)
- Cross-host numerical validation: cosine vs committed HOST A hidden_*_bridge.npy
  = 1.0000001 for P1/P2/P3 → identical token semantics + same model artifact.
- host caveat recorded: absolute latencies are HOST B's own; HOST A numbers are
  never re-measured or overwritten.

## Phase 1 — CPU backend linkage

Verifier: independent read-only subagent (2026-09-29, session ses_f134e39...)
Verdict: PASS
Evidence checked: backend_linkage.txt (ldd no libdnnl, readelf NEEDED, 192 vendored
thirdparty/onednn paths, SHA256); live ldd/strings reproduction; live verbose header
oneDNN v3.13.0 + runtime:threadpool + isa string.

Summary:
- results/openvino_root_cause/backend_linkage.txt: CPU plugin =
  libs/libopenvino_intel_cpu_plugin.so; ldd/readelf show NO libdnnl.so.2 —
  oneDNN is statically vendored (debug paths: src/plugins/intel_cpu/thirdparty/onednn).
- oneDNN version via verbose header: **oneDNN v3.13.0 (commit 1289c3b65dd6a119a5ed12a816517d9c3a21d81b)**.
- Threading runtime: bundled oneTBB (libtbb.so.12, libtbbmalloc); verbose says
  `cpu,runtime:threadpool`.
- No OpenMP. SHA256 of plugin/runtime/tbb recorded.

---

## Phase 2 — Runtime model + OpenVINO profiling

Verifier: independent read-only subagent ses_f1329b06 (round 1: FAIL on 3 issues) → ses_f132475c (round 2, after fixes)
Verdict: PASS (round 2)
Evidence checked: 253×brgemm_avx512_bf16 = 92.4–93.7% of profiled time across 6 CSVs;
DQ property =32; verbose raw strings; isa_matrix medians recomputed from raw;
err-config failure modes vs raw logs; JIT JSONs vs independent objdump spot checks;
provenance of default dump (ONEDNN_MAX_CPU_ISA empty).
Summary:
- investigate_openvino_runtime.py enables ov.properties.enable_profiling and
  dumps per-node get_profiling_info() (profiling_P*_*.csv) + get_runtime_model()
  (runtime_model_*.xml + summary JSON) with supported-properties census.
- All 3 fresh baseline procs: 253/253 fully-connected-class nodes execute as
  execType `brgemm_avx512_bf16` (254 brgemm nodes incl. lm_head), ~92% of
  total inference time. No int8-named exec types in the hot path.
- supported_properties includes DYNAMIC_QUANTIZATION_GROUP_SIZE (readable,
  default 32), INFERENCE_PRECISION_HINT (default bfloat16), KV cache precision
  u8, TBB_PARTITIONER=STATIC, RUNTIME_REQUIREMENTS isa string.
- IR census: 506 U8 weight constants (7.570 GB) + 506 Convert→f16 + Multiply.

## Phase 3 — oneDNN verbose

Verifier: independent read-only subagent ses_f1329b06 → fixes → ses_f132475c
Verdict: PASS (round 2)
Summary (results/openvino_isa/onednn_verbose_findings.md):
- Static oneDNN v3.13.0; verbose partially usable: ONEDNN_VERBOSE=profile_exec
  emits header + first exec lines (u8 weights reordered to blocked
  AB4b32a4b 4096x4096; f32→bf16 activation 1x39x4096) then the process aborts
  mid first inference; ONEDNN_VERBOSE=1/2 fail at compile
  (Reorder u8_i32 primitive creation); action/create/forward/oneline produce
  no output (not accepted). DNNL_VERBOSE=1 fails like =1.
- Consequence: dispatch proof carried by OV profiling execTypes (Level B),
  JIT machine code (Level C), ISA/DQ counterfactuals (Level E).

## Phase 4 — ISA ablation

Verifier: independent read-only subagent ses_f1329b06 → fixes → ses_f132475c
Verdict: PASS (round 2)
Summary (results/openvino_isa/isa_matrix.csv, 3 procs × 2 warm + 10 measured,
P1/P2/P3, CV ≤ 3%):
- default 0.1922/0.4375/0.9311 s; isa_bf16 ceiling 0.1869/0.4181/0.8821 (≈ equal,
  within proc noise); vnni_f32 0.1971/0.4466/0.9618; core_f32 0.1773/0.4422/0.9039;
  dq0 0.2632/0.6425/1.3067; dq128 0.1565/0.3786/0.7393.
- AVX2 / AVX2_VNNI / bare AVX512_CORE ceilings: process fails at the first
  FullyConnectedCompressed node ("could not create a primitive descriptor") —
  the compressed-weight FC kernels need at least AVX512_CORE_VNNI on this
  build; no AVX2 fallback exists.
- bare AVX512_CORE_VNNI ceiling: DIFFERENT mechanism — FCCompressed compiles
  (VNNI alone suffices for the int8 DQ kernels; further evidence the FC hot
  path is VNNI-based), the run then dies at first inference in
  ScaledDotProductAttentionWithKVCache: "brgemm bf16 kernel could only be used
  above avx512_bf16" (the attention bf16 brgemm needs AVX512_CORE_BF16).
- CRITICAL: ONEDNN_MAX_CPU_ISA=AVX512_CORE does NOT remove VNNI on this build —
  JIT dump still contains vpdpbusd kernels. The ceiling only gates bf16.
  Therefore "AVX512_CORE vs AVX512_CORE_VNNI" cannot serve as a no-VNNI
  counterfactual here; the valid counterfactual is DQ=0 (vdpbf16ps only).

## Phase 5 — Dynamic quantization ablation

Verifier: independent read-only subagent ses_f1329b06 → fixes → ses_f132475c
Verdict: PASS (round 2)
Summary (investigate_dq_*.json + matrix):
- DYNAMIC_QUANTIZATION_GROUP_SIZE is supported and settable; requested value
  is reflected by get_property (override verified applied, not ignored).
- DQ=0: same execType label but ZERO vpdpbusd in JIT dump (18 vdpbf16ps
  kernels instead), gemm time 241.6 ms vs 170.7 ms (P1 probe runs). End-to-end:
  matrix medians 1.37× slower (0.2632 vs 0.1922); single-probe runs
  0.2638/0.1872 = 1.41×. Both raw sets committed.
- DQ=32 (default): 12 brgemm kernels × 208 vpdpbusd + 6
  brgemm_src_quantization_kernel_t instances.
- DQ=64: 0.166/0.394/0.869; DQ=128: 0.157/0.379/0.739 (faster than default).
- Cosine vs HOST A npy unchanged (DQ path numerically equivalent, cos 1.0000001).

## Phase 6 — JIT machine-code evidence

Verifier: independent read-only subagent ses_f1329b06 → fixes → ses_f132475c
Verdict: PASS (round 2)
Summary (jit_vnni_hits*.txt/json, jit_instruction_summary*.txt):
- DEFAULT dump: 137 binaries; 12 `jit_brgemm_kernel_t` with 208 vpdpbusd each
  (+vcvtneps2bf16 epilogue); 6 src_quantization kernels; separate pure-bf16
  vdpbf16ps kernels also present.
- DQ=0 dump: 141 binaries, ZERO vpdpbusd, vdpbf16ps only.
- AVX512_CORE_VNNI+f32 hint dump: 12 kernels × 208 vpdpbusd, no vdpbf16ps.
- AVX512_CORE+f32 hint dump: 12 kernels × 304 vpdpbusd (ceiling does not
  remove VNNI — discovery).
- FP16-model dump: 132 binaries, ZERO vpdpbusd, 18 vdpbf16ps.

## Phase 7/8 — perf hotspot / PMU census

Verifier: independent read-only subagent ses_f131e0fb
Verdict: PASS (arithmetic and file evidence re-derived independently; one wording fix applied: fp16/int8 byte ratio is 1.999x, not '2.000x exact')
Summary: NOT AVAILABLE — perf binary absent, perf_event_paranoid=4, no
CAP_PERFMON; exact errors recorded in results/openvino_isa/phase7_8_perf_unavailable.md.
Evidence level capped at B+C+E; reproduction command documented for perf-enabled hosts.

## Phase 9 — weight-compression / memory traffic

Verifier: independent read-only subagent ses_f131e0fb
Verdict: PASS (arithmetic and file evidence re-derived independently; one wording fix applied: fp16/int8 byte ratio is 1.999x, not '2.000x exact')
Summary: results/openvino_isa/memory_traffic_accounting.md — exact byte
counts (7.573.989.065 B int8 bin; FP16 control exactly 2.000×; lm_head 622 MB
= 8.2% of bytes), effective weight-stream rates (P1: 39–48 GB/s int8 paths,
57–65 GB/s 2-byte-weight paths; HOST A ≈ 40.5 GB/s), RSS accounting
(14.8 GiB int8 all DQ configs; 28.7 GiB FP16).

## Phase 10 — FP16 OpenVINO control

Verifier: independent read-only subagent ses_f131e0fb
Verdict: PASS (arithmetic and file evidence re-derived independently; one wording fix applied: fp16/int8 byte ratio is 1.999x, not '2.000x exact')
Summary: OpenVINO/Qwen3-VL-8B-Instruct-fp16-ov (same export pipeline, same
architecture) downloaded from ModelScope to /root/models (outside repo; SHA256
in results/openvino_root_cause/fp16_control_sha256.txt — to be written).
Same bridge semantics (cos vs HOST-A int8 npy = 0.9971, within the repo's
known int8-vs-reference range). 3 procs: P1 0.232 / P2 0.645 / P3 1.149 s;
RSS 28.7 GiB; execType brgemm_avx512_bf16 but ZERO vpdpbusd in JIT dump →
int8+DQ-VNNI default is 1.21×/1.47×/1.23× faster than the FP16 control.

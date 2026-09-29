#!/usr/bin/env python3
"""Aggregate the ISA/DQ matrix + profiling evidence into the root-cause tables.

Inputs (all produced by investigate_openvino_runtime.py / run_openvino_isa_matrix.sh):
  results/openvino_isa/isa_matrix.csv
  results/openvino_isa/jit_vnni_hits*.json
  results/openvino_root_cause/profiling_P1_*.csv (default vs dq0 vs dq128)
  results/openvino_root_cause/investigate_*.json

Outputs:
  results/openvino_root_cause/root_cause_factor_table.md
  results/openvino_root_cause/vnni_contribution.json
"""
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISA = os.path.join(ROOT, "results", "openvino_isa")
RC = os.path.join(ROOT, "results", "openvino_root_cause")


def load_matrix():
    rows = []
    p = os.path.join(ISA, "isa_matrix.csv")
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["prompt"] == "FAILED":
                rows.append({"config": r["config"], "proc": r["proc"], "failed": True, "rc": r["rc"]})
                continue
            rows.append({**r, "failed": False,
                         "p50": float(r["warm_p50_s"]), "mean": float(r["warm_mean_s"]),
                         "std": float(r["warm_std_s"]), "min": float(r["warm_min_s"]),
                         "max": float(r["warm_max_s"]), "cv": float(r["cv"])})
    return rows


def config_stats(rows, cfg, prompt):
    vals = [r["p50"] for r in rows if r["config"] == cfg and not r["failed"] and r["prompt"] == prompt]
    if not vals:
        return None
    means = [r["mean"] for r in rows if r["config"] == cfg and not r["failed"] and r["prompt"] == prompt]
    cvs = [r["cv"] for r in rows if r["config"] == cfg and not r["failed"] and r["prompt"] == prompt]
    return {"n_procs": len(vals), "p50_of_p50": statistics.median(vals),
            "min": min(vals), "max": max(vals), "mean_of_means": statistics.mean(means),
            "cv_range": [min(cvs), max(cvs)],
            "proc_p50s": [round(v, 4) for v in vals]}


def failed_configs(rows):
    out = defaultdict(list)
    for r in rows:
        if r.get("failed"):
            out[r["config"]].append(r["rc"])
    return dict(out)


def main():
    rows = load_matrix()
    prompts = ["P1", "P2", "P3"]
    cfgs = ["default", "isa_bf16", "vnni_f32", "core_f32", "dq0", "dq128"]
    table = {}
    for cfg in cfgs:
        for p in prompts:
            s = config_stats(rows, cfg, p)
            if s:
                table[f"{cfg}/{p}"] = s

    contrib = {"matrix_table": table, "failed_configs": failed_configs(rows)}

    # comparisons (median-of-3-process p50 basis)
    def ratio(a, b, p):
        if f"{a}/{p}" in table and f"{b}/{p}" in table:
            return round(table[f"{a}/{p}"]["p50_of_p50"] / table[f"{b}/{p}"]["p50_of_p50"], 4)
        return None

    contrib["comparisons"] = {}
    for p in prompts:
        def sr(a, b):  # speedup of a over b (latency_b / latency_a)
            if f"{a}/{p}" in table and f"{b}/{p}" in table:
                return round(table[f"{b}/{p}"]["p50_of_p50"] / table[f"{a}/{p}"]["p50_of_p50"], 4)
            return None
        contrib["comparisons"][p] = {
            "dq_vnni_path_vs_dq0_bf16_speedup": sr("default", "dq0"),
            "dq128_vs_dq0_bf16_speedup": sr("dq128", "dq0"),
            "dq128_vs_default_dq32_speedup": sr("dq128", "default"),
            "default_vs_isa_bf16_ceiling_speedup": sr("default", "isa_bf16"),
            "default_vs_vnni_f32_speedup": sr("default", "vnni_f32"),
            "vnni_f32_vs_core_f32_speedup": sr("vnni_f32", "core_f32"),
        }

    # JIT census
    jit = {}
    for name, path in [("default_dump", os.path.join(ISA, "jit_vnni_hits.json")),
                       ("dq0_dump", os.path.join(ISA, "jit_vnni_hits_dq0.json")),
                       ("vnni_f32_dump", os.path.join(ISA, "jit_vnni_hits_vnni_f32.json"))]:
        if os.path.exists(path):
            d = json.load(open(path))
            jit[name] = {k: d[k] for k in ("n_binaries", "n_binaries_with_vnni",
                                           "total_instruction_counts", "files_with_vnni")}
    contrib["jit_census"] = jit

    # profiling gemm-time table
    prof = {}
    for f in sorted(glob.glob(os.path.join(RC, "profiling_P1_*.csv"))) + \
               sorted(glob.glob(os.path.join(ISA, "profiling_P1_*.csv"))):
        tag = os.path.basename(f)[len("profiling_P1_"):-len(".csv")]
        rr = list(csv.DictReader(open(f)))
        gemm = [(r["node_name"], float(r["real_time"]), r["exec_type"]) for r in rr
                if "brgemm" in r.get("exec_type", "")]
        tot = sum(float(r["real_time"]) for r in rr if r.get("real_time"))
        c = Counter(e for _, _, e in gemm)
        prof[tag] = {"gemm_nodes": len(gemm), "gemm_ms": round(sum(t for _, t, _ in gemm) / 1000, 1),
                     "all_ms": round(tot / 1000, 1), "gemm_exec_types": dict(c)}
    contrib["profiling_gemm_P1"] = prof

    with open(os.path.join(RC, "vnni_contribution.json"), "w") as f:
        json.dump(contrib, f, indent=1)

    # human-readable factor table (markdown)
    lines = ["# Root-cause factor evidence (machine-generated, HOST B)", "",
             f"generated: {__import__('datetime').datetime.now().isoformat()}", "",
             "## ISA/DQ matrix (warm p50, median of fresh processes)", "",
             "| config | P1 | P2 | P3 |", "|---|---:|---:|---:|"]
    for cfg in cfgs:
        cells = []
        for p in prompts:
            s = table.get(f"{cfg}/{p}")
            cells.append(f"{s['p50_of_p50']:.4f}s (n={s['n_procs']})" if s else "—")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    fc = failed_configs(rows)
    if fc:
        lines += ["", "Non-runnable ISA-ceiling configs (dispatch evidence):"]
        failmode = {
            "err_avx2": "FullyConnectedCompressed could not create a primitive descriptor (needs >=AVX512_CORE_VNNI for its int8 DQ kernels)",
            "err_avx2_vnni": "FullyConnectedCompressed could not create a primitive descriptor",
            "err_core": "FullyConnectedCompressed could not create a primitive descriptor",
            "err_vnni": "FCCompressed compiles (VNNI suffices for the int8 DQ kernels); model fails at first ScaledDotProductAttentionWithKVCache: 'brgemm bf16 kernel could only be used above avx512_bf16' (attention selected bf16)",
        }
        for k, v in fc.items():
            lines.append(f"- `{k}`: process exit non-zero — {failmode.get(k, 'see stdout log')}")
    lines += ["", "## Key ratios", ""]
    for p in prompts:
        for k, v in contrib["comparisons"][p].items():
            if v is not None:
                lines.append(f"- {p} {k}: {v}")
    with open(os.path.join(RC, "root_cause_factor_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[saved] {RC}/vnni_contribution.json")


if __name__ == "__main__":
    main()

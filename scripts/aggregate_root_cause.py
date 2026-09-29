#!/usr/bin/env python3
"""Aggregate the root-cause investigation artifacts.

Phase 4: ISA matrix (results/openvino_isa/profiling_isa_*_summary.json) ->
  isa_matrix.csv / isa_matrix.json + pairwise speedup comparisons.
Phase 5: dynamic-quantization matrix (results/openvino_root_cause/
  profiling_dq*_summary.json) -> dynamic_quant_matrix.csv.
Dispatch verification: per config, the FullyConnected primitiveType distribution
from runtime_nodes_<tag>.json (proves ONEDNN_MAX_CPU_ISA changed dispatch).
"""
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISA_DIR = os.path.join(ROOT, "results", "openvino_isa")
RC_DIR = os.path.join(ROOT, "results", "openvino_root_cause")

ISA_ORDER = ["DEFAULT", "BF16CEIL", "DEFAULT_F32", "VNNI512_F32", "AVX512_F32",
             "AVX2VNNI_F32"]


def load_summaries(pattern, tag_re):
    rows = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(tag_re, os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            d = json.load(f)
        rows.append((m.group(1), m.group(2), d))
    return rows


def fc_dispatch(tag_dir, tag):
    """FullyConnected primitiveType counter from runtime_nodes_<tag>.json."""
    p = os.path.join(tag_dir, f"runtime_nodes_{tag}.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        nodes = json.load(f)
    c = Counter((n.get("layerType"), n.get("primitiveType"), n.get("runtimePrecision"),
                 n.get("outputPrecisions")) for n in nodes if n.get("layerType") == "FullyConnected")
    return {str(k): v for k, v in c.most_common()}


def stats_from(d, pid):
    s = d["prompts"][pid]
    it = s["iterations"]
    return {
        "n_tokens": s["n_tokens_full"],
        "p50_s": s["warm_p50_s"],
        "mean_s": s["warm_mean_s"],
        "std_s": s["warm_std_s"],
        "cv": s["cv"],
        "min_s": s["warm_min_s"],
        "max_s": s["warm_max_s"],
        "n_iters": len(it),
        "compile_s": d["meta"]["compile_s"],
    }


def agg_matrix(rows, key_extract):
    """rows: (config, run, summary_dict). Aggregate per config+prompt across runs."""
    per = defaultdict(list)   # (config,prompt) -> [stats per run]
    dispatch = {}
    for config, run, d in rows:
        for pid, st in d["prompts"].items():
            per[(config, pid)].append(stats_from(d, pid))
        if run == "r1":
            tag = key_extract(config, run)
            dispatch[config] = fc_dispatch(ISA_DIR if tag.startswith("isa_") else RC_DIR, tag)
    out = {}
    for (config, pid), runs in per.items():
        p50s = [r["p50_s"] for r in runs]
        means = [r["mean_s"] for r in runs]
        all_iters = []
        # merge iterations for pooled dispersion
        out[(config, pid)] = {
            "runs": len(runs),
            "p50_median_of_runs_s": statistics.median(p50s),
            "p50_per_run_s": p50s,
            "mean_of_means_s": statistics.mean(means),
            "pooled_min_s": min(r["min_s"] for r in runs),
            "pooled_max_s": max(r["max_s"] for r in runs),
            "run_p50_spread_pct": (max(p50s) - min(p50s)) / min(p50s) * 100,
            "mean_cv_within_run": statistics.mean(r["cv"] for r in runs),
            "compile_s_runs": [round(r["compile_s"], 3) for r in runs],
        }
    return out, dispatch


def speedup(a, b, agg, pid):
    """latency(a)/latency(b) using median-of-runs p50."""
    try:
        la = agg[(a, pid)]["p50_median_of_runs_s"]
        lb = agg[(b, pid)]["p50_median_of_runs_s"]
        return la / lb, la, lb
    except KeyError:
        return None


def main():
    # ---- Phase 4 ISA matrix ----
    rows = load_summaries(os.path.join(ISA_DIR, "profiling_isa_*_summary.json"),
                          r"profiling_isa_(.+?)_r(\d+)_summary\.json")
    rows = [(c, f"r{r}", d) for c, r, d in rows]
    agg, dispatch = agg_matrix(rows, lambda c, r: f"isa_{c}_{r}")
    configs = sorted({c for c, _ in agg}, key=lambda c: ISA_ORDER.index(c) if c in ISA_ORDER else 99)

    csv_path = os.path.join(ISA_DIR, "isa_matrix.csv")
    with open(csv_path, "w") as f:
        f.write("config,run,prompt,n_tokens,p50_s,mean_s,std_s,cv,min_s,max_s,compile_s\n")
        for config, run, d in sorted(rows):
            for pid in sorted(d["prompts"]):
                s = stats_from(d, pid)
                f.write(f"{config},{run},{pid},{s['n_tokens']},{s['p50_s']:.6f},"
                        f"{s['mean_s']:.6f},{s['std_s']:.6f},{s['cv']:.4f},"
                        f"{s['min_s']:.6f},{s['max_s']:.6f},{s['compile_s']:.3f}\n")

    comparisons = {}
    for pid in ("P1", "P2", "P3"):
        comparisons[pid] = {}
        pairs = [
            # same ISA (avx512_core_bf16 HW), bf16 vs f32 compute — BF16 unit contribution
            ("BF16_vs_F32_same_ISA", "DEFAULT_F32", "DEFAULT"),
            # f32 compute: AVX-512 width vs AVX2_VNNI (256-bit)
            ("AVX512_vs_AVX2VNNI_f32", "AVX2VNNI_F32", "AVX512_F32"),
            # f32 compute: VNNI-capable ceiling vs non-VNNI ceiling (both f32 kernels)
            ("VNNI512_vs_AVX512_f32", "AVX512_F32", "VNNI512_F32"),
            # DEFAULT vs explicit BF16 ceiling (sanity: should be ~1.0)
            ("DEFAULT_vs_BF16CEIL", "BF16CEIL", "DEFAULT"),
        ]
        for name, a, b in pairs:
            r = speedup(a, b, agg, pid)
            if r:
                sp, la, lb = r
                comparisons[pid][name] = {
                    "latency_a_s": round(la, 6), "latency_b_s": round(lb, 6),
                    "speedup_a_over_b": round(sp, 4),
                    "abs_delta_s": round(la - lb, 6),
                }

    out = {
        "aggregates": {f"{c}|{p}": v for (c, p), v in sorted(agg.items())},
        "fc_dispatch_by_config": dispatch,
        "comparisons_latency_ratio": comparisons,
    }
    with open(os.path.join(ISA_DIR, "isa_matrix.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {csv_path} and isa_matrix.json ({len(rows)} runs)")

    # console digest
    for pid in ("P1", "P2", "P3"):
        print(f"\n== {pid} (median-of-3-runs p50, s) ==")
        for c in configs:
            if (c, pid) in agg:
                a = agg[(c, pid)]
                print(f"  {c:18s} {a['p50_median_of_runs_s']:.4f}  "
                      f"(runs {','.join(f'{x:.4f}' for x in a['p50_per_run_s'])})")
    print("\n== speedups (latency ratio) ==")
    for pid, cc in comparisons.items():
        print(f" {pid}: " + "  ".join(
            f"{k}={v['speedup_a_over_b']:.3f}x" for k, v in cc.items()))

    # ---- Phase 5 dynamic quantization matrix ----
    dq_rows = load_summaries(os.path.join(RC_DIR, "profiling_dq*_summary.json"),
                             r"profiling_(dq[a-z0-9_]+?)_r(\d+)_summary\.json")
    if dq_rows:
        dq_rows = [(c, f"r{r}", d) for c, r, d in dq_rows]
        dagg, ddispatch = agg_matrix(dq_rows, lambda c, r: f"dq{c}_{r}")
        dq_csv = os.path.join(RC_DIR, "dynamic_quant_matrix.csv")
        with open(dq_csv, "w") as f:
            f.write("config,run,prompt,n_tokens,p50_s,mean_s,std_s,cv,min_s,max_s,compile_s\n")
            for config, run, d in sorted(dq_rows):
                for pid in sorted(d["prompts"]):
                    s = stats_from(d, pid)
                    f.write(f"{config},{run},{pid},{s['n_tokens']},{s['p50_s']:.6f},"
                            f"{s['mean_s']:.6f},{s['std_s']:.6f},{s['cv']:.4f},"
                            f"{s['min_s']:.6f},{s['max_s']:.6f},{s['compile_s']:.3f}\n")
        dq_out = {
            "aggregates": {f"{c}|{p}": v for (c, p), v in sorted(dagg.items())},
            "fc_dispatch_by_config": ddispatch,
        }
        with open(os.path.join(RC_DIR, "dynamic_quant_matrix.json"), "w") as f:
            json.dump(dq_out, f, indent=2)
        print(f"\n[saved] {dq_csv} ({len(dq_rows)} dq runs)")
        for (c, pid), a in sorted(dagg.items()):
            print(f"  dq {c:6s} {pid} p50(median of runs)={a['p50_median_of_runs_s']:.4f}")


if __name__ == "__main__":
    main()

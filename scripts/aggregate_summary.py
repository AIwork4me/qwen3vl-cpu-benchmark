#!/usr/bin/env python3
"""Aggregate all benchmark JSONs into results/comparison/summary.{csv,json}.

Reads:
  results/comfy_cpu/comfy_run{1,2,3}.json  (A2 product path, default threads)
  results/comfy_cpu/comfy_t{8,32}.json     (A2 thread sweep)
  results/comfy_cpu/comfy_a1_t{16,8,32}.json (A1-forced true-int8 variant)
  results/openvino_cpu/openvino_{default,t8,t16,t32}.json (standalone)
  results/openvino_cpu/openvino_bridge.json (conditioning bridge)

Emits one row per (route, config, prompt): warm mean/p50/min/max, first,
load, RSS peaks, counters; plus cross-route comparison table rows.
"""
import csv
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "comparison")


def load(path):
    with open(path) as f:
        return json.load(f)


def comfy_rows():
    rows = []
    for jpath in sorted(glob.glob(os.path.join(RES, "comfy_cpu", "comfy_*.json"))) + \
            sorted(glob.glob(os.path.join(RES, "comfy_gpu", "comfy_gpu_*.json"))):
        tag = os.path.basename(jpath)[6:-5]
        if "smoke" in tag:
            continue
        d = load(jpath)
        meta = d["meta"]
        if meta.get("force_quant_mm"):
            route = "comfy_int8_convrot_A1_forced"
        elif "gpu_name" in meta:
            route = "comfy_bf16_radeon_gpu"
        else:
            route = "comfy_int8_convrot_A2_product"
        for pid, pd in d["prompts"].items():
            w = pd["warm_encode_s"]
            rows.append({
                "route": route,
                "config": tag,
                "prompt": pid,
                "n_tokens": pd["n_tokens"],
                "model_load_s": round(meta["model_load_s"], 4),
                "first_encode_s": round(pd["first_encode_s"], 4) if pd["first_encode_s"] else "",
                "warm_mean_s": round(statistics.mean(w), 4),
                "warm_p50_s": round(statistics.median(w), 4),
                "warm_min_s": round(min(w), 4),
                "warm_max_s": round(max(w), 4),
                "peak_rss_gb": round((pd["peak_rss_bytes"] if "peak_rss_bytes" in pd
                                      else pd["interval_stats"]["peak_rss_bytes"]) / 2**30, 3),
                "peak_gpu_alloc_gb": (round(meta["peak_gpu_alloc_bytes"] / 2**30, 3)
                                      if "peak_gpu_alloc_bytes" in meta else ""),
                "gpu_busy_pct_mean": meta.get("gpu_busy_pct_mean", ""),
                "post_load_rss_gb": round(meta.get("post_first_encode_rss_bytes", 0) / 2**30, 3),
                "baseline_rss_gb": round((meta.get("baseline_rss_bytes") or 0) / 2**30, 3),
                "int_mm_calls": pd.get("compute_counters", {}).get("int_mm", ""),
                "dequant_calls": pd.get("compute_counters", {}).get("dequant", ""),
                "dequant_bytes": pd.get("compute_counters", {}).get("dequant_bytes", ""),
                "threads": meta["threads"]["torch_num_threads"],
                "cpu_util_mean_pct": (pd["interval_stats"] or {}).get("proc_cpu_mean_pct"),
                "model_disk_gb": round(meta["model_disk_bytes"] / 1e9, 3),
            })
    return rows


def openvino_rows():
    rows = []
    for jpath in sorted(glob.glob(os.path.join(RES, "openvino_cpu", "openvino_*.json"))):
        tag = os.path.basename(jpath)[9:-5]
        if tag == "properties" or tag.startswith("smoke"):
            continue
        d = load(jpath)
        meta = d["meta"]
        variant = meta.get("variant", "")
        route = "openvino_int8_bridge" if "bridge" in tag else "openvino_int8_standalone"
        for pid, pd in d["prompts"].items():
            if "warm_encode_s" in pd:
                w = pd["warm_encode_s"]; first_s = pd["first_encode_s"]
            else:
                it = pd["iters"]; w = it["warm_s"]; first_s = it["first_s"]
            ntok = pd.get("n_tokens", pd.get("n_tokens_full"))
            rows.append({
                "route": route,
                "config": tag,
                "prompt": pid,
                "n_tokens": ntok,
                "model_load_s": round(meta["model_load_s"], 4),
                "first_encode_s": round(first_s, 4),
                "warm_mean_s": round(statistics.mean(w), 4),
                "warm_p50_s": round(statistics.median(w), 4),
                "warm_min_s": round(min(w), 4),
                "warm_max_s": round(max(w), 4),
                "peak_rss_gb": round(pd["interval_stats"]["peak_rss_bytes"] / 2**30, 3),
                "post_load_rss_gb": round(meta.get("post_load_peak_rss_bytes", 0) / 2**30, 3),
                "baseline_rss_gb": round((meta.get("baseline_rss_bytes") or 0) / 2**30, 3),
                "int_mm_calls": "",
                "dequant_calls": "",
                "dequant_bytes": "",
                "threads": meta.get("threads") if isinstance(meta.get("threads"), int) else meta.get("threads"),
                "cpu_util_mean_pct": (pd.get("interval_stats") or {}).get("proc_cpu_mean_pct"),
                "model_disk_gb": round(8810 / 1000, 3),
                "_variant": variant,
            })
    return rows


def main():
    rows = comfy_rows() + openvino_rows()
    os.makedirs(OUT, exist_ok=True)
    cols = ["route", "config", "prompt", "n_tokens", "model_load_s", "first_encode_s",
            "warm_mean_s", "warm_p50_s", "warm_min_s", "warm_max_s", "peak_rss_gb",
            "peak_gpu_alloc_gb", "gpu_busy_pct_mean",
            "post_load_rss_gb", "baseline_rss_gb", "int_mm_calls", "dequant_calls",
            "dequant_bytes", "threads", "cpu_util_mean_pct", "model_disk_gb"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"WROTE {OUT}/summary.csv ({len(rows)} rows) and summary.json")
    for r in rows:
        print(f"{r['route']:32s} {r['config']:8s} {r['prompt']} warm={r['warm_mean_s']}s "
              f"peak={r['peak_rss_gb']}GiB threads={r['threads']} int8mm={r['int_mm_calls']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 33: auto-generate the e2e primary metrics tables from raw run JSONs.

Never hand-edited: every number traces to results/e2e/<group>/<tag>.json images[]
entries + results/e2e/raw/process_walls.csv.

Outputs (results/e2e/summary/):
  per_image.csv     one row per generated image, all stages + route/dq/prompt/seed
  stats.csv         per (group,route,dq,prompt) warm-image stats (median/mean/std/min/max/CV)
  headline.csv      cold TTFI + warm E2E (median of fresh-process medians) per route
  primary.md        the spec's primary-metrics table + stage table (markdown)
"""
import csv
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E2E = os.path.join(ROOT, "results", "e2e")
SUM = os.path.join(E2E, "summary")
GROUPS = ["ab", "matrix", "throughput", "dq", "quality", "smoke"]


def load_runs():
    runs = []
    for g in GROUPS:
        for f in sorted(glob.glob(os.path.join(E2E, g, "*.json"))):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            r["_group"] = g
            r["_tag"] = os.path.basename(f)[:-5]
            runs.append(r)
    return runs


def load_walls():
    walls = {}
    wf = os.path.join(E2E, "raw", "process_walls.csv")
    if os.path.exists(wf):
        for row in csv.reader(open(wf)):
            if len(row) == 3 and row[0] != "tag":
                walls[row[0]] = (float(row[1]), float(row[2]))
    return walls


def route_name(r):
    m = r["meta"]
    if m["route"] == "ov":
        dq = m.get("dq")
        if dq is None and isinstance(m.get("ov"), dict):
            dq = m["ov"].get("dq_requested", 32)
        return f"ov_dq{dq if dq is not None else 32}"
    return m["route"]


def boot_ci(vals, n=2000, seed=7):
    import random
    rng = random.Random(seed)
    if len(vals) < 3:
        return None
    meds = []
    for _ in range(n):
        s = [rng.choice(vals) for _ in vals]
        meds.append(st.median(s))
    meds.sort()
    return (round(meds[int(0.025 * n)], 3), round(meds[int(0.975 * n)], 3))


def main():
    os.makedirs(SUM, exist_ok=True)
    runs = load_runs()
    walls = load_walls()

    per_image = []
    for r in runs:
        m = r["meta"]
        dqv = m.get("dq")
        if dqv is None and m["route"] == "ov" and isinstance(m.get("ov"), dict):
            dqv = m["ov"].get("dq_requested")
        base = {"group": r["_group"], "tag": r["_tag"], "route": route_name(r),
                "dq": dqv if m["route"] == "ov" else "",
                "steps": m["steps"], "res": f'{m["width"]}x{m["height"]}',
                "aotriton": m.get("aotriton", "?")}
        for im in r.get("images", []):
            row = dict(base)
            row.update({k: im.get(k) for k in (
                "prompt_id", "seed", "iter", "label",
                "tokenize_s", "encode_s", "encode_pos_s", "encode_neg_s", "cond_prep_s",
                "dit_s", "dit_preload_in_window_s", "vae_s", "png_save_s",
                "total_gen_s", "total_save_inclusive_s", "n_tokens_kept",
                "cuda_peak_alloc_gib", "gtt_free_gib_after", "img_sha16", "png",
                "wait_s", "total_from_submit_s", "standalone_s")})
            wd = im.get("windows", {}).get("dit", {})
            row["dit_gpu_busy_mean"] = (wd.get("gpu_busy_pct") or {}).get("mean")
            row["dit_gtt_used_max_gib"] = ((wd.get("gtt_used_bytes") or {}).get("max") or 0) / 2**30 or None
            row["contend_active"] = im.get("contend_active", False)
            per_image.append(row)
    if per_image:
        with open(os.path.join(SUM, "per_image.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            w.writeheader()
            w.writerows(per_image)

    # warm stats per (group, route, steps, res, prompt): iters>=1, skip iter 0 (cold)
    stats = []
    key = lambda r: (r["group"], r["route"], r["steps"], r["res"], r["prompt_id"])
    groups = {}
    for r in per_image:
        if r["iter"] and int(r["iter"]) > 0 and r["total_gen_s"] is not None:
            groups.setdefault(key(r), []).append(r)
    for k in sorted(groups):
        rows = groups[k]
        vals = [float(r["total_gen_s"]) for r in rows]
        enc = [float(r["encode_s"]) for r in rows if r["encode_s"] is not None]
        dit = [float(r["dit_s"]) for r in rows if r["dit_s"] is not None]
        vae = [float(r["vae_s"]) for r in rows if r["vae_s"] is not None]
        st_row = {"group": k[0], "route": k[1], "steps": k[2], "res": k[3], "prompt": k[4],
                  "n": len(vals),
                  "e2e_median_s": round(st.median(vals), 3),
                  "e2e_mean_s": round(st.mean(vals), 3),
                  "e2e_std_s": round(st.stdev(vals), 3) if len(vals) > 1 else "",
                  "e2e_min_s": round(min(vals), 3), "e2e_max_s": round(max(vals), 3),
                  "e2e_cv_pct": round(100 * st.stdev(vals) / st.mean(vals), 2) if len(vals) > 1 else "",
                  "e2e_ci95": boot_ci(vals),
                  "enc_median_s": round(st.median(enc), 3) if enc else "",
                  "dit_median_s": round(st.median(dit), 3) if dit else "",
                  "vae_median_s": round(st.median(vae), 3) if vae else ""}
        stats.append(st_row)
    if stats:
        with open(os.path.join(SUM, "stats.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
            w.writeheader()
            w.writerows(stats)

    # headline: cold TTFI (fresh-process, first image) + warm E2E per route from group=ab
    headline = []
    ab_runs = [r for r in runs if r["_group"] == "ab" and r["meta"]["steps"] == 20
               and f'{r["meta"]["width"]}x{r["meta"]["height"]}' == "1024x1024"]
    by_route = {}
    for r in ab_runs:
        by_route.setdefault(route_name(r), []).append(r)
    for route, rs in sorted(by_route.items()):
        colds, warms, encs, dits, vaes = [], [], [], [], []
        for r in rs:
            ims = r.get("images", [])
            if not ims:
                continue
            first = ims[0]
            colds.append(first["total_gen_s"] + r["loads"]["total_load_s"])
            wt = [im["total_gen_s"] for im in ims[1:]]
            warms.append(st.median(wt) if wt else None)
            encs.append(st.median([im["encode_s"] for im in ims[1:]]) if wt else None)
            dits.append(st.median([im["dit_s"] for im in ims[1:]]) if wt else None)
            vaes.append(st.median([im["vae_s"] for im in ims[1:]]) if wt else None)
            # TTFI from process spawn, uniform epoch anchor:
            # prefer explicit first_image_ready_epoch; fall back to meta.timestamp
            # (UTC wall clock at main() start) + monitor-relative marks
            tag = r["_tag"]
            first_ready = r.get("first_image_ready_epoch")
            anchor = first_ready
            if anchor is None and "mon_t0_epoch" in r:
                anchor = r["mon_t0_epoch"]
            if anchor is None:
                try:
                    from datetime import datetime
                    anchor = datetime.fromisoformat(r["meta"]["timestamp"]).timestamp()
                except Exception:
                    anchor = None
            if anchor is not None and tag in walls:
                m0 = dict(r.get("marks_ns", [])) if isinstance(r.get("marks_ns"), dict) else {}
                t0mark = r.get("marks_ns", {}).get("start")
                im0 = ims[0]
                off = 0.0
                if first_ready is None and t0mark is not None and \
                        m0.get(f"img0_{im0['label']}_start") is not None:
                    off = (m0[f"img0_{im0['label']}_start"] + im0["total_gen_s"] * 1e9 - t0mark) / 1e9
                colds[-1] = round(anchor + off - walls[tag][0], 2)
                r["_cold_epoch_based"] = True
        warm_vals = [w for w in warms if w is not None]
        headline.append({
            "route": route, "n_proc": len(rs),
            "cold_ttfi_median_s": round(st.median(colds), 2) if colds else "",
            "warm_e2e_median_of_proc_medians_s": round(st.median(warm_vals), 2) if warm_vals else "",
            "warm_e2e_ci95": boot_ci(warm_vals),
            "warm_enc_median_s": round(st.median([e for e in encs if e is not None]), 3) if any(encs) else "",
            "warm_dit_median_s": round(st.median([d for d in dits if d is not None]), 2) if any(dits) else "",
            "warm_vae_median_s": round(st.median([v for v in vaes if v is not None]), 2) if any(vaes) else "",
        })
    if headline:
        with open(os.path.join(SUM, "headline.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(headline[0].keys()))
            w.writeheader()
            w.writerows(headline)

    # primary.md
    lines = ["# E2E auto-generated summary (Phase 33)", "",
             "All numbers computed by scripts/aggregate_e2e.py from results/e2e/*/＋raw; do not edit by hand.", "",
             "## Headline (group=ab, 1024x1024, 20 steps, P3)", "",
             "| route | n_proc | cold TTFI (s) | warm E2E (s, median-of-medians) | CI95 | enc (s) | DiT (s) | VAE (s) |",
             "|---|---:|---:|---:|---|---:|---:|---:|"]
    for h in headline:
        lines.append(f"| {h['route']} | {h['n_proc']} | {h['cold_ttfi_median_s']} | "
                     f"{h['warm_e2e_median_of_proc_medians_s']} | {h['warm_e2e_ci95']} | "
                     f"{h['warm_enc_median_s']} | {h['warm_dit_median_s']} | {h['warm_vae_median_s']} |")
    lines += ["", "## Warm stats (per group/route/prompt)", "",
              "See stats.csv (full) — excerpt below:", ""]
    for srow in stats[:40]:
        lines.append(f"- {srow['group']}/{srow['route']} {srow['prompt']} @{srow['steps']}st {srow['res']}: "
                     f"n={srow['n']} e2e_med={srow['e2e_median_s']}s enc={srow['enc_median_s']} "
                     f"dit={srow['dit_median_s']} vae={srow['vae_median_s']}")
    with open(os.path.join(SUM, "primary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"[saved] {SUM}/per_image.csv stats.csv headline.csv primary.md ({len(per_image)} images)")


if __name__ == "__main__":
    main()

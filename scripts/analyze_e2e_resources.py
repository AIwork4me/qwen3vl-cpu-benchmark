#!/usr/bin/env python3
"""Phases 22/23/36/37: resource, GTT, UMA, thermal and power analysis.

Derives everything from results/e2e/raw/<tag>.csv (20 Hz monitor) + <tag>.json
marks. No estimates: only sampled values; gaps recorded as unavailable.

Outputs: results/e2e/summary/resources.csv (one row per run) and
          results/e2e/summary/thermal_drift.csv (per image in long runs)
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "e2e", "raw")
SUM = os.path.join(ROOT, "results", "e2e", "summary")
GROUPS = ["ab", "matrix", "throughput", "dq", "quality", "smoke"]


def read_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (None if v in ("", "None") else v) for k, v in r.items()})
    return rows


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def window(rows, ta, tb, key):
    vals = [fnum(r[key]) for r in rows if fnum(r["t_ns"]) is not None
            and ta <= float(r["t_ns"]) <= tb and fnum(r[key]) is not None]
    if not vals:
        return None, None, 0
    return sum(vals) / len(vals), max(vals), len(vals)


def main():
    os.makedirs(SUM, exist_ok=True)
    out_rows = []
    drift_rows = []
    for g in GROUPS:
        for jf in sorted(glob.glob(os.path.join(ROOT, "results", "e2e", g, "*.json"))):
            tag = os.path.basename(jf)[:-5]
            csv_path = os.path.join(RAW, f"{tag}.csv")
            if not os.path.exists(csv_path):
                continue
            run = json.load(open(jf))
            rows = read_csv(csv_path)
            marks = {}
            for line in open(csv_path + ".marks"):
                t, label = line.strip().split(",", 1)
                marks.setdefault(label, int(t))
            t_load0, t_load1 = marks.get("load_start"), marks.get("load_end")
            row = {"group": g, "tag": tag,
                   "route": run["meta"]["route"],
                   "dq": run["meta"].get("dq", "") if run["meta"]["route"] == "ov" else ""}
            # GTT / RSS / MemAvailable: after load vs whole-run peak vs baseline (first sample)
            base = rows[0] if rows else {}
            for key, name in (("gtt_used_bytes", "gtt"), ("rss_bytes", "rss"),
                              ("sys_available_bytes", "memavail")):
                whole_mean, whole_max, n = window(rows, 0, float("inf"), key)
                _, post_load, _ = window(rows, t_load1 or 0, float("inf"), key)
                row[f"{name}_baseline"] = round(fnum(base.get(key)) / 2**30, 2) if fnum(base.get(key)) else None
                row[f"{name}_post_load_gib"] = round(post_load / 2**30, 2) if post_load else None
                row[f"{name}_peak_gib"] = round(whole_max / 2**30, 2) if whole_max else None
                row[f"{name}_delta_peak_gib"] = (round((whole_max - fnum(base.get(key))) / 2**30, 2)
                                                 if whole_max and fnum(base.get(key)) else None)
            # GPU busy during all DiT windows combined vs whole run
            dit_windows = []
            for i in range(len(run.get("images", []))):
                a, b = marks.get(f"img{i}_dit_start"), marks.get(f"img{i}_dit_end")
                if a and b:
                    dit_windows.append((a, b))
            busy_vals = [fnum(r["gpu_busy_pct"]) for r in rows
                         if fnum(r["gpu_busy_pct"]) is not None
                         and any(a <= float(r["t_ns"]) <= b for a, b in dit_windows)]
            row["gpu_busy_dit_mean"] = round(sum(busy_vals) / len(busy_vals), 1) if busy_vals else None
            row["gpu_busy_dit_max"] = max(busy_vals) if busy_vals else None
            # power/thermal during DiT
            pw = [fnum(r["gpu_power_w"]) for r in rows if fnum(r.get("gpu_power_w")) is not None
                  and any(a <= float(r["t_ns"]) <= b for a, b in dit_windows)]
            row["gpu_power_dit_mean_w"] = round(sum(pw) / len(pw), 1) if pw else None
            # RAPL energy over whole run (first vs last sample)
            rapl = [fnum(r["rapl_energy_uj"]) for r in rows if fnum(r.get("rapl_energy_uj")) is not None]
            if len(rapl) > 1 and rapl[-1] >= rapl[0]:
                row["cpu_pkg_energy_wh"] = round((rapl[-1] - rapl[0]) / 3.6e9, 3)
            else:
                row["cpu_pkg_energy_wh"] = None
            out_rows.append(row)

            # thermal drift per image for runs with >=8 images
            ims = run.get("images", [])
            if len(ims) >= 8:
                for i, im in enumerate(ims):
                    a, b = marks.get(f"img{i}_dit_start"), marks.get(f"img{i}_dit_end")
                    if not (a and b):
                        continue
                    rowd = {"tag": tag, "img": i, "label": im["label"]}
                    for key in ("gpu_temp_c", "cpu_temp_c", "cpu_freq_mhz", "gpu_power_w"):
                        m, mx, _ = window(rows, a, b, key)
                        rowd[f"{key}_mean"] = round(m, 1) if m else None
                        rowd[f"{key}_max"] = round(mx, 1) if mx else None
                    rowd["dit_s"] = im.get("dit_s")
                    drift_rows.append(rowd)

    if out_rows:
        with open(os.path.join(SUM, "resources.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    if drift_rows:
        with open(os.path.join(SUM, "thermal_drift.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(drift_rows[0].keys()))
            w.writeheader()
            w.writerows(drift_rows)
    print(f"[saved] resources.csv ({len(out_rows)} runs), thermal_drift.csv ({len(drift_rows)} images)")


if __name__ == "__main__":
    main()

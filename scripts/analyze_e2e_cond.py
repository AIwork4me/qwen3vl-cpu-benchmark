#!/usr/bin/env python3
"""Phase 4/30: conditioning semantic identity across routes.

Compares the saved cond npys (results/e2e/cond/cond_<tag>_<PID>_pos.npy):
shape equality, token-count equality, cosine, RMSE, relative L2, max abs,
mean/std per tensor — NOT cosine alone (task rule).

Reference = the ComfyUI product path on the same machine. If a native (CPU TE)
npy exists for the prompt it is the reference; else gpu.

Output: results/e2e/cond/identity.csv + identity.md
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COND = os.path.join(ROOT, "results", "e2e", "cond")


def metrics(ref, x):
    a, b = ref.astype(np.float64).ravel(), x.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    return {"cosine": round(cos, 6), "rmse": round(rmse, 4),
            "relative_l2": round(rel, 6), "max_abs": round(float(np.max(np.abs(a - b))), 4),
            "mean_x": round(float(x.mean()), 5), "std_x": round(float(x.std()), 5),
            "shape_equal": str(ref.shape) == str(x.shape),
            "n_tokens": int(x.shape[1])}


def main():
    rows = []
    for polarity in ("pos", "neg"):
        files = sorted(glob.glob(os.path.join(COND, f"cond_*_{polarity}.npy")))
        groups = {}
        for f in files:
            m = re.match(r"cond_(.+?)_(P\d+)_" + polarity + r"\.npy", os.path.basename(f))
            if not m:
                continue
            tag, pid = m.group(1), m.group(2)
            groups.setdefault(pid, {})[tag] = np.load(f)
        rows.extend(tabulate(groups, polarity))
    if not rows:
        print("no cond npy found")
        sys.exit(1)
    import csv
    with open(os.path.join(COND, "identity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(COND, "identity.md"), "w") as f:
        f.write("# Conditioning identity (Phase 4/30)\n\n")
        f.write("| polarity | prompt | route | ref | shape_eq | n_tok | cosine | RMSE | relL2 | max_abs | mean | std |\n")
        f.write("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['polarity']} | {r['prompt']} | {r['route_tag']} | {r['reference']} | {r['shape_equal']} | "
                    f"{r['n_tokens']} | {r['cosine']:.6f} | {r['rmse']} | {r['relative_l2']:.6f} | "
                    f"{r['max_abs']} | {r['mean_x']} | {r['std_x']} |\n")
    print(open(os.path.join(COND, "identity.md")).read())


def tabulate(groups, polarity):
    out = []
    for pid in sorted(groups):
        tags = groups[pid]
        ref_tag = "native" if any(k.endswith("native") for k in tags) else (
            "gpu" if any("gpu" in k and "native" not in k for k in tags) else sorted(tags)[0])
        # prefer explicit baseline tags from the core A/B (ab_p*_gpu etc.)
        for k in tags:
            if "native" in k:
                ref_tag = k
                break
        else:
            for k in tags:
                if "gpu" in k:
                    ref_tag = k
                    break
        ref = tags[ref_tag]
        for tag in sorted(tags):
            r = {"polarity": polarity, "prompt": pid, "route_tag": tag, "reference": ref_tag}
            r.update(metrics(ref, tags[tag]))
            out.append(r)
    return out


if __name__ == "__main__":
    main()

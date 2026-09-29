#!/usr/bin/env python3
"""Compare conditioning embeddings across variants against the BF16 reference.

Inputs (results/):
  comparison/cond_{P}_bf16ref.npy           BF16 reference (ComfyUI product path, bf16 weights)
  comfy_cpu/cond_{P}.npy                    ComfyUI INT8 ConvRot product path (A2)
  comfy_cpu/cond_{P}_a1.npy                 ComfyUI INT8 ConvRot forced-int8 path (A1)
  openvino_cpu/hidden_{P}_bridge.npy        OpenVINO bridge hidden state

Metrics per task section 16: shape equality, finite, mean, std, max_abs,
mean_abs_error, RMSE, relative_L2, cosine_similarity.

Output: results/comparison/embedding_accuracy.csv (+ console table)
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "comparison")

VARIANTS = [
    ("comfy_int8_convrot_A2_product", os.path.join(RES, "comfy_cpu", "cond_{P}.npy")),
    ("comfy_int8_convrot_A1_forced", os.path.join(RES, "comfy_cpu", "cond_{P}_a1.npy")),
    ("openvino_int8_bridge", os.path.join(RES, "openvino_cpu", "hidden_{P}_bridge.npy")),
]


def metrics(ref, x):
    ref = ref.astype(np.float64)
    x = x.astype(np.float64)
    n = min(ref.shape[1], x.shape[1])
    if x.shape[1] != ref.shape[1]:
        # semantic misalignment — still compare the overlapping prefix
        ref = ref[:, :n, :]
        x = x[:, :n, :]
    d = (ref - x).ravel()
    rmse = float(np.sqrt(np.mean(d ** 2)))
    cos = float(np.sum(ref.ravel() * x.ravel()) /
                (np.linalg.norm(ref.ravel()) * np.linalg.norm(x.ravel()) + 1e-30))
    return {
        "shape_equal": ref.shape == x.shape,
        "finite": bool(np.isfinite(x).all()),
        "ref_mean": float(ref.mean()), "x_mean": float(x.mean()),
        "ref_std": float(ref.std()), "x_std": float(x.std()),
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs_error": float(np.mean(np.abs(d))),
        "rmse": rmse,
        "relative_l2": float(np.linalg.norm(d) / (np.linalg.norm(ref) + 1e-30)),
        "cosine_similarity": cos,
    }


def main():
    rows = []
    for vid, path_t in VARIANTS:
        for p in ("P1", "P2", "P3"):
            xpath = path_t.format(P=p)
            rpath = os.path.join(OUT, f"cond_{p}_bf16ref.npy")
            if not os.path.exists(xpath):
                rows.append({"variant": vid, "prompt": p, "error": f"missing {os.path.relpath(xpath, ROOT)}"})
                continue
            if not os.path.exists(rpath):
                rows.append({"variant": vid, "prompt": p, "error": "missing bf16 reference"})
                continue
            ref = np.load(rpath)
            x = np.load(xpath)
            m = metrics(ref, x)
            m.update({"variant": vid, "prompt": p,
                      "shape_ref": str(ref.shape), "shape_x": str(x.shape)})
            rows.append(m)

    os.makedirs(OUT, exist_ok=True)
    cols = ["variant", "prompt", "shape_equal", "finite", "ref_mean", "x_mean", "ref_std", "x_std",
            "max_abs", "mean_abs_error", "rmse", "relative_l2", "cosine_similarity",
            "shape_ref", "shape_x", "error"]
    with open(os.path.join(OUT, "embedding_accuracy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if "error" in r and r.get("error"):
            print(f"{r['variant']:34s} {r['prompt']}  ERROR: {r['error']}")
        else:
            print(f"{r['variant']:34s} {r['prompt']}  cos={r['cosine_similarity']:.6f} "
                  f"rmse={r['rmse']:.6f} relL2={r['relative_l2']:.6f} maxabs={r['max_abs']:.4f} "
                  f"shape={r['shape_x']}")

    with open(os.path.join(OUT, "embedding_accuracy.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("WROTE", os.path.join(OUT, "embedding_accuracy.csv"))


if __name__ == "__main__":
    main()

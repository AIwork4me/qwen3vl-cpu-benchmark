#!/usr/bin/env python3
"""Cosine similarity of investigation hidden-state npys vs the BF16 reference.

Reference: results/comparison/cond_P{1,2,3}_bf16ref.npy (same template/path as
the bridge benchmark). Inputs: results/openvino_root_cause/hidden_P*_<tag>.npy
produced by investigate_openvino_runtime.py --save-hidden.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(ROOT, "results", "comparison")
RC_DIR = os.path.join(ROOT, "results", "openvino_root_cause")


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    tags = sys.argv[1:] or sorted(
        os.path.basename(p).split("_", 2)[2][:-4]
        for p in glob.glob(os.path.join(RC_DIR, "hidden_P*_*.npy")))
    out = {}
    for tag in tags:
        for pid in ("P1", "P2", "P3"):
            p = os.path.join(RC_DIR, f"hidden_{pid}_{tag}.npy")
            if not os.path.exists(p):
                continue
            x = np.load(p)
            r = np.load(os.path.join(REF_DIR, f"cond_{pid}_bf16ref.npy"))
            out.setdefault(tag, {})[pid] = {
                "cosine": round(cosine(x, r), 8),
                "shape_equal": list(x.shape) == list(r.shape),
                "shape": list(x.shape),
            }
    print(json.dumps(out, indent=1))
    with open(os.path.join(RC_DIR, "hidden_cosine_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

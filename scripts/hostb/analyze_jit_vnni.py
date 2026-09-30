#!/usr/bin/env python3
# Frozen HOST-B snapshot — paths assume this file lives at scripts/<name>;
# copy there (or adjust ROOT) before running.
"""Scan oneDNN JIT-dump binaries for AVX-512 VNNI instructions.

Usage: analyze_jit_vnni.py <jit_dump_dir> [--out-dir DIR]

For every *.bin produced by ONEDNN_JIT_DUMP=1:
  objdump -D -b binary -m i386:x86-64 <file>
  search (case-insensitive) for: vpdpbusd, vpdpbusds, vpdpwssd, vpdpwssds

Writes:
  jit_vnni_hits.txt       — every hit with file, offset, +-10 instruction context
  jit_instruction_summary.txt — per-file census of all vector instruction classes
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter

VNNI = ["vpdpbusd", "vpdpbusds", "vpdpwssd", "vpdpwssds"]
BF16_DP = ["vdpbf16ps"]
VECTOR_CLASSES = [
    ("vpdpbusd", re.compile(r"\bvpdpbusd\b", re.I)),
    ("vpdpbusds", re.compile(r"\bvpdpbusds\b", re.I)),
    ("vpdpwssd", re.compile(r"\bvpdpwssd\b", re.I)),
    ("vpdpwssds", re.compile(r"\bvpdpwssds\b", re.I)),
    ("vdpbf16ps", re.compile(r"\bvdpbf16ps\b", re.I)),
    ("vcvtneps2bf16", re.compile(r"\bvcvtneps2bf16\b", re.I)),
    ("vcvtne2ps2bf16", re.compile(r"\bvcvtne2ps2bf16\b", re.I)),
    ("vpmaddubsw", re.compile(r"\bvpmaddubsw\b", re.I)),
    ("vpmaddwd", re.compile(r"\bvpmaddwd\b", re.I)),
    ("vpmaddwss", re.compile(r"\bvpmaddubsw\b\s", re.I)),
    ("zmm_usage", re.compile(r"\bzmm\d+\b", re.I)),
    ("ymm_usage", re.compile(r"\bymm\d+\b", re.I)),
    ("evex_masking", re.compile(r"\{k[0-7]\}", re.I)),
]


def disasm(path):
    r = subprocess.run(["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run(["objdump", "-D", "-b", "binary", "-m", "i386:x86-64:x86-64", path],
                           capture_output=True, text=True)
    return r.stdout.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jit_dir")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      "results", "openvino_isa"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bins = sorted(glob.glob(os.path.join(args.jit_dir, "*.bin")))
    print(f"[jit] {len(bins)} JIT binaries in {args.jit_dir}")

    hits_path = os.path.join(args.out_dir, "jit_vnni_hits.txt")
    summary_path = os.path.join(args.out_dir, "jit_instruction_summary.txt")
    json_path = os.path.join(args.out_dir, "jit_vnni_hits.json")

    per_file = []
    with open(hits_path, "w") as hits_f, open(summary_path, "w") as sum_f:
        for b in bins:
            lines = disasm(b)
            insts = [(i, l) for i, l in enumerate(lines) if re.match(r"\s+[0-9a-f]+:\s", l)]
            counts = Counter()
            vnni_lines = []
            for i, l in insts:
                for name, rx in VECTOR_CLASSES:
                    if rx.search(l):
                        counts[name] += 1
                        if name in VNNI or name in BF16_DP:
                            vnni_lines.append((i, l))
            sha = hashlib.sha256(open(b, "rb").read()).hexdigest()[:16]
            sz = os.path.getsize(b)
            per_file.append({"file": os.path.basename(b), "size": sz, "sha256_16": sha,
                             "counts": dict(counts), "n_vnni": sum(counts[v] for v in VNNI)})
            sum_f.write(f"{os.path.basename(b)}  size={sz} sha256[:16]={sha}\n")
            for name, _ in VECTOR_CLASSES:
                if counts.get(name):
                    sum_f.write(f"    {name:16s} {counts[name]}\n")
            for i, l in vnni_lines:
                hits_f.write(f"=== {os.path.basename(b)} line {i} ===\n")
                for j in range(max(0, i - 10), min(len(lines), i + 11)):
                    marker = ">>" if j == i else "  "
                    hits_f.write(f"{marker} {lines[j]}\n")
                hits_f.write("\n")
    total_vnni = {v: sum(f["counts"].get(v, 0) for f in per_file) for v in VNNI}
    with_vnni = [f for f in per_file if f["n_vnni"] > 0]
    summary = {
        "jit_dir": args.jit_dir,
        "n_binaries": len(bins),
        "n_binaries_with_vnni": len(with_vnni),
        "total_instruction_counts": total_vnni,
        "files_with_vnni": [f["file"] for f in with_vnni],
        "per_file": per_file,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[jit] binaries={len(bins)} with_vnni={len(with_vnni)} totals={total_vnni}")
    print(f"[jit] wrote {hits_path}\n[jit] wrote {summary_path}\n[jit] wrote {json_path}")
    for f in with_vnni[:20]:
        print(f"    VNNI: {f['file']}  {f['counts']}")


if __name__ == "__main__":
    main()

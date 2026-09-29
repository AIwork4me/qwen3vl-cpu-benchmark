#!/usr/bin/env python3
"""Analyze oneDNN JIT-dump binaries for VNNI / BF16 dot-product instructions.

Usage: analyze_jit_vnni.py <jit_dump_dir> [--out <dir>]

For every *.bin in the dump dir:
  objdump -D -b binary -m i386:x86-64 <file>
and search (case-insensitive) for:
  AVX-512 VNNI:  vpdpbusd  vpdpbusds  vpdpwssd  vpdpwssds
  AVX VNNI(256): vpdpbssd family appears via EVEX encoding of the same mnemonics
  AVX512_BF16:   vdpbf16ps (positive identification of the bf16 dot path)
  plain FMA:     vfmadd*231ps etc. (context only)

Outputs (in --out, default <jit_dump_dir>):
  jit_instruction_summary.txt   per-file instruction-class counts + totals
  jit_vnni_hits.txt             every hit with file, offset, +/-10 instruction context
  jit_dump_manifest.json        file list, sizes, sha256, per-file verdicts
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter

VNNI = ["vpdpbusd", "vpdpbusds", "vpdpwssd", "vpdpwssds"]  # VNNI_512 (zmm/EVEX) and VNNI_256 (ymm/VEX) split in classify
BF16 = ["vdpbf16ps"]
FMA = ["vfmadd"]  # substring match

INSTR_RE = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+)+\s*(.+)$")


def parse_mnem(field):
    # objdump prints ISA hints like "{vex}" / "{evex}" before the mnemonic:
    # strip them so 256-bit VEX-encoded VNNI is not missed.
    while field.startswith("{") and "}" in field:
        field = field.split("}", 1)[1].strip()
    return field.split()[0].lower() if field else ""


def disasm(path):
    try:
        out = subprocess.run(["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", path],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception as e:
        return None, f"objdump failed: {e}"
    lines = []
    for line in out.splitlines():
        m = INSTR_RE.match(line)
        if m:
            rest = m.group(2)
            mnem = parse_mnem(rest)
            lines.append((int(m.group(1), 16), mnem, rest))
    return lines, None


def classify(mnem, ops):
    for v in VNNI:
        if mnem.startswith(v):
            return "VNNI_256" if "%ymm" in ops else "VNNI_512"
    for v in BF16:
        if mnem.startswith(v):
            return "BF16"
    for v in FMA:
        if mnem.startswith(v):
            return "FMA"
    if mnem.startswith("vpmadd") or mnem.startswith("pmadd"):
        return "PMADD"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dump_dir = os.path.abspath(args.dump_dir)
    out_dir = os.path.abspath(args.out or dump_dir)
    os.makedirs(out_dir, exist_ok=True)

    bins = sorted(p for p in os.listdir(dump_dir) if p.endswith(".bin"))
    summary_lines = []
    hits_lines = []
    manifest = {"dump_dir": dump_dir, "files": [], "totals": Counter()}
    files_with_vnni = 0
    files_with_bf16 = 0

    for name in bins:
        path = os.path.join(dump_dir, name)
        lines, err = disasm(path)
        rec = {"file": name, "size": os.path.getsize(path),
               "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
        if err:
            rec["error"] = err
            manifest["files"].append(rec)
            summary_lines.append(f"{name}: {err}")
            continue
        counts = Counter()
        hit_idx = []
        for i, (off, mnem, ops) in enumerate(lines):
            cls = classify(mnem, ops)
            if cls:
                counts[cls] += 1
                if cls in ("VNNI_256", "VNNI_512", "BF16"):
                    hit_idx.append(i)
        rec["counts"] = dict(counts)
        rec["vnni"] = counts.get("VNNI_512", 0) + counts.get("VNNI_256", 0)
        rec["bf16_dot"] = counts.get("BF16", 0)
        rec["fma"] = counts.get("FMA", 0)
        rec["pmadd"] = counts.get("PMADD", 0)
        manifest["files"].append(rec)
        for k, v in counts.items():
            manifest["totals"][k] += v
        if rec["vnni"]:
            files_with_vnni += 1
        if rec["bf16_dot"]:
            files_with_bf16 += 1
        summary_lines.append(
            f"{name}: size={rec['size']:8d} vnni={rec['vnni']:5d} "
            f"(512b={counts.get('VNNI_512',0):5d}, 256b={counts.get('VNNI_256',0):4d}) "
            f"bf16_dot={rec['bf16_dot']:5d} fma={rec['fma']:5d} pmadd={rec['pmadd']:3d}")
        # context for every hit
        for i in hit_idx:
            off, mnem, ops = lines[i]
            hits_lines.append(f"=== {name} @ file-offset 0x{off:x}: {mnem} {ops} ===")
            lo, hi = max(0, i - 10), min(len(lines), i + 11)
            for j in range(lo, hi):
                o2, m2, p2 = lines[j]
                marker = ">>" if j == i else "  "
                hits_lines.append(f"{marker} 0x{o2:06x}: {m2} {p2}")
            hits_lines.append("")

    header = [
        f"# JIT dump instruction summary — {dump_dir}",
        f"# files analyzed: {len(bins)}",
        f"# files containing VNNI instructions (vpdpbusd/vpdpbusds/vpdpwssd/vpdpwssds): {files_with_vnni}",
        f"# files containing BF16 dot (vdpbf16ps): {files_with_bf16}",
        f"# totals: {dict(manifest['totals'])}",
        "",
    ]
    with open(os.path.join(out_dir, "jit_instruction_summary.txt"), "w") as f:
        f.write("\n".join(header + summary_lines) + "\n")
    with open(os.path.join(out_dir, "jit_vnni_hits.txt"), "w") as f:
        f.write("\n".join(hits_lines) + "\n" if hits_lines else "# no VNNI/BF16-dot hits\n")
    manifest["totals"] = dict(manifest["totals"])
    manifest["files_with_vnni"] = files_with_vnni
    manifest["files_with_bf16_dot"] = files_with_bf16
    with open(os.path.join(out_dir, "jit_dump_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("\n".join(header))
    print(f"[saved] jit_instruction_summary.txt / jit_vnni_hits.txt / jit_dump_manifest.json in {out_dir}")
    return 0 if bins else 1


if __name__ == "__main__":
    sys.exit(main())

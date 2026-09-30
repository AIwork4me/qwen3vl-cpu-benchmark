#!/usr/bin/env python3
# Frozen HOST-B snapshot — paths assume this file lives at scripts/<name>;
# copy there (or adjust ROOT) before running.
"""Parse oneDNN verbose logs (onednn:verb or dnnl_verbose format).

oneDNN verbose lines look like:
  onednn_verbose,info,oneDNN v3.7.0 (commit ...)
  onednn_verbose,info,cpu,isa:AVX2 ... or cpu,isa:AVX512_CORE
  onednn_verbose,exec,cpu,matmul,forward,...</prop>/...</attr>:jit:avx512_core_amx_int8,...</shapes>:16.7,0.05,34.68
  oneDNN 3.x: "onednn_verbose,exec,cpu,matmul,...name:jit:avx512_core,dt:f32..." etc.

This parser is defensive: it records raw lines plus structured fields
(primitive, implementation/alg, dtypes when present, exec time) and produces a
summary JSON. It NEVER guesses fields not present in the log.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict


def parse(path):
    rows = []
    headers = []
    with open(path, errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if "onednn_verbose" not in line and "dnnl_verbose" not in line:
                continue
            parts = line.split(",")
            kind = parts[1] if len(parts) > 1 else "?"
            if kind in ("info", "warn", "error"):
                headers.append({"lineno": lineno, "raw": line})
                continue
            if kind == "exec":
                rec = {"lineno": lineno, "raw": line, "engine": parts[2] if len(parts) > 2 else "",
                       "primitive": parts[3] if len(parts) > 3 else ""}
                impl_m = re.search(r"jit:[A-Za-z0-9_]+|ref|gemm:[A-Za-z0-9_]+", line)
                rec["impl"] = impl_m.group(0) if impl_m else ""
                # dtypes like dt:f32 or s8:u8:s8:u8 tokens anywhere
                dt = re.findall(r"(?:dt:)?((?:u8|s8|f16|bf16|f32|f64)(?::(?:u8|s8|f16|bf16|f32|f64))*)", line)
                rec["dtypes"] = dt[:3]
                # exec times: trailing numbers "16.7,0.05,34.68"
                tail = re.search(r":([\d.eE+-]+,[\d.eE+-]+(?:,[\d.eE+-]+)?)\s*$", line)
                rec["exec_times"] = tail.group(1).split(",") if tail else []
                # isa mentions
                isa = re.findall(r"avx[0-9a-z_]+|sve[0-9]*", line, re.I)
                rec["isa_tokens"] = sorted(set(isa), key=str.lower)
                rows.append(rec)
    return headers, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    headers, rows = parse(args.log)

    by_prim_impl = Counter()
    by_prim_impl_time = defaultdict(float)
    isa_counter = Counter()
    dtype_counter = Counter()
    for r in rows:
        key = (r["primitive"], r["impl"])
        by_prim_impl[key] += 1
        t = 0.0
        if r["exec_times"]:
            try:
                t = float(r["exec_times"][-1])
            except ValueError:
                t = 0.0
        by_prim_impl_time[key] += t
        for isa in r["isa_tokens"]:
            isa_counter[isa] += 1
        for d in r["dtypes"]:
            dtype_counter[d] += 1

    summary = {
        "log": args.log,
        "n_header_lines": len(headers),
        "n_exec_lines": len(rows),
        "headers": headers[:20],
        "primitive_impl_counts": [{"primitive": k[0], "impl": k[1], "count": v}
                                  for k, v in by_prim_impl.most_common()],
        "primitive_impl_total_exec_time_us_last_field": [
            {"primitive": k[0], "impl": k[1], "sum_last_time_field_us": round(v, 1)}
            for k, v in sorted(by_prim_impl_time.items(), key=lambda x: -x[1])[:args.top]],
        "isa_token_counts": dict(isa_counter),
        "dtype_token_counts": dict(dtype_counter),
        "distinct_exec_lines": [
            {"primitive": r["primitive"], "impl": r["impl"], "dtypes": r["dtypes"],
             "isa": r["isa_tokens"], "raw": r["raw"]}
            for r in rows[:200]],
    }
    out = args.out or (args.log.rsplit(".", 1)[0] + "_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[verbose] headers={len(headers)} exec_lines={len(rows)} -> {out}")
    for h in headers[:10]:
        print("   HDR:", h["raw"][:160])
    print("  top primitive/impl:")
    for k, v in by_prim_impl.most_common(12):
        print(f"    {v:5d}  {k[0]:12s} {k[1]}")
    print("  isa tokens:", dict(isa_counter))
    print("  dtypes:", dict(dtype_counter))
    if not rows:
        print("  NOTE: zero oneDNN verbose exec lines parsed — see report for possible causes.")


if __name__ == "__main__":
    main()

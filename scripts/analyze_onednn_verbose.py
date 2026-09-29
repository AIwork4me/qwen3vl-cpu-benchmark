#!/usr/bin/env python3
"""Parse oneDNN verbose logs (ONEDNN_VERBOSE / onednn_verbose,v1 format).

Handles: header info lines (version / cpu runtime / isa / gpu), primitive
create:* lines (operation,engine,primitive,implementation,...) and primitive
exec lines. Produces a summary JSON. Works on partial logs (e.g. logs that
end in an OpenVINO crash) — the crash itself is recorded, not hidden.

Usage: analyze_onednn_verbose.py <log...> [--out summary.json]
"""
import argparse
import json
import re
import sys
from collections import Counter

LINE_RE = re.compile(r"^onednn_verbose,v1,(?P<kind>info|primitive),(?P<rest>.*)$")


def parse_log(path):
    out = {"file": path, "header": {}, "create_lines": [], "exec_lines": [],
           "template": None, "other_lines": 0, "crash": None}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if "could not execute a primitive" in line or "RuntimeError" in line:
                out["crash"] = out["crash"] or line.strip()
                continue
            m = LINE_RE.match(line)
            if not m:
                if "onednn_verbose" in line:
                    out["other_lines"] += 1
                continue
            kind, rest = m.group("kind"), m.group("rest")
            if kind == "info":
                if rest.startswith("oneDNN v"):
                    out["header"]["onednn_version"] = rest
                elif rest.startswith("cpu,"):
                    parts = rest.split(",", 2)
                    val = parts[2] if len(parts) > 2 else ""
                    key = parts[1].split(":", 1)[0]
                    out["header"][f"cpu_{key}"] = val or parts[1]
                elif rest.startswith("gpu,"):
                    out["header"]["gpu"] = rest
                elif rest.startswith("template:"):
                    out["template"] = rest
                continue
            # primitive lines: rest = "<stage>,<payload>"
            stage, _, payload = rest.partition(",")
            if stage.startswith("create:"):
                fields = payload.split(",")
                rec = {"mode": stage, "raw": payload}
                if fields and fields[0] in ("cpu", "gpu"):
                    rec["engine"], rec["operation"] = fields[0], fields[1]
                    rec["implementation"] = fields[2] if len(fields) > 2 else ""
                    rec["prop_kind"] = fields[3] if len(fields) > 3 else ""
                    rec["memory_descriptors"] = fields[4] if len(fields) > 4 else ""
                else:
                    # dispatch rejection: "<op>,<reason>"
                    rec["engine"] = ""
                    rec["operation"] = fields[0] if fields else ""
                    rec["implementation"] = ""
                    rec["rejected_reason"] = fields[1] if len(fields) > 1 else ""
                out["create_lines"].append(rec)
            elif stage == "info" and payload.startswith("template:"):
                out["template"] = payload
            else:
                out["exec_lines"].append(payload)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    results = [parse_log(p) for p in args.logs]
    summary = {
        "logs": results,
        "combined": {
            "create_by_operation_impl": dict(Counter(
                f"{c.get('operation')}:{c.get('implementation') or c.get('rejected_reason','')}"
                for r in results for c in r["create_lines"])),
            "exec_line_count": sum(len(r["exec_lines"]) for r in results),
        },
    }
    js = json.dumps(summary, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js)
        print(f"[saved] {args.out}")
    else:
        print(js)


if __name__ == "__main__":
    main()

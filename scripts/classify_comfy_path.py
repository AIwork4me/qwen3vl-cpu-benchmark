#!/usr/bin/env python3
"""Classify the ComfyUI INT8 ConvRot CPU compute path as A1/A2/A3 from measured evidence.

Reads results/comfy_cpu/comfy_<tag>.json files produced by bench_comfy_qwen3vl_cpu.py
and emits results/comfy_cpu/runtime_path.md + dispatch.log summary.

Classification rules (task section 8):
  A1 TRUE_INT8_CPU_COMPUTE          int_mm calls per encode > 0 and dequant == 0 (or ~0)
  A2 INT8_WEIGHT_STORAGE_FP_COMPUTE int_mm == 0 and dequant > 0 with quantized weights present
  A3 UNSUPPORTED_OR_ERROR           encode failed / no quantized weights on CPU
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "comfy_cpu")


def classify(tag):
    jpath = os.path.join(OUT_DIR, f"comfy_{tag}.json")
    if not os.path.exists(jpath):
        return None
    with open(jpath) as f:
        r = json.load(f)
    audit = r["meta"]["audit"]
    n_q = audit["n_quantized_weights"]
    devs = audit["weight_devices"]
    per_prompt = {}
    for pid, d in r["prompts"].items():
        c = d["compute_counters"]
        per_prompt[pid] = c
    c0 = next(iter(per_prompt.values())) if per_prompt else {}
    int_mm = c0.get("int_mm", 0)
    dequant = c0.get("dequant", 0)

    if n_q == 0 or any("cpu" not in d for d in devs):
        verdict = "A3 UNSUPPORTED_OR_ERROR"
    elif int_mm > 0 and dequant == 0:
        verdict = "A1 TRUE_INT8_CPU_COMPUTE"
    elif int_mm == 0 and dequant > 0:
        verdict = "A2 INT8_WEIGHT_STORAGE_FP_COMPUTE"
    else:
        verdict = "A2 INT8_WEIGHT_STORAGE_FP_COMPUTE (mixed: int_mm=%d dequant=%d)" % (int_mm, dequant)
    return {
        "tag": tag,
        "verdict": verdict,
        "n_quantized_weights": n_q,
        "convrot_layers": audit["convrot_layers"],
        "quant_formats": audit["quant_formats"],
        "weight_devices": devs,
        "load_device": audit["patcher_load_device"],
        "offload_device": audit["patcher_offload_device"],
        "per_prompt_counters": per_prompt,
        "force_quant_mm": r["meta"].get("force_quant_mm", False),
        "output_dtype": next(iter(r["prompts"].values()))["output"]["dtype"],
    }


def main():
    tags = sys.argv[1:] or ["default"]
    rows = []
    lines = ["# ComfyUI INT8 ConvRot CPU — runtime compute-path classification\n"]
    for tag in tags:
        res = classify(tag)
        if res is None:
            continue
        rows.append(res)
        lines.append(f"## run tag: {tag}  (force_quant_mm={res['force_quant_mm']})\n")
        lines.append(f"- VERDICT: **{res['verdict']}**")
        lines.append(f"- quantized weights: {res['n_quantized_weights']} (convrot: {res['convrot_layers']}), formats: {res['quant_formats']}")
        lines.append(f"- load_device={res['load_device']} offload_device={res['offload_device']} weight_devices={res['weight_devices']}")
        lines.append(f"- per-encode counters: {res['per_prompt_counters']}")
        lines.append(f"- conditioning output dtype: {res['output_dtype']}")
        lines.append("")
    out = "\n".join(lines)
    path = os.path.join(OUT_DIR, "runtime_path.md")
    with open(path, "w") as f:
        f.write(out)
    # dispatch log summary
    disp = []
    for tag in tags:
        jpath = os.path.join(OUT_DIR, f"comfy_{tag}.json")
        if os.path.exists(jpath):
            disp.append(f"tag={tag} verdict={rows[[r_['tag'] for r_ in rows].index(tag)]['verdict']}")
    with open(os.path.join(OUT_DIR, "dispatch.log"), "a") as f:
        f.write("\n".join(disp) + "\n")
    print(out)
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()

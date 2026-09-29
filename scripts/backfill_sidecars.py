#!/usr/bin/env python3
"""Backfill/repair per-PNG sidecar JSONs from the run JSONs (Phase 20 integrity)."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E2E = os.path.join(ROOT, "results", "e2e")

n = 0
for jf in glob.glob(os.path.join(E2E, "*", "*.json")):
    name = os.path.basename(jf)
    if name.startswith(("e2e_", "server_")) or name in ("model_sha16.json",):
        continue
    try:
        r = json.load(open(jf))
    except Exception:
        continue
    m = r.get("meta", {})
    for im in r.get("images", []):
        png = os.path.join(ROOT, im.get("png", "")) if im.get("png") else None
        if not png or not os.path.exists(png):
            continue
        side_path = png.replace(".png", ".json")
        old = json.load(open(side_path)) if os.path.exists(side_path) else {}
        old.update({
            "prompt": im.get("prompt_text_hint", old.get("prompt")),
            "seed": im["seed"], "route": m["route"],
            "dq": m.get("dq") if m["route"] == "ov" else None,
            "resolution": [m["width"], m["height"]], "steps": m["steps"],
            "cfg": m["cfg"], "sampler": m["sampler"], "scheduler": m["scheduler"],
            "encode_s": im.get("encode_s"), "dit_s": im.get("dit_s"),
            "vae_s": im.get("vae_s"), "postprocess_s": im.get("postprocess_s"),
            "png_save_s": im.get("png_save_s"),
            "total_gen_s": im.get("total_gen_s"),
            "total_save_inclusive_s": im.get("total_save_inclusive_s"),
            "img_sha16": im.get("img_sha16"), "tag": m["tag"],
        })
        json.dump(old, open(side_path, "w"), indent=1)
        n += 1
print(f"[backfill] {n} sidecars updated")

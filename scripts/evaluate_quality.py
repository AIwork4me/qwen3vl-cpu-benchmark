#!/usr/bin/env python3
"""Phase 19: image quality evaluation for the e2e quality dataset.

Inputs : results/e2e/images/q_<route>_<QID>_s<seed>_i0.png (+ sidecar .json)
          reference route = q_gpu (ComfyUI product path)
Outputs: results/e2e/quality/quality_metrics.csv
          results/e2e/quality/blind/ contact sheets (A/B/C/D anonymized)

Metrics per (prompt, seed):
  A. pixel:   PSNR, SSIM (scikit-image), MAE vs the gpu-route image of the same
              prompt+seed (deterministic same-seed sampling => same noise;
              differences come from conditioning deltas + GPU nondeterminism)
  B. prompt-image alignment: CLIP ViT-L/14 cos(text, image) — same evaluator
     for every route; reported per route (absolute) and delta vs gpu.
  C. pairwise image similarity: CLIP image-embedding cosine vs gpu image.
  D. blind package: per prompt, 2x2 contact sheet with one shuffled label per present
     route (A–D on the Q01–Q12 four-route subset; A/B only for Q13–Q30),
     mapping stored in results/e2e/quality/blind/key.csv (labels randomized with
     a fixed seed; no subjective scores are invented here).

Runs in .venv-comfy (torch 2.9.1+cpu + transformers + scikit-image + PIL).
"""
import csv
import glob
import hashlib
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "results", "e2e", "images")
OUT = os.path.join(ROOT, "results", "e2e", "quality")
CLIP_DIR = os.path.join(ROOT, "models", "clip-vit-large-patch14")
REF_ROUTE = "q_gpu"
ROUTES = ["q_gpu", "q_dq32", "q_dq64", "q_dq128"]
LBL = ["A", "B", "C", "D"]


def load_img(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "blind"), exist_ok=True)
    from skimage.metrics import structural_similarity as ssim_fn

    clip_model = None
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(CLIP_DIR).eval()
        proc = CLIPProcessor.from_pretrained(CLIP_DIR)
        have_clip = True
    except Exception as e:
        print(f"[warn] CLIP evaluator unavailable ({e}); pixel metrics only")
        have_clip = False

    MAXTOK = clip_model.config.text_config.max_position_embeddings - 2

    def pooled_norm(out):
        if torch.is_tensor(out):
            t = out
        else:
            t = getattr(out, "pooler_output", None)
            if t is None:
                t = out[0]
            if not torch.is_tensor(t):
                t = t[0]
        return t / t.norm(dim=-1, keepdim=True)

    def chunks(text):
        ids = proc.tokenizer(text, add_special_tokens=False)["input_ids"]
        out, cur = [], []
        for t in ids:
            cur.append(t)
            if len(cur) >= MAXTOK:
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return [proc.tokenizer.decode(c) for c in out] or [text]

    def clip_scores(imgs, texts):
        """Chunked CLIP scoring: prompts longer than the CLIP context are split
        into <=max_pos windows; per-image score = mean cosine over chunks.
        Same evaluator + same truncation for every route."""
        img_embs = []
        with torch.no_grad():
            for im in imgs:
                inp = proc(text=["x"], images=[im], return_tensors="pt")
                e = clip_model.get_image_features(pixel_values=inp["pixel_values"])
                img_embs.append(pooled_norm(e))
            scores, iemb_rows = [], []
            for im_idx, text in enumerate(texts):
                cs = []
                for ch in chunks(text):
                    inp = proc(text=[ch], return_tensors="pt", padding=True)
                    e = clip_model.get_text_features(**{k: v for k, v in inp.items() if k != "pixel_values"})
                    e = pooled_norm(e)
                    cs.append(float((img_embs[im_idx] @ e.T).item()))
                scores.append(sum(cs) / len(cs))
            iemb = torch.cat(img_embs, 0)
        return scores, iemb

    # collect images per (QID, seed)
    entries = {}
    for route in ROUTES:
        for f in sorted(glob.glob(os.path.join(IMG, f"{route}_Q*_i0.png"))):
            base = os.path.basename(f)[:-7]          # e.g. q_dq32_Q01_s101
            assert base.startswith(route + "_")
            rest = base[len(route) + 1:]              # Q01_s101
            qid, seed = rest.rsplit("_s", 1)
            entries.setdefault((qid, seed), {})[route] = f

    rows = []
    blind_key = []
    rng = random.Random(20260929)
    for (qid, seed), routes in sorted(entries.items()):
        ref_path = routes.get(REF_ROUTE)
        if ref_path is None:
            continue
        sidecar = ref_path.replace(".png", ".json")
        prompt = json.load(open(sidecar))["prompt"] if os.path.exists(sidecar) else ""
        ref_img = load_img(ref_path)
        imgs_pil = [Image.open(routes[r]).convert("RGB") for r in ROUTES if r in routes]
        clip_ti, iemb = (None, None)
        if have_clip and prompt:
            clip_ti, iemb = clip_scores(imgs_pil, [prompt] * len(imgs_pil))
        row = {"prompt": qid, "seed": seed}
        present = [r for r in ROUTES if r in routes]
        for i, r in enumerate(present):
            img = load_img(routes[r])
            tag = r.replace("q_", "").replace("dq", "DQ")
            if r != REF_ROUTE:
                row[f"psnr_{tag}"] = round(psnr(ref_img, img), 2)
                row[f"ssim_{tag}"] = round(ssim_fn(ref_img, img, channel_axis=2, data_range=1.0), 4)
                row[f"mae_{tag}"] = round(float(np.abs(ref_img - img).mean()) * 255, 2)
            if clip_ti is not None:
                row[f"clip_ti_{tag}"] = round(clip_ti[i], 4)
                if iemb is not None and i > 0:
                    row[f"clip_imgcos_{tag}"] = round(float((iemb[0] @ iemb[i]).item()), 4)
        rows.append(row)

        # blind contact sheet: 2x2 labeled A-D with shuffled route order
        order = present[:]
        rng.shuffle(order)
        cell = 512
        sheet = Image.new("RGB", (cell * 2, cell * 2 + 40), "white")
        d = ImageDraw.Draw(sheet)
        for slot, r in enumerate(order):
            im = Image.open(routes[r]).convert("RGB").resize((cell, cell))
            x, y = (slot % 2) * cell, (slot // 2) * cell + 40
            sheet.paste(im, (x, y))
            d.text((x + 8, y + 6), LBL[slot], fill="red")
        d.text((8, 8), f"{qid} s{seed} — which images differ? A/B/C/D", fill="black")
        sheet.save(os.path.join(OUT, "blind", f"blind_{qid}_s{seed}.jpg"), quality=90)
        blind_key.append({"prompt": qid, "seed": seed, **{LBL[i]: r for i, r in enumerate(order)}})

    if rows:
        with open(os.path.join(OUT, "quality_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(OUT, "blind", "key.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(blind_key[0].keys()))
            w.writeheader()
            w.writerows(blind_key)
    print(f"[done] {len(rows)} prompt-seed groups -> {OUT}/quality_metrics.csv + blind/")


if __name__ == "__main__":
    main()

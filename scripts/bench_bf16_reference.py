#!/usr/bin/env python3
"""BF16 reference encode (numerical reference only — not a timed benchmark).

Loads qwen3vl_8b_bf16.safetensors through the SAME ComfyUI product path
(load_clip / QwenImage21TEModel / CPU pinning), encodes P1-P3 once each,
saves cond_*.npy for the accuracy comparison in results/comparison/.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ComfyUI"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MODEL = os.path.join(ROOT, "models", "qwen-image-2.1", "text_encoders", "qwen3vl_8b_bf16.safetensors")
OUT_DIR = os.path.join(ROOT, "results", "comparison")


def main():
    import numpy as np
    import torch
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.args_parsing = True
    import comfy.cli_args
    assert comfy.cli_args.args.cpu is True
    import comfy.sd
    import logging
    logging.basicConfig(level=logging.INFO)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[bf16-ref] loading {MODEL}")
    t0 = time.perf_counter()
    clip = comfy.sd.load_clip(
        [MODEL], embedding_directory=None, clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")})
    print(f"[bf16-ref] loaded in {time.perf_counter()-t0:.1f}s")

    with open(os.path.join(ROOT, "prompts", "prompts.json")) as f:
        prompts = json.load(f)["prompts"]

    for p in prompts:
        tokens = clip.tokenize(p["text"])
        cond = clip.encode_from_tokens(tokens)
        arr = cond.to(torch.float32).numpy()
        np.save(os.path.join(OUT_DIR, f"cond_{p['id']}_bf16ref.npy"), arr)
        print(f"[{p['id']}] cond {arr.shape} {arr.dtype} mean={arr.mean():.6f} std={arr.std():.6f} finite={np.isfinite(arr).all()}")
    print("[bf16-ref] done")


if __name__ == "__main__":
    main()

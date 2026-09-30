#!/usr/bin/env python3
"""E2E step: compute OpenVINO conditioning tensors for the Qwen-Image 2.1 e2e A/B.

Runs in .venv-openvino. Replicates ComfyUI's TextEncodeQwenImage21 t2i semantics
EXACTLY (same T2I template incl. system turn, same tokenizer files, same trim from
the second <|im_start|>, same empty-text->" " handling via prevent_empty_text):

    positive = prompt text (default: P3 from prompts.json)
    negative = "" -> " " (matches ComfyUI prevent_empty_text=True)

For a given --dq (DYNAMIC_QUANTIZATION_GROUP_SIZE) it writes:
    results/e2e/cond/cond_{P3}_{pos|neg}_dq{N}.npy   [1, seq, 4096] fp32
    results/e2e/cond/cond_timing_dq{N}.json           per-call latency (warm x N)

The npy is the trimmed pre-norm layer-36 hidden state — the exact tensor ComfyUI's
encode_token_weights returns for t2i (verified: same template, same trim index,
cosine >= 0.997 vs ComfyUI native in prior rounds).
"""
import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import openvino as ov

MODEL_DIR = os.path.join(ROOT, "models", "qwen3vl-openvino-int8")
OUT_DIR = os.path.join(ROOT, "results", "e2e", "cond")
PROMPTS_FILE = os.path.join(ROOT, "prompts", "prompts.json")

T2I_TEMPLATE = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")


def find_pre_norm_node(xml_path):
    xml = open(xml_path).read()
    id_by_name = {}
    for m in re.finditer(r'<layer id="(\d+)" name="([^"]+)"', xml):
        id_by_name[m.group(2)] = m.group(1)
    name_by_id = {v: k for k, v in id_by_name.items()}
    norm_pow = None
    for name in id_by_name:
        if name.startswith("__module.model.language_model.norm/") and name.endswith("aten::pow/Power"):
            norm_pow = name
            break
    pow_id = id_by_name[norm_pow]
    m = re.search(r'<edge[^>]*to-layer="%s"[^>]*/>' % pow_id, xml)
    if m is None:
        m = re.search(r"<edge[^>]*to-layer=\"%s\"[^>]*>" % pow_id, xml)
    frm = re.search(r'from-layer="(\d+)"', m.group(0)).group(1)
    return name_by_id.get(frm), norm_pow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dq", type=int, default=32)
    ap.add_argument("--prompt-id", default="P3")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--measure", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    core = ov.Core()
    lm_xml = os.path.join(MODEL_DIR, "openvino_language_model.xml")
    lm_bin = os.path.join(MODEL_DIR, "openvino_language_model.bin")
    pre_node, norm_node = find_pre_norm_node(lm_xml)
    assert pre_node

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "ComfyUI", "comfy", "text_encoders", "qwen25_tokenizer"))

    model = core.read_model(lm_xml, lm_bin)
    picked = {}
    for op in model.get_ordered_ops():
        fn = op.get_friendly_name()
        if fn in (pre_node, norm_node) and fn not in picked:
            picked[fn] = op
    model.add_outputs([picked[pre_node].output(0)])

    cfg = {"PERFORMANCE_HINT": "LATENCY",
           "DYNAMIC_QUANTIZATION_GROUP_SIZE": args.dq}
    compiled = core.compile_model(model, "CPU", cfg)
    assert [str(d) for d in compiled.get_property("EXECUTION_DEVICES")] == ["CPU"]
    req = compiled.create_infer_request()
    emb_req = core.compile_model(
        core.read_model(os.path.join(MODEL_DIR, "openvino_text_embeddings_model.xml"),
                        os.path.join(MODEL_DIR, "openvino_text_embeddings_model.bin")),
        "CPU").create_infer_request()

    with open(PROMPTS_FILE) as f:
        text = [p["text"] for p in json.load(f)["prompts"] if p["id"] == args.prompt_id][0]
    texts = {"pos": text, "neg": " "}  # neg: prevent_empty_text semantics

    out = {"meta": {"dq": args.dq, "prompt_id": args.prompt_id,
                    "openvino": ov.__version__,
                    "effective_dq": str(compiled.get_property("DYNAMIC_QUANTIZATION_GROUP_SIZE"))},
           "calls": {}}
    for kind, txt in texts.items():
        enc = tok(T2I_TEMPLATE.format(txt), add_special_tokens=False)["input_ids"]
        input_ids = np.array([enc], dtype=np.int64)
        n_tok = input_ids.shape[1]
        embeds = np.asarray(emb_req.infer({0: input_ids})[0])
        feed = {"inputs_embeds": embeds, "attention_mask": np.ones_like(input_ids)}
        ar = np.arange(n_tok, dtype=np.int64)
        feed["position_ids"] = np.stack([ar, ar, ar]).reshape(3, 1, n_tok)
        feed["visual_pos_masks"] = np.zeros((1, n_tok), dtype=bool)
        feed["deepstack_visual_embeds"] = np.zeros((1, 0, 4096), dtype=np.float32)
        feed["beam_idx"] = np.array([0], dtype=np.int32)
        f = {k: v for k, v in feed.items()}
        idx_feed = {}
        for i, inp in enumerate(model.inputs):
            idx_feed[i] = f[list(inp.names)[0]]
        for _ in range(args.warm):
            req.reset_state()
            req.infer(idx_feed)
        lat = []
        for _ in range(args.measure):
            req.reset_state()
            t0 = time.perf_counter_ns()
            res = req.infer(idx_feed)
            lat.append((time.perf_counter_ns() - t0) / 1e9)
        hidden = np.asarray(res[list(res.keys())[-1]], dtype=np.float32)  # pre-norm output
        ids_list = input_ids[0].tolist()
        im = [i for i, t in enumerate(ids_list) if t == 151644]
        trim = im[1] if len(im) > 1 else 0
        trimmed = hidden[:, trim:, :]
        np.save(os.path.join(OUT_DIR, f"cond_{args.prompt_id}_{kind}_dq{args.dq}.npy"), trimmed)
        out["calls"][kind] = {
            "n_tokens_full": n_tok, "n_tokens_kept": int(trimmed.shape[1]),
            "shape": list(trimmed.shape),
            "warm_p50_s": sorted(lat)[len(lat) // 2], "warm_mean_s": sum(lat) / len(lat),
            "iterations": lat,
        }
        print(f"[dq{args.dq} {kind}] tokens={n_tok} kept={trimmed.shape[1]} "
              f"p50={out['calls'][kind]['warm_p50_s']:.4f}s mean={out['calls'][kind]['warm_mean_s']:.4f}s")
    with open(os.path.join(OUT_DIR, f"cond_timing_dq{args.dq}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {OUT_DIR}/cond_timing_dq{args.dq}.json")


if __name__ == "__main__":
    main()

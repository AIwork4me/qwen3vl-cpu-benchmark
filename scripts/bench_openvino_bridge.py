#!/usr/bin/env python3
"""OpenVINO conditioning bridge: expose the Qwen-Image 2.1 hidden state from the
official INT8-ov IR via model.add_outputs() — zero-recompute graph patch.

Semantics alignment with ComfyUI (see report/qwen_image_conditioning_semantics.md):
  - token sequence = ComfyUI T2I_TEMPLATE (system turn + user turn + assistant head),
    tokenized with ComfyUI's own qwen25_tokenizer (Qwen2Tokenizer), full sequence
    including the system turn (hidden state computed on it, trimmed afterwards —
    exactly what encode_token_weights does).
  - hidden state = layer-36 residual stream BEFORE final RMSNorm
    (ComfyUI: layer_idx=-1, layer_norm_hidden_state=False).
    Found dynamically in the IR: the node feeding the final norm's first op.
  - output trimmed from the second <|im_start|> (151644) onward -> [1, seq_kept, 4096].

This is the *comparable* OpenVINO conditioning-encoder benchmark (task section 17).
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import openvino as ov
from resource_monitor import ResourceMonitor

MODEL_DIR = os.path.join(ROOT, "models", "qwen3vl-openvino-int8")
RESULTS_DIR = os.path.join(ROOT, "results", "openvino_cpu")
PROMPTS_FILE = os.path.join(ROOT, "prompts", "prompts.json")

T2I_TEMPLATE = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")


def find_pre_norm_node(xml_path):
    """Return the IR node name that produces the input of the final RMSNorm."""
    xml = open(xml_path).read()
    # layer map: name -> id
    id_by_name = {}
    for m in re.finditer(r'<layer id="(\d+)" name="([^"]+)"', xml):
        id_by_name[m.group(2)] = m.group(1)
    name_by_id = {v: k for k, v in id_by_name.items()}
    # the final norm ops
    norm_pow = None
    for name in id_by_name:
        if name.startswith("__module.model.language_model.norm/") and name.endswith("aten::pow/Power"):
            norm_pow = name
            break
    if norm_pow is None:
        return None, None
    pow_id = id_by_name[norm_pow]
    # find edge: <edge from-layer="<pow_id input?>" ... to-layer="<pow_id>"> — actually we need
    # the edge whose to-layer is the pow node's layer; its from-layer is the pre-norm producer.
    m = re.search(r'<edge[^>]*to-layer="%s"[^>]*/>' % pow_id, xml)
    if m is None:
        m = re.search(r"<edge[^>]*to-layer=\"%s\"[^>]*>" % pow_id, xml)
    if m is None:
        return None, norm_pow
    frm = re.search(r'from-layer="(\d+)"', m.group(0)).group(1)
    return name_by_id.get(frm), norm_pow


def build_tokenizer():
    """Use ComfyUI's own qwen25_tokenizer via transformers Qwen2Tokenizer (exact product parity)."""
    from transformers import AutoTokenizer
    tok_dir = os.path.join(ROOT, "ComfyUI", "comfy", "text_encoders", "qwen25_tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-iters", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--hint", default="LATENCY")
    ap.add_argument("--tag", default="bridge")
    ap.add_argument("--save-hidden", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    core = ov.Core()
    print(f"[bridge] openvino {ov.__version__} devices={core.available_devices}")

    lm_xml = os.path.join(MODEL_DIR, "openvino_language_model.xml")
    lm_bin = os.path.join(MODEL_DIR, "openvino_language_model.bin")

    pre_node, norm_node = find_pre_norm_node(lm_xml)
    print(f"[bridge] final norm op: {norm_node}")
    print(f"[bridge] pre-norm hidden state node: {pre_node}")
    assert pre_node is not None

    monitor = ResourceMonitor(hz=20,
        output_csv=os.path.join(ROOT, "results", "raw", f"openvino_{args.tag}_resources.csv")).start()

    t0 = time.perf_counter_ns()
    tok = build_tokenizer()
    tok_s = (time.perf_counter_ns() - t0) / 1e9
    monitor.mark("tokenizer_ready")

    t0 = time.perf_counter_ns()
    model = core.read_model(lm_xml, lm_bin)
    read_s = (time.perf_counter_ns() - t0) / 1e9
    # attach outputs by Output object (layer names are friendly names, not tensor names)
    wanted = {pre_node, norm_node}
    picked = {}
    for op in model.get_ordered_ops():
        fn = op.get_friendly_name()
        if fn in wanted and fn not in picked:
            picked[fn] = op
    assert pre_node in picked and norm_node in picked, f"nodes not found: {wanted - set(picked)}"
    model.add_outputs([picked[pre_node].output(0), picked[norm_node].output(0)])
    model_inputs = [list(i.names)[0] for i in model.inputs]
    print(f"[bridge] model inputs: {model_inputs}")

    cfg = {"PERFORMANCE_HINT": args.hint}
    if args.threads:
        cfg["INFERENCE_NUM_THREADS"] = args.threads
    t0 = time.perf_counter_ns()
    compiled = core.compile_model(model, "CPU", cfg)
    compile_s = (time.perf_counter_ns() - t0) / 1e9
    load_s = read_s + compile_s
    monitor.mark("model_loaded")
    exec_devs = compiled.get_property("EXECUTION_DEVICES")
    if isinstance(exec_devs, str):
        exec_devs = [exec_devs]
    exec_devs = [str(d) for d in exec_devs]
    assert exec_devs == ["CPU"], f"FAIL: execution devices {exec_devs}"
    post_load = monitor.interval_stats("start", "model_loaded")
    print(f"[bridge] read {read_s:.3f}s compile {compile_s:.3f}s exec={exec_devs}")

    infer_req = compiled.create_infer_request()
    emb_compiled = core.compile_model(
        core.read_model(os.path.join(MODEL_DIR, "openvino_text_embeddings_model.xml"),
                        os.path.join(MODEL_DIR, "openvino_text_embeddings_model.bin")), "CPU")
    emb_req = emb_compiled.create_infer_request()
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)["prompts"]

    results = {"meta": {
        "timestamp": datetime.now().isoformat(),
        "model_dir": MODEL_DIR,
        "variant": "OpenVINO INT8 weight-compressed + add_outputs conditioning bridge",
        "pre_norm_node": pre_node,
        "norm_node": norm_node,
        "device": "CPU",
        "execution_devices": str(exec_devs),
        "hint": args.hint,
        "threads": args.threads or "default",
        "tokenizer_init_s": tok_s,
        "model_read_s": read_s,
        "model_compile_s": compile_s,
        "model_load_s": load_s,
        "post_load_peak_rss_bytes": post_load["peak_rss_bytes"] if post_load else None,
        "model_inputs": model_inputs,
    }, "prompts": {}}

    for p in prompts:
        text = T2I_TEMPLATE.format(p["text"])
        enc = tok(text, add_special_tokens=False, return_tensors=None)
        ids = enc["input_ids"]
        input_ids = np.array([ids], dtype=np.int64)
        n_tok = input_ids.shape[1]
        embeds = np.asarray(emb_req.infer({0: input_ids})[0])
        feed = {}
        for i, inp in enumerate(model.inputs):
            name = list(inp.names)[0]
            if name == "inputs_embeds":
                feed[i] = embeds
            elif name == "attention_mask":
                feed[i] = np.ones_like(input_ids)
            elif name == "position_ids":
                ar = np.arange(n_tok, dtype=np.int64)
                feed[i] = np.stack([ar, ar, ar]).reshape(3, 1, n_tok)
            elif name == "visual_pos_masks":
                feed[i] = np.zeros((1, n_tok), dtype=bool)
            elif name == "deepstack_visual_embeds":
                feed[i] = np.zeros((1, 0, 4096), dtype=np.float32)
            elif name == "beam_idx":
                feed[i] = np.array([0], dtype=np.int32)
            else:
                raise RuntimeError(f"unhandled model input: {name}")
        # second <|im_start|> position (token 151644) — trim point from ComfyUI semantics
        ids_list = input_ids[0].tolist()
        im_starts = [i for i, t in enumerate(ids_list) if t == 151644]
        trim_at = im_starts[1] if len(im_starts) > 1 else 0

        infer_req.reset_state()
        t0 = time.perf_counter_ns()
        res = infer_req.infer(feed)
        warmup_s = (time.perf_counter_ns() - t0) / 1e9
        monitor.mark(f"{p['id']}_warmup")

        out_names = [list(o.names)[0] for o in model.outputs]
        logits_idx = [i for i, n in enumerate(out_names) if "logits" in n][0]
        hidden = res[len(out_names) - 2]   # pre-norm output added last but one
        hidden_np = np.asarray(hidden, dtype=np.float32)
        trimmed = hidden_np[:, trim_at:, :]
        post_norm = np.asarray(res[len(out_names) - 1], dtype=np.float32)

        if args.save_hidden:
            np.save(os.path.join(RESULTS_DIR, f"hidden_{p['id']}_{args.tag}.npy"), trimmed)

        first_s = None
        t0 = time.perf_counter_ns()
        infer_req.reset_state()
        res1 = infer_req.infer(feed)
        first_s = (time.perf_counter_ns() - t0) / 1e9
        monitor.mark(f"{p['id']}_first")
        warm = []
        for _ in range(args.warm_iters):
            t0 = time.perf_counter_ns()
            infer_req.reset_state()
            infer_req.infer(feed)
            warm.append((time.perf_counter_ns() - t0) / 1e9)
        monitor.mark(f"{p['id']}_warm")

        st = monitor.interval_stats(f"{p['id']}_warmup", f"{p['id']}_warm")
        w = warm
        print(f"[{p['id']}] n_tok={n_tok} trim_at={trim_at} hidden {trimmed.shape} "
              f"mean={trimmed.mean():.6f} std={trimmed.std():.6f}")
        print(f"[{p['id']}] first={first_s:.3f}s warm mean={sum(w)/len(w):.3f}s p50={sorted(w)[len(w)//2]:.3f}s "
              f"min={min(w):.3f}s max={max(w):.3f}s peak_rss={st['peak_rss_bytes']/2**30:.2f}GiB")

        results["prompts"][p["id"]] = {
            "n_tokens_full": n_tok,
            "trim_index": trim_at,
            "n_tokens_kept": int(trimmed.shape[1]),
            "hidden_shape": list(trimmed.shape),
            "hidden_mean": float(trimmed.mean()),
            "hidden_std": float(trimmed.std()),
            "warmup_s": warmup_s,
            "first_encode_s": first_s,
            "warm_encode_s": warm,
            "interval_stats": st,
        }

    monitor.stop()
    results["meta"]["full_run"] = monitor.full_run()
    jpath = os.path.join(RESULTS_DIR, f"openvino_{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    cpath = os.path.join(RESULTS_DIR, f"openvino_{args.tag}.csv")
    with open(cpath, "w") as f:
        f.write("prompt_id,phase,iter,latency_s\n")
        for pid, d in results["prompts"].items():
            f.write(f"{pid},warmup,{d['warmup_s']}\n")
            f.write(f"{pid},first,{d['first_encode_s']}\n")
            for i, s_ in enumerate(d["warm_encode_s"]):
                f.write(f"{pid},warm_{i},{s_}\n")
    print(f"[saved] {jpath}\n[saved] {cpath}")


if __name__ == "__main__":
    main()

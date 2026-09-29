#!/usr/bin/env python3
"""Experiment B: OpenVINO INT8 weight-compressed Qwen3-VL-8B on Ryzen AI Max+ 395 CPU.

Standalone benchmark + (optional) conditioning hidden-state extraction.

Device is hard-pinned to "CPU" (never AUTO). Everything is loaded from the local
ModelScope checkout under models/qwen3vl-openvino-int8 — no network access.

Metrics mirror the ComfyUI harness exactly: baseline RSS -> load/compile ->
first inference -> warm x5, with 20 Hz ResourceMonitor sampling peak RSS.
All raw iterations saved as JSON; per-sample CSV via ResourceMonitor.

Usage:
  python scripts/bench_openvino_qwen3vl_cpu.py [--threads N] [--hint LATENCY|THROUGHPUT]
      [--streams 1] [--tag TAG] [--extract-hidden] [--hidden-mode before_lm_head|after_final_norm]
"""
import argparse
import json
import os
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

LM_XML = os.path.join(MODEL_DIR, "openvino_language_model.xml")
LM_BIN = os.path.join(MODEL_DIR, "openvino_language_model.bin")


def load_prompts():
    with open(PROMPTS_FILE) as f:
        return json.load(f)["prompts"]


def build_tokenizer():
    """Use HF fast tokenizer (tokenizer.json present in the repo)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    import jinja2
    with open(os.path.join(MODEL_DIR, "chat_template.jinja")) as f:
        tpl_src = f.read()
    tpl = jinja2.Template(tpl_src)
    return tok, tpl


def render_chat(tpl, text):
    """Apply the official chat template to a single user message, text-only."""
    msgs = [{"role": "user", "content": text}]
    return tpl.render(messages=msgs, add_generation_prompt=True)


def introspect(model):
    info = {"inputs": [], "outputs": []}
    for inp in model.inputs:
        info["inputs"].append({
            "names": list(inp.names) if inp.names else [],
            "pshape": str(inp.partial_shape),
            "dtype": str(inp.element_type),
        })
    for out in model.outputs:
        info["outputs"].append({
            "names": list(out.names) if out.names else [],
            "pshape": str(out.partial_shape),
            "dtype": str(out.element_type),
        })
    return info


def encode_single(tok, prompt_text, max_len=4096):
    ids = tok.encode(prompt_text).ids
    ids = ids[:max_len]
    input_ids = np.array([ids], dtype=np.int64)
    attn = np.ones_like(input_ids)
    return input_ids, attn


def run_bench(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "raw"), exist_ok=True)

    props_record = {}
    core = ov.Core()
    props_record["openvino_version"] = ov.__version__
    props_record["available_devices"] = core.available_devices
    props_record["requested_device"] = "CPU"

    print(f"[openvino] version={ov.__version__} available_devices={core.available_devices}")
    assert "CPU" in core.available_devices

    monitor = ResourceMonitor(hz=20,
        output_csv=os.path.join(ROOT, "results", "raw", f"openvino_{args.tag}_resources.csv")).start()

    # ---- tokenizer (before model so template render cost is separate) ----
    t0 = time.perf_counter_ns()
    tok, tpl = build_tokenizer()
    tok_s = (time.perf_counter_ns() - t0) / 1e9
    monitor.mark("tokenizer_ready")
    print(f"[tokenizer] ready in {tok_s:.3f}s")

    # ---- model read + optional output patching ----
    t0 = time.perf_counter_ns()
    model = core.read_model(LM_XML, LM_BIN)
    read_s = (time.perf_counter_ns() - t0) / 1e9
    monitor.mark("ir_read")
    print(f"[ir] read_model {read_s:.3f}s; inputs={[list(i.names)[0] for i in model.inputs]}")

    if args.extract_hidden:
        node = args.hidden_node or "__module.model.language_model.norm/aten::mul/Multiply_1"
        model.add_outputs(node)
        print(f"[bridge] added hidden-state output: {node}")

    model_info = introspect(model)

    # ---- compile (this is what "model load" means for OpenVINO: read+compile) ----
    cfg = {"PERFORMANCE_HINT": args.hint}
    if args.threads:
        cfg["INFERENCE_NUM_THREADS"] = args.threads
    if args.streams:
        cfg["NUM_STREAMS"] = str(args.streams)
    t0 = time.perf_counter_ns()
    compiled = core.compile_model(model, "CPU", cfg)
    compile_s = (time.perf_counter_ns() - t0) / 1e9
    load_s = read_s + compile_s
    monitor.mark("model_loaded")
    print(f"[load] compile {compile_s:.3f}s; total read+compile {load_s:.3f}s")

    # ---- record device + runtime properties (evidence of CPU execution) ----
    for k in ("INFERENCE_NUM_THREADS", "NUM_STREAMS", "PERFORMANCE_HINT",
              "PERFORMANCE_HINT_NUM_REQUESTS"):
        try:
            props_record[k] = str(compiled.get_property(k))
        except Exception as e:
            props_record[k] = f"n/a ({e.__class__.__name__})"
    exec_devs = compiled.get_property("EXECUTION_DEVICES")
    if isinstance(exec_devs, str):
        exec_devs = [exec_devs]
    props_record["execution_devices"] = [str(d) for d in exec_devs]
    assert props_record["execution_devices"] == ["CPU"], \
        f"expected CPU-only execution, got {props_record['execution_devices']}"
    print(f"[device] execution_devices={props_record['execution_devices']}")

    post_load = monitor.interval_stats("start", "model_loaded")
    baseline = monitor.baseline_rss()

    infer_req = compiled.create_infer_request()

    prompts = load_prompts()
    results = {"meta": {
        "timestamp": datetime.now().isoformat(),
        "model_dir": MODEL_DIR,
        "model_variant": "OpenVINO/Qwen3-VL-8B-Instruct-int8-ov (INT8_ASYM weight compression, group_size=-1)",
        "device": "CPU",
        "device_hard_pinned": True,
        "hint": args.hint,
        "threads": args.threads or "default",
        "streams": args.streams or "default",
        "extract_hidden": args.extract_hidden,
        "hidden_node": (args.hidden_node or "__module.model.language_model.norm/aten::mul/Multiply_1")
                       if args.extract_hidden else None,
        "tokenizer_ready_s": tok_s,
        "model_read_s": read_s,
        "model_compile_s": compile_s,
        "model_load_s": load_s,
        "post_load_rss_bytes": post_load["peak_rss_bytes"] if post_load else None,
        "baseline_rss_bytes": baseline,
        "openvino_properties": props_record,
        "model_io": model_info,
    }, "prompts": {}}

    # embed model (openvino_text_embeddings_model = embed_tokens lookup)
    emb_compiled = core.compile_model(
        core.read_model(os.path.join(MODEL_DIR, "openvino_text_embeddings_model.xml"),
                        os.path.join(MODEL_DIR, "openvino_text_embeddings_model.bin")), "CPU")
    emb_req = emb_compiled.create_infer_request()

    for p in prompts:
        chat = render_chat(tpl, p["text"])
        input_ids, attn = encode_single(tok, chat, max_len=args.max_len)
        n_tok = int(input_ids.shape[1])
        print(f"[{p['id']}] chat-template tokens: {n_tok}")

        # stage 1: ids -> embeds
        t_e0 = time.perf_counter_ns()
        embeds = np.asarray(emb_req.infer({0: input_ids})[0])
        embed_ms = (time.perf_counter_ns() - t_e0) / 1e6

        # stage 2: LM prefill — official multimodal signature
        feed = {}
        for i, inp in enumerate(model.inputs):
            name = list(inp.names)[0]
            if name == "inputs_embeds":
                feed[i] = embeds
            elif name == "attention_mask":
                feed[i] = attn.astype(np.int64)
            elif name == "position_ids":
                # M-RoPE [3, batch, seq]; text-only => all three = arange
                ar = np.arange(n_tok, dtype=np.int64)
                feed[i] = np.stack([ar, ar, ar])[None, ...] if False else np.stack([ar, ar, ar]).reshape(3, 1, n_tok)
            elif name == "visual_pos_masks":
                feed[i] = np.zeros((1, n_tok), dtype=bool)
            elif name == "deepstack_visual_embeds":
                feed[i] = np.zeros((1, 0, 4096), dtype=np.float32)
            elif name == "beam_idx":
                feed[i] = np.array([0], dtype=np.int32)
            else:
                raise RuntimeError(f"unhandled input {name}")

        # warm-up (stateful IR: reset KV states so every call is a clean prefill)
        t0 = time.perf_counter_ns()
        infer_req.reset_state()
        res = infer_req.infer(feed)
        warmup_s = (time.perf_counter_ns() - t0) / 1e9
        monitor.mark(f"{p['id']}_warmup_done")

        # capture hidden state / logits info from warm-up call
        out_names = [list(o.names)[0] if o.names else f"output_{i}" for i, o in enumerate(model.outputs)]
        out_info = {}
        for i, oname in enumerate(out_names):
            arr = res[i]
            out_info[oname] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                               "mean": float(np.mean(arr.astype(np.float64))),
                               "std": float(np.std(arr.astype(np.float64))),
                               "finite": bool(np.isfinite(arr.astype(np.float64)).all())}
        if args.extract_hidden and args.save_hidden:
            for i, oname in enumerate(out_names):
                if i == 0 and "logits" in oname:
                    continue  # skip the huge logits output
                arr = res[i]
                np.save(os.path.join(ROOT, "results", "openvino_cpu", f"hidden_{p['id']}.npy"),
                        arr.astype(np.float32))

        # measured: first (cold-in-process) + 5 warm  (LM prefill only; embed_ms recorded separately)
        iters = {"warmup_s": warmup_s, "first_s": None, "warm_s": [], "embed_ms": embed_ms}
        t0 = time.perf_counter_ns()
        infer_req.reset_state()
        infer_req.infer(feed)
        iters["first_s"] = (time.perf_counter_ns() - t0) / 1e9
        monitor.mark(f"{p['id']}_first_done")
        for k in range(args.warm_iters):
            t0 = time.perf_counter_ns()
            infer_req.reset_state()
            infer_req.infer(feed)
            iters["warm_s"].append((time.perf_counter_ns() - t0) / 1e9)
        monitor.mark(f"{p['id']}_warm_done")

        stats = monitor.interval_stats(f"{p['id']}_warmup_done", f"{p['id']}_warm_done")
        post_load_now = monitor.interval_stats("start", f"{p['id']}_warm_done")

        results["prompts"][p["id"]] = {
            "n_tokens": n_tok,
            "embed_ms": embed_ms,
            "iters": iters,
            "interval_stats": stats,
            "peak_rss_bytes": post_load_now["peak_rss_bytes"] if post_load_now else None,
            "outputs": out_info,
        }
        w = iters["warm_s"]
        print(f"[{p['id']}] first={iters['first_s']:.3f}s warm mean={sum(w)/len(w):.3f}s "
              f"p50={sorted(w)[len(w)//2]:.3f}s min={min(w):.3f}s max={max(w):.3f}s "
              f"peak_rss={(post_load_now['peak_rss_bytes'] if post_load_now else 0)/2**30:.2f}GiB")

    monitor.stop()
    full = monitor.full_run()
    results["meta"]["full_run"] = full
    results["meta"]["torch_num_threads"] = None
    results["meta"]["openvino_num_threads_reported"] = props_record.get("INFERENCE_NUM_THREADS")

    tag = args.tag
    jpath = os.path.join(RESULTS_DIR, f"openvino_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    # CSV of iterations
    cpath = os.path.join(RESULTS_DIR, f"openvino_{tag}.csv")
    with open(cpath, "w") as f:
        f.write("prompt_id,phase,iter,latency_s\n")
        for pid, d in results["prompts"].items():
            f.write(f"{pid},warmup,{d['iters']['warmup_s']}\n")
            f.write(f"{pid},first,{d['iters']['first_s']}\n")
            for i, s in enumerate(d["iters"]["warm_s"]):
                f.write(f"{pid},warm_{i},{s}\n")
    print(f"[saved] {jpath}\n[saved] {cpath}")
    props_path = os.path.join(RESULTS_DIR, "openvino_properties.json")
    with open(props_path, "w") as f:
        json.dump(props_record, f, indent=2)
    print(f"[saved] {props_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--hint", default="LATENCY", choices=["LATENCY", "THROUGHPUT", "UNDEFINED"])
    ap.add_argument("--streams", type=int, default=None)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--warm-iters", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--extract-hidden", action="store_true",
                    help="add hidden-state output via model.add_outputs (conditioning bridge)")
    ap.add_argument("--hidden-node", default=None)
    ap.add_argument("--save-hidden", action="store_true")
    args = ap.parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()

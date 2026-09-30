#!/usr/bin/env python3
# Frozen HOST-B snapshot — paths assume this file lives at scripts/<name>;
# copy there (or adjust ROOT) before running.
"""Root-cause investigation runner for the OpenVINO Qwen3-VL INT8 conditioning workload.

Reuses the exact token/bridge semantics of scripts/bench_openvino_bridge.py
(T2I_TEMPLATE, add_outputs pre-norm node, trim from second <|im_start|>) but adds:

  * enable_profiling -> per-node get_profiling_info() dump
  * compiled.get_runtime_model() export + exec-type/precision census
  * supported-properties census (incl. DYNAMIC_QUANTIZATION_GROUP_SIZE)
  * optional dynamic-quantization group-size override
  * bench mode (--mode bench) with fresh warm/measure iterations
  * optional cosine check vs the committed HOST-A hidden_*.npy artifacts

Outputs under results/openvino_root_cause/ (never overwrites legacy results/).
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import openvino as ov

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "qwen3vl-openvino-int8")
PROMPTS_FILE = os.path.join(ROOT, "prompts", "prompts.json")
HOST_A_NPY = os.path.join(ROOT, "results", "openvino_cpu")

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
    if norm_pow is None:
        return None, None
    pow_id = id_by_name[norm_pow]
    m = re.search(r'<edge[^>]*to-layer="%s"[^>]*/>' % pow_id, xml)
    if m is None:
        m = re.search(r'<edge[^>]*to-layer="%s"[^>]*>' % pow_id, xml)
    if m is None:
        return None, norm_pow
    frm = re.search(r'from-layer="(\d+)"', m.group(0)).group(1)
    return name_by_id.get(frm), norm_pow


def build_tokenizer():
    from transformers import AutoTokenizer
    tok_dir = os.path.join(ROOT, "ComfyUI", "comfy", "text_encoders", "qwen25_tokenizer")
    return AutoTokenizer.from_pretrained(tok_dir)


def rss_gib():
    with open("/proc/self/status") as f:
        m = re.search(r"VmRSS:\s+(\d+) kB", f.read())
    return int(m.group(1)) * 1024 / 2**30


def pi_to_dict(pi):
    """ProfilingInfo -> dict, robust to attribute naming across OV versions."""
    d = {}
    for attr in ("status", "node_name", "layer_type", "exec_type", "real_time"):
        try:
            v = getattr(pi, attr)
        except Exception:
            v = None
        if attr == "real_time" and v is not None:
            try:
                v = v.total_seconds() * 1e6  # timedelta -> us
            except AttributeError:
                v = float(v)
        d[attr] = str(v) if attr == "status" else v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["profile", "bench"], default="profile")
    ap.add_argument("--prompts", default="P1,P2,P3")
    ap.add_argument("--warm-iters", type=int, default=2)
    ap.add_argument("--measure-iters", type=int, default=10)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dq-group-size", type=str, default=None,
                    help="pass e.g. 0/32/64/128 to set DYNAMIC_QUANTIZATION_GROUP_SIZE")
    ap.add_argument("--precision-hint", default=None,
                    help="INFERENCE_PRECISION_HINT override: f32/f16/bf16/i8 (passed through raw)")
    ap.add_argument("--tag", default="inv")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="model directory (default: the INT8 weight-compressed OV model)")
    ap.add_argument("--compare-npy", action="store_true",
                    help="cosine vs committed HOST-A hidden_P*_bridge.npy")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "openvino_root_cause"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    core = ov.Core()
    print(f"[inv] openvino {ov.__version__} devices={core.available_devices}")
    lm_xml = os.path.join(args.model_dir, "openvino_language_model.xml")
    lm_bin = os.path.join(args.model_dir, "openvino_language_model.bin")

    pre_node, norm_node = find_pre_norm_node(lm_xml)
    print(f"[inv] pre-norm node: {pre_node}")
    assert pre_node is not None

    tok = build_tokenizer()
    t0 = time.perf_counter_ns()
    model = core.read_model(lm_xml, lm_bin)
    read_s = (time.perf_counter_ns() - t0) / 1e9
    wanted = {pre_node, norm_node}
    picked = {}
    for op in model.get_ordered_ops():
        fn = op.get_friendly_name()
        if fn in wanted and fn not in picked:
            picked[fn] = op
    assert pre_node in picked and norm_node in picked
    model.add_outputs([picked[pre_node].output(0), picked[norm_node].output(0)])

    cfg = {"PERFORMANCE_HINT": "LATENCY",
           "INFERENCE_NUM_THREADS": args.threads,
           ov.properties.enable_profiling: True}
    if args.dq_group_size is not None:
        cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = int(args.dq_group_size)
    if args.precision_hint is not None:
        cfg["INFERENCE_PRECISION_HINT"] = args.precision_hint
    t0 = time.perf_counter_ns()
    try:
        compiled = core.compile_model(model, "CPU", cfg)
    except Exception as e:
        print(f"[inv] compile with DQ override FAILED: {e}")
        if args.dq_group_size is not None:
            print("[inv] retrying without DYNAMIC_QUANTIZATION_GROUP_SIZE")
            cfg.pop("DYNAMIC_QUANTIZATION_GROUP_SIZE", None)
            compiled = core.compile_model(model, "CPU", cfg)
        else:
            raise
    compile_s = (time.perf_counter_ns() - t0) / 1e9
    print(f"[inv] read {read_s:.3f}s compile {compile_s:.3f}s")

    exec_devs = compiled.get_property("EXECUTION_DEVICES")
    if isinstance(exec_devs, str):
        exec_devs = [exec_devs]
    assert [str(d) for d in exec_devs] == ["CPU"], f"exec devices {exec_devs}"

    props = {}
    for pname in compiled.get_property(ov.properties.supported_properties):
        try:
            v = compiled.get_property(pname)
            props[pname] = v if isinstance(v, (int, float, str, bool)) else str(v)
        except Exception as e:
            props[pname] = f"<error {e}>"
    print("[inv] supported_properties:")
    for k, v in sorted(props.items()):
        print(f"    {k} = {v}")
    dq_supported = "DYNAMIC_QUANTIZATION_GROUP_SIZE" in props
    print(f"[inv] DYNAMIC_QUANTIZATION_GROUP_SIZE supported: {dq_supported}")

    # ---- runtime model census (exec types / precisions) ----
    rt = compiled.get_runtime_model()
    rt_path = os.path.join(args.out_dir, f"runtime_model_{args.tag}.xml")
    ov.serialize(rt, rt_path)
    census = {}
    op_rows = []
    for op in rt.get_ordered_ops():
        ri = op.get_rt_info()
        entry = {}
        for k in list(ri.keys()):
            try:
                entry[k] = str(ri[k])
            except Exception:
                pass
        rt_info_flat = dict(entry)
        # canonical places for exec info in OV CPU runtime graphs
        exec_type = None
        for path in (("Info", "execType"), ("execType",)):
            cur = ri
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok:
                exec_type = str(cur)
                break
        if exec_type is None:
            exec_type = rt_info_flat.get("execType") or rt_info_flat.get("implementationId") or ""
        op_type = op.get_type_name()
        census.setdefault((op_type, exec_type), 0)
        census[(op_type, exec_type)] += 1
        op_rows.append({"friendly_name": op.get_friendly_name(), "op_type": op_type,
                        "exec_type": exec_type, "rt_info": rt_info_flat})
    with open(os.path.join(args.out_dir, f"runtime_model_summary_{args.tag}.json"), "w") as f:
        json.dump({"op_census": [{"op_type": k[0], "exec_type": k[1], "count": v}
                                 for k, v in sorted(census.items(), key=lambda x: -x[1])],
                   "ops": op_rows}, f, indent=1)
    print("[inv] top runtime op/exec types:")
    for (ot, et), n in sorted(census.items(), key=lambda x: -x[1])[:15]:
        print(f"    {n:5d}  {ot:24s} {et}")

    # ---- embeddings sub-model ----
    emb_compiled = core.compile_model(
        core.read_model(os.path.join(args.model_dir, "openvino_text_embeddings_model.xml"),
                        os.path.join(args.model_dir, "openvino_text_embeddings_model.bin")), "CPU")
    emb_req = emb_compiled.create_infer_request()
    infer_req = compiled.create_infer_request()

    with open(PROMPTS_FILE) as f:
        prompts = [p for p in json.load(f)["prompts"] if p["id"] in args.prompts.split(",")]

    results = {"meta": {
        "timestamp": datetime.now().isoformat(),
        "openvino_version": ov.__version__,
        "mode": args.mode, "tag": args.tag, "threads": args.threads,
        "hint": "LATENCY", "device": "CPU",
        "execution_devices": [str(d) for d in exec_devs],
        "model_read_s": read_s, "model_compile_s": compile_s,
        "warm_iters": args.warm_iters, "measure_iters": args.measure_iters,
        "dq_group_size_requested": args.dq_group_size,
        "DYNAMIC_QUANTIZATION_GROUP_SIZE_supported": dq_supported,
        "properties": props,
        "env": {k: os.environ.get(k, "") for k in
                ("ONEDNN_MAX_CPU_ISA", "ONEDNN_VERBOSE", "ONEDNN_JIT_DUMP",
                 "ONEDNN_JIT_PROFILE", "OPENVINO_")},
        "rss_gib_after_compile": rss_gib(),
    }, "prompts": {}}

    for p in prompts:
        text = T2I_TEMPLATE.format(p["text"])
        ids = tok(text, add_special_tokens=False)["input_ids"]
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
                raise RuntimeError(f"unhandled input {name}")
        ids_list = input_ids[0].tolist()
        im_starts = [i for i, t in enumerate(ids_list) if t == 151644]
        trim_at = im_starts[1] if len(im_starts) > 1 else 0

        for _ in range(args.warm_iters):
            infer_req.reset_state()
            infer_req.infer(feed)

        warm = []
        for _ in range(args.measure_iters):
            t0 = time.perf_counter_ns()
            infer_req.reset_state()
            infer_req.infer(feed)
            warm.append((time.perf_counter_ns() - t0) / 1e9)

        # profiled infer for node-level timing
        infer_req.reset_state()
        res = infer_req.infer(feed)
        out_names = [list(o.names)[0] for o in model.outputs]
        hidden = np.asarray(res[len(out_names) - 2], dtype=np.float32)
        trimmed = hidden[:, trim_at:, :]
        prof = [pi_to_dict(x) for x in infer_req.get_profiling_info()]

        row = {
            "n_tokens_full": n_tok, "trim_index": trim_at,
            "n_tokens_kept": int(trimmed.shape[1]),
            "warm_s": warm,
            "warm_p50": sorted(warm)[len(warm) // 2],
            "warm_mean": sum(warm) / len(warm),
            "warm_min": min(warm), "warm_max": max(warm),
            "hidden_mean": float(trimmed.mean()), "hidden_std": float(trimmed.std()),
            "profile_rows": prof,
        }
        if args.compare_npy:
            ref_path = os.path.join(HOST_A_NPY, f"hidden_{p['id']}_bridge.npy")
            if os.path.exists(ref_path):
                ref = np.load(ref_path)
                a, b = trimmed.reshape(-1), ref.reshape(-1)
                row["cosine_vs_hostA_npy"] = float(
                    (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                row["shape"] = list(trimmed.shape)
                row["ref_shape"] = list(ref.shape)
        results["prompts"][p["id"]] = row
        print(f"[{p['id']}] n_tok={n_tok} warm_p50={row['warm_p50']:.4f}s "
              f"mean={row['warm_mean']:.4f}s min={row['warm_min']:.4f}s "
              f"cos_hostA={row.get('cosine_vs_hostA_npy')}")
        if prof:
            top = sorted(prof, key=lambda r: -(r["real_time"] or 0))[:12]
            for t in top:
                print(f"        {t['real_time']:>10} us  {t['node_name'][:70]:72s} {t['exec_type']}")

    # per-prompt profiling CSVs
    for pid, d in results["prompts"].items():
        with open(os.path.join(args.out_dir, f"profiling_{pid}_{args.tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["status", "node_name", "layer_type", "exec_type", "real_time"])
            w.writeheader()
            for r in d["profile_rows"]:
                w.writerow(r)

    jpath = os.path.join(args.out_dir, f"investigate_{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"[saved] {jpath}  (rss {rss_gib():.2f} GiB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Root-cause investigation: OpenVINO runtime dispatch + profiling for the
Qwen3-VL INT8 weight-compressed conditioning bridge workload.

Collects (per task Phase 2):
  - CPU plugin supported properties + effective config
  - per-node runtime model dispatch info: primitiveType, runtimePrecision,
    outputPrecisions, outputLayouts, layerType, originalLayersNames,
    execOrder/execTimeMcs (when profiling enabled)
  - infer_request.get_profiling_info() per prompt
  - warm latency statistics (P1/P2/P3)

Writes to results/openvino_root_cause/profiling_<tag>_P*.csv and
runtime_model_<tag>.xml + runtime_model_<tag>_summary.json.

The workload is bit-identical to scripts/bench_openvino_bridge.py (same
add_outputs patch, same tokenization, same feeds).
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import openvino as ov

MODEL_DIR = os.path.join(ROOT, "models", "qwen3vl-openvino-int8")
OUT_DIR = os.path.join(ROOT, "results", "openvino_root_cause")
PROMPTS_FILE = os.path.join(ROOT, "prompts", "prompts.json")

T2I_TEMPLATE = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")


def find_pre_norm_node(xml_path):
    """Same logic as bench_openvino_bridge.py (kept verbatim in behavior)."""
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
        m = re.search(r"<edge[^>]*to-layer=\"%s\"[^>]*>" % pow_id, xml)
    if m is None:
        return None, norm_pow
    frm = re.search(r'from-layer="(\d+)"', m.group(0)).group(1)
    return name_by_id.get(frm), norm_pow


def build_tokenizer():
    from transformers import AutoTokenizer
    tok_dir = os.path.join(ROOT, "ComfyUI", "comfy", "text_encoders", "qwen25_tokenizer")
    return AutoTokenizer.from_pretrained(tok_dir)


def anystr(v):
    try:
        v = v.value  # OVAny -> python value
    except Exception:
        pass
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default")
    ap.add_argument("--warm-iters", type=int, default=2)
    ap.add_argument("--measure-iters", type=int, default=10)
    ap.add_argument("--prompts", default="P1,P2,P3")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--hint", default="LATENCY")
    ap.add_argument("--dyn-quant-group-size", default=None,
                    help="DYNAMIC_QUANTIZATION_GROUP_SIZE override (Phase 5)")
    ap.add_argument("--inference-precision", default=None,
                    help="INFERENCE_PRECISION_HINT override (f32/bf16/i8); needed for ISA "
                         "ceilings without bf16 (OV default hint is bf16 on this HW)")
    ap.add_argument("--save-xml", action="store_true", help="serialize runtime model IR")
    ap.add_argument("--verbose-after-compile", default=None,
                    help="set ONEDNN_VERBOSE to this value AFTER compile_model (workaround: "
                         "verbose dispatch mode crashes a constant-fold u8->i32 reorder at compile)")
    ap.add_argument("--save-hidden", action="store_true",
                    help="save trimmed hidden-state npy per prompt (accuracy checks)")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="OpenVINO model dir (default: the INT8-WC checkout); Phase 10 "
                         "points this at the fp16-ov checkout for the precision control")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="output dir (default results/openvino_root_cause; the ISA matrix "
                         "runner passes results/openvino_isa so its artifacts stay together)")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    core = ov.Core()

    lm_xml = os.path.join(args.model_dir, "openvino_language_model.xml")
    lm_bin = os.path.join(args.model_dir, "openvino_language_model.bin")
    pre_node, norm_node = find_pre_norm_node(lm_xml)
    assert pre_node is not None

    tok = build_tokenizer()
    model = core.read_model(lm_xml, lm_bin)
    wanted = {pre_node, norm_node}
    picked = {}
    for op in model.get_ordered_ops():
        fn = op.get_friendly_name()
        if fn in wanted and fn not in picked:
            picked[fn] = op
    model.add_outputs([picked[pre_node].output(0), picked[norm_node].output(0)])

    cfg = {"PERFORMANCE_HINT": args.hint, ov.properties.enable_profiling(): True}
    if args.threads:
        cfg["INFERENCE_NUM_THREADS"] = args.threads
    if args.dyn_quant_group_size is not None:
        cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = int(args.dyn_quant_group_size)
    if args.inference_precision is not None:
        cfg["INFERENCE_PRECISION_HINT"] = {"f32": ov.Type.f32, "bf16": ov.Type.bf16,
                                           "i8": ov.Type.i8}[args.inference_precision]
    t0 = time.perf_counter_ns()
    compiled = core.compile_model(model, "CPU", cfg)
    compile_s = (time.perf_counter_ns() - t0) / 1e9
    exec_devs = compiled.get_property("EXECUTION_DEVICES")
    if isinstance(exec_devs, str):
        exec_devs = [exec_devs]
    assert [str(d) for d in exec_devs] == ["CPU"], f"not CPU: {exec_devs}"

    if args.verbose_after_compile is not None:
        os.environ["ONEDNN_VERBOSE"] = args.verbose_after_compile

    # effective properties snapshot
    props = {}
    for name in ("PERF_COUNT", "INFERENCE_NUM_THREADS", "NUM_STREAMS", "PERFORMANCE_HINT",
                 "INFERENCE_PRECISION_HINT", "DYNAMIC_QUANTIZATION_GROUP_SIZE",
                 "SCHEDULING_CORE_TYPE", "ENABLE_HYPER_THREADING", "TBB_PARTITIONER"):
        try:
            props[name] = anystr(compiled.get_property(name))
        except Exception as e:
            props[name] = f"UNAVAILABLE ({type(e).__name__})"
    print(f"[inv] compile {compile_s:.3f}s props={json.dumps(props)}")

    emb_compiled = core.compile_model(
        core.read_model(os.path.join(args.model_dir, "openvino_text_embeddings_model.xml"),
                        os.path.join(args.model_dir, "openvino_text_embeddings_model.bin")), "CPU")
    emb_req = emb_compiled.create_infer_request()

    infer_req = compiled.create_infer_request()
    with open(PROMPTS_FILE) as f:
        prompts = [p for p in json.load(f)["prompts"] if p["id"] in args.prompts.split(",")]

    summary = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "openvino_version": ov.__version__,
            "tag": args.tag,
            "model_dir": args.model_dir,
            "pre_norm_node": pre_node,
            "compile_s": compile_s,
            "execution_devices": str(exec_devs),
            "effective_properties": props,
            "warm_iters": args.warm_iters,
            "measure_iters": args.measure_iters,
            "env": {k: os.environ.get(k) for k in
                    ("ONEDNN_VERBOSE", "ONEDNN_MAX_CPU_ISA", "ONEDNN_CPU_ISA_HINTS",
                     "ONEDNN_JIT_DUMP", "DNNL_MAX_CPU_ISA") if os.environ.get(k)},
        },
        "prompts": {},
    }
    if args.dyn_quant_group_size is not None:
        summary["meta"]["dyn_quant_group_size_requested"] = int(args.dyn_quant_group_size)

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

        for _ in range(args.warm_iters):
            infer_req.reset_state()
            infer_req.infer(feed)
        lat = []
        for _ in range(args.measure_iters):
            infer_req.reset_state()
            t0 = time.perf_counter_ns()
            infer_req.infer(feed)
            lat.append((time.perf_counter_ns() - t0) / 1e9)
        # one extra profiled inference (profiling overhead excluded from lat)
        infer_req.reset_state()
        res = infer_req.infer(feed)
        prof = infer_req.get_profiling_info()

        if args.save_hidden:
            out_names = [list(o.names)[0] for o in model.outputs]
            hidden_np = np.asarray(res[len(out_names) - 2], dtype=np.float32)
            ids_list = input_ids[0].tolist()
            im_starts = [i for i, t in enumerate(ids_list) if t == 151644]
            trim_at = im_starts[1] if len(im_starts) > 1 else 0
            np.save(os.path.join(out_dir, f"hidden_{p['id']}_{args.tag}.npy"),
                    hidden_np[:, trim_at:, :])

        def ptime(x):
            try:
                return x.real_time.total_seconds() * 1e6  # us
            except Exception:
                return float(x)

        rows = []
        for it in prof:
            try:
                rows.append((it.node_name, it.node_type, it.exec_type, it.status, ptime(it)))
            except Exception:
                d = dict(it)
                rows.append((d.get("node_name"), d.get("node_type"), d.get("exec_type"),
                             d.get("status"), float(d.get("real_time"))))
        rows.sort(key=lambda r: -r[4])
        with open(os.path.join(out_dir, f"profiling_{args.tag}_{p['id']}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node_name", "node_type", "exec_type", "status", "real_time_us",
                        "real_time_pct_of_total"])
            tot = sum(r[4] for r in rows) or 1.0
            for r in rows:
                w.writerow([r[0], r[1], r[2], str(r[3]), f"{r[4]:.1f}", f"{100*r[4]/tot:.3f}"])

        lat_sorted = sorted(lat)
        stats = {
            "n_tokens_full": n_tok,
            "warm_p50_s": lat_sorted[len(lat) // 2],
            "warm_mean_s": statistics.mean(lat),
            "warm_min_s": min(lat),
            "warm_max_s": max(lat),
            "warm_std_s": statistics.stdev(lat) if len(lat) > 1 else 0.0,
            "cv": (statistics.stdev(lat) / statistics.mean(lat)) if len(lat) > 1 else 0.0,
            "iterations": lat,
            "top10_nodes": [{"node": r[0], "type": r[1], "exec_type": r[2],
                             "us": round(r[4], 1)} for r in rows[:10]],
        }
        summary["prompts"][p["id"]] = stats
        print(f"[{p['id']}] n_tok={n_tok} p50={stats['warm_p50_s']:.4f}s "
              f"mean={stats['warm_mean_s']:.4f} cv={stats['cv']:.4f}")
        print(f"[{p['id']}] top5: " + " | ".join(
            f"{r[0].split('/')[-1][:40]}[{r[2]}]{r[4]:.0f}us" for r in rows[:5]))

    # ---- runtime model node table (populated execTimeMcs from the profiled runs) ----
    rt = compiled.get_runtime_model()
    node_rows = []
    for op in rt.get_ordered_ops():
        ri = op.get_rt_info()
        rec = {
            "name": op.get_friendly_name(),
            "ov_type": op.get_type_name(),
        }
        for k in ("primitiveType", "runtimePrecision", "outputPrecisions", "outputLayouts",
                  "layerType", "originalLayersNames", "execOrder", "execTimeMcs",
                  "kv_cache_precision"):
            if k in ri:
                rec[k] = anystr(ri[k])
        node_rows.append(rec)
    with open(os.path.join(out_dir, f"runtime_nodes_{args.tag}.json"), "w") as f:
        json.dump(node_rows, f, indent=1)

    if args.save_xml:
        xml_path = os.path.join(out_dir, f"runtime_model_{args.tag}.xml")
        ov.serialize(rt, xml_path)
        print(f"[saved] {xml_path} ({os.path.getsize(xml_path)/1e6:.1f} MB)")

    # ---- aggregate exec-type statistics over profiled nodes (per prompt CSVs) ----
    agg = {}
    for p in prompts:
        with open(os.path.join(out_dir, f"profiling_{args.tag}_{p['id']}.csv")) as f:
            r = list(csv.DictReader(f))
        tot = sum(float(x["real_time_us"]) for x in r) or 1.0
        prim_count = {}
        for x in r:
            prim_count.setdefault(x["exec_type"], [0, 0.0])
            prim_count[x["exec_type"]][0] += 1
            prim_count[x["exec_type"]][1] += float(x["real_time_us"])
        agg[p["id"]] = {
            "total_us": tot,
            "n_nodes": len(r),
            "by_exec_type_us": {k: [v[0], round(v[1], 1), round(100 * v[1] / tot, 2)]
                                for k, v in sorted(prim_count.items(), key=lambda kv: -kv[1][1])},
        }
    summary["profiling_aggregate"] = agg

    with open(os.path.join(out_dir, f"profiling_{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] profiling_{args.tag}_summary.json + runtime_nodes_{args.tag}.json")

    # quick stdout digest of runtime precisions/primitive types on MatMul-ish nodes
    matmul_like = [n for n in node_rows
                   if n.get("layerType", "").lower() in ("matmul", "fullyconnected", "gemm")
                   or "matmul" in n.get("name", "").lower()]
    pt = {}
    for n in matmul_like:
        key = (n.get("layerType"), n.get("primitiveType"), n.get("runtimePrecision"))
        pt[key] = pt.get(key, 0) + 1
    print(f"[digest] matmul-like runtime nodes: {len(matmul_like)}")
    for k, v in sorted(pt.items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {v:5d} x layerType={k[0]} primitive={k[1]} rtPrecision={k[2]}")


if __name__ == "__main__":
    main()

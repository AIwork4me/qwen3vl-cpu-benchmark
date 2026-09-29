#!/usr/bin/env python3
"""Experiment A: ComfyUI Qwen3-VL-8B INT8 ConvRot text encoder on CPU (Ryzen AI Max+ 395).

Uses the *product* loader path of the ComfyUI checkout in ./ComfyUI:
  comfy.sd.load_clip(..., clip_type=CLIPType.QWEN_IMAGE)  ->  QwenImage21TEModel
with load_device/offload_device hard-pinned to CPU.

Compute-path forensics (task section 8) collected at runtime:
  - torch._int_mm call counter (true int8xint8 GEMM signal) incl. devices/shapes
  - QuantizedTensor.dequantize call counter (dequant fallback signal)
  - comfy_kitchen logging at DEBUG (dispatch evidence)
  - per-layer weight introspection (QuantizedTensor layout / quant_format / convrot)
  - output tensor device/dtype/shape audit
Classification produced by scripts/classify_comfy_path.py from this JSON.

Product-path vs forced-quant variant:
  ComfyUI's conditioning encode path (encode_from_tokens) runs with
  full_precision_mm=True (hardcoded in sd1_clip.py) -> dequant + FP matmul.
  --force-quant-mm wraps encode in comfy.ops.use_quantized_matmul, the same
  context manager ComfyUI itself uses for generate(), enabling the true-int8
  path (A1 variant) for comparison.

All page-cache caveats are reported honestly (no root: caches NOT dropped).
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.join(ROOT, "ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MODEL = os.path.join(ROOT, "models", "qwen3vl-comfy-int8", "text_encoders", "qwen3vl_8b_int8_convrot.safetensors")
RESULTS_DIR = os.path.join(ROOT, "results", "comfy_cpu")
PROMPTS_FILE = os.path.join(ROOT, "prompts", "prompts.json")


def load_prompts():
    with open(PROMPTS_FILE) as f:
        return json.load(f)["prompts"]


def instrument():
    """Wrap torch._int_mm and QuantizedTensor.dequantize with call counters."""
    import torch
    counters = {"_int_mm": 0, "_int_mm_devices": set(), "_int_mm_shapes": [],
                "dequant": 0, "dequant_bytes": 0, "linear_qt": 0}

    orig_int_mm = torch._int_mm
    def wrapped_int_mm(a, b):
        counters["_int_mm"] += 1
        counters["_int_mm_devices"].add(str(a.device))
        if len(counters["_int_mm_shapes"]) < 20:
            counters["_int_mm_shapes"].append([list(a.shape), list(b.shape)])
        return orig_int_mm(a, b)
    torch._int_mm = wrapped_int_mm

    from comfy_kitchen.tensor.base import QuantizedTensor
    orig_deq = QuantizedTensor.dequantize
    def wrapped_deq(self, *a, **kw):
        counters["dequant"] += 1
        try:
            counters["dequant_bytes"] += self._qdata.numel() * self._qdata.element_size()
        except Exception:
            pass
        return orig_deq(self, *a, **kw)
    QuantizedTensor.dequantize = wrapped_deq
    return counters, orig_int_mm, QuantizedTensor, orig_deq


def audit_model(clip, counters):
    """Per-layer introspection: quant formats, devices, dtypes."""
    import torch
    from comfy_kitchen.tensor.base import QuantizedTensor
    info = {"quant_formats": {}, "layouts": {}, "n_quantized_weights": 0,
            "n_total_modules": 0, "weight_devices": set(), "convrot_layers": 0,
            "quantized_layer_names_first10": [], "max_weight_dtype": set()}
    model = clip.cond_stage_model
    for name, m in model.named_modules():
        info["n_total_modules"] += 1
        qf = getattr(m, "quant_format", None)
        if qf is not None:
            info["quant_formats"][qf] = info["quant_formats"].get(qf, 0) + 1
            lt = getattr(m, "layout_type", None)
            info["layouts"][str(lt)] = info["layouts"].get(str(lt), 0) + 1
        w = getattr(m, "weight", None)
        if w is None:
            continue
        info["weight_devices"].add(str(w.device))
        if isinstance(w, QuantizedTensor) or isinstance(getattr(w, "_qdata", None), object) and type(w).__name__ == "QuantizedTensor":
            info["n_quantized_weights"] += 1
            params = getattr(w, "_params", None)
            if params is not None and getattr(params, "convrot", False):
                info["convrot_layers"] += 1
            if len(info["quantized_layer_names_first10"]) < 10:
                info["quantized_layer_names_first10"].append(name)
            info["max_weight_dtype"].add("int8_storage(orig=%s)" % getattr(getattr(w, "_params", None), "orig_dtype", "?"))
        else:
            info["max_weight_dtype"].add(str(w.dtype))
    info["weight_devices"] = sorted(info["weight_devices"])
    info["max_weight_dtype"] = sorted(info["max_weight_dtype"])
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--warm-iters", type=int, default=5)
    ap.add_argument("--force-quant-mm", action="store_true",
                    help="wrap encode in comfy.ops.use_quantized_matmul (A1 forced variant)")
    ap.add_argument("--profile", action="store_true", help="torch.profiler one encode")
    ap.add_argument("--save-npy", action="store_true")
    args = ap.parse_args()

    # comfy.cli_args parses sys.argv at import time -> inject --cpu BEFORE any
    # comfy import so model_management.get_torch_device() returns cpu at import.
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, (args.threads or 16) // 2))
    except RuntimeError:
        pass

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        force=True)
    logging.getLogger("comfy_kitchen").setLevel(logging.DEBUG)
    logging.getLogger("comfy").setLevel(logging.INFO)

    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.args_parsing = True   # make cli_args actually parse our --cpu
    import comfy.cli_args
    assert comfy.cli_args.args.cpu is True
    import comfy.sd

    counters, orig_int_mm, QuantizedTensor, orig_deq = instrument()

    from resource_monitor import ResourceMonitor

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "raw"), exist_ok=True)

    monitor = ResourceMonitor(hz=20,
        output_csv=os.path.join(ROOT, "results", "raw", f"comfy_{args.tag}_resources.csv")).start()
    baseline = monitor.baseline_rss()

    import comfy.utils
    # ---- model disk size ----
    disk_bytes = os.path.getsize(MODEL)

    # ---- tokenizer import time (product tokenizer init) ----
    from comfy.text_encoders import qwen_image21

    # ---- model load (product path) ----
    t0 = time.perf_counter_ns()
    clip = comfy.sd.load_clip(
        [MODEL],
        embedding_directory=None,
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        model_options={
            "load_device": torch.device("cpu"),
            "offload_device": torch.device("cpu"),
        },
    )
    load_s = (time.perf_counter_ns() - t0) / 1e9
    monitor.mark("model_loaded")
    post_load = monitor.interval_stats("start", "model_loaded")
    print(f"[load] model_load_s={load_s:.3f} peak_rss_during_load={post_load['peak_rss_bytes']/2**30:.2f}GiB")

    # ---- thread info ----
    thread_info = {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "requested": args.threads,
    }
    print(f"[threads] {thread_info}")

    # ---- device / quant audit ----
    audit = audit_model(clip, counters)
    audit["patcher_load_device"] = str(clip.patcher.load_device)
    audit["patcher_offload_device"] = str(clip.patcher.offload_device)
    print(f"[audit] load_device={audit['patcher_load_device']} offload={audit['patcher_offload_device']}")
    print(f"[audit] quant_formats={audit['quant_formats']} quantized_weights={audit['n_quantized_weights']} convrot={audit['convrot_layers']}")
    print(f"[audit] weight_devices={audit['weight_devices']}")

    # ---- tokenizer ----
    t0 = time.perf_counter_ns()
    prompts = load_prompts()
    tokens = {}
    for p in prompts:
        tok = clip.tokenize(p["text"])
        tokens[p["id"]] = tok
    tok_s = (time.perf_counter_ns() - t0) / 1e9
    monitor.mark("tokenized")
    # token count audit
    tok_counts = {}
    for pid, tok in tokens.items():
        tk = tok["qwen3vl_8b"][0] if "qwen3vl_8b" in tok else next(iter(tok.values()))[0]
        tok_counts[pid] = len(tk)
    print(f"[tokens] {tok_counts} (template+prompt) tokenizer_init+tokenize={tok_s:.3f}s")

    # ---- encode ----
    import comfy.model_management as mm
    import contextlib
    n_flagged = 0
    if args.force_quant_mm:
        # Materialize weights once (runs load_reg which sets comfy_force_cast_weights=True),
        # then flip the per-module flags off so _use_quantized can go True.
        # NOTE: this measures ComfyUI's OWN int8 kernels (comfy-kitchen eager ->
        # torch._int_mm) with the force-cast blocker removed. NOT the product path.
        _ = clip.encode_from_tokens(tokens["P1"])
        import comfy.model_patcher as _mp
        clip.patcher.force_cast_weights = False
        for m in clip.cond_stage_model.modules():
            if hasattr(m, "comfy_force_cast_weights"):
                m.comfy_force_cast_weights = False
                n_flagged += 1
        print(f"[force-quant-mm] flipped comfy_force_cast_weights on {n_flagged} modules")
    def qm_ctx():
        # @contextmanager CMs are single-use: build a fresh one per encode
        if args.force_quant_mm:
            return comfy.ops.use_quantized_matmul(clip.cond_stage_model, torch.device("cpu"))
        return contextlib.nullcontext()

    results = {"meta": {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL,
        "model_disk_bytes": disk_bytes,
        "comfyui_version": open(os.path.join(COMFY, "comfyui_version.py")).read().split('"')[1],
        "torch_version": torch.__version__,
        "comfy_kitchen_backends": None,
        "threads": thread_info,
        "force_quant_mm": args.force_quant_mm,
        "force_quant_mm_modules_flipped": n_flagged,
        "tokenizer_init_and_tokenize_s": tok_s,
        "model_load_s": load_s,
        "post_load_peak_rss_bytes": post_load["peak_rss_bytes"],
        "baseline_rss_bytes": baseline,
        "audit": audit,
        "page_cache_note": "NOT cleared (no root); 'cold' = fresh process, file may be in OS page cache",
    }, "prompts": {}}

    import comfy_kitchen as ck
    results["meta"]["comfy_kitchen_backends"] = {k: v["available"] for k, v in ck.list_backends().items()}

    first_done = False
    for p in prompts:
        pid = p["id"]
        counters["_int_mm"] = 0
        counters["dequant"] = 0
        counters["dequant_bytes"] = 0
        pre_int_mm = counters["_int_mm"]

        # warmup (1)
        with qm_ctx():
            _ = clip.encode_from_tokens(tokens[pid])
        monitor.mark(f"{pid}_warmup")

        # first encode (process-first only, for P1)
        first_s = None
        if not first_done:
            t0 = time.perf_counter_ns()
            with qm_ctx():
                cond = clip.encode_from_tokens(tokens[pid])
            first_s = (time.perf_counter_ns() - t0) / 1e9
            first_done = True
            monitor.mark(f"{pid}_first")
            first_out = {"shape": list(cond.shape), "dtype": str(cond.dtype), "device": str(cond.device)}
            print(f"[{pid}] FIRST encode {first_s:.3f}s out={first_out}")
        else:
            with qm_ctx():
                cond = clip.encode_from_tokens(tokens[pid])
            monitor.mark(f"{pid}_first")

        # capture output info + counters from the last warmup/first call
        out_info = {"shape": list(cond.shape), "dtype": str(cond.dtype), "device": str(cond.device)}
        counters_after_first = {"int_mm": counters["_int_mm"] - pre_int_mm,
                                "dequant": counters["dequant"],
                                "dequant_bytes": counters["dequant_bytes"]}
        assert str(cond.device) == "cpu", f"FAIL: cond on {cond.device}"
        if pid == "P1":
            monitor.mark("first_encode_done")
            results["meta"]["post_first_encode_rss_bytes"] = monitor.interval_stats("start", "first_encode_done")["peak_rss_bytes"]

        if args.save_npy:
            np.save(os.path.join(RESULTS_DIR, f"cond_{pid}{('_a1' if args.force_quant_mm else '')}.npy"),
                    cond.to(torch.float32).numpy())

        # measured warm x N
        warm = []
        for i in range(args.warm_iters):
            t0 = time.perf_counter_ns()
            with qm_ctx():
                _ = clip.encode_from_tokens(tokens[pid])
            warm.append((time.perf_counter_ns() - t0) / 1e9)
        monitor.mark(f"{pid}_warm")

        st = monitor.interval_stats(f"{pid}_warmup", f"{pid}_warm")
        results["prompts"][pid] = {
            "n_tokens": tok_counts[pid],
            "first_encode_s": first_s,
            "warm_encode_s": warm,
            "warmup_s": None,
            "interval_stats": st,
            "peak_rss_bytes": st["peak_rss_bytes"],
            "output": out_info,
            "compute_counters": counters_after_first,
        }
        w = warm
        print(f"[{pid}] warm mean={sum(w)/len(w):.3f}s p50={sorted(w)[len(w)//2]:.3f}s "
              f"min={min(w):.3f}s max={max(w):.3f}s peak_rss={st['peak_rss_bytes']/2**30:.2f}GiB "
              f"counters(int_mm={counters_after_first['int_mm']}, dequant={counters_after_first['dequant']})")

        # optional profiler on one encode of P1
        if args.profile and pid == "P1":
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                with qm_ctx():
                    _ = clip.encode_from_tokens(tokens[pid])
            table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
            ppath = os.path.join(RESULTS_DIR, f"profiler_{args.tag}.txt")
            with open(ppath, "w") as f:
                f.write(table)
            print(f"[profiler] saved {ppath}")
            ops = {}
            for ev in prof.key_averages():
                ops[ev.key] = ev.self_cpu_time_total
            results["prompts"][pid]["profiler_top_ops"] = dict(
                sorted(ops.items(), key=lambda x: -x[1])[:25])

    monitor.stop()
    results["meta"]["full_run"] = monitor.full_run()

    monitor.save_csv(os.path.join(RESULTS_DIR, f"resources_{args.tag}.csv"))
    jpath = os.path.join(RESULTS_DIR, f"comfy_{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    cpath = os.path.join(RESULTS_DIR, f"comfy_{args.tag}.csv")
    with open(cpath, "w") as f:
        f.write("prompt_id,phase,iter,latency_s\n")
        for pid, d in results["prompts"].items():
            if d["first_encode_s"] is not None:
                f.write(f"{pid},first,{d['first_encode_s']}\n")
            for i, s in enumerate(d["warm_encode_s"]):
                f.write(f"{pid},warm_{i},{s}\n")
    print(f"[saved] {jpath}\n[saved] {cpath}")


if __name__ == "__main__":
    main()

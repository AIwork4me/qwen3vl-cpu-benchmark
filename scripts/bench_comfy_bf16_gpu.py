#!/usr/bin/env python3
"""Experiment C: BF16 Qwen3-VL-8B text encoder on the local Radeon 8060S (gfx1151, UMA/GTT).

Same ComfyUI product path as experiment A (load_clip / QwenImage21TEModel), but on the
HIP device via ComfyUI's own device selection (ROCm torch: torch.device('cuda')==HIP).

Measured: encode latency (torch.cuda.synchronize-bracketed), GPU tensor peak
(torch.cuda.max_memory_allocated), GTT pool headroom (mem_get_info), GPU busy %
(sysfs gpu_busy_percent sampled at 10 Hz in a thread), compute dtype forensics
(forward-pre-hook on the first Linear layers records the ACTUAL GEMM input dtype),
CPU-side RSS via the shared ResourceMonitor.

Output: results/comfy_gpu/comfy_gpu_<tag>.json|csv + cond npy for accuracy compare.
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.join(ROOT, "ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MODEL = os.path.join(ROOT, "models", "qwen-image-2.1", "text_encoders", "qwen3vl_8b_bf16.safetensors")
RESULTS_DIR = os.path.join(ROOT, "results", "comfy_gpu")
PROMPTS_FILE = os.path.join(ROOT, "prompts", "prompts.json")


def gpu_busy_sampler(hz=10.0, stop=None, out=None):
    """Sample /sys/class/drm/card1/device/gpu_busy_percent into a list."""
    path = "/sys/class/drm/card1/device/gpu_busy_percent"
    while not stop.is_set():
        try:
            with open(path) as f:
                out.append((time.perf_counter_ns(), int(f.read().strip())))
        except Exception:
            out.append((time.perf_counter_ns(), None))
        stop.wait(1.0 / hz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpu_run1")
    ap.add_argument("--warm-iters", type=int, default=5)
    ap.add_argument("--save-npy", action="store_true")
    args = ap.parse_args()

    os.environ.pop("OMP_NUM_THREADS", None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    import torch
    print(f"[gpu] torch {torch.__version__} hip={torch.version.hip} "
          f"cuda_avail={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}")
    assert torch.cuda.is_available() and torch.version.hip, "ROCm/HIP torch required"

    sys.argv = [sys.argv[0]]  # no --cpu: let model_management select the HIP device
    import comfy.options
    comfy.options.args_parsing = True
    import comfy.cli_args
    import comfy.sd
    import comfy.model_management as mm
    dev = mm.get_torch_device()
    print(f"[gpu] comfy torch_device = {dev}")
    assert str(dev).startswith("cuda"), f"expected HIP device, got {dev}"

    from resource_monitor import ResourceMonitor
    os.makedirs(RESULTS_DIR, exist_ok=True)

    monitor = ResourceMonitor(hz=20,
        output_csv=os.path.join(ROOT, "results", "raw", f"comfy_{args.tag}_resources.csv")).start()

    # GPU busy% sampler thread
    stop = threading.Event()
    busy = []
    th = threading.Thread(target=gpu_busy_sampler, kwargs=dict(stop=stop, out=busy), daemon=True)
    th.start()

    free0, total0 = torch.cuda.mem_get_info()
    print(f"[gpu] GTT pool: free {free0/2**30:.1f} / total {total0/2**30:.1f} GiB")

    # ---- load (product path, weights direct to HIP device) ----
    t0 = time.perf_counter_ns()
    clip = comfy.sd.load_clip(
        [MODEL], embedding_directory=None, clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        model_options={"load_device": torch.device("cuda"), "offload_device": torch.device("cuda")})
    load_s = (time.perf_counter_ns() - t0) / 1e9
    torch.cuda.synchronize()
    monitor.mark("model_loaded")
    post_load = monitor.interval_stats("start", "model_loaded")
    print(f"[load] model_load_s={load_s:.3f} peak_rss_during_load={post_load['peak_rss_bytes']/2**30:.2f}GiB")

    # ---- device + dtype audit ----
    audit = {"weight_devices": set(), "weight_dtypes": set(), "n_modules": 0,
             "compute_input_dtypes": [], "hooked": []}
    hooks = []
    def pre_hook(module, inp):
        if len(audit["compute_input_dtypes"]) < 12:
            audit["compute_input_dtypes"].append(str(inp[0].dtype))
    for name, m in clip.cond_stage_model.named_modules():
        audit["n_modules"] += 1
        w = getattr(m, "weight", None)
        if w is not None:
            audit["weight_devices"].add(str(w.device))
            audit["weight_dtypes"].add(str(w.dtype))
        if type(m).__name__ == "Linear" and len(audit["hooked"]) < 3 and "layers.0." in name:
            hooks.append(m.register_forward_pre_hook(pre_hook))
            audit["hooked"].append(name)
    audit["weight_devices"] = sorted(audit["weight_devices"])
    audit["weight_dtypes"] = sorted(audit["weight_dtypes"])
    audit["patcher_load_device"] = str(clip.patcher.load_device)
    audit["patcher_offload_device"] = str(clip.patcher.offload_device)
    print(f"[audit] {audit}")

    # ---- tokenize ----
    t0 = time.perf_counter_ns()
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)["prompts"]
    tokens = {p["id"]: clip.tokenize(p["text"]) for p in prompts}
    tok_s = (time.perf_counter_ns() - t0) / 1e9
    monitor.mark("tokenized")
    tok_counts = {pid: len(next(iter(t.values()))[0]) for pid, t in tokens.items()}
    print(f"[tokens] {tok_counts} tokenize={tok_s:.3f}s")

    torch.cuda.reset_peak_memory_stats()
    alloc_after_load = torch.cuda.memory_allocated()

    results = {"meta": {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL,
        "model_disk_bytes": os.path.getsize(MODEL),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "gpu_name": torch.cuda.get_device_name(0),
        "comfy_device": str(dev),
        "threads": {"torch_num_threads": torch.get_num_threads()},
        "tokenizer_init_and_tokenize_s": tok_s,
        "model_load_s": load_s,
        "post_load_peak_rss_bytes": post_load["peak_rss_bytes"],
        "gtt_pool_free_at_start_gib": round(free0 / 2**30, 2),
        "gtt_pool_total_gib": round(total0 / 2**30, 2),
        "alloc_after_load_bytes": alloc_after_load,
        "audit": audit,
    }, "prompts": {}}

    first_done = False
    for p in prompts:
        pid = p["id"]
        with torch.no_grad():
            # warmup
            torch.cuda.synchronize(); t0 = time.perf_counter_ns()
            _ = clip.encode_from_tokens(tokens[pid])
            torch.cuda.synchronize()
            warmup_s = (time.perf_counter_ns() - t0) / 1e9
            monitor.mark(f"{pid}_warmup")

            first_s = None
            if not first_done:
                torch.cuda.synchronize(); t0 = time.perf_counter_ns()
                cond = clip.encode_from_tokens(tokens[pid])
                torch.cuda.synchronize()
                first_s = (time.perf_counter_ns() - t0) / 1e9
                first_done = True
                monitor.mark(f"{pid}_first")
            else:
                torch.cuda.synchronize(); t0 = time.perf_counter_ns()
                cond = clip.encode_from_tokens(tokens[pid])
                torch.cuda.synchronize()
                monitor.mark(f"{pid}_first")

            warm = []
            for _ in range(args.warm_iters):
                torch.cuda.synchronize(); t0 = time.perf_counter_ns()
                _ = clip.encode_from_tokens(tokens[pid])
                torch.cuda.synchronize()
                warm.append((time.perf_counter_ns() - t0) / 1e9)
        monitor.mark(f"{pid}_warm")

        st = monitor.interval_stats(f"{pid}_warmup", f"{pid}_warm")
        n0 = len(busy)
        # busy% during the last encode window is not separable; report whole-run below
        if args.save_npy:
            import numpy as np
            np.save(os.path.join(RESULTS_DIR, f"cond_{pid}_gpu.npy"), cond.to(torch.float32).cpu().numpy())

        results["prompts"][pid] = {
            "n_tokens": tok_counts[pid],
            "first_encode_s": first_s,
            "warm_encode_s": warm,
            "warmup_s": warmup_s,
            "interval_stats": st,
            "output": {"shape": list(cond.shape), "dtype": str(cond.dtype), "device": str(cond.device)},
            "cuda_memory": {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            },
        }
        w = warm
        print(f"[{pid}] first={first_s if first_s is None else round(first_s,3)} "
              f"warm mean={sum(w)/len(w):.3f}s p50={sorted(w)[len(w)//2]:.3f}s min={min(w):.3f}s max={max(w):.3f}s "
              f"out={cond.dtype}/{cond.device} peak_gpu={torch.cuda.max_memory_allocated()/2**30:.2f}GiB")

    stop.set()
    # GPU busy% over the run
    vals = [v for _, v in busy if v is not None]
    results["meta"]["gpu_busy_pct_mean"] = round(sum(vals) / len(vals), 1) if vals else None
    results["meta"]["gpu_busy_pct_max"] = max(vals) if vals else None
    results["meta"]["gpu_busy_n_samples"] = len(vals)
    results["meta"]["peak_gpu_alloc_bytes"] = torch.cuda.max_memory_allocated()
    results["meta"]["gtt_pool_free_at_end_gib"] = round(torch.cuda.mem_get_info()[0] / 2**30, 2)
    results["meta"]["full_run"] = monitor.full_run()
    print(f"[gpu-busy] mean={results['meta']['gpu_busy_pct_mean']}% max={results['meta']['gpu_busy_pct_max']}% "
          f"n={results['meta']['gpu_busy_n_samples']} | peak_gpu_alloc={torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
          f"| GTT free end={results['meta']['gtt_pool_free_at_end_gib']}GiB")

    monitor.stop()
    monitor.save_csv(os.path.join(RESULTS_DIR, f"resources_{args.tag}.csv"))
    jpath = os.path.join(RESULTS_DIR, f"comfy_gpu_{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    cpath = os.path.join(RESULTS_DIR, f"comfy_gpu_{args.tag}.csv")
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

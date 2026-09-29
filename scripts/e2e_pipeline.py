#!/usr/bin/env python3
"""Unified Qwen-Image 2.1 end-to-end harness (GPU-heavy G / Hybrid H / CPU-TE C).

Runs in .venv-comfy-rocm. One process = one route (clean A/B; the driver spawns
fresh processes for cold-start repetition). Stage timing per image:

  T1 tokenize -> T2 text-encode -> T3 cond prep -> T4 DiT first step ...
  T5 DiT final step -> T6 VAE start -> T7 image ready (headline) -> T8 PNG saved

GPU-side boundaries are wrapped in torch.cuda.synchronize() (Phase 6). A 20 Hz
monitor (e2e_monitor) writes per-run CSV with CPU/GPU/GTT/temp/power; marks align
stage windows to the monitor timeline.

Routes:
  gpu     (G) ComfyUI product path: BF16 qwen3vl TE via comfy.sd.load_clip with
             DEFAULT device selection (TE lands on the Radeon), DiT+VAE on Radeon.
  native  (C) same product path but TE forced to CPU (int8-convrot file) — the
             route the repo already characterizes; context reference.
  ov      (H) Hybrid: OpenVINO INT8 qwen3vl encoder in-process on the Zen 5 CPU
             (scripts/ov_encoder.py, ComfyUI's own tokenizer), conditioning
             injected as [[tensor, {}]] — structurally identical to the product
             conditioning; DiT+VAE on Radeon.

Everything except the conditioning route (and --dq) is held fixed: same DiT/VAE
files, sampler=euler scheduler=simple, steps, cfg, resolution, seed, latent.

Extra modes:
  --pipelined       (ov) encode prompt n+1 on CPU while the GPU samples image n
  --contend-text T  (ov) run CPU encodes of T continuously during sampling
                    (CPU/GPU UMA contention experiment; pair with a run without)
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.join(ROOT, "ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DIT = os.path.join(ROOT, "models", "qwen-image-2.1-full", "diffusion_models", "qwen_image_2.1_bf16.safetensors")
VAE_M = os.path.join(ROOT, "models", "qwen-image-2.1-full", "vae", "qwen_image_2.1_vae_bf16.safetensors")
TE_BF16 = os.path.join(ROOT, "models", "qwen-image-2.1", "text_encoders", "qwen3vl_8b_bf16.safetensors")
TE_INT8 = os.path.join(ROOT, "models", "qwen3vl-comfy-int8", "text_encoders", "qwen3vl_8b_int8_convrot.safetensors")
OV_MODEL_DIR = os.path.join(ROOT, "models", "qwen3vl-openvino-int8")
PROMPTS_FILE = os.path.join(ROOT, "prompts", "prompts.json")
OV_CACHE = os.path.join(ROOT, "models", ".ov_cache")

NEG = " "  # TextEncodeQwenImage21 prevent_empty_text semantics


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def now_ns():
    return time.perf_counter_ns()


def load_prompt_set(path, ids=None):
    with open(path) as f:
        data = json.load(f)
    items = data["prompts"] if isinstance(data, dict) else data
    if ids:
        want = set(ids)
        items = [p for p in items if p["id"] in want]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--group", default="ab", help="subdir under results/e2e")
    ap.add_argument("--route", required=True, choices=["gpu", "native", "ov"])
    ap.add_argument("--dq", type=int, default=32)
    ap.add_argument("--ov-cache-dir", default=OV_CACHE)
    ap.add_argument("--prompts", default="P3", help="comma ids from prompts.json")
    ap.add_argument("--prompt-file", default=PROMPTS_FILE)
    ap.add_argument("--seeds", default="20260929")
    ap.add_argument("--iters", type=int, default=1, help="images per (prompt,seed)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--save-cond", action="store_true")
    ap.add_argument("--pipelined", action="store_true")
    ap.add_argument("--contend-text", default=None)
    ap.add_argument("--no-dit-preload", action="store_true",
                    help="do not force_full_load the DiT before sampling (probe streaming behavior)")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="throwaway sampling pass at N steps before the plan (triggers kernel "
                         "compilation/autotune so the FIRST plan image matches steady-state determinism); "
                         "not included in any timing")
    args = ap.parse_args()

    outdir = os.path.join(ROOT, "results", "e2e", args.group)
    imgdir = os.path.join(ROOT, "results", "e2e", "images")
    rawdir = os.path.join(ROOT, "results", "e2e", "raw")
    conddir = os.path.join(ROOT, "results", "e2e", "cond")
    for d in (outdir, imgdir, rawdir, conddir):
        os.makedirs(d, exist_ok=True)

    sys.argv = [sys.argv[0]]
    import comfy.options
    comfy.options.args_parsing = True
    import comfy.cli_args  # noqa: F401  (initializes comfy arg state)
    import comfy.sd
    import comfy.sample
    import comfy.model_management as mm
    import torch
    import numpy as np
    from PIL import Image
    from e2e_monitor import E2EMonitor, window_stats

    dev = mm.get_torch_device()
    assert str(dev).startswith("cuda"), f"expected HIP device, got {dev}"

    mon = E2EMonitor(os.path.join(rawdir, f"{args.tag}.csv"), hz=20.0)
    mon.start()
    t_proc0 = now_ns()
    epoch_at_t0 = time.time() - (t_proc0 - mon.t0) / 1e9  # epoch sec at monitor t0

    free0, total0 = torch.cuda.mem_get_info()
    meta = {
        "tag": args.tag, "group": args.group, "route": args.route,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "gtt_free_gib_start": round(free0 / 2**30, 2), "gtt_total_gib": round(total0 / 2**30, 2),
        "vram_state": mm.vram_state.name if hasattr(mm.vram_state, "name") else str(mm.vram_state),
        "te_device_selected": str(mm.text_encoder_device()),
        "dit": DIT, "vae": VAE_M,
        "dit_sha16": sha16(DIT) if not os.path.exists(os.path.join(rawdir, "model_sha16.json")) else None,
        "steps": args.steps, "cfg": args.cfg, "sampler": args.sampler,
        "scheduler": args.scheduler, "width": args.width, "height": args.height,
        "prompts": args.prompts, "seeds": args.seeds, "iters": args.iters,
        "pipelined": args.pipelined, "contend_text": bool(args.contend_text),
        "comfyui_version": "0.37.0",
        "aotriton": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0"),
    }
    for sysf, name in (("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "cpu_governor"),
                       ("/sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference", "cpu_epp")):
        try:
            meta[name] = open(sysf).read().strip()
        except OSError:
            meta[name] = "unavailable"
    # model hashes once (cached, they are big files)
    sha_cache = os.path.join(rawdir, "model_sha16.json")
    hashes = {}
    if os.path.exists(sha_cache):
        hashes = json.load(open(sha_cache))
    for name, p in (("dit", DIT), ("vae", VAE_M), ("te_bf16", TE_BF16), ("te_int8", TE_INT8)):
        if name not in hashes:
            hashes[name] = sha16(p)
    json.dump(hashes, open(sha_cache, "w"), indent=1)
    meta["dit_sha16"] = hashes["dit"]
    meta["vae_sha16"] = hashes["vae"]

    # ---------------- model loading (T0 window) ----------------
    mon.mark("load_start")
    loads = {}
    t0 = now_ns()
    model = comfy.sd.load_diffusion_model(DIT)
    loads["dit_read_s"] = (now_ns() - t0) / 1e9
    t0 = now_ns()
    vae_sd, vae_meta = comfy.utils.load_torch_file(VAE_M, return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_meta)
    vae.throw_exception_if_invalid()
    loads["vae_read_s"] = (now_ns() - t0) / 1e9

    encoder = None
    clip = None
    if args.route == "gpu":
        t0 = now_ns()
        clip = comfy.sd.load_clip([TE_BF16], embedding_directory=None,
                                  clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
        loads["te_read_s"] = (now_ns() - t0) / 1e9
        meta["te_model"] = TE_BF16
        meta["te_sha16"] = hashes["te_bf16"]
    elif args.route == "native":
        t0 = now_ns()
        clip = comfy.sd.load_clip(
            [TE_INT8], embedding_directory=None, clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
            model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")})
        loads["te_read_s"] = (now_ns() - t0) / 1e9
        meta["te_model"] = TE_INT8
        meta["te_sha16"] = hashes["te_int8"]
    else:  # ov
        from ov_encoder import OVQwenEncoder
        t0 = now_ns()
        encoder = OVQwenEncoder(OV_MODEL_DIR, dq=args.dq, cache_dir=args.ov_cache_dir)
        loads["ov_total_s"] = (now_ns() - t0) / 1e9
        loads["ov_compile_read_s"] = encoder.compile_read_s
        loads["ov_compile_s"] = encoder.compile_s
        meta["ov"] = encoder.meta()
        meta["te_model"] = OV_MODEL_DIR
    mon.mark("load_end")
    loads["total_load_s"] = (mon.marks[-1][1] - mon.marks[-2][1]) / 1e9

    # DiT full GPU load AFTER TE encode would be product order; but TE encode needs
    # the plan below. We preload the DiT lazily at the first sampling call, timed
    # inside the T4 window of the first image (records real transfer cost), unless
    # disabled for probing.
    dit_preloaded = False
    dit_preload_s = None

    def ensure_dit_loaded():
        """Returns True iff the load happened in THIS call (first image only)."""
        nonlocal dit_preloaded, dit_preload_s
        if dit_preloaded or args.no_dit_preload:
            return False
        torch.cuda.synchronize()
        t0 = now_ns()
        mm.load_models_gpu([model], force_full_load=True)
        torch.cuda.synchronize()
        dit_preload_s = (now_ns() - t0) / 1e9
        dit_preloaded = True
        return True

    ids = None if args.prompts == "all" else [p for p in args.prompts.split(",") if p]
    prompt_items = load_prompt_set(args.prompt_file, ids)
    seeds = [int(s) for s in args.seeds.split(",")]
    plan = []
    for pi in prompt_items:
        seed_list = [pi["seed"]] if "seed" in pi else seeds
        for seed in seed_list:
            for k in range(args.iters):
                plan.append({"id": pi["id"], "text": pi["text"], "seed": seed, "iter": k})

    latent0 = torch.zeros([1, 64, args.height // 16, args.width // 16],
                          device=mm.intermediate_device())
    latent0 = comfy.sample.fix_empty_latent_channels(model, latent0)

    def encode_cond(text):
        """Route-specific (T1+T2+T3). Returns (cond, stage_dict)."""
        st = {}
        if clip is not None:
            t1 = now_ns()
            toks_pos = clip.tokenize(text, keep_vision=True, prevent_empty_text=True)
            toks_neg = clip.tokenize(NEG, keep_vision=True, prevent_empty_text=True)
            st["tokenize_s"] = (now_ns() - t1) / 1e9
            torch.cuda.synchronize()
            t2 = now_ns()
            pos = clip.encode_from_tokens_scheduled(toks_pos)
            neg = clip.encode_from_tokens_scheduled(toks_neg)
            torch.cuda.synchronize()
            st["encode_s"] = (now_ns() - t2) / 1e9
            st["n_tokens_kept"] = int(pos[0][0].shape[1])
            st["cond_pos_shape"] = list(pos[0][0].shape)
            cond = (pos, neg)
        else:
            t2 = now_ns()
            rp = encoder.encode(text, synchronize_gpu=torch.cuda.synchronize)
            rn = encoder.encode(NEG, synchronize_gpu=torch.cuda.synchronize)
            st["encode_s"] = (now_ns() - t2) / 1e9
            st["encode_pos_s"] = rp["encode_s"]
            st["encode_neg_s"] = rn["encode_s"]
            st["n_tokens_full"] = rp["n_tokens_full"]
            st["n_tokens_kept"] = rp["n_tokens_kept"]
            t3 = now_ns()
            pos = ([[torch.from_numpy(rp["hidden"].copy()).float(), {}]],)
            neg = ([[torch.from_numpy(rn["hidden"].copy()).float(), {}]],)
            st["cond_prep_s"] = (now_ns() - t3) / 1e9
            st["cond_pos_shape"] = list(rp["hidden"].shape)
            st["ov_ids_pos"] = rp["ids"][:8]  # first ids for alignment debugging
            cond = (pos[0], neg[0])
        return cond, st

    def sample_one(cond, seed, idx):
        """T4+T5 (+T6+T7 VAE). Returns (stage dict, image ndarray, t_ready_ns)."""
        st = {}
        noise = comfy.sample.prepare_noise(latent0, seed)
        loaded_now = ensure_dit_loaded()
        step_times = []

        def cb(i, denoised, x, total_steps):
            step_times.append((int(i), now_ns()))

        mon.mark(f"img{idx}_dit_start")
        torch.cuda.synchronize()
        t4 = now_ns()
        samples = comfy.sample.sample(
            model, noise, args.steps, args.cfg, args.sampler, args.scheduler,
            cond[0], cond[1], latent0, callback=cb, disable_pbar=True, seed=seed)
        torch.cuda.synchronize()
        t5 = now_ns()
        st["dit_s"] = (t5 - t4) / 1e9
        st["dit_steps"] = step_times
        if loaded_now:
            st["dit_preload_in_window_s"] = dit_preload_s  # first-image transfer cost
        mon.mark(f"img{idx}_dit_end")
        torch.cuda.synchronize()
        t6 = now_ns()
        imgs = vae.decode(samples)
        torch.cuda.synchronize()
        t7 = now_ns()
        st["vae_s"] = (t7 - t6) / 1e9
        img = imgs[0].detach().float().cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype("uint8")
        t_post = now_ns()
        st["postprocess_s"] = (t_post - t7) / 1e9  # D2H + uint8 conversion
        return st, img, t7

    if args.warmup_steps > 0:
        wc, _ = encode_cond(plan[0]["text"])
        ensure_dit_loaded()
        wnoise = comfy.sample.prepare_noise(latent0, plan[0]["seed"])
        torch.cuda.synchronize()
        comfy.sample.sample(model, wnoise, args.warmup_steps, args.cfg, args.sampler,
                            args.scheduler, wc[0], wc[1], latent0, disable_pbar=True,
                            seed=plan[0]["seed"])
        torch.cuda.synchronize()
        del wc, wnoise
        print(f"[warmup] {args.warmup_steps}-step throwaway pass done", flush=True)

    images_meta = []
    cond_shape_ref = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if args.pipelined else None
    next_future = None

    try:
        for idx, item in enumerate(plan):
            label = f"{item['id']}_s{item['seed']}_i{item['iter']}"
            mon.mark(f"img{idx}_{label}_start")
            rec = {"idx": idx, "label": label, "prompt_id": item["id"],
                   "seed": item["seed"], "iter": item["iter"]}
            t_img0 = now_ns()

            if args.pipelined:
                # Hybrid pipelined: cond for image i must be ready (from previous
                # iteration's future); submit encode of i+1 BEFORE sampling image i.
                if next_future is None:
                    t0c = now_ns()
                    cond, st_enc = encode_cond(item["text"])
                    st_enc["standalone_s"] = (now_ns() - t0c) / 1e9
                else:
                    t_wait0 = now_ns()
                    cond, st_enc = next_future.result()
                    st_enc["wait_s"] = (now_ns() - t_wait0) / 1e9
                    st_enc["total_from_submit_s"] = (now_ns() - st_enc.pop("submit_t_ns", now_ns())) / 1e9
                if idx + 1 < len(plan):
                    nxt = plan[idx + 1]
                    submit_t = now_ns()

                    def job(txt=nxt["text"], t=submit_t):
                        c, s = encode_cond(txt)
                        s["submit_t_ns"] = t
                        return c, s
                    next_future = executor.submit(job)
                st, img, t_ready = sample_one(cond, item["seed"], idx)
                rec.update(st_enc)
                rec.update(st)
            else:
                cond, st_enc = encode_cond(item["text"])
                if args.contend_text:
                    stop_ev = threading.Event()

                    def spin():
                        while not stop_ev.is_set():
                            encoder.encode(args.contend_text)
                    th = threading.Thread(target=spin, daemon=True)
                    th.start()
                    st, img, t_ready = sample_one(cond, item["seed"], idx)
                    stop_ev.set()
                    th.join(timeout=30)
                    rec["contend_active"] = True
                else:
                    st, img, t_ready = sample_one(cond, item["seed"], idx)
                rec.update(st_enc)
                rec.update(st)

            # conditioning identity bookkeeping (per prompt id: token counts
            # legitimately differ ACROSS prompts; within one prompt they must match)
            pos_t = cond[0][0][0]
            ref_for_prompt = cond_shape_ref.setdefault(item["id"], list(pos_t.shape))
            assert list(pos_t.shape) == ref_for_prompt, \
                f"cond shape changed for {item['id']}: {pos_t.shape} vs {ref_for_prompt}"
            rec["cond_shape"] = list(pos_t.shape)
            rec["cond_pos_mean"] = float(pos_t.float().mean())
            rec["cond_pos_std"] = float(pos_t.float().std())
            if args.save_cond and idx == 0:
                np.save(os.path.join(conddir, f"cond_{args.tag}_{item['id']}_pos.npy"),
                        pos_t.float().numpy())
                np.save(os.path.join(conddir, f"cond_{args.tag}_{item['id']}_neg.npy"),
                        cond[1][0][0].float().numpy())

            rec["total_gen_s"] = (t_ready - t_img0) / 1e9  # T1..T7 (image ready, save excluded)
            t8 = now_ns()
            png = os.path.join(imgdir, f"{args.tag}_{label}.png")
            Image.fromarray(img).save(png)
            rec["png_save_s"] = (now_ns() - t8) / 1e9
            rec["png"] = os.path.relpath(png, ROOT)
            rec["img_sha16"] = sha16(png)
            rec["total_save_inclusive_s"] = (now_ns() - t_img0) / 1e9
            sidecar = {
                "prompt": item["text"], "prompt_id": item["id"], "seed": item["seed"],
                "route": args.route, "dq": args.dq if args.route == "ov" else None,
                "resolution": [args.width, args.height], "steps": args.steps,
                "cfg": args.cfg, "sampler": args.sampler, "scheduler": args.scheduler,
                "negative": NEG, "dit_sha16": meta["dit_sha16"],
                "vae_sha16": meta["vae_sha16"],
                "te_sha16": meta.get("te_sha16"), "model_dir": meta.get("te_model"),
                "comfyui_version": meta["comfyui_version"], "torch": meta["torch"],
                "aotriton": meta["aotriton"],
                "encode_s": rec.get("encode_s"), "dit_s": rec.get("dit_s"),
                "vae_s": rec.get("vae_s"), "postprocess_s": rec.get("postprocess_s"),
                "png_save_s": rec.get("png_save_s"),
                "total_gen_s": rec.get("total_gen_s"),
                "total_save_inclusive_s": rec.get("total_save_inclusive_s"),
                "tag": args.tag,
            }
            with open(png.replace(".png", ".json"), "w") as f:
                json.dump(sidecar, f, indent=1)
            free1, _ = torch.cuda.mem_get_info()
            rec["gtt_free_gib_after"] = round(free1 / 2**30, 2)
            rec["cuda_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()
            images_meta.append(rec)
            print(f"[{args.tag} {label}] enc={rec.get('encode_s', 0):.3f}s "
                  f"dit={rec['dit_s']:.2f}s vae={rec['vae_s']:.2f}s "
                  f"total={rec['total_gen_s']:.2f}s peak={rec['cuda_peak_alloc_gib']}GiB",
                  flush=True)

        if executor:
            executor.shutdown(wait=True)
    finally:
        mon.stop()
        mon.flush()

    # ---------------- post-run: monitor windows ----------------
    marks = dict(mon.marks)
    run = {"meta": meta, "loads": loads, "images": images_meta, "marks_ns": {
        k: v for k, v in mon.marks}}

    def win(a, b, key):
        ta = marks.get(a)
        tb = marks.get(b)
        if ta is None or tb is None:
            return None
        return window_stats(mon.samples, ta, tb, key)

    run["monitor_summary"] = {
        "whole_run": {k: window_stats(mon.samples, mon.t0, marks["stop"], k)
                      for k in ("proc_cpu_pct", "sys_cpu_pct", "gpu_busy_pct",
                                "gtt_used_bytes", "rss_bytes", "sys_available_bytes",
                                "gpu_temp_c", "gpu_power_w", "cpu_temp_c", "cpu_freq_mhz")},
        "n_samples": len(mon.samples),
    }
    for i, rec in enumerate(images_meta):
        ws = {}
        for a, b, name in ((f"img{i}_{rec['label']}_start", f"img{i}_dit_start", "enc"),
                           (f"img{i}_dit_start", f"img{i}_dit_end", "dit")):
            ws[name] = {k: win(a, b, k) for k in ("gpu_busy_pct", "proc_cpu_pct", "gtt_used_bytes")}
        rec["windows"] = ws

    run["epoch_at_t0"] = epoch_at_t0
    if images_meta:
        # epoch when the FIRST image became ready (for cold TTFI reconstruction)
        m0 = dict(mon.marks)
        t0_img = m0.get("img0_" + images_meta[0]["label"] + "_start", mon.t0)
        run["first_image_ready_epoch"] = epoch_at_t0 + (t0_img + images_meta[0]["total_gen_s"] * 1e9 - mon.t0) / 1e9
        run["mon_t0_epoch"] = epoch_at_t0
    out_json = os.path.join(outdir, f"{args.tag}.json")
    with open(out_json, "w") as f:
        json.dump(run, f, indent=1, default=str)
    print(f"[saved] {out_json} ({len(images_meta)} images, "
          f"wall={(now_ns() - t_proc0) / 1e9:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

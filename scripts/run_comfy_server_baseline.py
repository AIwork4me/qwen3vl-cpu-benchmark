#!/usr/bin/env python3
"""Phase 2: run the REAL ComfyUI server (product path) for the GPU-heavy baseline.

Starts ComfyUI headless on 127.0.0.1:8199, submits the Qwen-Image 2.1 workflow
(official node set: UNETLoader/CLIPLoader(qwen_image)/VAELoader/
TextEncodeQwenImage21/KSampler/VAEDecode/SaveImage) for P1/P2/P3, records:
  - workflow JSON (API format) per prompt
  - ComfyUI startup log (device selection, model management behavior)
  - per-prompt prompt-execution time from the server log
  - output PNGs + sidecar meta
Results -> results/e2e/gpu_baseline/
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.join(ROOT, "ComfyUI")
OUT = os.path.join(ROOT, "results", "e2e", "gpu_baseline")
PY = os.path.join(ROOT, ".venv-comfy-rocm", "bin", "python")
PORT = 8199
STEPS = 20
CFG = 2.5
SEED = 20260929
RES = 1024


def workflow(prompt_text, prefix):
    return {
        "7": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}},
        "8": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image", "device": "default"}},
        "9": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "10": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["8", 0], "prompt": prompt_text, "negative_prompt": " ",
            "resolution": RES}},
        "11": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["8", 0], "prompt": " ", "negative_prompt": " ",
            "resolution": RES}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": SEED, "steps": STEPS, "cfg": CFG, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0, "model": ["7", 0],
            "positive": ["10", 0], "negative": ["11", 0], "latent_image": ["10", 2]}},
        "4": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["9", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": prefix}},
    }


def wait_ready(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/system_stats", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def post_prompt(wf):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/prompt",
                                 data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["prompt_id"]


def get_history(pid, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/history/{pid}", timeout=5) as r:
                h = json.load(r)
            if pid in h:
                return h[pid]
        except Exception:
            pass
        time.sleep(2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="P1,P2,P3")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    prompts = {p["id"]: p["text"] for p in json.load(open(os.path.join(ROOT, "prompts", "prompts.json")))["prompts"]}
    ids = [p for p in args.prompts.split(",") if p]

    log_path = os.path.join(OUT, "comfyui_server.log")
    logf = open(log_path, "w")
    env = dict(os.environ, TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1")
    srv = subprocess.Popen(
        [PY, "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
         "--output-directory", OUT],
        cwd=COMFY, stdout=logf, stderr=subprocess.STDOUT, env=env)
    try:
        assert wait_ready(), "server did not become ready"
        print("[server] ready", flush=True)
        results = {}
        for pid_text in ids:
            wf = workflow(prompts[pid_text], f"server_{pid_text}")
            wf_path = os.path.join(OUT, f"workflow_{pid_text}.json")
            json.dump(wf, open(wf_path, "w"), indent=1)
            t0 = time.time()
            prompt_id = post_prompt(wf)
            hist = get_history(prompt_id)
            wall = time.time() - t0
            assert hist, f"no history for {pid_text}"
            status = hist.get("status", {})
            ok = status.get("status_str") == "success"
            outputs = hist.get("outputs", {})
            imgs = []
            for node_out in outputs.values():
                for im in node_out.get("images", []):
                    imgs.append(im)
            # server-reported per-node + prompt times
            times = {}
            for st in status.get("messages", []):
                if st[0] == "execution_start":
                    times["exec_start"] = st[1].get("prompt_id")
            results[pid_text] = {
                "prompt_id": prompt_id, "success": ok, "wall_s": round(wall, 2),
                "images": imgs,
            }
            print(f"[{pid_text}] success={ok} wall={wall:.1f}s images={len(imgs)}", flush=True)
        # pull "Prompt executed in" lines from the server log
        time.sleep(1)
        logf.flush()
        exec_times = re.findall(r"Prompt executed in (\d+\.\d+) seconds", open(log_path).read())
        results["_server_prompt_exec_s"] = [float(x) for x in exec_times]
        json.dump(results, open(os.path.join(OUT, "server_run.json"), "w"), indent=1)
        print("[saved]", os.path.join(OUT, "server_run.json"), flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=20)
        except subprocess.TimeoutExpired:
            srv.kill()
        logf.close()
    print("[done] server baseline", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 41: benchmark the Hybrid architecture through the REAL ComfyUI server
using the custom node (Qwen3VLOpenVINOCPUEncoder) — proves the harness advantage
is not from bypassing ComfyUI.

Workflow: UNETLoader + custom node (OV CPU encoder, dq configurable) + KSampler +
VAEDecode + SaveImage — NO CLIPLoader at all. P3, 20 steps, 1024², same seed as the
GPU server baseline for a true product-stack A/B.

Results -> results/e2e/gpu_baseline/custom_node_dq{dq}_run.json (+ PNGs, workflow JSON, log)
"""
import json
import os
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


def workflow(prompt_text, prefix, dq):
    return {
        "7": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}},
        "9": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "10": {"class_type": "Qwen3VLOpenVINOCPUEncoder", "inputs": {
            "prompt": prompt_text, "negative_prompt": " ",
            "width": 1024, "height": 1024, "dq_group": dq,
            "model_dir": "../models/qwen3vl-openvino-int8",
            "cache_dir": "../models/.ov_cache", "cpu_threads": 0}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": SEED, "steps": STEPS, "cfg": CFG, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0, "model": ["7", 0],
            "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["10", 2]}},
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


def get_history(pid, timeout=1800):
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
    dq = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    os.makedirs(OUT, exist_ok=True)
    prompt = [p["text"] for p in json.load(open(os.path.join(ROOT, "prompts", "prompts.json")))["prompts"] if p["id"] == "P3"][0]
    wf = workflow(prompt, f"customnode_dq{dq}", dq)
    json.dump(wf, open(os.path.join(OUT, f"workflow_custom_node_dq{dq}.json"), "w"), indent=1)

    log_path = os.path.join(OUT, f"custom_node_dq{dq}.log")
    logf = open(log_path, "w")
    env = dict(os.environ, TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1")
    srv = subprocess.Popen(
        [PY, "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
         "--output-directory", OUT],
        cwd=COMFY, stdout=logf, stderr=subprocess.STDOUT, env=env)
    results = {}
    try:
        assert wait_ready(), "server not ready"
        print("[server] ready", flush=True)
        # 3 sequential generations of the same prompt (server keeps models cached)
        for i in range(3):
            t0 = time.time()
            pid = post_prompt(wf)
            hist = get_history(pid)
            wall = time.time() - t0
            ok = hist and hist.get("status", {}).get("status_str") == "success"
            results[f"gen{i}"] = {"success": bool(ok), "wall_s": round(wall, 2)}
            print(f"[gen{i}] success={ok} wall={wall:.1f}s", flush=True)
        time.sleep(1)
        logf.flush()
        import re
        exec_times = re.findall(r"Prompt executed in (\d+\.\d+) seconds", open(log_path).read())
        results["_server_prompt_exec_s"] = [float(x) for x in exec_times]
        json.dump(results, open(os.path.join(OUT, f"custom_node_dq{dq}_run.json"), "w"), indent=1)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=20)
        except subprocess.TimeoutExpired:
            srv.kill()
        logf.close()
    print("[done]", results, flush=True)


if __name__ == "__main__":
    main()

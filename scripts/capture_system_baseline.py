#!/usr/bin/env python3
"""Capture hardware/software baseline into environment/system.json + system.md.

Stdlib-only where possible; falls back gracefully and records 'unavailable'
when a metric cannot be read reliably (e.g. memory speed needs root dmidecode).
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return f"ERROR: {e}"


def cpu_info():
    lscpu = sh("lscpu")
    fields = {}
    for line in lscpu.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    flags = fields.get("Flags", "")
    isa = {
        "avx": " avx " in f" {flags} ",
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
        "avx512_vnni": "avx512_vnni" in flags,
        "avx_vnni": "avx_vnni" in flags,
        "avx512_bf16": "avx512_bf16" in flags,
        "amx": any(f.startswith("amx") for f in flags.split()),
    }
    gov = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    epp = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
    profile = sh("cat /sys/firmware/acpi/platform_profile 2>/dev/null") or sh("powerprofilesctl get 2>/dev/null")
    cur_freqs = sh("cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq").split()
    freqs_mhz = [int(f) / 1000 for f in cur_freqs if f.isdigit()]
    return {
        "model": fields.get("Model name"),
        "vendor": fields.get("Vendor ID"),
        "physical_cores": fields.get("Core(s) per socket"),
        "sockets": fields.get("Socket(s)"),
        "logical_threads": fields.get("CPU(s)"),
        "threads_per_core": fields.get("Thread(s) per core"),
        "numa_nodes": fields.get("NUMA node(s)"),
        "l1d_cache": fields.get("L1d cache"),
        "l1i_cache": fields.get("L1i cache"),
        "l2_cache": fields.get("L2 cache"),
        "l3_cache": fields.get("L3 cache"),
        "max_mhz": fields.get("CPU max MHz"),
        "min_mhz": fields.get("CPU min MHz"),
        "isa": isa,
        "scaling_governor": gov,
        "energy_performance_preference": epp,
        "power_profile": profile,
        "current_freq_mhz_sample": {
            "min": round(min(freqs_mhz), 1) if freqs_mhz else None,
            "max": round(max(freqs_mhz), 1) if freqs_mhz else None,
            "mean": round(sum(freqs_mhz) / len(freqs_mhz), 1) if freqs_mhz else None,
        },
    }


def mem_info():
    memtotal_kb = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                memtotal_kb = int(line.split()[1])
                break
    free_h = sh("free -h")
    # memory speed: dmidecode requires root
    dmi = sh("sudo -n dmidecode --type 17 2>/dev/null", timeout=5)
    speed = None
    if "Speed:" in dmi and "Unknown" not in dmi:
        speeds = re.findall(r"Speed:\s*(\d+)\s*MT/s", dmi)
        speed = speeds[0] + " MT/s" if speeds else None
    numa = sh("numactl --hardware 2>/dev/null || lscpu | grep NUMA", timeout=5)
    return {
        "total_gib": round(memtotal_kb / 1024 ** 2, 2) if memtotal_kb else None,
        "free_h": free_h.splitlines()[1] if len(free_h.splitlines()) > 1 else free_h,
        "speed": speed if speed else "unavailable (needs root dmidecode); UMA shared with iGPU",
        "numa": numa.splitlines()[:6],
        "swap": sh("free -h | tail -1"),
    }


def gpu_info():
    return {
        "lspci_vga": sh("lspci | grep -Ei 'vga|display|3d'"),
        "rocm": sh("rocminfo 2>/dev/null | grep -E 'Marketing|Name' | head -4"),
        "note": "GPU recorded for completeness only; text-encoder benchmark forces CPU. UMA: iGPU shares system RAM.",
    }


def sw_info():
    return {
        "os": sh("cat /etc/os-release | head -2").replace("\n", " | "),
        "kernel": sh("uname -r"),
        "hostname": socket.gethostname(),
        "python3": sh("python3 --version"),
        "pip3": sh("pip3 --version 2>/dev/null | head -c 120"),
        "uv": sh("uv --version 2>/dev/null"),
        "gcc": sh("gcc --version 2>/dev/null | head -1"),
    }


def main():
    data = {
        "timestamp": datetime.now().isoformat(),
        "cpu": cpu_info(),
        "memory": mem_info(),
        "gpu": gpu_info(),
        "software": sw_info(),
    }
    os.makedirs(os.path.join(ROOT, "environment"), exist_ok=True)
    jpath = os.path.join(ROOT, "environment", "system.json")
    with open(jpath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    md = os.path.join(ROOT, "environment", "system.md")
    c, m, g, s = data["cpu"], data["memory"], data["gpu"], data["software"]
    with open(md, "w") as f:
        f.write(f"# System Baseline ({data['timestamp']})\n\n")
        f.write(f"- CPU: **{c['model']}** — {c['physical_cores']} cores / {c['logical_threads']} threads, "
                f"NUMA={c['numa_nodes']}\n")
        f.write(f"- L1d {c['l1d_cache']} | L2 {c['l2_cache']} | L3 {c['l3_cache']}\n")
        f.write(f"- ISA: avx2={c['isa']['avx2']} avx512f={c['isa']['avx512f']} "
                f"avx512_vnni={c['isa']['avx512_vnni']} avx_vnni={c['isa']['avx_vnni']} "
                f"avx512_bf16={c['isa']['avx512_bf16']}\n")
        f.write(f"- Governor={c['scaling_governor']}, EPP={c['energy_performance_preference']}, "
                f"profile={c['power_profile']}, freq now {c['current_freq_mhz_sample']}\n")
        f.write(f"- RAM: **{m['total_gib']} GiB** (speed: {m['speed']})\n")
        f.write(f"- Swap: {m['swap']}\n")
        f.write(f"- GPU (not used in benchmark): {g['lspci_vga']}\n")
        f.write(f"- OS: {s['os']} kernel {s['kernel']}\n")
        f.write(f"- Python: {s['python3']} | uv: {s['uv']}\n")
    print(open(md).read())
    print(f"WROTE {jpath} and {md}")


if __name__ == "__main__":
    main()

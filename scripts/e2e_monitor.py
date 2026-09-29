#!/usr/bin/env python3
"""E2E resource monitor: 20 Hz CPU + GPU + power sampling for the Qwen-Image e2e runs.

Samples (all best-effort; unavailable metrics are recorded as None, never estimated):
  CPU: proc cpu %, RSS/VSZ bytes, threads, per-core freq mean MHz, sys cpu %,
       sys used / available bytes
  GPU: amdgpu busy %, GTT used bytes, visible VRAM used bytes (sysfs card1/card0),
       amdgpu hwmon temp + power, k10temp
  PWR: RAPL package energy uJ (powercap) -> cumulative energy counter

Every sample row -> CSV (one file per run). Marks (label, perf_counter_ns) align the
harness stage timing with the monitor timeline (same clock).
"""
import csv
import glob
import os
import threading
import time

import psutil

HZ = 20.0

GPU_SYSFS_CANDIDATES = [
    "/sys/class/drm/card1/device",
    "/sys/class/drm/card0/device",
]


def _find(path_candidates, name):
    for d in path_candidates:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _hwmon_by_name(name):
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            with open(os.path.join(h, "name")) as f:
                if f.read().strip() == name:
                    return h
        except OSError:
            continue
    return None


def _read_int(path, scale=1.0):
    try:
        with open(path) as f:
            return int(float(f.read().strip()) * scale)
    except Exception:
        return None


def _read_float(path):
    try:
        with open(path) as f:
            return float(f.read().strip())
    except Exception:
        return None


FIELDS = [
    "t_ns", "rel_s",
    "proc_cpu_pct", "rss_bytes", "vms_bytes", "nthreads", "cpu_freq_mhz",
    "sys_cpu_pct", "sys_used_bytes", "sys_available_bytes",
    "gpu_busy_pct", "gtt_used_bytes", "vram_used_bytes",
    "gpu_temp_c", "gpu_power_w", "cpu_temp_c", "rapl_energy_uj",
]


class E2EMonitor:
    def __init__(self, out_csv, hz=HZ, pids=None):
        self.out_csv = out_csv
        self.interval = 1.0 / hz
        self.pids = set(pids or [os.getpid()])
        self.samples = []
        self.marks = []  # (label, t_ns)
        self._stop = threading.Event()
        self._thread = None
        self.t0 = None

        self.busy_path = _find(GPU_SYSFS_CANDIDATES, "gpu_busy_percent")
        self.gtt_path = _find(GPU_SYSFS_CANDIDATES, "mem_info_gtt_used")
        self.vram_path = _find(GPU_SYSFS_CANDIDATES, "mem_info_vis_vram_used")
        amdgpu = _hwmon_by_name("amdgpu")
        k10 = _hwmon_by_name("k10temp")
        self.gpu_temp = self.gpu_power = self.cpu_temp = self.rapl = None
        if amdgpu:
            t = sorted(glob.glob(os.path.join(amdgpu, "temp*_input")))
            p = sorted(glob.glob(os.path.join(amdgpu, "power*_average"))) or \
                sorted(glob.glob(os.path.join(amdgpu, "power*_input")))
            self.gpu_temp = t[0] if t else None
            self.gpu_power = p[0] if p else None
        if k10:
            t = sorted(glob.glob(os.path.join(k10, "temp*_input")))
            self.cpu_temp = t[0] if t else None
        rapl = sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"))
        self.rapl = rapl[0] if rapl else None

        self.procs = [psutil.Process(p) for p in self.pids if psutil.pid_exists(p)]

    # -- lifecycle -------------------------------------------------------
    def start(self):
        self.t0 = time.perf_counter_ns()
        self.marks.append(("start", self.t0))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def mark(self, label):
        self.marks.append((label, time.perf_counter_ns()))

    def stop(self):
        self.marks.append(("stop", time.perf_counter_ns()))
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # -- sampling --------------------------------------------------------
    def _one(self):
        row = {"t_ns": time.perf_counter_ns()}
        row["rel_s"] = (row["t_ns"] - self.t0) / 1e9
        pcpu = rss = vms = nth = 0
        for pr in list(self.procs):
            try:
                with pr.oneshot():
                    pcpu += pr.cpu_percent(None)
                    rss += pr.memory_info().rss
                    vms += pr.memory_info().vms
                    nth += pr.num_threads()
            except psutil.Error:
                pass
        row["proc_cpu_pct"] = round(pcpu, 1)
        row["rss_bytes"] = rss
        row["vms_bytes"] = vms
        row["nthreads"] = nth
        freqs = psutil.cpu_freq(percpu=True)
        row["cpu_freq_mhz"] = round(sum(f.current for f in freqs) / len(freqs), 0) if freqs else None
        row["sys_cpu_pct"] = psutil.cpu_percent(None)
        vm = psutil.virtual_memory()
        row["sys_used_bytes"] = vm.used
        row["sys_available_bytes"] = vm.available
        row["gpu_busy_pct"] = _read_int(self.busy_path) if self.busy_path else None
        row["gtt_used_bytes"] = _read_int(self.gtt_path) if self.gtt_path else None
        row["vram_used_bytes"] = _read_int(self.vram_path) if self.vram_path else None
        if self.gpu_temp:
            v = _read_float(self.gpu_temp)
            row["gpu_temp_c"] = v / 1000.0 if v is not None else None
        else:
            row["gpu_temp_c"] = None
        if self.gpu_power:
            v = _read_float(self.gpu_power)
            row["gpu_power_w"] = v / 1e6 if v is not None else None
        else:
            row["gpu_power_w"] = None
        if self.cpu_temp:
            v = _read_float(self.cpu_temp)
            row["cpu_temp_c"] = v / 1000.0 if v is not None else None
        else:
            row["cpu_temp_c"] = None
        row["rapl_energy_uj"] = _read_int(self.rapl) if self.rapl else None
        return row

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self._one())
            self._stop.wait(self.interval)

    def flush(self):
        if not self.samples:
            return
        new = not os.path.exists(self.out_csv)
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
        with open(self.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.samples)
        with open(self.out_csv + ".marks", "w") as f:
            for label, t in self.marks:
                f.write(f"{t},{label}\n")


def window_stats(samples, t_from_ns, t_to_ns, key):
    """Mean/max of a metric over the [t_from, t_to] window (None-safe)."""
    vals = [s[key] for s in samples
            if s[key] is not None and t_from_ns <= s["t_ns"] <= t_to_ns]
    if not vals:
        return {"mean": None, "max": None, "n": 0}
    return {"mean": round(sum(vals) / len(vals), 2), "max": round(max(vals), 2), "n": len(vals)}

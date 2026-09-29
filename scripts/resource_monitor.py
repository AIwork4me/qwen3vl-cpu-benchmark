#!/usr/bin/env python3
"""Unified resource monitor for both benchmark harnesses.

Runs a daemon thread sampling at `hz` (default 20 Hz, low overhead):
  timestamp_ns, proc_rss_bytes, proc_vms_bytes, sys_used_bytes,
  sys_available_bytes, cpu_util_pct, proc_cpu_pct, num_threads, cpu_freq_mhz

Provides:
  monitor = ResourceMonitor(hz=20); monitor.start()
  ...work...
  monitor.mark("model_load_end")
  monitor.stop()
  monitor.summary()  -> per-interval (between marks + whole run) peak RSS etc.

Peak RAM is derived from continuous sampling over the interval — never from
start/end snapshots only. Wall-clock uses time.perf_counter_ns().
"""
import os
import threading
import time

import psutil

DEFAULT_HZ = 20.0


class ResourceMonitor:
    def __init__(self, hz: float = DEFAULT_HZ, output_csv: str | None = None):
        self.hz = hz
        self.interval = 1.0 / hz
        self.output_csv = output_csv
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None
        self.samples = []          # list of dicts
        self.marks = []            # (label, t_ns)
        self.t0 = None

    # ---- lifecycle ----
    def start(self):
        self.t0 = time.perf_counter_ns()
        self.mark("start")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def mark(self, label: str):
        self.marks.append((label, time.perf_counter_ns()))

    def stop(self):
        self.mark("stop")
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ---- sampler ----
    def _run(self):
        if self.output_csv:
            with open(self.output_csv, "w") as f:
                f.write("timestamp_ns,rel_s,rss_bytes,vms_bytes,sys_used_bytes,"
                        "sys_available_bytes,cpu_util_pct,proc_cpu_pct,num_threads,cpu_freq_mhz\n")
        last_freq_check = 0.0
        cpu_freq = None
        n = 0
        while not self._stop.is_set():
            t = time.perf_counter_ns()
            try:
                with self.proc.oneshot():
                    rss = self.proc.memory_info().rss
                    vms = self.proc.memory_info().vms
                    threads = self.proc.num_threads()
                vm = psutil.virtual_memory()
                # cpu percent over the sampling interval (non-blocking second call pattern)
                pcpu = self.proc.cpu_percent(None)
                scpu = psutil.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            # CPU freq: read every ~0.5s (sysfs reads are cheap but not free)
            if t - last_freq_check > 5e8:
                cpu_freq = self._read_freq()
                last_freq_check = t
            row = (t, (t - self.t0) / 1e9, rss, vms, vm.used, vm.available,
                   scpu, pcpu, threads, cpu_freq)
            self.samples.append(row)
            if self.output_csv:
                with open(self.output_csv, "a") as f:
                    f.write(",".join(str(x) for x in row) + "\n")
            n += 1
            # sleep to next tick
            now = time.perf_counter_ns()
            target = self.t0 + (n + 1) * self.interval * 1e9
            delay = (target - now) / 1e9
            if delay > 0:
                time.sleep(delay)

    @staticmethod
    def _read_freq():
        try:
            import glob
            vals = []
            for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"):
                with open(p) as f:
                    vals.append(int(f.read().strip()) / 1000.0)
            return round(sum(vals) / len(vals), 1) if vals else None
        except Exception:
            return None

    # ---- reporting ----
    def _slice(self, start_ns, end_ns):
        return [s for s in self.samples if start_ns <= s[0] <= end_ns]

    def interval_stats(self, start_label: str, end_label: str):
        """Stats over [mark(start_label), mark(end_label)]."""
        marks = {lbl: t for lbl, t in self.marks}
        if start_label not in marks or end_label not in marks:
            return None
        s0, s1 = marks[start_label], marks[end_label]
        rows = self._slice(s0, s1)
        if not rows:
            return None
        return {
            "start_label": start_label,
            "end_label": end_label,
            "wall_s": (s1 - s0) / 1e9,
            "peak_rss_bytes": max(r[2] for r in rows),
            "peak_vms_bytes": max(r[3] for r in rows),
            "sys_used_peak_bytes": max(r[4] for r in rows),
            "sys_available_min_bytes": min(r[5] for r in rows),
            "cpu_util_mean_pct": round(sum(r[6] for r in rows) / len(rows), 2),
            "cpu_util_max_pct": max(r[6] for r in rows),
            "proc_cpu_mean_pct": round(sum(r[7] for r in rows) / len(rows), 2),
            "proc_cpu_max_pct": max(r[7] for r in rows),
            "threads_max": max(r[8] for r in rows),
            "cpu_freq_mean_mhz": round(sum(r[9] for r in rows if r[9]) / max(1, sum(1 for r in rows if r[9])), 1),
            "n_samples": len(rows),
        }

    def baseline_rss(self):
        """RSS at 'start' mark (process baseline, before any model work)."""
        marks = {lbl: t for lbl, t in self.marks}
        if "start" not in marks or not self.samples:
            return None
        s0 = marks["start"]
        rows = [r for r in self.samples if r[0] <= s0 + 2 * self.interval * 1e9]
        return min(r[2] for r in rows) if rows else None

    def full_run(self):
        return self.interval_stats("start", "stop")

    def save_csv(self, path: str):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ns", "rel_s", "rss_bytes", "vms_bytes", "sys_used_bytes",
                        "sys_available_bytes", "cpu_util_pct", "proc_cpu_pct", "num_threads", "cpu_freq_mhz"])
            w.writerows(self.samples)


if __name__ == "__main__":
    # smoke test
    m = ResourceMonitor(hz=50).start()
    time.sleep(1.0)
    m.mark("phase1")
    time.sleep(0.5)
    m.stop()
    print(m.interval_stats("start", "phase1"))
    print(m.full_run())
    print("baseline_rss", m.baseline_rss())

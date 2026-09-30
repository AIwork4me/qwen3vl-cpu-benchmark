#!/usr/bin/env python3
"""Full-scope audit of the OpenVINO CPU Plugin support on the local CPU (HOST B).

Sections (all evidence written to results/openvino_cpu_support/):
  1. environment + device-level property census
  2. ISA ceiling matrix (ONEDNN_MAX_CPU_ISA, one fresh subprocess per value;
     tiny f32 matmul; capture oneDNN verbose isa header + impl line)
  3. precision-hint matrix on tiny models (f32 model x {f32,bf16,f16,i8},
     plus dedicated f64 and i8-input models)
  4. operator smoke suite vs numpy references (exec types from profiling)
  5. threading behavior (1/4/16 threads on tiny matmul)

The real Qwen3-VL workload capstone runs separately via
scripts/investigate_openvino_runtime.py (see audit shell steps in the report).
"""
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import openvino as ov

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "openvino_cpu_support")
PY = sys.executable
os.makedirs(OUT, exist_ok=True)

TINY_ISA_VALUES = [
    "SSE41", "AVX", "AVX2", "AVX2_VNNI", "AVX512_CORE", "AVX512_CORE_VNNI",
    "AVX512_CORE_BF16", "AVX512_CORE_FP16", "AVX512_CORE_AMX",
    "AVX512_CORE_AMX_INT8", "AVX512_CORE_AMX_BF16", "BEST", "ALL",
    "NOT_A_REAL_ISA",
]
HINTS = ["f32", "bf16", "f16", "i8"]


def tiny_matmul_model():
    rng = np.random.default_rng(7)
    w = rng.standard_normal((256, 256)).astype(np.float32)
    param = ov.opset13.parameter([1, 256], np.float32)
    mm = ov.opset13.matmul(param, ov.opset13.constant(w, dtype=np.float32), False, True)
    return ov.Model(mm, [param], "tiny_mm"), w


def run_tiny_once(env_extra=None, hint=None, with_profiling=False):
    """Tiny FC-style matmul (transpose_b=True -> FullyConnected path) in a fresh
    subprocess. Reports correctness + (optionally) per-node exec types."""
    code = r"""
import numpy as np, openvino as ov, time, json
rng = np.random.default_rng(7)
w = rng.standard_normal((256, 256)).astype(np.float32)
x = rng.standard_normal((1, 256)).astype(np.float32)
param = ov.opset13.parameter([1, 256], np.float32)
mm = ov.opset13.matmul(param, ov.opset13.constant(w, dtype=np.float32), False, True)
m = ov.Model(mm, [param], "tiny_mm")
cfg = {"PERFORMANCE_HINT": "LATENCY"}
if %r: cfg["INFERENCE_PRECISION_HINT"] = %r
if %d: cfg[ov.properties.enable_profiling] = True
try:
    c = ov.Core().compile_model(m, "CPU", cfg)
    r = c.create_infer_request()
    out = np.asarray(r.infer({0: x})[0])
    ref = x @ w.T
    err = float(np.abs(out-ref).max()); scale = float(np.abs(ref).max())
    ok = bool(err <= 0.02*scale + 1e-4)
    res = {"ok": ok, "max_abs_err": err, "ref_scale": scale}
    if %d:
        types = {}
        for pi in r.get_profiling_info():
            types[pi.exec_type] = types.get(pi.exec_type, 0) + 1
        res["exec_types"] = types
    print("AUDIT_RESULT", json.dumps(res))
except Exception as e:
    print("AUDIT_RESULT", json.dumps({"ok": None, "error": repr(e)[-450:]}))
"""
    code = code % (hint if hint else None, hint if hint else None,
                   1 if with_profiling else 0, 1 if with_profiling else 0)
    env = dict(os.environ)
    env.pop("ONEDNN_MAX_CPU_ISA", None)
    env.pop("ONEDNN_VERBOSE", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env, timeout=120)


def section1_device_properties():
    core = ov.Core()
    dev_props = {"available_devices": core.available_devices}
    try:
        names = core.get_property("CPU", ov.properties.supported_properties)
    except Exception as e:
        names = []
        dev_props["_supported_properties_error"] = str(e)
    for n in names:
        try:
            v = core.get_property("CPU", n)
            dev_props[n] = v if isinstance(v, (int, float, str, bool, list)) else str(v)
        except Exception as e:
            dev_props[n] = f"<error {e}>"
    # compiled-model-level properties (defaults on tiny model, no hint)
    m, _ = tiny_matmul_model()
    c = core.compile_model(m, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    cm = {}
    for n in c.get_property(ov.properties.supported_properties):
        try:
            v = c.get_property(n)
            cm[n] = v if isinstance(v, (int, float, str, bool, list)) else str(v)
        except Exception as e:
            cm[n] = f"<error {e}>"
    rec = {"timestamp": datetime.now().isoformat(),
           "openvino_version": ov.__version__,
           "cpu": open("/proc/cpuinfo").read().split("model name\t:")[1].split("\n")[0].strip(),
           "kernel": os.uname().release,
           "device_properties": dev_props, "compiled_model_defaults": cm}
    return rec


def run_verbose_matmul(env):
    """transpose_b=False plain matmul (non-FC): verbose=1 emits header/impl for it."""
    code = r"""
import numpy as np, openvino as ov
rng = np.random.default_rng(7)
w = rng.standard_normal((256, 256)).astype(np.float32)
x = rng.standard_normal((1, 256)).astype(np.float32)
param = ov.opset13.parameter([1, 256], np.float32)
mm = ov.opset13.matmul(param, ov.opset13.constant(w, dtype=np.float32), False, False)
m = ov.Model(mm, [param], "tiny_mm_tbf")
c = ov.Core().compile_model(m, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
c.create_infer_request().infer({0: x})
"""
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env, timeout=120)


def _probe_one(isa):
    """Per ceiling: (a) no-verbose default & f32-hint probes with exec types;
    (b) verbose=1 f32-hint subprocess for the oneDNN isa header + impl line."""
    out = {}
    for mode, hint in (("default", None), ("f32hint", "f32")):
        env = {"ONEDNN_MAX_CPU_ISA": isa} if isa is not None else {}
        r = run_tiny_once(env_extra=env, hint=hint, with_profiling=True)
        mm = re.search(r"AUDIT_RESULT (\{.*\})", r.stdout)
        d = json.loads(mm.group(1)) if mm else {"ok": None, "error": "no result"}
        out[mode] = {"run_ok": d.get("ok") is not None, "correct": d.get("ok"),
                     "exec_types": d.get("exec_types"), "error": d.get("error", "")}
    env = {"ONEDNN_VERBOSE": "1"}
    if isa is not None:
        env["ONEDNN_MAX_CPU_ISA"] = isa
    rv = run_verbose_matmul(env)
    isa_hdr, impl = "", ""
    for line in (rv.stdout + rv.stderr).splitlines():
        m = re.search(r"cpu,isa:([^,]+)", line)
        if m and not isa_hdr:
            isa_hdr = m.group(1).strip()
        m2 = re.search(r"(brgemm:[a-z0-9_]+|gemm:[a-z0-9_]+|jit:[a-z0-9_]+)", line)
        if m2 and not impl:
            impl = m2.group(1)
    return {"isa_setting": isa if isa is not None else "(none)",
            "default_run_ok": out["default"]["run_ok"],
            "default_correct": out["default"]["correct"],
            "default_exec_types": out["default"]["exec_types"],
            "default_error": out["default"]["error"],
            "f32_run_ok": out["f32hint"]["run_ok"], "f32_correct": out["f32hint"]["correct"],
            "f32_exec_types": out["f32hint"]["exec_types"], "f32_error": out["f32hint"]["error"],
            "verbose_isa_header": isa_hdr, "verbose_impl_f32": impl}


def section2_isa_matrix():
    rows = []
    for isa in TINY_ISA_VALUES + [None]:
        row = _probe_one(isa)
        rows.append(row)
        print(f"  ISA={row['isa_setting']:22s} default_ok={row['default_run_ok']} "
              f"default={str(row['default_exec_types'])[:52]:54s} f32_ok={row['f32_run_ok']} "
              f"isa_hdr={row['verbose_isa_header'][:30]!r}", flush=True)
    return rows


def section3_precision_hints():
    rows = []
    for h in HINTS + ["__default__"]:
        hint = None if h == "__default__" else h
        r = run_tiny_once(hint=hint, with_profiling=True)
        res = {}
        mm = re.search(r"AUDIT_RESULT (\{.*\})", r.stdout)
        if mm:
            res = json.loads(mm.group(1))
        rows.append({"hint": h, "rc": r.returncode,
                     "compile_ok": res.get("ok") is not None,
                     "correct": res.get("ok"),
                     "max_abs_err": res.get("max_abs_err"),
                     "ref_scale": res.get("ref_scale"),
                     "exec_types": res.get("exec_types"),
                     "error": res.get("error", "")})
        print(f"  hint={h:10s} correct={res.get('ok')} err={res.get('max_abs_err')} {res.get('exec_types','')} {res.get('error','')[:80]}", flush=True)
    # dedicated f64 model (does the plugin honor f64 compute?)
    code = r"""
import numpy as np, openvino as ov, json
rng = np.random.default_rng(3)
w = rng.standard_normal((64,64)); x = rng.standard_normal((1,64))
p = ov.opset13.parameter([1,64], np.float64)
mm = ov.opset13.matmul(p, ov.opset13.constant(w), False, True)
try:
    c = ov.Core().compile_model(ov.Model(mm,[p],"f64mm"), "CPU",
                                {ov.properties.enable_profiling: True})
    r = c.create_infer_request()
    out = np.asarray(r.infer({0:x})[0])
    ref = x @ w.T
    types = {}
    for pi in r.get_profiling_info():
        types[pi.exec_type] = types.get(pi.exec_type, 0) + 1
    err = float(np.abs(out-ref).max()); scale = float(np.abs(ref).max())
    print("AUDIT_RESULT", json.dumps({"compile_ok": True, "out_dtype": str(out.dtype),
        "rel_err": err/scale, "exec_types": types}))
except Exception as e:
    print("AUDIT_RESULT", json.dumps({"compile_ok": False, "error": repr(e)[-300:]}))
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120)
    mm = re.search(r"AUDIT_RESULT (\{.*\})", r.stdout)
    f64 = json.loads(mm.group(1)) if mm else {"error": r.stderr[-200:]}
    # dedicated i8-input model (i8 activations + i8 weights converted in-graph)
    code = r"""
import numpy as np, openvino as ov, json
rng = np.random.default_rng(5)
w = rng.integers(-127,127,(64,64)).astype(np.int8); x = rng.integers(-127,127,(1,64)).astype(np.int8)
p = ov.opset13.parameter([1,64], np.int8)
mm = ov.opset13.matmul(ov.opset13.convert(p, np.float32), ov.opset13.constant(w, dtype=np.float32), False, True)
try:
    c = ov.Core().compile_model(ov.Model(mm,[p],"i8mm"), "CPU", {ov.properties.enable_profiling: True})
    r = c.create_infer_request()
    out = np.asarray(r.infer({0:x})[0])
    ref = x.astype(np.float32) @ w.astype(np.float32).T
    types = {}
    for pi in r.get_profiling_info():
        types[pi.exec_type] = types.get(pi.exec_type, 0) + 1
    print("AUDIT_RESULT", json.dumps({"compile_ok": True, "out_dtype": str(out.dtype),
        "correct": bool(np.allclose(out, ref, rtol=1e-3, atol=1e-1)), "exec_types": types}))
except Exception as e:
    print("AUDIT_RESULT", json.dumps({"compile_ok": False, "error": repr(e)[-300:]}))
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120)
    mm = re.search(r"AUDIT_RESULT (\{.*\})", r.stdout)
    i8 = json.loads(mm.group(1)) if mm else {"error": r.stderr[-200:]}
    return {"hint_matrix": rows, "f64_model": f64, "i8_input_model": i8}


def _ref_conv(x, w):
    n, c, h, wd = x.shape
    oc, ic, kh, kw = w.shape
    oh, ow = h - kh + 1, wd - kw + 1
    out = np.zeros((n, oc, oh, ow), dtype=np.float32)
    for o in range(oc):
        for i in range(c):
            for y in range(oh):
                for xx in range(ow):
                    out[0, o, y, xx] += (x[0, i, y:y+kh, xx:xx+kw] * w[o, i]).sum()
    return out


def section4_op_smoke():
    core = ov.Core()
    rng = np.random.default_rng(11)
    tests = []

    def run_case(name, model, feed, ref, strict_rtol=1e-4, strict_atol=1e-4):
        ref = np.asarray(ref)
        for mode in ("default", "f32hint"):
            rec = {"op": name, "mode": mode}
            try:
                cfg = {"PERFORMANCE_HINT": "LATENCY", ov.properties.enable_profiling: True}
                if mode == "f32hint":
                    cfg["INFERENCE_PRECISION_HINT"] = "f32"
                c = core.compile_model(model, "CPU", cfg)
                r = c.create_infer_request()
                out = np.asarray(r.infer(feed)[0])
                err = float(np.abs(out.astype(np.float64) - ref.astype(np.float64)).max()) if out.shape == ref.shape else None
                scale = float(np.abs(ref).max())
                rec["exec_ok"] = True
                rec["max_abs_err"] = err
                rec["correct"] = bool(out.shape == ref.shape and err <= 0.02 * scale + 1e-4)
                rec["strict_pass"] = bool(out.shape == ref.shape and
                                          np.allclose(out, ref, rtol=strict_rtol, atol=strict_atol))
                rec["out_dtype"] = str(out.dtype)
                types = {}
                for pi in r.get_profiling_info():
                    types[pi.exec_type] = types.get(pi.exec_type, 0) + 1
                rec["exec_types"] = types
            except Exception as e:
                rec["exec_ok"] = False
                rec["error"] = str(e)[:200]
            tests.append(rec)
            print(f"  {name:20s} {mode:8s} ok={rec.get('exec_ok')} correct={rec.get('correct')} "
                  f"strict={rec.get('strict_pass')} {str(rec.get('exec_types',''))[:90]}", flush=True)

    # matmul
    w = rng.standard_normal((128, 128)).astype(np.float32)
    x = rng.standard_normal((1, 128)).astype(np.float32)
    p = ov.opset13.parameter([1, 128], np.float32)
    run_case("matmul_f32", ov.Model(ov.opset13.matmul(p, ov.opset13.constant(w), False, True), [p]), {0: x}, x @ w.T)
    # conv
    xc = rng.standard_normal((1, 4, 16, 16)).astype(np.float32)
    wc = rng.standard_normal((8, 4, 3, 3)).astype(np.float32)
    pc = ov.opset13.parameter([1, 4, 16, 16], np.float32)
    run_case("conv2d_f32", ov.Model(ov.opset13.convolution(pc, ov.opset13.constant(wc), [1, 1], [0, 0], [0, 0], [1, 1]), [pc]), {0: xc}, _ref_conv(xc, wc))
    # sdpa-lite: qk softmax av
    d = 32
    q = rng.standard_normal((1, 4, d)).astype(np.float32)
    k = rng.standard_normal((1, 4, d)).astype(np.float32)
    v = rng.standard_normal((1, 4, d)).astype(np.float32)
    pq = ov.opset13.parameter([1, 4, d], np.float32)
    pk = ov.opset13.parameter([1, 4, d], np.float32)
    pv = ov.opset13.parameter([1, 4, d], np.float32)
    scores = ov.opset13.matmul(pq, ov.opset13.constant(k, dtype=np.float32), False, True)
    probs = ov.opset13.softmax(scores, -1)
    av = ov.opset13.matmul(probs, ov.opset13.constant(v, dtype=np.float32), False, False)
    s = q[0] @ k[0].T
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    run_case("sdpa_lite_f32", ov.Model(av, [pq, pk, pv]), {0: q, 1: k, 2: v}, (s @ v[0])[None])
    # elementwise add/mul
    a = rng.standard_normal((1, 64)).astype(np.float32)
    b = rng.standard_normal((1, 64)).astype(np.float32)
    pa = ov.opset13.parameter([1, 64], np.float32)
    pb = ov.opset13.parameter([1, 64], np.float32)
    run_case("add_mul_f32", ov.Model(ov.opset13.multiply(ov.opset13.add(pa, pb), ov.opset13.constant(np.float32(2.0))), [pa, pb]), {0: a, 1: b}, (a + b) * 2.0)
    # rmsnorm chain: pow/mean/sqrt/div/mul
    def rms(x):
        ms = (x ** 2).mean(-1, keepdims=True)
        return x / np.sqrt(ms + 1e-6)
    pr = ov.opset13.parameter([1, 64], np.float32)
    sq = ov.opset13.multiply(pr, pr)
    ms = ov.opset13.reduce_mean(sq, ov.opset13.constant(np.array([-1]), dtype=np.int64), True)
    den = ov.opset13.sqrt(ov.opset13.add(ms, ov.opset13.constant(np.float32(1e-6))))
    run_case("rmsnorm_chain_f32", ov.Model(ov.opset13.divide(pr, den), [pr]), {0: a}, rms(a))
    # gather + transpose
    ids = np.array([[3, 1, 0, 2]], dtype=np.int64)
    tab = rng.standard_normal((8, 16)).astype(np.float32)
    pg = ov.opset13.parameter([1, 4], np.int64)
    run_case("gather_f32",
             ov.Model(ov.opset13.gather(ov.opset13.constant(tab), pg, ov.opset13.constant(0, dtype=np.int64)), [pg]),
             {0: ids}, tab[ids])
    return tests


def section5_threading():
    rows = []
    for t in (1, 4, 16, 0):  # 0 = plugin default
        code = r"""
import numpy as np, openvino as ov, time, json
rng = np.random.default_rng(7)
w = rng.standard_normal((256,256)).astype(np.float32); x = rng.standard_normal((1,256)).astype(np.float32)
p = ov.opset13.parameter([1,256], np.float32)
m = ov.Model(ov.opset13.matmul(p, ov.opset13.constant(w), False, True), [p])
cfg = {"PERFORMANCE_HINT": "LATENCY"}
if %d: cfg["INFERENCE_NUM_THREADS"] = %d
c = ov.Core().compile_model(m, "CPU", cfg)
nthr = c.get_property("INFERENCE_NUM_THREADS")
r = c.create_infer_request(); r.infer({0:x})
t0=time.perf_counter()
for _ in range(200): r.infer({0:x})
print("AUDIT_RESULT", json.dumps({"nthr": str(nthr), "mean_ms": (time.perf_counter()-t0)/200*1e3}))
""" % (t, t)
        r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120)
        mm = re.search(r"AUDIT_RESULT (\{.*\})", r.stdout)
        d = json.loads(mm.group(1)) if mm else {"error": r.stderr[-150:]}
        d["requested"] = t or "default"
        rows.append(d)
        print(f"  threads_req={d['requested']} -> {d.get('nthr')} ({d.get('mean_ms','?')} ms)", flush=True)
    return rows


def main():
    print("[audit] section 1: device properties", flush=True)
    s1 = section1_device_properties()
    print("[audit] section 2: ISA ceiling matrix (fresh subprocess per value)", flush=True)
    s2 = section2_isa_matrix()
    print("[audit] section 3: precision hints", flush=True)
    s3 = section3_precision_hints()
    print("[audit] section 4: operator smoke vs numpy", flush=True)
    s4 = section4_op_smoke()
    print("[audit] section 5: threading", flush=True)
    s5 = section5_threading()

    full = {"section1_device": s1, "section2_isa_matrix": s2,
            "section3_precision": s3, "section4_op_smoke": s4,
            "section5_threading": s5}
    with open(os.path.join(OUT, "cpu_support_audit.json"), "w") as f:
        json.dump(full, f, indent=1, default=str)
    with open(os.path.join(OUT, "isa_ceiling_matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["isa_setting", "default_run_ok", "default_correct", "default_exec_types", "default_error", "f32_run_ok", "f32_correct", "f32_exec_types", "f32_error", "verbose_isa_header", "verbose_impl_f32"])
        w.writeheader()
        for r in s2:
            w.writerow(r)
    with open(os.path.join(OUT, "op_smoke_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["op", "mode", "exec_ok", "correct", "strict_pass", "max_abs_err", "out_dtype", "exec_types", "error"])
        w.writeheader()
        for r in s4:
            w.writerow({**r, "exec_types": str(r.get("exec_types"))})
    with open(os.path.join(OUT, "device_properties.txt"), "w") as f:
        f.write(json.dumps(s1, indent=1, default=str))
    print(f"[audit] saved -> {OUT}/cpu_support_audit.json")


if __name__ == "__main__":
    main()

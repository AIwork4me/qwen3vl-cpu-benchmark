# Phase 7/8 — instruction-level execution verification availability (HOST B)

Goal of Phases 7–8: prove with Linux perf + oneDNN JIT profiling that CPU
samples actually land inside the vpdpbusd-containing JIT kernels
(Level D evidence), and check for a PMU event that counts VNNI retired
instructions.

## Result: NOT AVAILABLE in this environment — exact errors

- `perf` binary: **not installed** on this container host (`which perf` empty).
- `perf_event_paranoid = 4` (`/proc/sys/kernel/perf_event_paranoid`): even with
  a perf binary, kernel-wide PMU access is disabled for unprivileged users.
  (4 = most restrictive; requires CAP_PERFMON / root to change.)
- Container restriction: no CAP_PERFMON (we do not have root; and per task
  rules we do NOT modify system security policy).
- Consequently `perf list`, `perf record`, `perf stat`, `perf annotate` are all
  unavailable — the PMU event census required by Phase 8 cannot be performed.

## Consequence for the evidence chain (honest downgrade)

Level D (executed-hotspot) evidence was NOT obtained on HOST B. The VNNI
execution claim therefore rests on:

  Level B — runtime dispatch: 253/253 fully-connected nodes execType
            `brgemm_avx512_bf16` (OpenVINO profiling CSVs), plus the
            DYNAMIC_QUANTIZATION_GROUP_SIZE=32 property and 6 JIT'd
            `brgemm_src_quantization_kernel_t` instances in the default dump.
  Level C — JIT machine code: 12 JIT brgemm kernels containing 208 `vpdpbusd`
            each in the default run; ZERO in the DQ=0 and FP16-model dumps;
            12 kernels (208–304 each) in both f32-hint dumps.
  Level E — counterfactual: setting DYNAMIC_QUANTIZATION_GROUP_SIZE=0 removes
            the VNNI kernels from the JIT dump AND slows the workload
            1.37–1.47× (3 fresh processes × 10 iters, CV ≤ 2%); DQ=128 gives a
            further 1.20–1.26× over the DQ=32 default.

Suggested on-host command for anyone reproducing on a perf-enabled machine
(from the oneDNN JIT profiling documentation):

    mkdir -p perf && cd perf
    JITDUMPDIR=. ONEDNN_JIT_PROFILE=6 perf record -k 1 -g -- \
      taskset -c 0-15 .venv-openvino/bin/python scripts/investigate_openvino_runtime.py \
        --mode bench --tag perf_p1 --threads 16 --warm-iters 5 --measure-iters 100 \
        --prompts P1
    perf inject -j -i perf.data -o perf.data.j && perf report -i perf.data.j --stdio

Expected symbol names to look for: jit: [brgemm_kernel_t...] and
[src_quantization...]; annotated instructions should include vpdpbusd.

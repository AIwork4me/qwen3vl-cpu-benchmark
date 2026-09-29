# Host-B investigation environment (2026-09-29)

This root-cause investigation runs on a **different physical host** than the
original benchmark (README.md / report/RESULTS.md). Both are recorded here so
no conclusion is silently transferred between machines.

## HOST A (original benchmark host — NOT this machine)

- CPU: AMD Ryzen AI Max+ PRO 395 w/ Radeon 8060S (Strix Halo, Zen 5, 16C/32T, NUMA=1)
- RAM: 94.06 GiB LPDDR5X UMA
- OS: Ubuntu 24.04.4 LTS, kernel 6.17.0-1032-oem
- OpenVINO: 2026.4.0
- Recorded in `environment/system.md` (committed upstream)

## HOST B (this root-cause investigation host)

- CPU: 2× AMD EPYC 9334 (Zen 4, 32C/64T per socket, 64C/128T total, NUMA=2)
- ISA flags (lscpu): avx2, avx512f/dq/bw/vl/cd, **avx512_vnni**, avx512_bf16,
  av512vbmi/vbmi2/vpopcntdq/bitalg, gfni, vaes — same ISA superset relevant to
  this investigation as HOST A (both expose AVX512_CORE + AVX512_VNNI +
  AVX512_BF16 to oneDNN; neither has AMX)
- RAM: 503 GiB DDR5 (12 channels/sock, no UMA)
- L3: 256 MiB (8 instances); L2 64 MiB
- governor/EPP: recorded below at run time (cpupower needs root → frequency
  read from /proc/cpuinfo if available)
- OS/kernel/python: recorded in `environment_root_cause/host_b_system.json`
- OpenVINO wheel: openvino==2026.4.0 from PyPI (identical version string to
  HOST A: `2026.4.0-22959-99c81491cc3-releases/2026/4`)
- Model: identical artifact family — `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov`
  re-downloaded from ModelScope on HOST B (SHA256 recorded in
  `results/openvino_root_cause/host_b_model_sha256.txt`)

## Why HOST B is acceptable for the *mechanistic* questions

Q1/Q2 (does the OpenVINO CPU plugin dispatch/execute an AVX512_VNNI path for
this workload; what is its incremental contribution) are decided by the
OpenVINO+oneDNN **runtime code path**, which is a property of the same wheel +
same IR + an ISA-identical CPU (AVX512_VNNI present, AMX absent). HOST B
satisfies all three. Absolute latencies (Q3, the 0.187 s headline) are NOT
re-measured as "the" baseline here; HOST-B latencies are reported as their own
series and all Zen 5-specific absolute numbers remain attributed to HOST A
only.

## Consequences for claims

- "Executed AVX512_VNNI on HOST A (Ryzen AI Max+ 395)": only claimable via
  transfer argument (same wheel, same IR, same dispatch decision inputs) —
  labeled STRONG EVIDENCE, not direct measurement.
- "Executed AVX512_VNNI on HOST B (EPYC 9334)": directly measured here.
- ISA-ablation speedups: measured on HOST B; HOST-A magnitudes may differ
  (memory subsystem, 2-socket NUMA). Labeled per-host everywhere.

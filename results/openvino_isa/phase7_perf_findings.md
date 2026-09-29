# Phase 7 — Linux perf hotspot evidence: NOT OBTAINABLE (documented)

## What was attempted (2026-09-29)

Planned: `JITDUMPIR=. ONEDNN_JIT_PROFILE=6 perf record -k 1 -g -- <P1 loop>` +
`perf inject -j` + `perf report --stdio` to place CPU samples on the oneDNN JIT kernels
found in Phase 6, and `perf annotate` on the VNNI MAC loops.

## Exact blockers (recorded verbatim)

1. `/proc/sys/kernel/perf_event_paranoid` = **4** — most restrictive setting; unprivileged
   perf_event_open for userspace sampling is denied. Changing it would require passworded
   sudo (task rules forbid modifying system security policy), and the user account has no
   passwordless sudo (`sudo -n true` -> "a password is required").
2. The installed `perf` front-end does not match the running kernel:
   ```
   WARNING: perf not found for kernel 6.17.0-1032-oem
   You may need to install the following packages for this specific kernel:
     linux-tools-6.17.0-1032-oem
     linux-cloud-tools-6.17.0-1032-oem
   ```
   (`/usr/lib/linux-tools/6.17.0-1032-oem/` exists but the wrapper refuses to run for the
   running kernel; and even a working binary would still be gated by paranoid=4.)
3. Consequently `perf stat` (Phase 9 counters) also returns only the same warning — no
   cycles/instructions/cache-misses data without privileges.

## Evidence-level impact

- Level D (executed hotspot samples on the VNNI JIT kernels) is **NOT OBTAINED** in this
  environment. Per task rules, no sudo/system changes were attempted.
- The execution claim rests on Level B (runtime dispatch: brgemm kernels carry 91–92% of
  node time, Phase 2), Level C (JIT machine code contains vpdpbusd MAC loops in exactly
  those kernel classes, Phase 6), and Level E (counterfactual: disabling dynamic
  quantization removes all VNNI kernels from the JIT set and costs +23–43% latency,
  Phase 5). Executed-sample confirmation remains open for an environment with
  perf_event_paranoid<=2 and matching linux-tools.

## Phase 8 (PMU events for VNNI retired-instruction counting)

`perf list` is unavailable for the same reasons (no working perf for this kernel). No PMU
event enumeration was possible, so per the task's anti-fabrication rule:

> No reliable architectural PMU counter exposure could be verified on this environment
> (perf tooling non-functional for the running kernel; perf_event_paranoid=4). Direct
> VNNI retired-instruction counting was not attempted and must not be inferred.

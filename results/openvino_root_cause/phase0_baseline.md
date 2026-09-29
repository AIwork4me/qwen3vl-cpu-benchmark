# Phase 0 — Baseline reproduction (root-cause investigation)

Date: 2026-09-29. Command (unmodified existing bridge script, fresh process):

```bash
.venv-openvino/bin/python scripts/bench_openvino_bridge.py --tag rc_p0_baseline --warm-iters 10
```

Fixed conditions: device=CPU, PERFORMANCE_HINT=LATENCY, NUM_STREAMS=1 (LATENCY default),
threads=default (OV auto-selects 16 = physical cores, per openvino_properties.json),
same model (`models/qwen3vl-openvino-int8`, SHA256 in environment/model_sha256.txt),
same bridge (layer-35 residual `__module.model.language_model.layers.35/aten::add/Add_1`),
same prompts P1/P2/P3, same T2I token semantics.

## Reproduced numbers (this run, p50 of 10 warm iters after 1 warmup + 1 first)

| Prompt | tokens | this run p50 | historical (results/comparison/summary.csv `bridge`) | delta |
|---|---:|---:|---:|---:|
| P1 | 25 kept (39 full) | 0.183 s | 0.187 s mean / 0.183 p50 | -2.1% vs mean, 0% vs p50 |
| P2 | 67 kept (81 full) | 0.372 s | 0.377 s mean / 0.377 p50 | -1.3% |
| P3 | 171 kept (185 full) | 0.675 s | 0.687 s mean / 0.697 p50 | -1.7% vs mean |

Verdict: baseline REPRODUCED within run-to-run noise (< ±3%). Hidden-state statistics
match the committed accuracy data bit-for-bit at print precision (P1 mean 0.131684 vs
0.131684 in embedding_accuracy.csv) — same model, same tokenization, same output node.

## Environment at run time

- governor: powersave; EPP: balance_performance (unchanged from original benchmark)
- CPU freq at idle sample: ~3537 MHz max (lscpu MHz); during this run's encode intervals the
  20 Hz monitor recorded sustained ~3.1–3.2 GHz means (resources CSV), i.e. no depressed clocks.
  (The 3.90 GHz figure elsewhere in the repo belongs to the ComfyUI CPU runs, not the OV bridge.)
- temperature: readable via hwmon (e.g. hwmon1 47.0 C at idle; the run-time resources CSV has no
  temperature column, so run-time temperature is unavailable rather than measured)
- OpenVINO: 2026.4.0-22959-99c81491cc3-releases/2026/4
- Python: 3.12.3 (.venv-openvino)
- kernel: 6.17.0-1032-oem, Ubuntu 24.04.4
- CPU flags: avx2, avx512f, av512_vnni (avx512_vnni), avx_vnni, avx512_bf16 — full set in environment/system.json
- EXECUTION_DEVICES assert: ['CPU'] (in-script hard assert passed)

Raw artifacts: openvino_rc_p0_baseline.json / .csv / _resources.csv (20 Hz sampling) in this directory.

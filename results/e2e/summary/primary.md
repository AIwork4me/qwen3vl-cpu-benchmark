# E2E auto-generated summary (Phase 33)

All numbers computed by scripts/aggregate_e2e.py from results/e2e/*/＋raw; do not edit by hand.

## Headline (group=ab, 1024x1024, 20 steps, P3)

| route | n_proc | cold TTFI (s) | warm E2E (s, median-of-medians) | CI95 | enc (s) | DiT (s) | VAE (s) |
|---|---:|---:|---:|---|---:|---:|---:|
| gpu | 3 | 176.11 | 154.17 | (144.532, 154.386) | 2.618 | 149.05 | 2.43 |
| native | 1 | 255.03 | 248.87 | None | 99.436 | 148.84 | 2.22 |
| ov_dq32 | 3 | 168.82 | 151.25 | (148.617, 152.517) | 0.993 | 147.95 | 2.25 |

## Warm stats (per group/route/prompt)

See stats.csv (full) — excerpt below:

- ab/gpu P3 @20st 1024x1024: n=9 e2e_med=154.173s enc=2.642 dit=149.031 vae=2.431
- ab/native P3 @20st 1024x1024: n=3 e2e_med=248.868s enc=99.436 dit=148.843 vae=2.221
- ab/ov_dq32 P3 @20st 1024x1024: n=9 e2e_med=151.254s enc=1.006 dit=147.952 vae=2.255
- dq/ov_dq0 P3 @20st 1024x1024: n=2 e2e_med=153.782s enc=1.436 dit=150.081 vae=2.262
- dq/ov_dq128 P3 @20st 1024x1024: n=2 e2e_med=153.362s enc=0.834 dit=150.097 vae=2.429
- dq/ov_dq64 P3 @20st 1024x1024: n=2 e2e_med=153.183s enc=0.884 dit=150.003 vae=2.294
- matrix/gpu P1 @20st 1024x1024: n=1 e2e_med=155.178s enc=1.351 dit=151.441 vae=2.382
- matrix/gpu P2 @20st 1024x1024: n=1 e2e_med=154.177s enc=2.003 dit=149.944 vae=2.225
- matrix/gpu P3 @20st 1024x1024: n=1 e2e_med=155.071s enc=2.691 dit=150.139 vae=2.237
- matrix/gpu P4 @20st 1024x1024: n=1 e2e_med=152.56s enc=1.296 dit=149.023 vae=2.234
- matrix/gpu P5 @20st 1024x1024: n=1 e2e_med=157.063s enc=4.142 dit=150.662 vae=2.253
- matrix/gpu P3 @20st 1328x1328: n=1 e2e_med=384.007s enc=2.758 dit=375.244 vae=5.95
- matrix/gpu P3 @40st 1024x1024: n=1 e2e_med=305.805s enc=2.653 dit=300.464 vae=2.684
- matrix/ov_dq32 P1 @20st 1024x1024: n=1 e2e_med=152.421s enc=0.348 dit=149.657 vae=2.414
- matrix/ov_dq32 P2 @20st 1024x1024: n=1 e2e_med=153.679s enc=0.613 dit=150.565 vae=2.488
- matrix/ov_dq32 P3 @20st 1024x1024: n=5 e2e_med=156.235s enc=0.99 dit=151.095 vae=2.348
- matrix/ov_dq32 P4 @20st 1024x1024: n=1 e2e_med=150.984s enc=0.33 dit=148.437 vae=2.216
- matrix/ov_dq32 P5 @20st 1024x1024: n=1 e2e_med=153.645s enc=1.49 dit=149.918 vae=2.235
- matrix/ov_dq32 P3 @20st 1328x1328: n=1 e2e_med=380.748s enc=0.996 dit=375.033 vae=4.709
- matrix/ov_dq32 P3 @40st 1024x1024: n=1 e2e_med=304.625s enc=1.047 dit=301.248 vae=2.319
- smoke/gpu P1 @4st 1024x1024: n=1 e2e_med=31.53s enc=1.26 dit=27.355 vae=2.895
- smoke/ov_dq32 P1 @4st 1024x1024: n=1 e2e_med=29.833s enc=0.361 dit=27.065 vae=2.405
- throughput/ov_dq32 P1 @20st 1024x1024: n=2 e2e_med=152.539s enc=0.713 dit=149.745 vae=2.596
- throughput/ov_dq32 P2 @20st 1024x1024: n=2 e2e_med=156.572s enc=2.645 dit=153.404 vae=2.873
- throughput/ov_dq32 P3 @20st 1024x1024: n=6 e2e_med=162.656s enc=1.03 dit=158.838 vae=3.127
- throughput/ov_dq32 P4 @20st 1024x1024: n=2 e2e_med=162.21s enc=0.473 dit=159.411 vae=2.633
- throughput/ov_dq32 P5 @20st 1024x1024: n=2 e2e_med=165.026s enc=3.673 dit=161.674 vae=2.514

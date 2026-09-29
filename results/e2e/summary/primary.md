# E2E auto-generated summary (Phase 33)

All numbers computed by scripts/aggregate_e2e.py from results/e2e/*/＋raw; do not edit by hand.

## Headline (group=ab, 1024x1024, 20 steps, P3)

| route | n_proc | cold TTFI (s) | warm E2E (s, median-of-medians) | CI95 | enc (s) | DiT (s) | VAE (s) |
|---|---:|---:|---:|---|---:|---:|---:|
| gpu | 3 | 176.11 | 154.17 | (144.532, 154.386) | 2.618 | 149.05 | 2.43 |
| ov_dq32 | 3 | 168.82 | 151.25 | (148.617, 152.517) | 0.993 | 147.95 | 2.25 |

## Warm stats (per group/route/prompt)

See stats.csv (full) — excerpt below:

- ab/gpu P3 @20st 1024x1024: n=9 e2e_med=154.173s enc=2.642 dit=149.031 vae=2.431
- ab/ov_dq32 P3 @20st 1024x1024: n=9 e2e_med=151.254s enc=1.006 dit=147.952 vae=2.255
- smoke/gpu P1 @4st 1024x1024: n=1 e2e_med=31.53s enc=1.26 dit=27.355 vae=2.895
- smoke/ov_dq32 P1 @4st 1024x1024: n=1 e2e_med=29.833s enc=0.361 dit=27.065 vae=2.405

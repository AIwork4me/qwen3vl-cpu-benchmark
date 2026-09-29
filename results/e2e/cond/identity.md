# Conditioning identity (Phase 4/30)

| polarity | prompt | route | ref | shape_eq | n_tok | cosine | RMSE | relL2 | max_abs | mean | std |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| pos | P1 | smoke_gpu | smoke_gpu | True | 25 | 1.000000 | 0.0 | 0.000000 | 0.0 | 0.13493 | 12.10996 |
| pos | P1 | smoke_ov | smoke_gpu | True | 25 | 0.997244 | 0.8989 | 0.074223 | 53.4953 | 0.13168 | 12.05269 |
| pos | P3 | ab_p1_gpu | ab_p1_gpu | True | 171 | 1.000000 | 0.0 | 0.000000 | 0.0 | 0.10129 | 11.00108 |
| pos | P3 | ab_p1_ov | ab_p1_gpu | True | 171 | 0.998451 | 0.6127 | 0.055689 | 53.4928 | 0.09718 | 11.00863 |
| neg | P1 | smoke_gpu | smoke_gpu | True | 9 | 1.000000 | 0.0 | 0.000000 | 0.0 | -0.0197 | 12.23197 |
| neg | P1 | smoke_ov | smoke_gpu | True | 9 | 0.971018 | 2.9248 | 0.239112 | 186.5498 | -0.01816 | 11.96368 |
| neg | P3 | ab_p1_gpu | ab_p1_gpu | True | 9 | 1.000000 | 0.0 | 0.000000 | 0.0 | -0.0197 | 12.23197 |
| neg | P3 | ab_p1_ov | ab_p1_gpu | True | 9 | 0.971018 | 2.9248 | 0.239112 | 186.5498 | -0.01816 | 11.96368 |

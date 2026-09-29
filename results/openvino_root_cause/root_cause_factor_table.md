# Root-cause factor evidence (machine-generated, HOST B)

generated: 2026-09-29T11:09:54.829673

## ISA/DQ matrix (warm p50, median of fresh processes)

| config | P1 | P2 | P3 |
|---|---:|---:|---:|
| default | 0.1922s (n=3) | 0.4375s (n=3) | 0.9311s (n=3) |
| isa_bf16 | 0.1869s (n=3) | 0.4181s (n=3) | 0.8821s (n=3) |
| vnni_f32 | 0.1971s (n=3) | 0.4466s (n=3) | 0.9618s (n=3) |
| core_f32 | 0.1773s (n=3) | 0.4422s (n=3) | 0.9039s (n=3) |
| dq0 | 0.2632s (n=3) | 0.6425s (n=3) | 1.3067s (n=3) |
| dq128 | 0.1565s (n=3) | 0.3786s (n=3) | 0.7393s (n=3) |

Non-runnable ISA-ceiling configs (dispatch evidence):
- `err_avx2`: process exit non-zero — FullyConnectedCompressed could not create a primitive descriptor (needs >=AVX512_CORE_VNNI for its int8 DQ kernels)
- `err_avx2_vnni`: process exit non-zero — FullyConnectedCompressed could not create a primitive descriptor
- `err_core`: process exit non-zero — FullyConnectedCompressed could not create a primitive descriptor
- `err_vnni`: process exit non-zero — FCCompressed compiles (VNNI suffices for the int8 DQ kernels); model fails at first ScaledDotProductAttentionWithKVCache: 'brgemm bf16 kernel could only be used above avx512_bf16' (attention selected bf16)

## Key ratios

- P1 dq_vnni_path_vs_dq0_bf16_speedup: 1.37
- P1 dq128_vs_dq0_bf16_speedup: 1.6822
- P1 dq128_vs_default_dq32_speedup: 1.2279
- P1 default_vs_isa_bf16_ceiling_speedup: 0.9727
- P1 default_vs_vnni_f32_speedup: 1.0258
- P1 vnni_f32_vs_core_f32_speedup: 0.8994
- P2 dq_vnni_path_vs_dq0_bf16_speedup: 1.4685
- P2 dq128_vs_dq0_bf16_speedup: 1.697
- P2 dq128_vs_default_dq32_speedup: 1.1556
- P2 default_vs_isa_bf16_ceiling_speedup: 0.9556
- P2 default_vs_vnni_f32_speedup: 1.0206
- P2 vnni_f32_vs_core_f32_speedup: 0.9903
- P3 dq_vnni_path_vs_dq0_bf16_speedup: 1.4034
- P3 dq128_vs_dq0_bf16_speedup: 1.7674
- P3 dq128_vs_default_dq32_speedup: 1.2594
- P3 default_vs_isa_bf16_ceiling_speedup: 0.9474
- P3 default_vs_vnni_f32_speedup: 1.0329
- P3 vnni_f32_vs_core_f32_speedup: 0.9398

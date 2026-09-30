# System Baseline (2026-09-29T13:25:46.528129)

- CPU: **AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S** — 16 cores / 32 threads, NUMA=1
- L1d 768 KiB (16 instances) | L2 16 MiB (16 instances) | L3 64 MiB (2 instances)
- ISA: avx2=True avx512f=True avx512_vnni=True avx_vnni=True avx512_bf16=True
- Governor=powersave, EPP=balance_performance, profile=balanced, freq now {'min': 1774.5, 'max': 3550.3, 'mean': 2772.2}
- RAM: **94.06 GiB** (speed: unavailable (needs root dmidecode); UMA shared with iGPU)
- Swap: Swap:          8.0Gi          0B       8.0Gi
- GPU (not used in benchmark): c3:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 1586 (rev d1)
  > 注（2026-09-30 补）：本文件为 2026-09-29T13:25 快照；"GPU not used in benchmark" 仅对当时的实验 A/B 成立。同日实验 C（GPU BF16 编码器，results/comfy_gpu/）与 e2e 轮（GPU 跑 DiT/VAE，results/e2e/）均大量使用该 GPU（Radeon 8060S，GTT 100 GiB）。
- OS: PRETTY_NAME="Ubuntu 24.04.4 LTS" | NAME="Ubuntu" kernel 6.17.0-1032-oem
- Python: Python 3.12.3 | uv: uv 0.12.3 (x86_64-unknown-linux-gnu)

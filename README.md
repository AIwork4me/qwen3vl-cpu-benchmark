# Qwen3-VL Ryzen AI Max+ 395 CPU Benchmark

对比两条 CPU 路线在 AMD Ryzen AI Max+ 395 (Strix Halo, 16C/32T) 上的真实测量数据:

- **A**: `qwen3vl_8b_int8_convrot.safetensors` + ComfyUI + CPU
- **B**: `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov` + OpenVINO + Ryzen CPU

所有结论必须由日志 / 源码路径 / profiler 证据支持。不修改任何现有项目；
ComfyUI 核心全新安装在 `ComfyUI/`（本目录内，独立 venv `.venv-comfy`），
OpenVINO 使用独立 venv `.venv-openvino`。

## 目录

```
environment/    系统基线 (system.json / system.md / model_inventory.md)
scripts/        resource_monitor.py + 两个 benchmark 脚本
prompts/        prompts.json (P1/P2/P3 统一 prompt)
workflows/      ComfyUI workflow JSON (如需)
models/         本地模型 (ModelScope 下载, 带校验)
results/raw/    原始 JSON/CSV iteration 数据
results/comfy_cpu/       实验 A 原始结果 + dispatch.log
results/openvino_cpu/    实验 B 原始结果 + openvino_properties.json
results/comparison/      summary.csv / summary.json / embedding_accuracy.csv
report/         qwen_image_conditioning_semantics.md / RESULTS.md
logs/           下载日志 / 运行日志
ComfyUI/        本实验专用的 ComfyUI checkout (master tarball)
```

## 运行约定

- 模型只从 ModelScope 下载（HF 被墙），见 `environment/model_inventory.md`。
- benchmark 一律加载本地文件，禁止运行时联网。
- CPU pinning 通过 ComfyUI `--cpu` 参数与源码级验证双重确认；
  日志打印 load_device / offload_device / 参数与输入输出 tensor device。
- INT8 计算路径分类: `TRUE_INT8_CPU_COMPUTE` / `INT8_WEIGHT_STORAGE_FP_COMPUTE` /
  `UNSUPPORTED_OR_ERROR`，证据存 `results/comfy_cpu/`。
- OpenVINO 显式 `device="CPU"`，禁止 AUTO。

## 主要脚本

```
python scripts/bench_comfy_qwen3vl_cpu.py     # 实验 A
python scripts/bench_openvino_qwen3vl_cpu.py  # 实验 B
```

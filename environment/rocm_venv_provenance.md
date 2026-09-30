# .venv-comfy-rocm 来源声明

- torch 2.12.0+rocm7.14.0 (HIP 7.14.60850) 来自本机已有 `SenseNova-U1.5-ROCm/.venv` 的
  site-packages **硬链接复制** (`cp -al`)，随后仅在新 venv 内补装 ComfyUI requirements。
- 未修改源 venv：安装前后各导出 145 个包的 `importlib.metadata` 清单，`diff` 为空
  （验证时间 2026-09-29）。
- 源 venv 清单快照: sensenova_venv_packages_before.txt / after.txt（逐字节相同）。

> 追记（2026-09-30，e2e 轮）：`.venv-comfy-rocm` 内另补装 `openvino==2026.4.0`（与 `.venv-openvino` 同版本 wheel，用于 in-process Hybrid 桥与流水线实验；见 environment/e2e_checkpoints.md Gate-1 finding 9）。源 venv 其余未动。

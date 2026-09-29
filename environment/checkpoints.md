# Subagent Checkpoint 验收记录

按任务协议，每个阶段由独立 Subagent 只读验核，全部 PASS 后才进入下一阶段。
记录日期: 2026-09-29。

## CHECKPOINT 1 — 目录隔离与下载完整性 → PASS

- 目录隔离：邻居项目 git 状态实核未变（muse-rocm HEAD e6eb7c2d、porcelain 0 行；
  SenseNova-U1.5-ROCm-work 非 git 仓库，如实记录）。
- 模型完整性：`qwen3vl_8b_int8_convrot.safetensors` = 9,350,798,360 bytes（与 ModelScope API
  声明一致）；bf16 = 17,534,334,616 bytes；OpenVINO `openvino_language_model.bin` =
  7,573,989,065 bytes（Subagent 更正了验收规格中的笔误常数，文件与上游 SHA256 逐字吻合）。
- 无 .incomplete 残留、无 modelscope 进程残留；model_inventory.md 声明了
  ModelScope 优先级与 INT8_ASYM（非 W8A8）。

## CHECKPOINT 2 — ComfyUI 路线 → PASS

- CPU 证据：weight_devices=['cpu']、load/offload device=cpu、torch 2.9.1+cpu、
  comfy-kitchen backends cuda/hip=false、输出 device=cpu（脚本硬断言）。
- A2 取证链：int_mm=0、dequant=504（窗口=warmup+first 两次 encode，=252 层/encode）、
  **dequant_bytes=13,891,534,848 与 (int8 载荷 8,190,427,136 − 2×embed/lm_head 622,329,856) × 2
  字节级精确对账**。
- A1 取证链：int_mm=504/dequant=0、profiler 中 `aten::_int_mm` 252 次调用、
  `comfy_kitchen::int8_linear` 252 次。
- 测量方法：20Hz 采样（n_samples/wall_s 精确=20.0）、warm×5、first 仅 P1、
  page cache 未清空已如实声明。
- 精度：embedding_accuracy.csv 9 行全部 shape_equal=True、cosine>0.997。

## CHECKPOINT 3 — 最终报告忠实性 → PASS

- 抽查 6 组数字全部可由底层 JSON/CSV 精确复现：
  A2 warm 中位数 12.978/13.729/15.428 s（重算一致）；
  A1 P1 7.322 s；OV bridge 0.1871/0.3765/0.6872 s、load 1.2625 s；
  RSS 7.587 vs 15.142 GiB；cosine 范围与模型大小逐格一致。
- 可比性边界诚实：standalone 标 NOT DIRECTLY COMPARABLE、OpenVINO True INT8 GEMM 标
  UNKNOWN（未做内部 dump）、量化方案注明"非 W8A8"。
- OpenVINO 设备证据：execution_devices=["CPU"]、INFERENCE_NUM_THREADS=16、NUM_STREAMS=1。
- 文件清单完整无缺；runtime_path.md 分类 A2×5 + A1×3 与 JSON meta 一致。
- 次要备注（不影响判定）：§1 加速比按中位数精确为 69.4×/36.5×/22.4×，报告取保守值 22–68×。

## 结论

三重独立验核全部 PASS；报告数字可复现、取证链完整、诚实性声明齐备。

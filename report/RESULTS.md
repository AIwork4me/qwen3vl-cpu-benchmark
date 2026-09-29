# Qwen3-VL Ryzen AI Max+ 395 CPU Benchmark — 最终报告

> **更新（2026-09-29 晚，root-cause investigation）**：本报告 §1/§7 中的

日期: 2026-09-29 · 全部数据为真实测量（20 Hz psutil 采样 + `time.perf_counter_ns()`），
原始数据见 `results/raw/` 与 `results/*/`，逐 iteration 数据完整保留。

## 0. 环境基线（实测）

| 项 | 值 |
|---|---|
| CPU | AMD Ryzen AI Max+ PRO 395 w/ Radeon 8060S（16C/32T，NUMA=1） |
| ISA | AVX2 ✓ AVX-512 ✓ AVX512_VNNI ✓ AVX_VNNI ✓ AVX512_BF16 ✓（无 AMX） |
| 频率/功耗 | governor=powersave，EPP=balance_performance；encode 期间实测均值 3.90 GHz |
| RAM | 94.06 GiB（UMA，与 iGPU 共享；速度需 root 不可读，如实标注） |
| OS | Ubuntu 24.04.4 LTS，kernel 6.17.0-1032-oem |
| ComfyUI | 0.37.0（master tarball 2026-09-29，本实验专用 checkout，独立 .venv-comfy） |
| PyTorch | 2.9.1+cpu（`torch.version.cuda=None` → comfy-kitchen CUDA 后端自动禁用） |
| Comfy Kitchen | 0.2.36（本机仅 eager 后端可用：cuda/hip/triton/ascend 均 unavailable） |
| OpenVINO | 2026.4.0（独立 .venv-openvino），`execution_devices=['CPU']` 运行时断言 |
| 模型来源 | 全部 ModelScope（HF 不可达）：`Comfy-Org/Qwen-Image-2.1`、`OpenVINO/Qwen3-VL-8B-Instruct-int8-ov`；SHA256 记录于 `environment/model_sha256.txt`，大小与上游逐字节一致 |

统一 prompts P1/P2/P3（`prompts/prompts.json`），三条 runtime 使用完全相同字符串。

---

## 1. 主对比表（conditioning-encoder 语义，可比部分）

| Metric | ComfyUI INT8 ConvRot CPU（产品路径 A2） | OpenVINO INT8 CPU（conditioning bridge） |
|---|---:|---:|
| Runtime | ComfyUI 0.37.0 + torch 2.9.1+cpu | OpenVINO 2026.4.0 |
| Quant scheme | int8_tensorwise + ConvRot（per-layer `.comfy_quant`，groupsize 256） | INT8_ASYM weight compression（NNCF，group_size=-1，**非 W8A8**） |
| True INT8 GEMM | **NO**（int_mm=0，252 线性层/encode 全部 dequant） | UNKNOWN¹（IR 内部实现未剖析；权重 int8 常驻） |
| Model disk | 9.351 GB | 8.810 GB |
| Model load | 0.30 s（mmap assign；真实 page-in 发生在首次 encode） | **1.26 s**（read 0.10 + compile 1.16） |
| First encode（冷） | 12.11 s（P1，含 page-in） | **0.18 s** |
| RAM after load | 7.48 GiB（首次 encode 后 page 物化） | ~14.5 GiB（峰值，含运行时缓冲） |
| Peak RAM | 7.59 GiB | 14.57 GiB |
| RAM delta（baseline→peak） | ≈7.4 GiB | ≈14.4 GiB |
| Warm encode P1（25 tok） | 12.98 s² | **0.187 s** |
| Warm encode P2（67 tok） | 13.73 s² | **0.377 s** |
| Warm encode P3（171 tok） | 15.43 s² | **0.687 s** |
| CPU utilization | ~1318%（16 线程，均值 3.9 GHz） | ~1332%（16 线程） |
| Threads（最优） | 16（默认；8/32 均更慢） | 16（默认；8 更慢，32 持平） |
| Output semantics identical | YES（本机同一路径） | **YES³**（token 级对齐 + shape 一致） |
| Embedding cosine vs BF16 | 0.9989 / 0.9992 / 0.9993 | 0.9972 / 0.9981 / 0.9985 |
| Exact Qwen-Image compatible | YES | **YES³**（`add_outputs` bridge，零重算） |

¹ OpenVINO INT8 GEMM 是否真实执行需 oneDNN 内部 dump 才能定性；本实验证明的是权重 int8 常驻 + CPU 执行。
² ComfyUI 列为 3 个独立 fresh process 的 warm 均值中位数（run1/2/3）。
³ bridge 条件：完全复刻 ComfyUI T2I_TEMPLATE（39/81/185 token，含 system turn）→ 同为 layer-36 residual stream、**无 final RMSNorm** → 同样裁掉 system turn（第二个 `<|im_start|>` 起），输出 [1,25/67/171,4096] 与 ComfyUI 逐 prompt 同形。数值 cosine vs BF16 参考 0.997+，语义对齐成立。

**可比结论**（语义一致前提下）：OpenVINO bridge 比 ComfyUI 产品路径快 **22–68×**（P1 68× / P2 35× / P3 22×），加载快 ~10×（首次可用视角），代价是峰值 RAM 高约 2×（15 GiB vs 7.6 GiB，绝对值仍只占 94 GiB 的 16%）。

---

## 2. OpenVINO standalone 表（官方 chat 模板，语义与 ComfyUI 不同 — NOT DIRECTLY COMPARABLE）

官方 IR 是多模态生成导出（inputs_embeds + M-RoPE position_ids[3,b,s] + KV cache states），
standalone 数值 = 单次 prefill（reset_state 后前向一次，无自回归生成）。
Token 数（官方模板）25/67/171 ≠ ComfyUI 的 25/67/171（裁剪后，巧合一致）——模板语义不同，仅几何可比：

| Config | P1 warm | P2 warm | P3 warm | Peak RSS |
|---|---:|---:|---:|---:|
| default（LATENCY hint，自动 16 线程，1 stream） | 0.140 s | 0.370 s | 0.568 s | 15.14 GiB |
| threads=8 | 0.201 s | 0.535 s | 0.831 s | 14.98 GiB |
| threads=16 | 0.146 s | 0.383 s | 0.585 s | 14.98 GiB |
| threads=32 | 0.145 s | 0.378 s | 0.600 s | 15.08 GiB |

最优 = 默认（LATENCY hint 自动选择 16 物理核）；LATENCY hint 默认 NUM_STREAMS=1（已记录属性），单用户 latency 语义。

---

## 3. ComfyUI INT8 计算路径取证（任务 Q1）

### 产品 conditioning 路径判定：**A2 — INT8_WEIGHT_STORAGE_FP_COMPUTE**

证据链（全部来自本机实测/源码）：

1. **权重确为 INT8+ConvRot 存储**：safetensors 头解析 254 组 `comfy_quant` JSON =
   `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`；
   运行时 254 个 QuantizedTensor（TensorWiseINT8Layout），`weight_devices=['cpu']`。
2. **运行时计数器**（每次 encode，3 prompt × 8 run 一致）：
   `torch._int_mm` 调用 = **0**；`QuantizedTensor.dequantize` 调用 = **252**；
   反量化字节数 = **13,891,534,848 B/encode** —— 与
   `(int8 线性层载荷 8,190,427,136 − embed 622,329,856 − lm_head 622,329,856) × 2（bf16）`
   分毫不差：每次 encode 全部 252 个 int8 线性权重（36 层 × 7 投影）被完整反量化。
3. **torch profiler**（A1 变体 run 中对照）：A2 run 无 `aten::_int_mm`，矩阵乘为 `aten::mm`（FP）。
4. **源码机制**：
   - `comfy/sd1_clip.py:114` — 文本编码器硬编码 `mixed_precision_ops(..., full_precision_mm=True)`
   - `comfy/model_patcher.py:1019` — `set_model_compute_dtype()` → `force_cast_weights=True` →
     每个模块 `comfy_force_cast_weights=True`
   - `comfy/ops.py:1448` — `_use_quantized` 条件含 `not comfy_force_cast_weights` → 恒 False
   - `comfy/ops.py:437`（`cast_bias_weight` generic tail）— dtype 不匹配 → `weight.dequantize()` → FP `F.linear`
   - 输出 dtype 实测 `torch.float32`（sd1_clip.py:279 强制 fp32 前向）
5. **CPU 设备证明**：`--cpu` 注入 cli_args + `model_options["load_device"/"offload_device"]=cpu`；
   ComfyUI 自身日志 `CLIP/text encoder model load device: cpu, offload device: cpu, current: cpu`；
   全部权重 device=cpu；输出 tensor device=cpu（脚本硬断言）；
   comfy-kitchen backends：cuda=false / hip=false / triton=false / eager=true。

### 强制变体判定：**A1 — TRUE_INT8_CPU_COMPUTE**（非产品路径，作为能力对照）

用 ComfyUI 自带的 `comfy.ops.use_quantized_matmul` 上下文（generate() 同款）+ 关闭
`comfy_force_cast_weights`（575 个模块）后：`aten::_int_mm` = 252 次/encode（profiler 实证），
dequant = 0，warm P1 12.98→7.32 s（**1.77×**）。证明本机 CPU（AVX512_VNNI）具备真 INT8 GEMM
能力（`torch._int_mm`/oneDNN），ComfyUI 的 int8 内核在 CPU 可用，只是被 conditioning 路径的
force-cast 设计关闭。

---

## 4. 精度验证（vs BF16 参考，`results/comparison/embedding_accuracy.csv`）

| 变体 | P1 cos | P2 cos | P3 cos | RMSE 范围 | shape_equal |
|---|---:|---:|---:|---|---|
| ComfyUI A2（产品，fp32 计算） | 0.99893 | 0.99921 | 0.99926 | 0.42–0.56 | ✓ |
| ComfyUI A1-forced（真 int8 GEMM） | 0.99783 | 0.99845 | 0.99859 | 0.59–0.80 | ✓ |
| OpenVINO INT8 bridge | 0.99724 | 0.99806 | 0.99845 | 0.61–0.90 | ✓ |
| ComfyUI BF16（Radeon GPU，fp32 计算） | **1.000000** | **1.000000** | **1.000000** | 0.00006–0.00008 | ✓ |

三者与 BF16 参考均高度一致（cos ≥ 0.997）；A2 最高（本就是 FP 计算）符合预期。
bridge 若 tokenization/hidden-state 层错位，cosine 不可能达到 0.997+ —— 语义对齐由数据证实。

---

## 4.5 实验 C：BF16 模型在本机 Radeon 8060S GPU 上的实测（对照路线）

**环境**：torch 2.12.0+rocm7.14.0（HIP 7.14.60850），独立 venv `.venv-comfy-rocm`
（site-packages 硬链接复用本机 SenseNova venv，源 venv 经 145 包清单 diff 验证未修改）。
ComfyUI 产品路径不变（load_clip / QwenImage21TEModel），load_device=offload_device=cuda(HIP)。

**取证**：权重全部 `cuda:0 / torch.bfloat16`；forward pre-hook 捕获前 12 个 Linear 输入 +
`F.linear` 包装计数 —— **252 个线性层全部以 `input=fp32 × weight=fp32` 在 GPU 上执行**
（`sd1_clip.py:279` 强制 fp32 的同一设计在 GPU 上生效）。即 GPU 路线的真实形态是
**"BF16 存储 → cast → FP32 GEMM"**，不是 bf16 tensor-core 计算。

| Metric | ComfyUI BF16 → GPU（产品路径） | OpenVINO INT8 bridge（CPU） |
|---|---:|---:|
| Model load（真实 disk→显存） | 4.48 s | 1.26 s |
| Warm P1（25 tok） | 0.601 s（3-run 中位） | **0.187 s** |
| Warm P2（67 tok） | 1.228 s | **0.377 s** |
| Warm P3（171 tok） | 1.834 s | **0.687 s** |
| 计算单元占用 | GPU busy 84.5–85.6%（max 100%） | CPU ~1332%（16 线程，GPU 空闲） |
| 内存占用 | 16.63 GiB GPU 侧（GTT，常驻） | 14.57 GiB 系统侧（CPU 进程） |
| Cosine vs CPU-BF16 参考 | **1.000000**（RMSE 6–8e-5） | 0.9972–0.9985 |
| Compute dtype 实测 | fp32 × fp32（252/252 线性层） | int8 权重常驻（内部未 dump） |

**对比结论**：

1. **速度：OpenVINO INT8 CPU 比 GPU BF16 还快 2.7–3.3×**（P1 3.2× / P2 3.3× / P3 2.7×）。
   GPU 慢的原因是产品路径把全部权重 cast 成 fp32 计算（流量 ×2 且用不到 bf16 吞吐）；
   理论上纯 bf16 compute 会显著更快，但那不是当前 ComfyUI conditioning 路径的行为。
2. **精度：GPU BF16 与 CPU BF16 参考互为镜像验证**（cos=1.000000，仅 GEMM 求和顺序差异）——
   同时印证了"CPU int8 路线实际是 FP32 计算"与"GPU BF16 路线实际也是 FP32 计算"两个取证结论。
3. **资源竞争：GPU 路线把 16.63 GiB 常驻在 GPU 侧 GTT 并占用 85% GPU**，与 Qwen-Image DiT
   直接抢算力和显存；**OV CPU 路线完全不碰 GPU**，把全部 GTT 留给 DiT。
4. UMA 视角：两条路线最终共享同一 LPDDR5X 池（GPU 侧 16.6 GiB + 系统 14.6 GiB 并存时仍余 ~60 GiB），
   但 OV 路线的绝对内存成本更低且不产生 GPU 上下文切换。
5. **推荐不变且更强**：conditioning 放 CPU（OV INT8 bridge）+ DiT 放 GPU，比 conditioning 放 GPU
   更快、更省 GPU 资源——实测推翻了"文本编码器放 GPU 更快"的直觉。

## 5. 工程判断（Q1–Q4，全部基于实测）

**Q1：ComfyUI + qwen3vl_8b_int8_convrot + CPU 是真 INT8 CPU GEMM 吗？**
否。是 **INT8 storage → 每 encode 全量 dequant（252 层，13.9 GB/次）→ FP32 GEMM**（A2）。
证据：`torch._int_mm`=0、dequant=252/encode、字节级对账、profiler、源码五重证据（§3）。

**Q2：OpenVINO INT8 在本机的表现？**
加载（read+compile）1.26 s；常驻权重 7.57 GB（bin）；prefill 峰值 RSS 15.1 GiB；
单次 prefill（官方模板 171 tok）0.57 s / （ComfyUI 模板 bridge 171 tok）0.69 s；
最优线程 = 默认（LATENCY hint 自动 16 线程、NUM_STREAMS=1）；8 线程明显变慢，32 线程无收益。

**Q3：谁 RAM 更低 / 启动更快 / encode 更快？**
- RAM：ComfyUI CPU 路线最低（峰值 7.6 GiB）< OV bridge（15.1 GiB）< GPU BF16（16.63 GiB GPU 侧常驻）。
- 启动（首次可用 encode）：OpenVINO 快约 10×（0.18 vs 12.11 s）；GPU 路线 load 4.48 s。
- encode（语义一致前提）：OV bridge 0.19–0.69 s，比 ComfyUI A2 快 22–69×，比 A1-forced 快 12–39×，
  **比 GPU BF16 产品路径快 2.7–3.3×**（§4.5）。

**Q4：推荐架构（Ryzen AI Max+ 395 CPU → Qwen3-VL conditioning → Radeon GPU → DiT）？**
**推荐 OpenVINO INT8 weight-compressed + `add_outputs` conditioning bridge 跑在 CPU**：
- 唯一同时满足"低延迟 conditioning + CPU 执行 + 语义对齐验证通过"的方案（0.19–0.69 s vs 12.7–15.4 s）；
- **对照实验 C 后结论更强：它甚至比把 BF16 编码器放上 GPU 还快 2.7–3.3×，且不占 GPU（GPU 路线
  常驻 16.63 GiB GTT + busy 85%，会直接挤压 DiT）**；
- bridge 零重算（IR 图内已有 layers.35 残差节点），无需重导出模型；
- 16 GiB 峰值对 94 GiB UMA 无压力，且不与 GPU 侧 DiT 争夺 VRAM 之外的 CPU 资源（编码期间 CPU util ~42%）；
- 若必须留在 ComfyUI 生态：短期可接受 A2 现状（慢 22–68×，但功能正确），
  中期工程项是让 conditioning 路径走 comfy-kitchen 的 eager int8_linear（A1-forced 已证明可行且 1.77× 加速、精度 0.997+）。

**下一步工程项**（bridge 已验证，剩余为产品化）：
1. end-to-end sanity（需下载 33 GB Qwen-Image-2.1 本体 + Radeon GPU 侧环境）——本轮按任务要求未做；
2. ComfyUI 侧若走 A1：需要上游接受"conditioning 路径关闭 force_cast_weights"的改动或提供开关；
3. OpenVINO IR 的 logits 输出对 conditioning 无用，可裁掉再省 ~0.6 GB/次分配（可选微优化）。

## 6. 完整产物索引

```
environment/system.json|system.md   硬件/软件基线
environment/model_inventory.md      ModelScope 下载清单 + SHA256 + 优先级声明
environment/model_sha256.txt        4 个主权重校验和
scripts/resource_monitor.py         20Hz 统一监控器
scripts/bench_comfy_qwen3vl_cpu.py  实验 A（产品路径 + A1 强制变体 + profiler）
scripts/bench_openvino_qwen3vl_cpu.py  实验 B standalone
scripts/bench_openvino_bridge.py    add_outputs conditioning bridge
scripts/bench_bf16_reference.py     BF16 数值参考
scripts/classify_comfy_path.py      A1/A2/A3 分类器
scripts/compare_embeddings.py       精度对比
scripts/aggregate_summary.py        汇总
results/comfy_cpu/                  8 个正式 run JSON/CSV + runtime_path.md + dispatch.log + profiler_a1_t16.txt
results/openvino_cpu/               default/t8/t16/t32/bridge JSON/CSV + openvino_properties.json
results/comparison/                 summary.csv|json + embedding_accuracy.csv|json + cond_*.npy
results/raw/                        20Hz 资源采样 CSV（每个 run 一个）
report/qwen_image_conditioning_semantics.md
report/RESULTS.md                   本报告
```

## 7. 诚实性声明 / caveats

- page cache **未**清空（无免密 sudo）："cold" = 新进程加载，模型文件可能驻留 OS page cache；
  warm encode 数值不受影响。
- ComfyUI `model_load_s=0.30s` 是 safetensors mmap assign；真实磁盘 page-in 计入 first encode（12.11s）。
- OpenVINO standalone 与 ComfyUI 的 token 数差异来自官方 chat 模板 vs ComfyUI T2I_TEMPLATE；
  可比基准一律使用 bridge（§1 主表）。
- OpenVINO 内部是否 oneDNN 真 INT8 GEMM 未做内部 dump 定性（标 UNKNOWN），
  但权重 int8 常驻 + CPU 执行 + 与 A1 的精度差异（cos 0.997 vs 0.998，同数量级）一致。
- 本轮未做 end-to-end 生成质量对比（conditioning 语义一致性已证明，但按任务 §18 属可选且需 33 GB 本体模型）。

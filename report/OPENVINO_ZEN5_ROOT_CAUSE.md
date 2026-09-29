# Why OpenVINO is Fast on Ryzen AI Max+ PRO 395

> **双主机交叉验证**：同一 16-phase 协议已在第二台机器（HOST B = 2× EPYC 9334 / Zen 4，
> 模型逐字节一致、OpenVINO wheel 相同、hidden-state cosine 1.0000001）独立执行并得到
> 一致结论——见 [OPENVINO_ZEN5_ROOT_CAUSE.hostB.md](OPENVINO_ZEN5_ROOT_CAUSE.hostB.md)。
> 本报告为 HOST A（Ryzen AI Max+ PRO 395 / Zen 5，原 benchmark 主机）的记录。


Qwen3-VL-8B INT8 weight-compressed conditioning workload · root-cause investigation ·
2026-09-29 · branch `investigate/openvino-zen5-vnni-root-cause`。

本报告回答任务北极星问题 Q1–Q3：OpenVINO 在本机执行这个 workload 时**到底有没有用
AVX512_VNNI**、贡献多少、0.187 s 的根本原因是什么。每个结论都标注证据等级
（Level A 硬件能力 / B 运行时派发 / C JIT 机器码 / D 执行热点采样 / E 反事实），
全部数字可追到 `results/openvino_isa/` 与 `results/openvino_root_cause/` 的原始文件，
每个 Phase 均经过独立只读 subagent 核验（`environment/root_cause_checkpoints.md`，全 PASS）。

## TL;DR

1. **AVX512_VNNI 确实在执行**（Level B+C+E 证据链；Level D perf 采样因
   `perf_event_paranoid=4` + 内核不匹配的 perf 工具不可用，如实缺失）：
   默认路径下 253 个权重压缩 FullyConnected 节点（占节点时间 91–92%）的 oneDNN JIT
   brgemm 内核中有 **12 个内核、每个 208 条 `vpdpbusd`（zmm, u8×s8→int32）**，MAC 循环
   里带 `vpbroadcastd` 组缩放——这是"激活动态量化(组=32)→int8 点积"的机器码形态。
2. **但 VNNI 不是 22–69× 的主角**。同栈反事实测量：VNNI int8 引擎比 BF16 引擎快
   **1.23–1.43×**；INT8 权重压缩+动态量化路径整体比 FP16 对照快 **1.41–1.67×**；
   AVX-512 向量宽度贡献 **1.38–1.55×**。大头是架构性的：OpenVINO **从不物化反量化权重**
   （每 encode 8.5 GB u8 流量 vs ComfyUI A2 的 34.7 GB = 4.1×）+ 预编译图 + 融合内核。
3. `brgemm_avx512_bf16` 这个 primitive 名字**不代表 MAC 用 bf16**——它只是 ISA 家族标签；
   真正决定计算引擎的是 `DYNAMIC_QUANTIZATION_GROUP_SIZE`（默认 32=int8/VNNI 路径，
   设 0=纯 BF16 路径且 VNNI 指令完全消失）。
4. 顺带发现：`DYNAMIC_QUANTIZATION_GROUP_SIZE=128` 比 32 默认值再快 **11–21%**
   （cosine 0.9959–0.9971，仍与 BF16 参考同阶），是免费的调优空间。

## What was known before this investigation

见 README 与 report/RESULTS.md（2026-09-29 上午完成）：ComfyUI A2 取证、A1 对照、
OpenVINO bridge 语义对齐（cosine ≥0.997）、GPU 实验 C。当时明确标注两个 UNKNOWN：
OpenVINO internal GEMM dtype = UNKNOWN；是否实际执行 AVX512_VNNI = 未证明。
本轮把这两个 UNKNOWN 变成证据链结论（见下），未改动任何既有实验记录。

## Hardware capability

(Level A，仅证明能力，不作为执行证据) AMD Ryzen AI Max+ PRO 395（Zen 5, 16C/32T,
94 GiB UMA）：AVX2 / AVX-512 / AVX512_VNNI / AVX_VNNI / AVX512_BF16 全部在位
（environment/system.md）。无 AMX。

## OpenVINO CPU backend

(Phase 1, PASS) `CPU` 设备由 `libopenvino_intel_cpu_plugin.so` 提供；**oneDNN v3.13.0
静态嵌入**该插件（无动态 libdnnl，0 个导出 dnnl_* 符号）；线程运行时为随轮分发的
**oneTBB 2021.13.1**（无 OpenMP）。`ONEDNN_VERBOSE / MAX_CPU_ISA / JIT_DUMP /
JIT_PROFILE / CPU_ISA_HINTS` 均编译在内。SHA256 与 ldd/readelf 证据：
results/openvino_root_cause/backend_linkage.txt。

## Runtime execution graph

(Phase 2, PASS) 2721 个运行时节点中 253 个为权重压缩 FC（`FullyConnectedCompressed`，
36 层×7 投影 + lm_head）：primitive=`brgemm_avx512_bf16`、**rtPrecision=u8（权重 int8
常驻）**、输出 bf16。其余图基本 bf16（RMS/SDPA/RoPE/reorder）。brgemm 占节点时间
91.35/92.40/92.20%（P1/P2/P3）。有效属性：threads=16、NUM_STREAMS=1、LATENCY、
**INFERENCE_PRECISION_HINT=bfloat16（默认）**、**DYNAMIC_QUANTIZATION_GROUP_SIZE=32（默认）**。
文件：profiling_default_P*.csv、runtime_nodes_default.json、runtime_model_default.xml。

## oneDNN dispatch evidence

(Phase 3, PASS——部分) ONEDNN_VERBOSE 头部（在 exec 模式崩溃前捕获）：
`oneDNN v3.13.0`、`cpu,runtime:threadpool,nthr:16`、
`isa:Intel AVX-512 with Intel DL Boost and bfloat16 support`（进程级能力，非逐算子结论）。
逐 primitive 的 verbose 输出在本 build 上**不可得**：任何含 exec 日志的模式都会在
编译期常量折叠 reorder（u8→i32）上崩溃（与 ISA/DQ 无关，dtype 驱动；已用 5 种模式×
多种配置复现并记录）。按任务规则，这不解读为"没用 oneDNN"——派发拒绝日志
（brgemm_matmul_reorders.cpp:292 ×79）本身就是 oneDNN 在参与的证据。
文件：results/openvino_isa/onednn_verbose_exec_crash.log、phase3_verbose_findings.md。

## Is AVX512_VNNI actually executed?

**是 —— Case A（带一个如实记录的证据缺口）。**

- Level B：253 个 u8 权重 FC 节点（91–92% 节点时间）派发到 brgemm 内核族；
- Level C：这些内核的 JIT 机器码含 `vpdpbusd %zmm…`（12 内核×208 条；dq=0 时为 0 条）；
- Level E：开关 DQ 恰好增删这批 VNNI 内核，延迟 +23–43% 且**输出数值单调变化**
  （组越大越粗）——循环确实在跑；
- Level D：**未获得**（perf 不可用，见下）。无任何 sudo/系统策略改动。

反例警示（已证伪的直觉）：`ONEDNN_MAX_CPU_ISA=AVX512_CORE`（不含 VNNI 的上限）下
JIT 里**仍有 vpdpbusd** 且跑得更快——oneDNN 的权重压缩 GEMM 路径不受该上限剥离 VNNI，
因此"VNNI 上限 vs 非 VNNI 上限"的延迟对比**不能**当作 VNNI 的隔离实验（两档都执行
VNNI 代码，仅代码生成形态不同：304 vs 208 条/内核，32/126 个二进制 SHA 不同）。

## JIT instruction evidence

(Phase 6, 两轮核验 PASS) `ONEDNN_JIT_DUMP=1` 对 6 种配置抓全量 JIT 二进制并 objdump：

| 配置 | VNNI (vpdpbusd) | BF16 dot (vdpbf16ps) | 判读 |
|---|---|---|---|
| DEFAULT (DQ=32) | **12 内核×208 (zmm)** | 6 内核/988 条 | 默认=动态量化 int8 点积 |
| DQ=0 | **0** | 18 内核/2,716 条 | 纯 BF16 引擎 |
| DQ=128 | 12×**832**（4×展开） | 6/988 | 更粗组→更长 VNNI 直线码→更快 |
| AVX512_CORE 上限+f32 | 12×304 | 0 | 上限不剥离 VNNI |
| AVX512_CORE_VNNI 上限+f32 | 12×208 | 0 | 与上行仅代码生成不同 |
| AVX2_VNNI 上限+f32 | 12×96（`{vex}` ymm 256-bit） | 0 | int8 引擎保留，只丢宽度 |

MAC 循环样例（default, jit_brgemm_kernel_t.100.bin @0x557）：
`vmovups (%r10),%zmm4 / vpbroadcastd (%r11),%zmm0 / prefetcht0 … / vpdpbusd %zmm0,%zmm4,%zmm30`。
二进制不提交（gitignore），提交 SHA256 清单+摘要+命中摘录。文件：
results/openvino_isa/jit_dump/<cfg>_analysis/、phase6_jit_findings.md。

## perf hotspot evidence

(Phase 7/8, PASS=受阻如实记录) **未获得**：`/proc/sys/kernel/perf_event_paranoid=4`、
无免密 sudo、且 `perf not found for kernel 6.17.0-1032-oem`（linux-tools 不匹配）。
`perf list` 同因不可用 → Phase 8 无 PMU 事件可核实，按反编造规则写明
"No reliable architectural PMU counter exposure could be verified"。未做任何系统改动。
文件：results/openvino_isa/phase7_perf_findings.md。

## ISA ablation

(Phase 4, PASS) 6 个可运行配置×3 fresh process（runner: scripts/run_openvino_isa_matrix.sh；
每档派发表都变——MAX_CPU_ISA 确认生效）。P2（数据最稳，run 间离散 <2.4%）：

| 对比 | P1 | P2 | P3 | 结论 |
|---|---:|---:|---:|---|
| AVX-512 宽度 vs 256-bit（f32, 同 int8 引擎） | 1.43× | 1.38× | 1.55× | 唯一显著的 ISA 因子 |
| BF16 vs F32 计算（同 ISA 同 u8 流量） | 1.04× | **1.00×** | 1.03× | 计算精度几乎不影响 → 非算力瓶颈 |
| VNNI 上限 vs 非 VNNI 上限 | 0.92× | 1.00× | 0.93× | 非有效隔离（两档都执行 VNNI） |

可运行性边界（本身是派发证据）：纯 AVX2 **跑不起来**（primitive descriptor 创建失败，
isa_AVX2_NOT_RUNNABLE.log）；bf16 hint 在无 BF16 单元的上限下同样失败；
`INFERENCE_PRECISION_HINT=i8` 被插件拒绝（只收 bf16/f16/f32）。文件：isa_matrix.csv/json、
phase4_isa_matrix.md。

## Dynamic quantization ablation

(Phase 5, PASS) 派发表在 g=0/32/64/128 下逐字节不变，但：

| DQ group | P1 | P2 | P3 | cosine(P1/P2/P3) |
|---|---:|---:|---:|---|
| 0（禁用） | 0.2274 | 0.5371 | 1.0101 | 0.99764/0.99838/0.99879（最好） |
| 32（默认） | 0.1854 | 0.3770 | 0.7064 | 0.99724/0.99806/0.99845 |
| 64 | 0.1672 | 0.3414 | 0.6037 | 0.99687/0.99780/0.99821 |
| 128 | 0.1590 | 0.3359 | 0.5559 | 0.99592/0.99690/0.99707 |

教科书式速度/精度权衡曲线 = 激活按组动态量化的直接证据（与 Phase 6 机器码互证）。
文件：dynamic_quant_matrix.csv/json、phase5_dynamic_quant.md、hidden_cosine_summary.json。

## Weight-compression effect

(Phase 9+10, PASS) 每 encode 权重字节：OV u8 = **8,474,066,944 B**；ComfyUI A2 =
6.95 GB int8 读 + 13.89 GB 反量化写 + 13.89 GB 读回 = **34.7 GB（4.1×）**。
受控 FP16 对照（ModelScope `OpenVINO/Qwen3-VL-8B-Instruct-fp16-ov`，与 int8-ov 的
config.json 逐字节相同，同栈同图跑 3 进程）：

| | P1 | P2 | P3 |
|---|---:|---:|---:|
| OV FP16（bf16 引擎, f16 存储） | 0.2609 | 0.6289 | 1.0241 |
| OV INT8-WC（dq+VNNI） | 0.1854 | 0.3770 | 0.7064 |
| **INT8-WC 加速** | **1.41×** | **1.67×** | **1.45×** |

隐含利用率：3.6–4.7 TFLOP/s ≈ 12.7 TOPS 理论包络的 28–37% → **非算力饱和**；
隐含权重流 46.3/22.8/12.6 GB/s（一次触达假设下的推算值，非实测带宽）。
文件：phase9_memory_traffic.{md,json}、phase10_fp16_control.md。

## Threading / runtime contribution

线程：16（默认）最优（既有 sweep：t8 慢 ~1.4×，t32 无增益——本轮引用未重测，
STRONG EVIDENCE）。oneTBB threadpool nthr=16（verbose 头部）。NUM_STREAMS=1/LATENCY。

## Root-cause decomposition

（全部因子表见 results/openvino_root_cause/phase11_12_factor_attribution.md，含
Evidence/Counterfactual/Confidence 列）

OpenVINO 0.185 s（P1）的构成 = **INT8 权重常驻（u8，不物化反量化副本）** →
**copy-B/内核内反量化 + 激活动态量化（组 32）** → **vpdpbusd int8 点积（VNNI）** →
**AVX-512 宽度** → **预编译图+16 线程调度**。各因子量级（重叠、非独立乘子）：
压缩+引擎 1.41–1.67×（vs FP16 对照）；VNNI 引擎增量 1.23–1.43×（vs BF16 引擎）；
AVX-512 宽度 1.38–1.55×；流量 4.1×（vs A2）。

## What explains the 22–69× vs ComfyUI CPU result

分两段（P1: 12.98 s → 0.185 s）：
1. **A2→A1（1.77×）**：停掉每 encode 对 252 层、13.9 GB 的全量反量化（既有取证）。
2. **A1→OV（~40×）**：同一份 u8 权重上，torch eager 逐算子 vs 推理引擎流水线——
   预编译图、权重 repack 缓存、融合动态量化、brgemm/VNNI、16 线程调度。
   其中可测的引擎因子如上；无单一"因为 VNNI"的解释成立。

## What explains the 2.7–3.3× vs current ComfyUI Radeon path

GPU 路线（实验 C 取证）把全部 252 个线性层以 fp32×fp32 跑在 GPU（产品路径强制 fp32），
即 BF16 存储→cast→FP32 GEMM：流量×2 且用不上 bf16/int8 吞吐，还常驻 16.6 GiB GTT、
GPU busy 85%。OV CPU 路线 0% GPU、u8 流量。0.60/1.23/1.83 s vs 0.19/0.38/0.69 s
= **2.7–3.3×**。此结论只针对该产品路径，不可泛化为"CPU 普遍快于 GPU"。

## What we can claim

- 默认 OpenVINO 路径执行 AVX512_VNNI（vpdpbusd）——Level B+C+E 完整，且数值行为随
  DQ 组大小单调变化证明循环执行；
- 该 workload 的 MAC 引擎由 DYNAMIC_QUANTIZATION_GROUP_SIZE 决定（32=VNNI int8，
  0=BF16 dot），primitive 名不带语义；
- 权重全程 u8、从不物化 13.9 GB 反量化副本；
- VNNI 增量 1.23–1.43×，压缩+引擎 1.41–1.67×，AVX-512 宽度 1.38–1.55×（均有
  反事实与原始数据）；
- 22–69× 的主因是架构性（免反量化 + 推理引擎流水线），不是 VNNI 单因子。

## What we cannot claim

- 不能声称 perf 采样级（Level D）执行证据——环境不可用，未获得；
- 不能声称实测 DRAM 带宽 / cache counters（perf stat 同因不可用；GB/s 全为推算值）；
- 不能把 ISA 上限对比当作 VNNI 隔离实验（两档都执行 VNNI）；
- 不能说 ComfyUI GPU 路线慢代表所有 GPU 路线（bf16 tensor-core 路线未测，超范围）；
- 不能把本机（Zen 5, 16C）结论外推到其他 CPU；
- oneDNN 逐 primitive verbose 明细在本 build 上不可得（崩溃已记录）。

## Reproduce

```bash
# Phase 0 baseline（未改动的既有 bridge）
.venv-openvino/bin/python scripts/bench_openvino_bridge.py --tag rc_p0_baseline --warm-iters 10

# Phase 2 profiling / Phase 5 DQ / Phase 10 FP16（同一脚本不同参数）
.venv-openvino/bin/python scripts/investigate_openvino_runtime.py --tag default --warm-iters 2 --measure-iters 10 --save-xml
.venv-openvino/bin/python scripts/investigate_openvino_runtime.py --tag dq_g0_r1 --dyn-quant-group-size 0 --warm-iters 2 --measure-iters 12 --save-hidden
.venv-openvino/bin/python scripts/investigate_openvino_runtime.py --tag fp16_r1 --model-dir models/qwen3vl-openvino-fp16 --warm-iters 2 --measure-iters 12 --save-hidden

# Phase 4 ISA 矩阵（6 配置 × 3 fresh process，串行）
bash scripts/run_openvino_isa_matrix.sh && .venv-openvino/bin/python scripts/aggregate_root_cause.py

# Phase 6 JIT dump（每配置一个目录；见 phase6_jit_findings.md 的逐配置参数）
cd results/openvino_isa/jit_dump/default && JITDUMPDIR=. ONEDNN_JIT_DUMP=1 \
  ../../../../.venv-openvino/bin/python ../../../../scripts/investigate_openvino_runtime.py \
  --tag jitdump_default --prompts P1 --warm-iters 1 --measure-iters 2
.venv-openvino/bin/python scripts/analyze_jit_vnni.py results/openvino_isa/jit_dump/default --out /tmp/out

# Phase 3 verbose（会按文档崩溃——保留崩溃日志即证据）
ONEDNN_VERBOSE=all .venv-openvino/bin/python scripts/investigate_openvino_runtime.py --tag vt --prompts P1 --measure-iters 1

# 精度
.venv-openvino/bin/python scripts/check_hidden_cosine.py
```

模型：`modelscope download --model OpenVINO/Qwen3-VL-8B-Instruct-fp16-ov --local_dir models/qwen3vl-openvino-fp16`（int8-ov 见 README）。

## Raw evidence index

```
results/openvino_root_cause/
  phase0_baseline.md + openvino_rc_p0_baseline.*      基线复现
  backend_linkage.txt                                   Phase 1 后端链接+SHA256
  phase2_findings.md profiling_default_* runtime_nodes_default.json runtime_model_default.xml
  phase5_dynamic_quant.md dynamic_quant_matrix.* profiling_dq_* hidden_*_dq_* hidden_cosine_summary.json
  phase9_memory_traffic.{md,json}                       流量/利用率分析
  phase10_fp16_control.md profiling_fp16_* runtime_nodes_fp16_*   FP16 对照
  phase11_12_factor_attribution.md                      因子表+VNNI 量化+Q1 判定
results/openvino_isa/
  phase3_verbose_findings.md onednn_verbose_exec_crash.log onednn_default_summary.json
  phase4_isa_matrix.md isa_matrix.{csv,json} profiling_isa_* runtime_nodes_isa_* runlogs/
  phase6_jit_findings.md jit_dump/<cfg>/（本地 .bin）与 jit_dump/<cfg>_analysis/
  phase7_perf_findings.md                               perf/PMU 受阻记录
environment/root_cause_checkpoints.md                   13 次 subagent 核验全记录
scripts/investigate_openvino_runtime.py run_openvino_isa_matrix.sh aggregate_root_cause.py
        analyze_onednn_verbose.py analyze_jit_vnni.py check_hidden_cosine.py
```

## Social-media-safe conclusions

### Confirmed facts（可直接写）

- Zen 5 提供 AVX-512 / AVX512_VNNI / AVX512_BF16；本次 Qwen3-VL OpenVINO CPU
  workload 中，我们从 runtime 派发、JIT 反汇编（vpdpbusd MAC 循环）和开关反事实
  三层证据确认了 **VNNI int8 点积路径实际在跑**（perf 采样级证据因系统权限不可用）。
- OpenVINO 把 INT8 权重压缩模型跑成"u8 常驻 + 激活动态量化(组32) + VNNI int8 点积"，
  关掉动态量化会退回纯 BF16 点积并慢 23–43%。
- 同栈对照：INT8-WC 比 FP16 快 1.41–1.67×；AVX-512 比 256-bit 快 1.38–1.55×。
- 22–69×（vs ComfyUI CPU 产品路径）的主因是免掉每轮全量反量化 + 推理引擎流水线；
  2.7–3.3×（vs ComfyUI Radeon 产品路径）是因为那条路径在 GPU 上也强制 fp32 计算。

### Strong evidence（需限定措辞）

- "VNNI 在此 workload 中实际执行"——建议写明证据层级：runtime dispatch + JIT 机器码 +
  反事实；perf 采样未获得（系统限制），不做指令级计数的宣称。
- "DYNAMIC_QUANTIZATION_GROUP_SIZE=128 还能再快 11–21%（精度 cos≥0.9959）"——
  本机本模型实测，别的模型需重测。
- 权重流量 4.1× 之差为字节对账推算；DRAM 带宽未实测。

### Do not claim（不能说）

- ❌ "OpenVINO 快 69× 是因为 VNNI"（单因子归因不成立）。
- ❌ "是 W8A8 量化模型"（是 INT8_ASYM weight-only + 运行时动态量化激活，激活端非静态量化）。
- ❌ "perf 证明 VNNI 指令执行了 X 次"（无 PMU 数据）。
- ❌ "CPU 比 GPU 快"（只对比了 ComfyUI 那条强制 fp32 的 GPU 产品路径）。
- ❌ "brgemm_avx512_bf16 说明用 BF16 计算"或反过来"名字无关紧要"——须按本报告的
  DQ 开关语义表述。

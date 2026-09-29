你现在负责继续完善 GitHub 仓库：

`AIwork4me/qwen3vl-cpu-benchmark`

这不是一次单点 benchmark。

这是一轮完整的 **End-to-End System Validation**。

---

# 北极星目标

通过真实、可复现、可审计的实验回答：

> **在 AMD Ryzen AI Max+ PRO 395 + Radeon 8060S 这台 APU 上，把 Qwen-Image 2.1 的 Qwen3-VL Text Encoder 从 GPU 搬到 Ryzen Zen 5 CPU，用 OpenVINO INT8 执行，并让 Radeon GPU 专门负责 DiT，是否能够让整条生成流水线更快、更省 GPU 资源、更适合本地 AI？**

最终目标不是证明：

> CPU > GPU

而是验证：

> **CPU 和 GPU 各自跑最适合自己的 workload，是否比“什么都塞 GPU”更合理。**

最终所有结论必须来自：

- 真机
- 真模型
- 真 prompt
- 真图片生成
- 真 latency
- 真 CPU/GPU utilization
- 真 memory/GTT 数据
- 真 output image
- 真 numerical evidence
- 真 raw logs

禁止使用：

- 理论估算替代实测
- 模拟 benchmark
- mock data
- synthetic result
- “应该”
- “预计”
- 根据之前结果直接推断 end-to-end 结论

---

# 当前已知事实

先完整阅读仓库，不允许凭记忆。

当前仓库已实验证明：

## HOST A

AMD Ryzen AI Max+ PRO 395

- Zen 5
- 16C / 32T
- Radeon 8060S
- 94 GiB UMA
- AVX-512
- AVX512_VNNI
- AVX512_BF16

## 已知 Text Encoder 路线

### Route A — ComfyUI CPU product path

`qwen3vl_8b_int8_convrot`

实际上：

```text
INT8 storage
→ 每 encode 全量 dequant
→ FP compute
```

P1：

```text
~12.98 s
```

### Route B — ComfyUI Radeon GPU product path

BF16 weights

实际：

```text
BF16 storage
→ cast
→ FP32 GEMM
```

P1/P2/P3：

```text
0.601 / 1.228 / 1.834 s
```

GPU busy：

```text
~85%
```

GPU allocation：

```text
~16.63 GiB
```

### Route C — OpenVINO CPU conditioning bridge

INT8 weight-compressed

默认：

```text
DYNAMIC_QUANTIZATION_GROUP_SIZE=32
```

实际已经通过：

- runtime dispatch
- JIT disassembly
- counterfactual

确认执行：

```text
AVX512_VNNI
vpdpbusd
```

P1/P2/P3：

```text
0.187 / 0.377 / 0.687 s
```

且：

```text
GPU 不参与 inference
```

---

# 这次真正的问题

之前只证明：

```text
Text Encoder alone
```

这次必须证明：

# 整个 Qwen-Image 2.1 pipeline

即：

```text
Prompt
↓
Qwen3-VL Text Encoder
↓
conditioning
↓
Qwen-Image 2.1 DiT
↓
VAE decode
↓
final image
```

---

# 两个核心架构

必须至少比较以下两种真实架构。

---

## Architecture G — GPU-heavy baseline

```text
Qwen3-VL Text Encoder
        ↓
Radeon 8060S GPU

Qwen-Image 2.1 DiT
        ↓
Radeon 8060S GPU

VAE
        ↓
Radeon 8060S GPU
```

即：

# Text Encoder + DiT 都尽可能放 GPU

这是 baseline。

必须使用当前 ComfyUI 产品路径。

不要自己构造一个更差的 GPU baseline。

如果 ComfyUI 默认会 offload 某些模块：

如实记录。

---

## Architecture H — Hybrid

```text
Ryzen Zen 5 CPU
        ↓
OpenVINO Qwen3-VL INT8
        ↓
conditioning

Radeon 8060S GPU
        ↓
Qwen-Image 2.1 DiT
        ↓
VAE
        ↓
final image
```

即：

# CPU 做 Text Encoder
# GPU 专心做 DiT / VAE

这才是实验核心。

---

# 最终必须回答的 10 个问题

## Q1

Hybrid 是否降低：

```text
time-to-first-image
```

---

## Q2

连续生成多张图片时：

```text
steady-state images/min
```

是否提高？

---

## Q3

GPU Text Encoder 路线是否因为：

```text
16+ GiB GTT resident
+
GPU compute contention
```

影响：

```text
DiT
```

的 latency？

---

## Q4

CPU Text Encoder 是否真正释放：

- GPU memory
- GPU busy time
- GPU scheduling pressure

---

## Q5

OpenVINO CPU encoder 是否与 DiT GPU 阶段存在：

```text
CPU/GPU overlap
```

潜力？

例如下一条 prompt conditioning 是否可以在上一张图的 DiT 期间提前算？

---

## Q6

Hybrid 的最终图片质量是否可接受？

不能只比较 conditioning cosine。

必须实际生成图片。

---

## Q7

DQ group：

```text
32
64
128
```

哪个是最佳：

```text
latency / quality
```

trade-off？

---

## Q8

Hybrid 是否在较长 prompt 下优势更明显？

---

## Q9

Hybrid 是否更适合：

```text
94 GiB UMA
```

这种 APU？

---

## Q10

最终能否有证据支持：

> 把 Text Encoder 从 GPU 搬回 Ryzen CPU，让 GPU 专门跑 DiT，是这台机器上更合理的 Qwen-Image 2.1 架构。

---

# 总原则

---

# Rule 1 — 每个结论必须有 Raw Evidence

最终 README 中出现的每一个数字：

必须能追溯到：

```text
JSON
CSV
log
PNG
metadata
```

例如：

```text
Hybrid first-image = 8.73 s
```

必须可以找到：

```text
results/e2e/hybrid/run03.json
```

里面有：

```json
"total_latency_s": 8.73
```

禁止手工填表。

所有 summary 必须脚本自动生成。

---

# Rule 2 — 不允许只跑一次

任何 headline latency：

至少：

```text
3 fresh processes
```

每个 process：

```text
1 cold generation
+
至少 5 warm generations
```

如果单张 Qwen-Image 太慢：

允许：

```text
3 fresh process
×
3 warm images
```

但必须在报告里解释。

---

# Rule 3 — 固定随机性

所有图片质量实验必须固定：

```text
prompt
negative prompt
seed
resolution
steps
CFG
sampler
scheduler
model
VAE
```

仅改变：

```text
Text Encoder execution path
```

或：

```text
DQ group
```

禁止同时修改多个变量。

---

# Rule 4 — 每 Phase 后必须 Subagent 审核

每完成一个 Phase：

启动：

```text
independent read-only subagent
```

它不得修改任何文件。

必须回答：

```text
PASS
```

或者：

```text
FAIL
reason
```

核验：

- experiment design
- raw data
- scripts
- calculations
- statistical treatment
- overclaim
- reproducibility
- hardware attribution

FAIL 必须修复后重新审。

记录：

```text
environment/e2e_checkpoints.md
```

---

# Rule 5 — 不破坏已有实验

开始前：

```bash
git status --short
git branch --show-current
git log -8 --oneline
```

禁止：

```text
reset --hard
force push
```

创建新 branch：

```text
experiment/qwen-image-e2e-hybrid
```

已有 root-cause 数据全部只读。

---

# Phase 0 — Repo Audit

完整阅读：

```text
README.md
report/RESULTS.md
report/OPENVINO_ZEN5_ROOT_CAUSE.md
report/qwen_image_conditioning_semantics.md
environment/system.md
environment/model_inventory.md
results/comparison/summary.csv
results/comparison/embedding_accuracy.csv
scripts/bench_openvino_bridge.py
scripts/bench_comfy_bf16_gpu.py
```

然后总结：

```text
KNOWN
UNKNOWN
TO VERIFY
```

写：

```text
report/E2E_EXPERIMENT_PLAN.md
```

Subagent 必须审阅实验设计。

---

# Phase 1 — 获取完整 Qwen-Image 2.1

优先：

# ModelScope

不要默认 Hugging Face。

下载完整：

```text
Qwen-Image 2.1
```

包括至少：

- DiT / transformer
- VAE
- text encoder artifacts
- config
- tokenizer
- scheduler config

记录：

```text
model source
revision
file size
SHA256
download date
```

更新：

```text
environment/model_inventory.md
environment/model_sha256.txt
```

禁止 commit 大模型。

---

# Phase 2 — 建立完全可运行的 baseline

必须先跑：

# 原生 ComfyUI GPU-heavy pipeline

不要先跑 Hybrid。

确保：

```text
prompt → text encoder → DiT → VAE → PNG
```

真实成功。

保存：

```text
results/e2e/gpu_baseline/
```

必须保存：

```text
workflow JSON
ComfyUI commit
torch version
ROCm version
GPU arch
model hashes
seed
sampler
steps
resolution
CFG
```

最终生成至少：

```text
P1
P2
P3
```

对应当前仓库统一 prompts。

---

# Phase 3 — 实现 Hybrid bridge

目标：

OpenVINO 输出必须真实进入：

```text
Qwen-Image 2.1 DiT
```

不能：

```text
OpenVINO benchmark
+
ComfyUI separately generate
```

必须真正接起来。

允许两种实现：

## Option A

直接修改 benchmark harness：

```text
OpenVINO encoder
→ numpy / torch tensor
→ Qwen-Image conditioning
→ ComfyUI / DiT
```

## Option B

实现 ComfyUI custom node。

优先：

# 最少侵入现有代码

第一阶段只求实验可验证。

不要先做漂亮 UI。

---

# Phase 4 — Conditioning Semantic Identity

再次确认：

Hybrid 输出：

```text
shape
token alignment
hidden layer
RMSNorm position
system token trim
dtype
```

必须与原 ComfyUI conditioning 一致。

记录：

```text
shape equality
cosine
RMSE
relative L2
```

禁止只看 cosine。

---

# Phase 5 — End-to-End Timing Instrumentation

必须把整条 pipeline 分阶段计时。

至少：

```text
T0 model init
T1 tokenizer
T2 text encoder
T3 conditioning transfer/preparation
T4 DiT first step
T5 DiT final step
T6 VAE start
T7 image ready
T8 PNG saved
```

最终每张图片记录：

```text
model_load_s
tokenize_s
text_encoder_s
conditioning_bridge_s
dit_s
vae_s
postprocess_s
png_save_s
total_generation_s
```

注意：

# headline total latency

必须使用：

```text
image ready
```

不要把 PNG encoding 混进去。

可以另外报告：

```text
save-inclusive latency
```

---

# Phase 6 — GPU Synchronization Correctness

所有 GPU timing 必须：

```python
torch.cuda.synchronize()
```

在计时边界前后正确使用。

禁止仅用：

```text
wall clock around async kernel launch
```

Subagent 必须核查 timing code。

---

# Phase 7 — Resource Monitoring

整个生成过程以：

```text
10–20 Hz
```

采样。

CPU：

```text
process CPU %
system CPU %
RSS
threads
frequency
```

GPU：

至少：

```text
GPU busy %
memory allocated
reserved
GTT usage
temperature
power
```

如果 AMD sysfs 能获取：

记录。

如果获取不到：

明确 unavailable。

保存：

```text
results/e2e/raw/
```

每次 run：

```text
resource timestamp CSV
```

---

# Phase 8 — Baseline A/B

现在做核心对照：

## G — GPU-heavy

```text
ComfyUI text encoder → Radeon
DiT → Radeon
```

## H — Hybrid

```text
OpenVINO text encoder → Ryzen CPU
DiT → Radeon
```

必须：

```text
same prompt
same seed
same steps
same resolution
same sampler
same scheduler
same CFG
same DiT
same VAE
```

---

# Phase 9 — Prompt Matrix

不要只测 3 条。

现有：

```text
P1 short
P2 medium
P3 long
```

保留。

再增加至少：

```text
P4 very short
P5 very long
```

建议最终覆盖：

```text
~10–20
~40
~80
~180
~300+
```

token length。

不要只看字符数。

保存真实 tokenizer input length。

---

# Phase 10 — Resolution Matrix

至少：

```text
1024×1024
```

作为主实验。

如果显存允许，再测：

```text
1328×1328
```

或者 Qwen-Image 官方推荐高分辨率。

禁止因为某路线 OOM 就偷偷降低 resolution。

OOM 本身就是结果。

---

# Phase 11 — Step Matrix

至少：

```text
20 steps
40 steps
```

如果官方默认不同：

使用：

```text
official recommended
+
one shorter configuration
```

目的是观察：

Text Encoder 越快之后，

随着 DiT workload 增大，

end-to-end gain 是否被摊薄。

---

# Phase 12 — Cold Start

必须单独比较：

# Cold pipeline

fresh process：

```text
start
→ model load
→ first image
```

测：

```text
time to first image
```

不要混入：

```text
OS page cache
```

无法 drop cache：

明确写：

```text
fresh-process cold
not disk-cold
```

---

# Phase 13 — Warm Single Image

模型全部 loaded。

测：

```text
single prompt
→ final image
```

至少：

```text
5 iterations
```

report：

```text
median
mean
std
min
max
CV
```

---

# Phase 14 — Continuous Batch Workflow

这是验证“流水线更合理”的关键。

测试：

```text
Prompt 1
Prompt 2
Prompt 3
Prompt 4
Prompt 5
```

连续生成。

每次 prompt 不同。

测：

```text
total wall time
images/min
average latency
GPU idle gaps
CPU utilization
```

---

# Phase 15 — Pipeline Overlap Experiment

非常重要。

验证：

当 GPU 正在生成第 N 张图片时：

Ryzen CPU 是否可以提前计算：

```text
N+1 prompt conditioning
```

建立：

# Sequential

```text
Encode P1
DiT P1

Encode P2
DiT P2
```

vs

# Pipelined

```text
CPU Encode P2
      ↘
GPU DiT P1

CPU Encode P3
      ↘
GPU DiT P2
```

如果可行：

测：

```text
5 images
10 images
```

总 wall time。

这很可能是 Hybrid 架构最大的产品价值。

---

# Phase 16 — CPU/GPU contention test

为了证明：

# CPU encoder 不会拖慢 GPU DiT

必须测：

## DiT alone

GPU 跑 DiT 时：

CPU idle。

## DiT + CPU encoder concurrent

GPU 跑 DiT：

同时 CPU 算下一 prompt。

比较：

```text
DiT latency
GPU busy
GPU clocks
CPU util
memory bandwidth side effects
```

由于是 UMA：

CPU/GPU 会共享 LPDDR5X。

所以这个实验非常重要。

如果 CPU encoder 和 GPU DiT 同时运行反而导致：

```text
memory contention
```

必须如实报告。

---

# Phase 17 — DQ Group Matrix

Hybrid 至少测试：

```text
DQ=0
DQ=32
DQ=64
DQ=128
```

主要关注：

```text
32
64
128
```

每个配置必须实际生成图片。

不能再只看 conditioning cosine。

---

# Phase 18 — Image Quality Dataset

至少：

# 30 prompts

最好：

# 50 prompts

覆盖：

```text
portrait
landscape
text rendering
complex composition
multiple objects
fine detail
photorealistic
illustration
long prompt
short prompt
Chinese
English
```

每个 prompt：

固定：

```text
seed
resolution
steps
CFG
```

生成：

```text
GPU baseline
DQ32
DQ64
DQ128
```

---

# Phase 19 — Image Quality Evaluation

至少做：

## A. Pixel / latent similarity

如果 deterministic path 允许：

```text
SSIM
PSNR
LPIPS
```

如果生成路径会由于 conditioning 差异产生较大结构变化：

说明这些指标局限。

---

## B. CLIP / SigLIP prompt-image score

使用同一个 evaluator。

所有图片统一。

---

## C. Pairwise image similarity

例如：

```text
CLIP image embedding cosine
```

---

## D. Blind human evaluation package

自动生成：

```text
comparison/contact sheet
```

隐藏：

```text
GPU / DQ32 / DQ64 / DQ128
```

改成：

```text
A/B/C/D
```

用户之后可以人工盲评。

不要由 Codex 编造主观分数。

---

# Phase 20 — Final Image Metadata

每张 PNG 必须写 sidecar：

```text
prompt
seed
route
DQ group
resolution
steps
CFG
sampler
scheduler
model SHA
encoder latency
DiT latency
total latency
```

例如：

```text
P03_seed42_hybrid_dq32.png
P03_seed42_hybrid_dq32.json
```

---

# Phase 21 — Quality Acceptance

不能只说：

```text
looks the same
```

定义门槛。

建议：

Hybrid DQ32：

必须：

```text
conditioning cosine >= 0.997
```

并且在：

```text
30–50 image test
```

中：

- 无系统性崩坏
- 无明显 prompt adherence 下降
- 无明显 text rendering regression
- quality metrics 无大规模异常

DQ64/128：

作为：

```text
performance/quality trade-off
```

单独报告。

不要预设 128 是最佳。

---

# Phase 22 — GPU Memory / GTT

必须证明：

GPU Text Encoder 会占掉多少 GPU-side allocation。

记录：

## GPU-heavy

```text
after encoder load
after DiT load
peak during generation
```

## Hybrid

```text
after OpenVINO CPU load
after DiT load
peak
```

比较：

```text
GPU memory/GTT saved
```

---

# Phase 23 — UMA Total Memory

因为是 Strix Halo：

CPU RAM 和 GPU GTT 共享物理内存。

所以不能简单说：

> Hybrid 少用了 16GB RAM

必须分别报告：

```text
CPU process RSS
GPU allocation
system MemAvailable
total UMA delta
```

区分：

# GPU address-space saving

和：

# physical memory saving

---

# Phase 24 — GPU contention analysis

必须回答：

GPU Text Encoder 是否造成：

```text
DiT cold-start delay
kernel scheduling contention
memory eviction
GTT pressure
```

如果 Text Encoder 只在 DiT 前运行：

不能误写成：

```text
simultaneous contention
```

要区分：

```text
temporal contention
memory residency contention
compute overlap contention
```

---

# Phase 25 — Model Residency

分别记录：

## GPU-heavy

Text Encoder 是否：

```text
stays resident
gets offloaded
gets evicted
```

## Hybrid

OpenVINO model：

```text
CPU RSS
resident weight
compile cache
```

确认后续图片是否需要 reload。

---

# Phase 26 — OpenVINO Compile Cache

测试：

```text
CACHE_DIR
```

如果支持：

测：

## First compile

vs

## cached compile

这会影响：

```text
time-to-first-image
```

报告：

```text
cold compile
warm compile cache
```

---

# Phase 27 — Model loading fair comparison

必须避免：

```text
ComfyUI mmap 0.3 s
```

和：

```text
OpenVINO read+compile
```

直接不公平比较。

最终用两个指标：

# API load call time

和：

# Time until first usable image

后者更重要。

---

# Phase 28 — DiT Isolation

单独记录：

# GPU-heavy route DiT latency

和：

# Hybrid route DiT latency

如果 Hybrid：

```text
DiT 本身更快
```

必须找原因。

例如：

```text
more free GTT
different allocation
cache effect
less pressure
```

不能直接归功 OpenVINO。

---

# Phase 29 — VAE Isolation

确认 VAE：

```text
same device
same dtype
same implementation
```

两路线必须一致。

---

# Phase 30 — Numerical Conditioning Inspection

每个 route：

保存部分 conditioning tensor。

比较：

```text
mean
std
cosine
RMSE
relative L2
max abs
```

避免因 bridge bug 导致图片结果不可比。

---

# Phase 31 — Repeated Seeds

至少选：

```text
5 representative prompts
```

每个：

```text
5 seeds
```

防止：

某一个 seed 恰好让 DQ128 看起来没有影响。

---

# Phase 32 — Statistical Analysis

最终 headline speedup 必须使用：

```text
median-of-fresh-process-medians
```

或者其它预先定义的 robust statistic。

不要：

```text
挑最快的一次
```

同时报告：

```text
95% CI
```

如果样本足够。

可以 bootstrap。

---

# Phase 33 — Primary Metrics

必须自动生成最终主表。

至少：

| Metric | GPU-heavy | Hybrid DQ32 | Hybrid DQ64 | Hybrid DQ128 |
|---|---:|---:|---:|---:|
| Time to first image | | | | |
| Warm image latency | | | | |
| Text Encoder latency | | | | |
| DiT latency | | | | |
| VAE latency | | | | |
| 5-image throughput | | | | |
| GPU busy | | | | |
| GPU peak alloc | | | | |
| CPU peak RSS | | | | |
| System MemAvailable delta | | | | |
| Conditioning cosine | | | | |
| Image quality metric | | | | |

---

# Phase 34 — “Is it more reasonable?” scorecard

不要做主观打分。

用事实维度：

```text
Latency
GPU occupancy
GPU memory residency
CPU utilization
quality
complexity
startup time
throughput
```

每项只写：

```text
measured fact
```

例如：

```text
Hybrid frees X GiB GPU allocation
Hybrid reduces warm E2E latency by Y%
Hybrid changes image metric by Z
```

不要写：

```text
Hybrid = 9/10
```

---

# Phase 35 — Failure cases

主动寻找：

```text
Hybrid worse than GPU
```

的情况。

例如：

- extremely short prompt
- long DiT-heavy workload
- high resolution
- simultaneous CPU/GPU memory contention
- low CPU frequency
- CPU thermal throttling

这是报告可信度非常关键的一部分。

---

# Phase 36 — Thermal stability

连续运行：

```text
10–20 images
```

观察：

```text
CPU frequency
GPU frequency
temperature
latency drift
```

防止：

第一张很快，

后来 thermal throttling。

---

# Phase 37 — Power

如果系统暴露：

```text
RAPL
amd_energy
sysfs power
```

可以记录：

```text
package power
GPU power
```

如果没有：

写 unavailable。

禁止估算功耗。

---

# Phase 38 — End-to-End correctness

最终每条路线至少实际生成：

```text
>= 30 images
```

整个实验总图片数最好：

```text
100+
```

不能只跑：

```text
3 benchmark prompts
```

然后声称“完整 Qwen-Image pipeline”。

---

# Phase 39 — 推荐配置

最终根据真实数据决定：

```text
Recommended
```

可能是：

```text
Hybrid DQ32
```

也可能：

```text
Hybrid DQ64
```

甚至：

```text
DQ128
```

但不能提前决定。

推荐必须同时考虑：

```text
end-to-end latency
image quality
memory
stability
```

---

# Phase 40 — 产品化 Experiment

只有主实验结论 PASS 后：

才允许开始做 ComfyUI custom node。

目标：

```text
Qwen3VL OpenVINO CPU Encoder
```

参数：

```text
model_dir
CPU threads
DQ group
OpenVINO cache dir
```

输出必须无缝兼容：

```text
Qwen-Image 2.1 conditioning
```

---

# Phase 41 — Custom Node benchmark

装上 custom node 后：

重新跑：

```text
GPU-heavy native
vs
Hybrid custom node
```

确保：

benchmark harness 的优势不是来自：

```text
绕开 ComfyUI overhead
```

---

# Phase 42 — One-click reproducibility

添加：

```text
scripts/run_e2e_gpu_baseline.sh
scripts/run_e2e_hybrid.sh
scripts/run_e2e_matrix.sh
scripts/run_quality_matrix.sh
scripts/aggregate_e2e.py
scripts/make_blind_comparison.py
```

最好做到：

```bash
bash scripts/run_e2e_matrix.sh
```

可以自动跑完整主 benchmark。

---

# Phase 43 — Results structure

建议：

```text
results/e2e/
  gpu_baseline/
  hybrid_dq32/
  hybrid_dq64/
  hybrid_dq128/
  pipelined/
  quality/
  raw/
  images/
```

---

# Phase 44 — Final Report

新增：

```text
report/QWEN_IMAGE_E2E_HYBRID.md
```

结构：

```markdown
# Qwen-Image 2.1 on Ryzen AI Max+ PRO 395:
# Should the Text Encoder Run on CPU?

## TL;DR

## Test platform

## Models and exact hashes

## Two system architectures

## Why this experiment exists

## Methodology

## Conditioning correctness

## Time-to-first-image

## Warm end-to-end latency

## Stage-by-stage latency

## GPU utilization

## GPU memory / GTT

## UMA memory

## Continuous-generation throughput

## CPU/GPU overlap

## DQ group 32/64/128

## Final image quality

## Failure cases

## Thermal stability

## What actually improved

## What did not improve

## Recommended architecture

## What we can claim

## What we cannot claim

## Reproduce

## Raw evidence index
```

---

# Phase 45 — README update

README 首页不要堆所有实验。

加一个新章节：

# End-to-End Qwen-Image 2.1

回答一句：

> Does moving Qwen3-VL conditioning to the Ryzen CPU improve the complete generation pipeline?

然后只放：

```text
3–5 个关键数字
```

和链接：

```text
report/QWEN_IMAGE_E2E_HYBRID.md
```

---

# Phase 46 — Social-media-safe conclusions

报告必须生成：

```text
## Social-media-safe conclusions
```

分：

## Confirmed

可以直接传播。

## Context-required

必须带限制条件。

## Do not claim

禁止使用。

---

# 允许出现的最终 headline

只有数据支持时：

> **我把 Qwen-Image 的 Text Encoder 从 GPU 搬回 Ryzen CPU，整条 AI 流水线反而更合理了。**

但必须有下列证据中的至少 3 项同时成立：

```text
Hybrid E2E latency lower
GPU memory/GTT lower
GPU busy time reduced
continuous throughput higher
quality remains acceptable
```

---

# 禁止的 headline

除非真实实验直接支持，否则不能写：

```text
CPU 比 GPU 快
```

不能写：

```text
OpenVINO 让整个 Qwen-Image 快 69×
```

69× 只属于：

```text
Text Encoder CPU route comparison
```

不能扩大到完整 image generation。

---

# 最终最重要的指标

最终必须输出一个极简主结论表：

```text
GPU-heavy
vs
Hybrid CPU+GPU
```

只保留：

```text
Time to first image
Warm image latency
5-image throughput
Text Encoder latency
DiT latency
GPU peak memory
GPU busy
CPU RSS
Quality
```

---

# 最终 Subagent Audit

最后启动一个完全独立的只读审计 agent。

角色：

```text
OpenVINO performance engineer
+
ComfyUI maintainer
+
ROCm performance engineer
+
ML benchmark reviewer
```

要求逐条检查：

1. GPU baseline 是否公平
2. 两路线是不是同一个模型
3. 同一个 DiT
4. 同一个 VAE
5. 同一个 seed
6. 同一个 sampler
7. 同一个 resolution
8. OpenVINO conditioning 是否真正进入 DiT
9. timing 是否正确 synchronize
10. warm/cold 是否区分
11. 是否 cherry-pick 最快数据
12. GPU utilization 是否窗口正确
13. memory 是否把 RSS/GTT/UMA 混为一谈
14. DQ quality 是否真的生成图片验证
15. 是否泛化成 CPU > GPU
16. 是否把 encoder speedup 当 E2E speedup
17. 所有 headline 数字是否能自动从 raw data 重算
18. 所有图片是否能追溯到 seed/config
19. 是否存在未披露失败 case
20. README 是否和 raw data 完全一致

只有：

```text
FINAL E2E AUDIT: PASS
```

才能进入 Git 提交。

---

# GitHub 提交

最终：

```bash
git status
git diff --check
git diff --stat
```

禁止提交：

```text
models/
huge caches
temporary JIT binaries
perf data
secrets
```

生成 commit：

```text
bench: validate hybrid CPU-encoder Radeon-DiT Qwen-Image pipeline
```

push：

```text
experiment/qwen-image-e2e-hybrid
```

创建 PR。

PR title：

```text
Validate OpenVINO CPU conditioning + Radeon GPU DiT for Qwen-Image 2.1
```

不要自动 merge。

---

# PR body

必须包含：

## Hypothesis

## Architecture A

## Architecture B

## Exact hardware/software

## E2E latency

## Stage-level latency

## GPU memory

## CPU/GPU utilization

## Pipeline-overlap result

## DQ group result

## Image-quality result

## Failure cases

## Reproduction

## Raw evidence

## Limitations

---

# 最终回复给我

任务全部完成后，只输出以下格式：

```text
E2E VERDICT

1. Does moving Qwen3-VL Text Encoder to Ryzen CPU improve full Qwen-Image 2.1?
YES / MIXED / NO

2. GPU-heavy architecture
Time to first image:
Warm E2E:
5-image throughput:
GPU peak allocation:
GPU busy:
CPU RSS:

3. Hybrid architecture
Time to first image:
Warm E2E:
5-image throughput:
GPU peak allocation:
GPU busy:
CPU RSS:

4. End-to-end speedup
Cold:
Warm:
Continuous:

5. Did DiT itself become faster?
YES / NO
Measured delta:
Reason:

6. Did Hybrid free GPU resources?
GPU memory saved:
GPU busy reduction:

7. Best DQ group
32 / 64 / 128
Why:
Latency:
Quality:

8. Image quality
Number of prompts:
Number of seeds:
Conditioning cosine:
CLIP/SigLIP:
LPIPS/other:
Human-blind package:

9. CPU/GPU overlap
Sequential:
Pipelined:
Speedup:

10. Failure cases
...

11. Safest public conclusion
...

12. What must NOT be claimed
...

13. Files added
...

14. Commit
...

15. Branch / PR
...
```

---

# 最终原则

不要为了证明：

> “CPU 搬回去更好”

而选择实验。

要验证它。

如果真实结果是：

```text
Text Encoder 更快
但 end-to-end 只快 2%
```

照实写。

如果：

```text
Hybrid 降低 GTT
但 CPU/GPU UMA contention 让 DiT 变慢
```

照实写。

如果：

```text
DQ128 更快
但图片质量下降明显
```

照实写。

如果：

```text
Hybrid 在连续生成时因为 CPU/GPU overlap 反而明显提高 throughput
```

把这个真正的系统价值找出来。

这次实验的目标不是赢 benchmark。

而是回答一个真正有价值的问题：

> **在 Ryzen AI Max+ PRO 395 这种 CPU + Radeon GPU + UMA 的本地 AI 机器上，Qwen-Image 2.1 的最佳系统架构到底应该是什么？**
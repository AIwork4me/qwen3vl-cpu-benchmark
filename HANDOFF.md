# 交接文档 — Qwen-Image 2.1 端到端 A/B 验证（未完成部分）

> 写于 2026-09-29 晚，因用户指示停止。前序工作（root-cause investigation）**已完成并合入 main**（PR #1，merge commit `3ee1388`）；本文档只覆盖**进行中的端到端 A/B 计划**的剩余部分。
> 交接原则沿用用户要求：**做一步 → 独立 subagent 核验 → PASS 后再下一步**。

## 0. 一句话状态

Step 0（PR 合并）✅ 完成；Step 1（DiT 下载）🟡 后台进行中（73%，约 3 分钟自行完成，无需干预）；Step 2（三个 e2e 脚本）🟡 已写好、**语法通过但未冒烟测试、未核验**（未提交，位于工作区）；Step 3–6 ⬜ 未开始。

## 1. 已完成且合入 main 的成果（无需重做）

- PR #1 `investigate/openvino-zen5-vnni-root-cause` 已 merge 到 main（`3ee1388`），包含：
  双主机（HOST A = 本机 Ryzen AI Max+ PRO 395 / Zen 5；HOST B = EPYC 9334 / Zen 4，合并保留）
  root-cause 全部证据链。核心结论：**AVX512_VNNI 确实执行**（vpdpbusd，Level B+C+E；
  perf Level D 因 `perf_event_paranoid=4` 不可得）；VNNI 增量仅 1.23–1.47×；
  22–69× 主因是免反量化架构；**DQ=128 免费再快 11–21%**（cos ≥0.9959）——
  这是端到端 A/B 要验收的产品发现。
- 报告：`report/OPENVINO_ZEN5_ROOT_CAUSE.md`（HOST A）/ `*.hostB.md`（HOST B）；
  核验记录 `environment/root_cause_checkpoints.md`；README 已更新（UNKNOWN→YES，保留历史；
  CPU util 1332% vs 1552–1572% 双窗口解释）。
- 本地 main 已同步至 `3ee1388`，工作区干净（除下述 3 个未提交脚本）。

## 2. 端到端 A/B 计划总表（含当前状态）

| Step | 内容 | 状态 | 备注 |
|---|---|---|---|
| 0 | PR #1 预合并审查 + merge + main 验证 | ✅ | subagent 审查 PASS 后 merge |
| 1 | 下载 DiT+VAE 并校验 | 🟡 73% | 后台 pid 46659 自行完成；完成后需校验（见 §4.1） |
| 2 | e2e 脚本编写 + 冒烟测试 | 🟡 脚本已写未测 | 3 个脚本在工作区未提交（见 §3） |
| 3 | 完整 A/B：3 路线 × 3 seed | ⬜ | 命令见 §4.3 |
| 4 | 质量分析（RMSE/SSIM/确定性）+ DQ=128 结论 | ⬜ | 方案见 §5 |
| 5 | e2e 报告 + README + 新分支 PR | ⬜ | 分支名建议 `e2e/qwen-image21-ab` |
| 6 | ComfyUI 上游提案草案（仓内存档不外发） | ⬜ | 素材：A1 取证 + root-cause §1/§6 |

## 3. 已写好但未测试的脚本（工作区，未提交）

- `scripts/prep_e2e_cond.py`（跑在 `.venv-openvino`）：按 ComfyUI `TextEncodeQwenImage21`
  **t2i 语义**逐位复刻（同 T2I 模板含 system turn、同 qwen25_tokenizer、同第二
  `<|im_start|>` 裁剪、空 negative 用 `" "` 对应 prevent_empty_text），输出
  `results/e2e/cond/cond_P3_{pos|neg}_dq{32,128}.npy` + 计时 json。
- `scripts/bench_e2e_qwen_image21.py`（跑在 `.venv-comfy-rocm`，HIP GPU）：加载
  DiT bf16 + VAE；路线 `native`（int8-convrot TE 在 CPU，产品路径 A2）/ `ov_dq32` /
  `ov_dq128`（注入 `[[tensor, {}]]`，与 ComfyUI 自身 cond 结构一致——已从源码核实
  `encode_from_tokens_scheduled` 无 hooks 时正是此结构）；采样参数
  euler/simple/20 步/cfg 2.5/1024×1024/latent [1,64,64,64]（已含
  `fix_empty_latent_channels`）；每路线 × seed {20260929,42,7} 计时 + 存 PNG + GPU busy。
- `scripts/run_e2e_ab.sh`：两阶段编排（stage1 OV conds → stage2a 冒烟 4 步 → stage2b 全量）。

**尚未做**：任何实际运行（冒烟也没跑）、Step 2 的 subagent 核验、git 提交。

## 4. 恢复操作手册（按序执行，每步完成后独立 subagent 核验）

### 4.1 Step 1 收尾（下载完成后）
```bash
tail -2 logs/dl_qwen_image_dit.log            # 应显示 Snapshot ready
ls -la models/qwen-image-2.1-full/diffusion_models/ models/qwen-image-2.1-full/vae/
# 校验：文件应 ≥ 下载日志声明大小；DiT bf16 = 14.23 GB(decimal)，VAE = 0.676 GB
# 建议对两个 safetensors 计算 SHA256 并记录（若 ModelScope API 可给哈希则比对）
```
核验点：文件大小/哈希与 ModelScope `Comfy-Org/Qwen-Image-2.1` 仓库声明一致；磁盘余量足够
（1.9T 盘当时剩 ~207G）。

### 4.2 Step 2 冒烟
```bash
bash scripts/run_e2e_ab.sh                    # 内含 stage1(conds) + stage2a(冒烟) + stage2b(全量)
# 或分步：先 stage1，再单独冒烟
.venv-comfy-rocm/bin/python scripts/bench_e2e_qwen_image21.py --tag smoke --steps 4 --seeds 7 --routes native,ov_dq32 --enc-warm 0 --enc-measure 1
```
核验点（subagent，只读）：三路线 cond shape 断言通过（native 与 OV 同形）；
`results/e2e/cond/` 的 npy 形状 = [1, kept_tokens, 4096]；冒烟 PNG 正常产出；
native encode 计时量级 ≈ 12–15 s（与已合入 main 的 A2 数据一致）。
**已知风险**（冒烟可能暴露）：
- `comfy.sample.sample` 在无 pbar 环境的 callback 兼容性（如报错，加 `disable_pbar=True`）；
- DiT 14.2 GB 装载 GTT（当时 free ≈ 60+ GiB，应够；不够则 `mm.unload_all_models()` 已在每轮前调用）；
- `clip.tokenize(prompt, prevent_empty_text=True)` 关键字在 0.37.0 的接受度（TextEncodeQwenImage21
  源码就是这样调的，应当可行）。

### 4.3 Step 3 全量 A/B
```bash
.venv-comfy-rocm/bin/python scripts/bench_e2e_qwen_image21.py --tag run1   # 默认 3 路线 × 3 seed, 20 步
```
产出：`results/e2e/e2e_run1.json` + 9 张 PNG（`images/run1_<route>_s<seed>.png`）。
核验点：9 张图齐全；每路线采样耗时相近（采样与 cond 无关，仅差噪声）；native 编码 ~13 s vs
OV DQ32 ~0.7 s / DQ128 ~0.56 s（P3 长度，fresh-process 计时来自 cond_timing json）；
PNG sha256 记录在 json。

### 4.4 Step 4 质量分析
写一个小脚本（建议 `scripts/analyze_e2e_images.py`，纯 numpy/PIL，任意 venv）：
- 确定性：同路线同 seed 重复采样的位一致（可在 run2 复跑 1 个 route×seed 验证）；
- 路线间：以 `native` 为参考，对每个 seed 计算 ov_dq32/ov_dq128 图像的 RMSE / PSNR /
  8×8 分块 SSIM（无 skimage 就手写或用 PIL+numpy 近似）；
- 判读标准（事先约定，避免事后找理由）：若 DQ128 vs DQ32 的差异 ≤ DQ32 vs native 差异
  的同数量级、且肉眼不可分（RMSE 阈值建议先看 DQ32-native 分布再定），则支持
  “DQ=128 可作产品默认”；另注意 PROMPT P3 有可数物体（两颗蓝 LED 眼/左右手物品/
  三块显示器）可人工抽查。

### 4.5 Step 5 归档
新分支 `e2e/qwen-image21-ab`（从最新 main 切出），提交：3 个脚本 + 分析脚本 +
`results/e2e/`（json/log/小图；**PNG 每张几百 KB 可提交**，cond npy ~2.8MB×4 可提交）+
`report/QWEN_IMAGE21_E2E_AB.md`（结构仿 OPENVINO_ZEN5_ROOT_CAUSE.md：结论先行、
分阶段计时表、质量矩阵、DQ=128 建议、限制与复现命令）+ README 增补一节。
推送 + PR（**不要自动 merge**；push 若遇 GitHub 网络慢，重试 2–3 次或设
`git config http.postBuffer 524288000`）。

### 4.6 Step 6 上游提案草案
仓内 `report/COMFYUI_UPSTREAM_PROPOSAL.md`：请求 conditioning 路径提供关闭
`force_cast_weights` 的开关（或复用 generate() 的 use_quantized_matmul）。
论据：A2→A1 实测 1.77×（torch._int_mm 252/encode、dequant 0、cos 0.9978）+
root-cause 对 OV 更快原因的解释。**只存档，不外发**（外发需用户明示授权）。

## 5. 关键设计决策记录（已从源码核实，续作不必重查）

1. **cond 注入点**：`encode_from_tokens_scheduled`（无 hooks）返回 `[[tensor, dict]]`；
   QwenImage21 t2i 时 dict 为空（attention_mask 全 1 被丢弃、无 image_slots）→ OV 注入
   `[[tensor.float(), {}]]` 与产品结构逐位同构（`comfy/text_encoders/qwen_image21.py`）。
2. **模板/裁剪**：t2i = T2I_TEMPLATE（system turn）+ 从第二个 `<|im_start|>`(151644) 裁剪；
   keep_vision=True（t2i 无参考图）。OV bridge 已按此实现且 cosine ≥0.997。
3. **空 negative**：`prevent_empty_text` 把 `""` 变 `" "`（`qwen3vl.py:168`）——prep 脚本已用 `" "`。
4. **latent**：`[1, 64, H/16, W/16]`（64 通道分层 latent），过 `fix_empty_latent_channels`。
5. **采样参数**：euler/simple/20/cfg2.5/1024²。绝对画质不重要（A/B 只比 conditioning 路线），
   三路线严格同参同 seed，噪声由 `prepare_noise(latent, seed)` 决定。
6. **计时口径**：OV 编码计时来自独立 fresh OV 进程（`cond_timing_dq*.json`，与已合入
   main 的 DQ 矩阵同协议）；native 编码在本进程内实测（TE load_device=offload=cpu）。
   汇报时明确两者口径。

## 6. 资产与路径索引

```
models/qwen-image-2.1-full/diffusion_models/qwen_image_2.1_bf16.safetensors   14.23GB(下载中)
models/qwen-image-2.1-full/vae/qwen_image_2.1_vae_bf16.safetensors            0.68GB
models/qwen3vl-openvino-int8/   OV INT8-WC 模型（已有）；models/qwen3vl-comfy-int8/  原生 TE（已有）
models/qwen3vl-openvino-fp16/   FP16 对照（已有，gitignored）
logs/dl_qwen_image_dit.log      下载日志
results/e2e/                    本计划全部输出（cond/ images/ *.json *.log）
scripts/prep_e2e_cond.py + bench_e2e_qwen_image21.py + run_e2e_ab.sh   未提交
venv: .venv-openvino(OV) / .venv-comfy-rocm(HIP GPU, DiT) / .venv-comfy(CPU torch)
GPU: Radeon 8060S (gfx1151)；GTT 当时 free≈60GiB；/sys/class/drm/card1/device/gpu_busy_percent
```

## 7. 其他注意事项

- GitHub push 偶发 "Operation too slow"：重试即可（本轮成功 2/3 次）。
- 严格串行跑基准（避免并发干扰计时）；每步 subagent 核验是硬性要求（用户指令）。
- 已合入 main 的历史数据（results/comfy_*、openvino_*）**禁止改动**（沿用调查期 Rule 3）。
- 若冒烟阶段 OV cond 与 native cond 形状不一致：核对 tokens 数（native
  `clip.tokenize` 的 llama_template 默认 t2i 与 prep 脚本的 T2I_TEMPLATE 必须逐字符一致——
  两者都取自 `comfy/text_encoders/qwen_image21.py` 的 T2I_TEMPLATE，理论上一致；
  不一致时打印两侧 token ids 的 diff 定位）。

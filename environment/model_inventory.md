# Model Inventory

下载日期: 2026-09-29 (所有模型来自 ModelScope，网络为受限环境: huggingface.co 不通)

下载优先级声明:
```
Download priority:
1. ModelScope            (used: all three artifacts found on ModelScope)
2. Existing local cache  (not needed)
3. Upstream fallback     (not needed)

Reason: exact artifacts exist on ModelScope mirrors; HF unreachable on this network.
```

## A. ComfyUI Qwen3-VL INT8 ConvRot (text encoder)

- ModelScope repo ID: `Comfy-Org/Qwen-Image-2.1`
- File: `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
- Local: `models/qwen3vl-comfy-int8/text_encoders/qwen3vl_8b_int8_convrot.safetensors`
- Declared size (ModelScope API): 9.35079836 GB (decimal)
- Expected official size: ~9.35 GB ✓
- SHA256: 见 `environment/model_sha256.txt`
- 量化格式（来自 ComfyUI 源码 `comfy/ops.py` 语义）: `int8_tensorwise` + `convrot` 参数
  （per-layer `.comfy_quant` JSON 元数据, per-tensor `weight_scale`, convrot_groupsize 默认 256）
- 下载命令: `modelscope download --model Comfy-Org/Qwen-Image-2.1 text_encoders/qwen3vl_8b_int8_convrot.safetensors --local_dir models/qwen3vl-comfy-int8`

## B. OpenVINO INT8 Qwen3-VL-8B-Instruct

- ModelScope repo ID: `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov`
- Local: `models/qwen3vl-openvino-int8/`（完整仓库, 共 8.8 GB decimal）
- 主要文件: `openvino_language_model.xml/.bin` (7.574 GB), `openvino_text_embeddings_model.xml/.bin` (0.623 GB),
  `openvino_tokenizer`, `openvino_vision_embeddings_*`, tokenizer/config JSONs
- 量化方案（来自仓库 `openvino_config.json`）: `dtype=int8`, `quant_method=default`
  → **OpenVINO INT8_ASYM weight compression**（NNCF weight compression, group_size=-1, 未量化 activation）
  → 报告用语: "OpenVINO INT8 weight-compressed"，**不是** W8A8
- `openvino_language_model`: stateful LLM 导出（ReadValue/Assign ×72 = 36 层 KV cache），
  唯一输出 `logits`；图中含 `__module.model.language_model.norm`（最终 RMSNorm）与独立 `__module.lm_head`
- `openvino_text_embeddings_model`: 仅 embed_tokens 查找（Gather+Multiply），非完整 backbone
- SHA256: 主要权重见 `environment/model_sha256.txt`

## C. BF16 参考模型（精度验证用）

- ModelScope repo ID: `Comfy-Org/Qwen-Image-2.1`
- File: `text_encoders/qwen3vl_8b_bf16.safetensors`
- Local: `models/qwen-image-2.1/text_encoders/qwen3vl_8b_bf16.safetensors`
- Declared size: 17.534334616 GB (decimal)
- 用途: 仅作为数值 reference（cosine/RMSE 对比），不参与主 benchmark 表
- SHA256: 见 `environment/model_sha256.txt`

## D. Qwen-Image 2.1 DiT + VAE（端到端实验，2026-09-29 晚补全）

- ModelScope repo ID: `Comfy-Org/Qwen-Image-2.1`
- Local:
  - `models/qwen-image-2.1-full/diffusion_models/qwen_image_2.1_bf16.safetensors`
    （14,230,280,616 B = 14.23 GB decimal，与 ModelScope API 声明逐字节一致）
  - `models/qwen-image-2.1-full/vae/qwen_image_2.1_vae_bf16.safetensors`
    （675,509,688 B = 0.676 GB decimal，与 API 一致）
- SHA256: `environment/model_sha256.txt`（ModelScope API 不提供文件哈希，故记录本地
  实测 SHA256 + 大小对账；下载命令见 logs/dl_qwen_image_dit.log）
- 下载命令: `modelscope download --model Comfy-Org/Qwen-Image-2.1 diffusion_models/qwen_image_2.1_bf16.safetensors vae/qwen_image_2.1_vae_bf16.safetensors --local_dir models/qwen-image-2.1-full`
- 同仓库还有 `qwen3vl_8b_w4a8`、`qwen3.5_9b` prompt-enhancer TE 与 DiT int8_convrot 变体，本轮不使用。

## E. OpenVINO FP16 对照模型（root-cause 轮，gitignored）

- ModelScope `OpenVINO/Qwen3-VL-8B-Instruct-fp16-ov` → `models/qwen3vl-openvino-fp16/`（17 GB，
  与 int8-ov 的 config.json 逐字节相同的受控对照；见 report/OPENVINO_ZEN5_ROOT_CAUSE.md Phase 10）。

## F. CLIP 质量评估器（e2e 轮，gitignored）

- ModelScope `AI-ModelScope/clip-vit-large-patch14` → `models/clip-vit-large-patch14/`（1.71 GB，
  30 提示质量数据集的统一 prompt-image 评分器，77-token 分块打分；见 scripts/evaluate_quality.py）。

## 校验与复用规则

- benchmark 一律从上述本地路径加载，禁止运行时联网。
- `test -d` / 文件大小 / SHA256 三重校验已执行（D 项为大小+SHA256，API 无哈希可比）。
- ~~未下载完整 Qwen-Image-2.1~~ 已于 2026-09-29 补全（DiT+VAE，见 D 项）。

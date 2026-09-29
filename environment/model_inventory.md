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

## 校验与复用规则

- benchmark 一律从上述本地路径加载，禁止运行时联网。
- `test -d` / 文件大小 / SHA256 三重校验已执行。
- 未下载完整 `Qwen/Qwen-Image-2.1`（33 GB），因端到端 sanity test 仅在 text-encoder
  阶段全部 PASS 后才考虑，且非必需。

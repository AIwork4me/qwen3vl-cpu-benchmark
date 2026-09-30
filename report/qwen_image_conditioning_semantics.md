# Qwen-Image 2.1 Conditioning 语义分析（基于本机 ComfyUI checkout 源码）

来源: `ComfyUI/comfy/text_encoders/qwen_image21.py`、`qwen3vl.py`、`sd1_clip.py`、`comfy/sd.py`（ComfyUI master 0.37.0 tarball，2026-09-29）。

## 1. Tokenization（QwenImage21Tokenizer）

```
T2I_TEMPLATE = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
             + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
```

- 固定 **system turn**："Comprehend and analyze the provided prompt."
- 用户文本填入 user turn，模板以 `<|im_start|>assistant\n` 结尾
- `thinking=True`（QwenImage21Tokenizer 覆盖默认值）→ **不追加** `<think>\n\n</think>` 空 think 块
- 纯文本时无 vision block；带参考图时在 user 文本前拼接 `<|vision_start|><|image_pad|><|vision_end|>`
- Tokenizer: Qwen2 BPE（`comfy/text_encoders/qwen25_tokenizer/`），pad token 151643，无 start/end token 自动添加
- 实测 token 数（本机，含模板）: P1=39, P2=81, P3=185

## 2. Hidden state 选取（QwenImage21Qwen3VLClipModel）

```python
class Qwen3VLClipModel(sd1_clip.SDClipModel):
    def __init__(..., layer="hidden", layer_idx=-1, ..., layer_norm_hidden_state=False, ...)
```

- `layer_idx=-1` → **最后一个 decoder layer（第 36 层）的 residual stream 输出**
- `layer_norm_hidden_state=False` → **跳过最终 RMSNorm（model.norm）**
- 源码注释原文: "last layer without the final RMSNorm: transformers 4.57 hidden_states[-1], which Qwen's results are tuned to"
- 即等价于 HF `transformers`（4.57 语义）`output_hidden_states=True` 的 `hidden_states[-1]`

## 3. Post-processing（QwenImage21TEModel.encode_token_weights）

1. **裁掉 system turn**: 保留从第二个 `<|im_start|>`（token 151644）开始的序列 → P1: 39 → 25 tokens
2. **裁掉 vision token**（无图时无影响），记录 image_slots
3. `attention_mask` 若全 1 则删除（全因果无 padding）
4. 输出: `cond` shape `[1, seq_kept, 4096]`，本机实测 **dtype=torch.float32**（sd1_clip.py:279 强制 `dtype=torch.float32` 传入 transformer）
5. pooled 输出对 Qwen-Image 主流程不使用（DiT 只吃 cond 序列）

## 4. 与 OpenVINO 官方 IR 的语义差异

`OpenVINO/Qwen3-VL-8B-Instruct-int8-ov` 的 `openvino_language_model`:
- 面向 **autoregressive generation** 导出：ReadValue/Assign ×72（36 层 KV cache 状态输入/输出）
- **唯一输出 = `logits`**（Result_44730, output_names="logits"）—— 经过独立 `lm_head`（`__module.lm_head/aten::linear/MatMul`）
- 官方 IR **没有**直接暴露 decoder hidden state
- 但图中存在 `__module.model.language_model.norm/...`（最终 RMSNorm 节点）→ 可用 `core.read_model()` + `model.add_outputs()` 零重算暴露最后一层 hidden state（norm 前后节点均可选）

结论: 不能直接比较 "ComfyUI conditioning latency" 与 "OpenVINO 完整生成/或 logits prefill latency"。
可比的 bridge = 用 `add_outputs()` 暴露 layer-36 residual stream（对应 ComfyUI 的 `layer_idx=-1` 且可选 norm 前/后），
输入用与 ComfyUI 完全相同的模板 token 序列。

## 5. Bridge 所需的精确对齐清单

| 维度 | ComfyUI | OpenVINO bridge 需要做 |
|---|---|---|
| 模板 | T2I_TEMPLATE（含 system turn）+ thinking=True（无 think 块） | 用 chat_template.jinja 渲染官方模板 ≠ ComfyUI 模板 → 必须按 ComfyUI 的 T2I_TEMPLATE 手工拼 token |
| token 序列 | 含 system turn 编码后裁掉 → 等价于只保留 user+assistant 头 | 需先编码完整模板再裁剪，或直接验证"裁剪后 token 等价于从 user 开始编码" |
| 位置 | 文本 only → 标准 1D position ids（mrope 退化为 text 模式） | position_ids = arange（需验证 IR 的 position_ids 输入约定） |
| hidden state | layer 36 输出、**无 final RMSNorm**、fp32 | add_outputs 选 norm **之前**的节点 |
| 输出裁剪 | 去掉 system 部分 | bridge 内复刻 encode_token_weights 的裁剪 |

## 6. 实测语义锚点（用于验证 bridge 正确性）

- ComfyUI P1 cond: shape [1, 25, 4096], dtype fp32, device cpu
- token 计数（含模板）: P1=39, P2=81, P3=185；裁剪后: P1=25
- 若 OpenVINO bridge 输出与 ComfyUI/BF16 参考的 cosine similarity ≈ 0.98+（同为 fp 精度），语义对齐成立
- 【e2e 轮补充 2026-09-30】负向提示 `" "`（prevent_empty_text，qwen3vl.py:168）裁剪后
  9 tokens；产品路径 `encode_from_tokens_scheduled` 无 hooks 时返回
  `[[cond, {"pooled_output": pooled}]]`，bridge 注入 `[[cond, {}]]` 对该 DiT 功能等价
  （只消费 tensor + attention_mask/reference_latents/image_slots 键）——实测锚点见
  results/e2e/cond/identity.csv 与 report/QWEN_IMAGE_E2E_HYBRID.md「Conditioning correctness」。

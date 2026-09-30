#!/usr/bin/env python3
"""In-process OpenVINO Qwen3-VL conditioning encoder (the Hybrid bridge).

Runs inside .venv-comfy-rocm next to the ROCm torch DiT (openvino wheel is
torch-independent). Tokenization uses ComfyUI's OWN QwenImage21Tokenizer
(comfy.text_encoders.qwen_image21) — the exact product tokenizer object — so the
token-id sequence fed to the OV IR is by construction the same one ComfyUI's
native route uses (output shape asserted per call; token-level equality verified
in results/e2e/cond/identity.csv).

Graph patch (identical to the validated bench_openvino_bridge.py):
  - add_outputs(<node feeding final RMSNorm's pow>) -> layer-36 residual stream
    BEFORE final RMSNorm == ComfyUI layer_idx=-1, layer_norm_hidden_state=False
  - trim from the second <|im_start|> (151644) == encode_token_weights t2i trim

encode(text) -> dict(hidden [1,seq,4096] fp32, ids, n_tokens_full,
                     n_tokens_kept, encode_s)
"""
import json
import os
import re
import time


def find_pre_norm_node(xml_path):
    """Return the IR node name that produces the input of the final RMSNorm."""
    xml = open(xml_path).read()
    id_by_name = {}
    for m in re.finditer(r'<layer id="(\d+)" name="([^"]+)"', xml):
        id_by_name[m.group(2)] = m.group(1)
    name_by_id = {v: k for k, v in id_by_name.items()}
    norm_pow = None
    for name in id_by_name:
        if name.startswith("__module.model.language_model.norm/") and name.endswith("aten::pow/Power"):
            norm_pow = name
            break
    pow_id = id_by_name[norm_pow]
    m = re.search(r'<edge[^>]*to-layer="%s"[^>]*/>' % pow_id, xml)
    if m is None:
        m = re.search(r"<edge[^>]*to-layer=\"%s\"[^>]*>" % pow_id, xml)
    frm = re.search(r'from-layer="(\d+)"', m.group(0)).group(1)
    return name_by_id.get(frm), norm_pow


class OVQwenEncoder:
    def __init__(self, model_dir, dq=32, cache_dir=None, log=print):
        import numpy as np
        import openvino as ov
        self.np = np
        self.ov = ov
        self.dq = dq
        self.model_dir = model_dir
        self.log = log

        core = ov.Core()
        if cache_dir:
            core.set_property({"CACHE_DIR": cache_dir})
        lm_xml = os.path.join(model_dir, "openvino_language_model.xml")
        lm_bin = os.path.join(model_dir, "openvino_language_model.bin")
        pre_node, norm_node = find_pre_norm_node(lm_xml)
        assert pre_node, "pre-norm node not found in IR"
        t0 = time.perf_counter_ns()
        model = core.read_model(lm_xml, lm_bin)
        picked = {}
        for op in model.get_ordered_ops():
            fn = op.get_friendly_name()
            if fn in (pre_node, norm_node) and fn not in picked:
                picked[fn] = op
        model.add_outputs([picked[pre_node].output(0)])
        self.compile_read_s = (time.perf_counter_ns() - t0) / 1e9

        t0 = time.perf_counter_ns()
        cfg = {"PERFORMANCE_HINT": "LATENCY",
               "DYNAMIC_QUANTIZATION_GROUP_SIZE": dq}
        self.compiled = core.compile_model(model, "CPU", cfg)
        self.compile_s = (time.perf_counter_ns() - t0) / 1e9
        assert [str(d) for d in self.compiled.get_property("EXECUTION_DEVICES")] == ["CPU"], \
            "OV bridge must execute on CPU only"
        self.effective_dq = str(self.compiled.get_property("DYNAMIC_QUANTIZATION_GROUP_SIZE"))
        self.req = self.compiled.create_infer_request()
        emb = core.compile_model(
            core.read_model(os.path.join(model_dir, "openvino_text_embeddings_model.xml"),
                            os.path.join(model_dir, "openvino_text_embeddings_model.bin")),
            "CPU")
        self.emb_req = emb.create_infer_request()
        self.in_names = [list(i.names)[0] for i in model.inputs]

        # ComfyUI's own tokenizer (product object) — no manual template copy
        from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer
        self.tokenizer = QwenImage21Tokenizer()

    def token_ids(self, text):
        toks = self.tokenizer.tokenize_with_weights(
            text, llama_template=None, prevent_empty_text=True, thinking=True, keep_vision=True)
        pairs = toks["qwen3vl_8b"][0]
        return [int(t[0]) for t in pairs]

    def encode(self, text, synchronize_gpu=None):
        """Returns dict with trimmed hidden state (numpy fp32) + timings.

        synchronize_gpu: optional callable (e.g. torch.cuda.synchronize) invoked
        just before t0 so the timing is not polluted by unrelated pending GPU work.
        """
        np = self.np
        ids = self.token_ids(text)
        input_ids = np.array([ids], dtype=np.int64)
        n_tok = input_ids.shape[1]
        embeds = np.asarray(self.emb_req.infer({0: input_ids})[0])
        feed = {"inputs_embeds": embeds, "attention_mask": np.ones_like(input_ids)}
        ar = np.arange(n_tok, dtype=np.int64)
        feed["position_ids"] = np.stack([ar, ar, ar]).reshape(3, 1, n_tok)
        feed["visual_pos_masks"] = np.zeros((1, n_tok), dtype=bool)
        feed["deepstack_visual_embeds"] = np.zeros((1, 0, 4096), dtype=np.float32)
        feed["beam_idx"] = np.array([0], dtype=np.int32)
        idx_feed = {i: feed[name] for i, name in enumerate(self.in_names)}
        self.req.reset_state()
        if synchronize_gpu is not None:
            synchronize_gpu()
        t0 = time.perf_counter_ns()
        res = self.req.infer(idx_feed)
        lat = (time.perf_counter_ns() - t0) / 1e9
        out = res[list(res.keys())[-1]]
        hidden = np.asarray(out, dtype=np.float32)
        assert hidden.shape == (1, n_tok, 4096), f"unexpected OV output shape {hidden.shape} for {n_tok} tokens"
        im = [i for i, t in enumerate(ids) if t == 151644]
        trim = im[1] if len(im) > 1 else 0
        trimmed = hidden[:, trim:, :]
        return {"hidden": trimmed, "ids": ids, "n_tokens_full": n_tok,
                "n_tokens_kept": int(trimmed.shape[1]), "encode_s": lat}

    def meta(self):
        return {"openvino": self.ov.__version__, "dq_requested": self.dq,
                "dq_effective": self.effective_dq,
                "compile_read_s": round(self.compile_read_s, 3),
                "compile_s": round(self.compile_s, 3),
                "model_dir": self.model_dir}

"""Qwen3-VL OpenVINO CPU conditioning encoder for Qwen-Image 2.1 (ComfyUI node).

Hybrid architecture product node: text conditioning computed by OpenVINO INT8 on
the Ryzen CPU (AVX512_VNNI), leaving the Radeon GPU exclusively for the DiT+VAE.
Outputs conditioning structurally identical to TextEncodeQwenImage21's t2i path
([[tensor, {}]] — the DiT consumes only the tensor + attention_mask/reference
keys), plus an empty Qwen-Image latent (64ch) at the requested resolution.

Semantics replicated from ComfyUI's own tokenizer/model (see repo
report/qwen_image_conditioning_semantics.md): T2I template, layer-36 pre-norm
hidden state, trim from second <|im_start|>, empty text -> " ".
"""
import os

import torch

import comfy.model_management as mm
import comfy.sample
import node_helpers


_ENCODER = None
_ENCODER_KEY = None


class Qwen3VLOpenVINOCPUEncoder:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # always re-encode (cheap, stateless node)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": " "}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "dq_group": ([32, 64, 128, 0], {"default": 32,
                            "tooltip": "OpenVINO DYNAMIC_QUANTIZATION_GROUP_SIZE (0 disables dynamic quantization)"}),
                "model_dir": ("STRING", {"default": "models/qwen3vl-openvino-int8"}),
                "cache_dir": ("STRING", {"default": "models/.ov_cache"}),
                "cpu_threads": ("INT", {"default": 0, "min": 0, "max": 64,
                              "tooltip": "0 = OpenVINO default (all physical cores)"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/qwen image"

    @classmethod
    def encode(cls, prompt, negative_prompt, width, height, dq_group,
               model_dir, cache_dir, cpu_threads):
        global _ENCODER, _ENCODER_KEY
        root = os.getcwd()  # ComfyUI working dir (repo root when run from harness)
        mdir = model_dir if os.path.isabs(model_dir) else os.path.join(root, model_dir)
        if not os.path.isdir(mdir):
            # ComfyUI launched inside ComfyUI/ -> resolve repo layout
            alt = os.path.abspath(os.path.join(root, "..", model_dir))
            if os.path.isdir(alt):
                mdir = alt
        cdir = cache_dir if os.path.isabs(cache_dir) else os.path.join(root, cache_dir)
        if not os.path.isdir(cdir):
            c2 = os.path.abspath(os.path.join(root, "..", cache_dir))
            cdir = c2 if os.path.isdir(os.path.dirname(c2)) else cdir

        key = (mdir, dq_group, cdir, cpu_threads)
        if _ENCODER is None or _ENCODER_KEY != key:
            import sys
            here = os.path.dirname(os.path.realpath(__file__))  # resolve symlink
            repo_root = os.path.dirname(os.path.dirname(here))
            scripts_dir = os.path.join(repo_root, "scripts")
            if os.path.isdir(scripts_dir) and scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from ov_encoder import OVQwenEncoder  # repo bridge module
            if cpu_threads > 0:
                os.environ["OV_CPU_THREADS_NUM"] = str(cpu_threads)
            _ENCODER = OVQwenEncoder(mdir, dq=dq_group, cache_dir=cdir)
            _ENCODER_KEY = key

        def enc(text):
            r = _ENCODER.encode(text)
            return [[torch.from_numpy(r["hidden"].copy()).float(), {}]]

        positive = enc(prompt)
        negative = enc(negative_prompt if negative_prompt.strip() else " ")
        latent = torch.zeros([1, 64, height // 16, width // 16],
                             device=mm.intermediate_device())
        return (positive, negative, {"samples": latent})


NODE_CLASS_MAPPINGS = {"Qwen3VLOpenVINOCPUEncoder": Qwen3VLOpenVINOCPUEncoder}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen3VLOpenVINOCPUEncoder": "Qwen3-VL OpenVINO CPU Encoder (Hybrid)"}

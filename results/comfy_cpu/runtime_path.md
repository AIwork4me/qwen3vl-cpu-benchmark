# ComfyUI INT8 ConvRot CPU — runtime compute-path classification

## run tag: run1  (force_quant_mm=False)

- VERDICT: **A2 INT8_WEIGHT_STORAGE_FP_COMPUTE**
- quantized weights: 254 (convrot: 254), formats: {'int8_tensorwise': 254}
- load_device=cpu offload_device=cpu weight_devices=['cpu']
- per-encode counters: {'P1': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P2': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P3': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}}
- conditioning output dtype: torch.float32

## run tag: run2  (force_quant_mm=False)

- VERDICT: **A2 INT8_WEIGHT_STORAGE_FP_COMPUTE**
- quantized weights: 254 (convrot: 254), formats: {'int8_tensorwise': 254}
- load_device=cpu offload_device=cpu weight_devices=['cpu']
- per-encode counters: {'P1': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P2': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P3': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}}
- conditioning output dtype: torch.float32

## run tag: run3  (force_quant_mm=False)

- VERDICT: **A2 INT8_WEIGHT_STORAGE_FP_COMPUTE**
- quantized weights: 254 (convrot: 254), formats: {'int8_tensorwise': 254}
- load_device=cpu offload_device=cpu weight_devices=['cpu']
- per-encode counters: {'P1': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P2': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P3': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}}
- conditioning output dtype: torch.float32

## run tag: t8  (force_quant_mm=False)

- VERDICT: **A2 INT8_WEIGHT_STORAGE_FP_COMPUTE**
- quantized weights: 254 (convrot: 254), formats: {'int8_tensorwise': 254}
- load_device=cpu offload_device=cpu weight_devices=['cpu']
- per-encode counters: {'P1': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P2': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P3': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}}
- conditioning output dtype: torch.float32

## run tag: t32  (force_quant_mm=False)

- VERDICT: **A2 INT8_WEIGHT_STORAGE_FP_COMPUTE**
- quantized weights: 254 (convrot: 254), formats: {'int8_tensorwise': 254}
- load_device=cpu offload_device=cpu weight_devices=['cpu']
- per-encode counters: {'P1': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P2': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}, 'P3': {'int_mm': 0, 'dequant': 504, 'dequant_bytes': 13891534848}}
- conditioning output dtype: torch.float32

## run tag: a1_t16  (force_quant_mm=True)

- VERDICT: **A1 TRUE_INT8_CPU_COMPUTE**
- quantized weights: 254 (convrot: 254), formats: {'int8_tensorwise': 254}
- load_device=cpu offload_device=cpu weight_devices=['cpu']
- per-encode counters: {'P1': {'int_mm': 504, 'dequant': 0, 'dequant_bytes': 0}, 'P2': {'int_mm': 504, 'dequant': 0, 'dequant_bytes': 0}, 'P3': {'int_mm': 504, 'dequant': 0, 'dequant_bytes': 0}}
- conditioning output dtype: torch.float32

## run tag: a1_t8  (force_quant_mm=True)

- VERDICT: **A1 TRUE_INT8_CPU_COMPUTE**
- quantized weights: 254 (convrot: 254), formats: {'int8_tensorwise': 254}
- load_device=cpu offload_device=cpu weight_devices=['cpu']
- per-encode counters: {'P1': {'int_mm': 504, 'dequant': 0, 'dequant_bytes': 0}, 'P2': {'int_mm': 504, 'dequant': 0, 'dequant_bytes': 0}, 'P3': {'int_mm': 504, 'dequant': 0, 'dequant_bytes': 0}}
- conditioning output dtype: torch.float32

## run tag: a1_t32  (force_quant_mm=True)

- VERDICT: **A1 TRUE_INT8_CPU_COMPUTE**
- quantized weights: 254 (convrot: 254), formats: {'int8_tensorwise': 254}
- load_device=cpu offload_device=cpu weight_devices=['cpu']
- per-encode counters: {'P1': {'int_mm': 504, 'dequant': 0, 'dequant_bytes': 0}, 'P2': {'int_mm': 504, 'dequant': 0, 'dequant_bytes': 0}, 'P3': {'int_mm': 504, 'dequant': 0, 'dequant_bytes': 0}}
- conditioning output dtype: torch.float32

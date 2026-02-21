"""Model loading nodes for TRELLIS.2.

This returns a lightweight config object - actual model loading
happens inside node methods.
"""

import logging
import torch
import comfy.model_management as mm

log = logging.getLogger("trellis2")

# Resolution modes (matching original TRELLIS.2)
RESOLUTION_MODES = ['512', '1024_cascade', '1536_cascade']

# Attention backend options (auto detects best available)
ATTN_BACKENDS = ['auto', 'flash_attn', 'xformers', 'sdpa', 'sageattn']


class LoadTrellis2Models:
    """Load TRELLIS.2 models for 3D generation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (RESOLUTION_MODES, {"default": '1024_cascade'}),
            },
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "attn_backend": (ATTN_BACKENDS, {
                    "default": "auto",
                    "tooltip": "Attention backend. auto: best available (sageattn > flash_attn > xformers > sdpa)."
                }),
            }
        }

    RETURN_TYPES = ("TRELLIS2_MODEL_CONFIG",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "load_models"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Load TRELLIS.2 model configuration for image-to-3D generation.

This node creates a configuration object that inference nodes use
to load models on-demand.

Resolution modes:
- 512: Fast, lower quality
- 1024_cascade: Best quality, uses 512->1024 cascade
- 1536_cascade: Highest resolution output

Attention backend:
- auto: Best available (sageattn > flash_attn > xformers > sdpa)
- flash_attn: FlashAttention (requires flash_attn package)
- xformers: Memory-efficient attention (requires xformers package)
- sdpa: PyTorch native scaled_dot_product_attention (PyTorch >= 2.0)
- sageattn: SageAttention (fastest, requires sageattention package)
"""

    def load_models(self, resolution='1024_cascade', precision="auto", attn_backend="auto", **kwargs):
        # Resolve precision to actual torch dtype
        device = mm.get_torch_device()
        if precision == "auto":
            if mm.should_use_bf16(device):
                dtype = torch.bfloat16
            elif mm.should_use_fp16(device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        log.info(f"Resolved precision: {precision} -> {dtype}")

        # Setup attention backend
        if attn_backend != "auto":
            try:
                from .trellis2.modules.attention import config as dense_config
                from .trellis2.modules.sparse import config as sparse_config
                dense_config.set_backend(attn_backend)
                # Sparse attention doesn't support sageattn
                if attn_backend in ('flash_attn', 'xformers', 'sdpa'):
                    sparse_config.set_attn_backend(attn_backend)
                log.info(f"Attention backend set to: {attn_backend}")
            except Exception as e:
                log.warning(f"Could not set attention backend '{attn_backend}': {e}")
        else:
            log.info("Attention backend: auto (will detect on first use)")

        # Store dtype as string for JSON-safe IPC across isolation boundary
        dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
        config = {
            "model_name": "microsoft/TRELLIS.2-4B",
            "resolution": resolution,
            "precision": precision,
            "dtype": dtype_str,
            "attn_backend": attn_backend,
        }
        return (config,)


NODE_CLASS_MAPPINGS = {
    "LoadTrellis2Models": LoadTrellis2Models,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTrellis2Models": "Load TRELLIS.2 Models",
}

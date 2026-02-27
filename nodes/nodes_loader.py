"""Model loading nodes for TRELLIS.2.

This returns a lightweight config object - actual model loading
happens inside node methods.
"""

import os
import logging
import torch
import comfy.model_management as mm
import folder_paths
from comfy_api.latest import io

# Register model folder with ComfyUI's folder_paths system
_trellis2_models_dir = os.path.join(folder_paths.models_dir, "trellis2")
os.makedirs(_trellis2_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("trellis2", _trellis2_models_dir)

log = logging.getLogger("trellis2")

# Resolution modes (matching original TRELLIS.2)
RESOLUTION_MODES = ['512', '1024_cascade', '1536_cascade']

# Attention backend options (auto detects best available)
ATTN_BACKENDS = ['auto', 'flash_attn', 'xformers', 'sdpa', 'sageattn']


class LoadTrellis2Models(io.ComfyNode):
    """Load TRELLIS.2 models for 3D generation."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadTrellis2Models",
            display_name="(Down)Load TRELLIS.2 Models",
            category="TRELLIS2",
            description="""Load TRELLIS.2 model configuration for image-to-3D generation.

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
- sageattn: SageAttention (fastest, requires sageattention package)""",
            inputs=[
                io.Combo.Input("resolution", options=RESOLUTION_MODES, default='1024_cascade'),
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"],
                               default="auto", optional=True,
                               tooltip="Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."),
                io.Combo.Input("attn_backend", options=ATTN_BACKENDS,
                               default="auto", optional=True,
                               tooltip="Attention backend. auto: best available (sageattn > flash_attn > xformers > sdpa)."),
            ],
            outputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Output(display_name="model_config"),
            ],
        )

    @classmethod
    def execute(cls, resolution='1024_cascade', precision="auto", attn_backend="auto", **kwargs):
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
        # Dense attention: handled by ComfyUI's optimized_attention_for_device (auto-selects)
        # Sparse/varlen attention: handled by attention_sparse.py (auto-detects best backend)
        from .trellis2.model import set_backend as set_dense_backend
        from .trellis2.sparse import set_attn_backend as set_sparse_backend
        set_dense_backend(attn_backend)
        set_sparse_backend(attn_backend)
        log.info(f"Attention backend configured: {attn_backend}")

        # Store dtype as string for JSON-safe IPC across isolation boundary
        dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]
        config = {
            "model_name": "microsoft/TRELLIS.2-4B",
            "resolution": resolution,
            "precision": precision,
            "dtype": dtype_str,
            "attn_backend": attn_backend,
        }
        return io.NodeOutput(config)


NODE_CLASS_MAPPINGS = {
    "LoadTrellis2Models": LoadTrellis2Models,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTrellis2Models": "(Down)Load TRELLIS.2 Models",
}

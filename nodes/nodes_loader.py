"""Model loading nodes for TRELLIS.2.

This returns a lightweight config object - actual model loading
happens inside @isolated subprocess methods.
"""

from .trellis2_config import Trellis2ModelConfig

# Resolution modes (matching original TRELLIS.2)
RESOLUTION_MODES = ['512', '1024_cascade', '1536_cascade']

# Attention backend options
ATTN_BACKENDS = ['auto', 'flash_attn', 'xformers', 'sdpa']


class LoadTrellis2Models:
    """Load TRELLIS.2 models for 3D generation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (RESOLUTION_MODES, {"default": '1024_cascade'}),
            },
            "optional": {
                "attn_backend": (ATTN_BACKENDS, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_MODEL_CONFIG",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "load_models"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Load TRELLIS.2 model configuration for image-to-3D generation.

This node creates a configuration object that inference nodes use
to load models on-demand inside isolated subprocess environments.

Resolution modes:
- 512: Fast, lower quality
- 1024_cascade: Best quality, uses 512->1024 cascade
- 1536_cascade: Highest resolution output

Attention backend:
- auto: Auto-detect best available (flash_attn > xformers > sdpa)
- flash_attn: FlashAttention (fastest, requires flash_attn package)
- xformers: Memory-efficient attention (requires xformers package)
- sdpa: PyTorch native scaled_dot_product_attention (PyTorch >= 2.0)
"""

    def load_models(self, resolution='1024_cascade', attn_backend="auto"):
        # Create lightweight config object
        # Actual model loading happens in @isolated subprocess methods
        config = Trellis2ModelConfig(
            model_name="microsoft/TRELLIS.2-4B",
            resolution=resolution,
            attn_backend=attn_backend,
        )
        return (config,)


NODE_CLASS_MAPPINGS = {
    "LoadTrellis2Models": LoadTrellis2Models,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTrellis2Models": "Load TRELLIS.2 Models",
}

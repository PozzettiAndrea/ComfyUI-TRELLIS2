"""Model loading nodes for TRELLIS.2."""
import torch

import comfy.model_management as mm

from .utils import logger

# Resolution modes (matching original TRELLIS.2)
RESOLUTION_MODES = ['512', '1024_cascade', '1536_cascade']

# Attention backend options
ATTN_BACKENDS = ['auto', 'flash_attn', 'xformers', 'sdpa']

# Shape models needed for each resolution mode
SHAPE_MODELS_BY_RESOLUTION = {
    '512': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
    ],
    '1024_cascade': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
    ],
    '1536_cascade': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
    ],
}

# Texture models needed for each resolution mode
# Note: Texture maxes out at 1024, even for 1536_cascade (matches original TRELLIS.2)
TEXTURE_MODELS_BY_RESOLUTION = {
    '512': [
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
    ],
    '1024_cascade': [
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ],
    '1536_cascade': [
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ],
}

# Texture resolution mapping (texture maxes at 1024)
TEXTURE_RESOLUTION_MAP = {
    '512': '512',
    '1024_cascade': '1024_cascade',
    '1536_cascade': '1024_cascade',  # Texture uses 1024 even for 1536 shape
}


class LoadTrellis2Models:
    """Load TRELLIS.2 models for 3D generation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (RESOLUTION_MODES, {"default": '1024_cascade'}),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "attn_backend": (ATTN_BACKENDS, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_DINOV3", "TRELLIS2_SHAPE_PIPELINE", "TRELLIS2_TEXTURE_PIPELINE")
    RETURN_NAMES = ("dinov3", "shape_pipeline", "texture_pipeline")
    FUNCTION = "load_models"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Load TRELLIS.2 models for image-to-3D generation.

This single node prepares all models needed for the full pipeline:
- DinoV3 feature extractor (for conditioning)
- Shape generation models (sparse structure + shape latent)
- Texture generation models (texture latent + decoder)

All models are lazy-loaded when needed by inference nodes, ensuring
optimal VRAM usage. With keep_model_loaded=False, each stage loads
its models, runs, then offloads before the next stage.

Resolution modes:
- 512: Fast, lower quality
- 1024_cascade: Best quality, uses 512->1024 cascade
- 1536_cascade: Highest resolution output

Memory options:
- keep_model_loaded=False (default): Models load on demand, offload after use.
  Peak VRAM = max(dinov3, shape, texture), not sum.
- keep_model_loaded=True: Models stay on GPU. Faster but uses more VRAM.

Attention backend:
- auto: Auto-detect best available (flash_attn > xformers > sdpa)
- flash_attn: FlashAttention (fastest, requires flash_attn package)
- xformers: Memory-efficient attention (requires xformers package)
- sdpa: PyTorch native scaled_dot_product_attention (PyTorch >= 2.0)
"""

    def load_models(self, resolution='1024_cascade', keep_model_loaded=False, attn_backend="auto"):
        # Set attention backend before any model loading
        if attn_backend != "auto":
            from ..trellis2.modules.attention import config as dense_config
            from ..trellis2.modules.sparse import config as sparse_config
            dense_config.set_backend(attn_backend)
            sparse_config.set_attn_backend(attn_backend)
            logger.info(f"Attention backend set to: {attn_backend}")

        device = mm.get_torch_device()
        model_name = 'microsoft/TRELLIS.2-4B'

        # Get models needed for this resolution
        shape_models = SHAPE_MODELS_BY_RESOLUTION.get(resolution, SHAPE_MODELS_BY_RESOLUTION['1024_cascade'])
        texture_models = TEXTURE_MODELS_BY_RESOLUTION.get(resolution, TEXTURE_MODELS_BY_RESOLUTION['1024_cascade'])
        texture_resolution = TEXTURE_RESOLUTION_MAP.get(resolution, '1024_cascade')

        logger.info(f"Preparing TRELLIS.2 configs (resolution={resolution}, lazy loading enabled)")

        # === DinoV3 config (lazy loads in GetConditioning) ===
        dinov3_config = {
            "model": None,  # Lazy loaded
            "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "device": device,
            "keep_model_loaded": keep_model_loaded,
        }

        # === Shape pipeline config (lazy loads in ImageToShape) ===
        shape_config = {
            "pipeline": None,  # Lazy loaded
            "model_name": model_name,
            "resolution": resolution,
            "device": device,
            "keep_model_loaded": keep_model_loaded,
            "models_to_load": shape_models,
            "attn_backend": attn_backend,
        }

        # === Texture pipeline config (lazy loads in ShapeToTexturedMesh) ===
        texture_config = {
            "pipeline": None,  # Lazy loaded
            "model_name": model_name,
            "resolution": texture_resolution,
            "device": device,
            "keep_model_loaded": keep_model_loaded,
            "models_to_load": texture_models,
            "attn_backend": attn_backend,
        }

        logger.info(f"TRELLIS.2 configs prepared (shape={resolution}, texture={texture_resolution})")

        return (dinov3_config, shape_config, texture_config)


NODE_CLASS_MAPPINGS = {
    "LoadTrellis2Models": LoadTrellis2Models,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTrellis2Models": "Load TRELLIS.2 Models",
}

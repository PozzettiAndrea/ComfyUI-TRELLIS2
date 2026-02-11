"""Model loading nodes for TRELLIS.2."""
import torch

import comfy.model_management as mm
import folder_paths

from .utils import logger

# Resolution modes
RESOLUTION_MODES = ['512', '1024', '1024_cascade', '1536_cascade']

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
    '1024': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_1024',
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
# Note: shape_slat_decoder is NOT needed here - texture node receives pre-computed 'subs' from shape node
TEXTURE_MODELS_BY_RESOLUTION = {
    '512': [
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
    ],
    '1024': [
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
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


class LoadTrellis2ShapeModel:
    """Load TRELLIS.2 shape generation models from HuggingFace."""

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

    RETURN_TYPES = ("TRELLIS2_SHAPE_PIPELINE",)
    RETURN_NAMES = ("shape_pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Load TRELLIS.2 shape generation models from HuggingFace.

This loads only the models needed for geometry generation:
- Sparse structure models
- Shape latent flow models
- Shape decoder

Resolution modes:
- 512: Fast, lower quality (~2.5GB)
- 1024: Higher quality (~3GB)
- 1024_cascade: Best quality, uses 512->1024 cascade (~4GB)
- 1536_cascade: Highest resolution output (~4GB)

Memory options:
- keep_model_loaded=False (default): Models are unloaded after each use
  and reloaded directly from disk to GPU. Zero RAM usage but ~2-3s reload time.
- keep_model_loaded=True: Models stay on GPU between uses. Faster but uses VRAM.

Attention backend:
- auto: Auto-detect best available (flash_attn > xformers > sdpa)
- flash_attn: FlashAttention (fastest, requires flash_attn package)
- xformers: Memory-efficient attention (requires xformers package)
- sdpa: PyTorch native scaled_dot_product_attention (PyTorch >= 2.0)

Connect to "Image to Shape" node for geometry generation.
"""

    def loadmodel(self, resolution='1024_cascade', keep_model_loaded=False, attn_backend="auto"):
        # Set attention backend before loading models
        if attn_backend != "auto":
            from ..trellis2.modules.attention import config as dense_config
            from ..trellis2.modules.sparse import config as sparse_config
            dense_config.set_backend(attn_backend)
            sparse_config.set_attn_backend(attn_backend)
            logger.info(f"Attention backend set to: {attn_backend}")

        device = mm.get_torch_device()
        model = 'microsoft/TRELLIS.2-4B'

        logger.info(f"Loading TRELLIS.2 shape pipeline (resolution={resolution}, attn_backend={attn_backend})")

        from ..trellis2.pipelines import Trellis2ImageTo3DPipeline

        # Get list of shape models needed for this resolution
        needed_models = SHAPE_MODELS_BY_RESOLUTION.get(resolution, SHAPE_MODELS_BY_RESOLUTION['1024_cascade'])

        # Load pipeline with disk offload support when not keeping models loaded
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            model,
            models_to_load=needed_models,
            enable_disk_offload=not keep_model_loaded
        )
        pipeline.keep_model_loaded = keep_model_loaded
        pipeline.default_pipeline_type = resolution

        # Move to device (only if keeping models loaded)
        if keep_model_loaded:
            if device.type == 'cuda':
                pipeline.cuda()
            else:
                pipeline.to(device)
        else:
            # Just set the device, models will be loaded on-demand
            pipeline._device = device

        logger.info(f"TRELLIS.2 shape pipeline loaded successfully (resolution={resolution}, keep_model_loaded={keep_model_loaded})")

        shape_model = {
            "pipeline": pipeline,
            "model_name": model,
            "resolution": resolution,
            "device": device,
        }

        return (shape_model,)


class LoadTrellis2TextureModel:
    """Load TRELLIS.2 texture generation models from HuggingFace."""

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

    RETURN_TYPES = ("TRELLIS2_TEXTURE_PIPELINE",)
    RETURN_NAMES = ("texture_pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Prepare TRELLIS.2 texture/PBR generation pipeline config.

Models are loaded LAZILY when the "Shape to Textured Mesh" node runs,
ensuring texture models don't compete with shape models for VRAM.

This will load:
- Texture latent flow models
- Texture decoder (outputs base_color, metallic, roughness, alpha)

Resolution modes:
- 512: Uses 512px texture models (~2.5GB)
- 1024/1024_cascade/1536_cascade: Uses 1024px texture models (~2.5GB)

Memory options:
- keep_model_loaded=False (default): Models are unloaded after each use
  and reloaded directly from disk to GPU. Zero RAM usage but ~2-3s reload time.
- keep_model_loaded=True: Models stay on GPU between uses. Faster but uses VRAM.

Attention backend:
- auto: Auto-detect best available (flash_attn > xformers > sdpa)
- flash_attn: FlashAttention (fastest, requires flash_attn package)
- xformers: Memory-efficient attention (requires xformers package)
- sdpa: PyTorch native scaled_dot_product_attention (PyTorch >= 2.0)

Connect to "Shape to Textured Mesh" node for PBR material generation.
"""

    def loadmodel(self, resolution='1024_cascade', keep_model_loaded=False, attn_backend="auto"):
        # Set attention backend before loading models
        if attn_backend != "auto":
            from ..trellis2.modules.attention import config as dense_config
            from ..trellis2.modules.sparse import config as sparse_config
            dense_config.set_backend(attn_backend)
            sparse_config.set_attn_backend(attn_backend)
            logger.info(f"Attention backend set to: {attn_backend}")

        device = mm.get_torch_device()
        model = 'microsoft/TRELLIS.2-4B'

        # Get list of texture models needed for this resolution
        needed_models = TEXTURE_MODELS_BY_RESOLUTION.get(resolution, TEXTURE_MODELS_BY_RESOLUTION['1024_cascade'])

        logger.info(f"Prepared TRELLIS.2 texture pipeline config (resolution={resolution}, models will load lazily)")

        # Don't load models here - defer to inference node for optimal VRAM usage
        # This ensures texture models don't compete with shape models during shape inference
        texture_model = {
            "pipeline": None,  # Will be loaded lazily in inference node
            "model_name": model,
            "resolution": resolution,
            "device": device,
            "keep_model_loaded": keep_model_loaded,
            "models_to_load": needed_models,
        }

        return (texture_model,)


class LoadTrellis2DinoV3:
    """Load DinoV3 feature extractor for TRELLIS.2 conditioning."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "low_vram": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_DINOV3",)
    RETURN_NAMES = ("dinov3",)
    FUNCTION = "loadmodel"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Load DinoV3 feature extractor for image conditioning.

This model extracts visual features from images that guide
the 3D generation process. Must be connected to the
"Get Conditioning" node.
"""

    def loadmodel(self, low_vram=True):
        device = mm.get_torch_device()

        logger.info("Loading DinoV3 feature extractor...")

        from ..trellis2.modules import image_feature_extractor

        # DinoV3 model config (from pipeline.json)
        dinov3 = image_feature_extractor.DinoV3FeatureExtractor(
            model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"
        )

        if not low_vram:
            dinov3.to(device)

        logger.info("DinoV3 feature extractor loaded successfully")

        dinov3_model = {
            "model": dinov3,
            "device": device,
            "low_vram": low_vram,
        }

        return (dinov3_model,)


NODE_CLASS_MAPPINGS = {
    "LoadTrellis2ShapeModel": LoadTrellis2ShapeModel,
    "LoadTrellis2TextureModel": LoadTrellis2TextureModel,
    "LoadTrellis2DinoV3": LoadTrellis2DinoV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTrellis2ShapeModel": "Load TRELLIS.2 Shape Model",
    "LoadTrellis2TextureModel": "Load TRELLIS.2 Texture Model",
    "LoadTrellis2DinoV3": "Load TRELLIS.2 DinoV3",
}

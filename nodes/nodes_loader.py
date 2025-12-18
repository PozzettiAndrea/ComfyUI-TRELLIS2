"""Model loading nodes for TRELLIS.2."""
import torch

import comfy.model_management as mm
import folder_paths

from .utils import logger

# Resolution modes
RESOLUTION_MODES = ['512', '1024', '1024_cascade', '1536_cascade']

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
# Note: shape_slat_decoder is needed for texture decoding (generates mesh structure)
TEXTURE_MODELS_BY_RESOLUTION = {
    '512': [
        'shape_slat_decoder',
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
    ],
    '1024': [
        'shape_slat_decoder',
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ],
    '1024_cascade': [
        'shape_slat_decoder',
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ],
    '1536_cascade': [
        'shape_slat_decoder',
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
                "low_vram": ("BOOLEAN", {"default": True}),
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

Connect to "Image to Shape" node for geometry generation.
"""

    def loadmodel(self, resolution='1024_cascade', low_vram=True):
        device = mm.get_torch_device()
        model = 'microsoft/TRELLIS.2-4B'

        logger.info(f"Loading TRELLIS.2 shape pipeline (resolution={resolution})")

        from ..trellis2.pipelines import Trellis2ImageTo3DPipeline

        # Get list of shape models needed for this resolution
        needed_models = SHAPE_MODELS_BY_RESOLUTION.get(resolution, SHAPE_MODELS_BY_RESOLUTION['1024_cascade'])

        # Load pipeline with only shape models
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            model,
            models_to_load=needed_models
        )
        pipeline.low_vram = low_vram
        pipeline.default_pipeline_type = resolution

        # Move to device
        if device.type == 'cuda':
            pipeline.cuda()
        else:
            pipeline.to(device)

        logger.info(f"TRELLIS.2 shape pipeline loaded successfully (resolution={resolution}, low_vram={low_vram})")

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
                "low_vram": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_TEXTURE_PIPELINE",)
    RETURN_NAMES = ("texture_pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Load TRELLIS.2 texture/PBR generation models from HuggingFace.

This loads models needed for texture generation:
- Shape decoder (needed to decode mesh structure)
- Texture latent flow models
- Texture decoder (outputs base_color, metallic, roughness, alpha)

Resolution modes:
- 512: Uses 512px texture models (~2.5GB)
- 1024/1024_cascade/1536_cascade: Uses 1024px texture models (~2.5GB)

Connect to "Shape to Texture" node for PBR material generation.
"""

    def loadmodel(self, resolution='1024_cascade', low_vram=True):
        device = mm.get_torch_device()
        model = 'microsoft/TRELLIS.2-4B'

        logger.info(f"Loading TRELLIS.2 texture pipeline (resolution={resolution})")

        from ..trellis2.pipelines import Trellis2ImageTo3DPipeline

        # Get list of texture models needed for this resolution
        needed_models = TEXTURE_MODELS_BY_RESOLUTION.get(resolution, TEXTURE_MODELS_BY_RESOLUTION['1024_cascade'])

        # Load pipeline with only texture models
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            model,
            models_to_load=needed_models
        )
        pipeline.low_vram = low_vram
        pipeline.default_pipeline_type = resolution

        # Move to device
        if device.type == 'cuda':
            pipeline.cuda()
        else:
            pipeline.to(device)

        logger.info(f"TRELLIS.2 texture pipeline loaded successfully (resolution={resolution}, low_vram={low_vram})")

        texture_model = {
            "pipeline": pipeline,
            "model_name": model,
            "resolution": resolution,
            "device": device,
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

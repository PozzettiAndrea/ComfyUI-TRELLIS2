"""Model loading nodes for TRELLIS.2."""
import torch

import comfy.model_management as mm
import folder_paths

from .utils import logger

# Resolution modes and which models they need
RESOLUTION_MODES = ['512', '1024', '1024_cascade', '1536_cascade']

# Models needed for each resolution mode
MODELS_BY_RESOLUTION = {
    '512': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
    ],
    '1024': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_1024',
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ],
    '1024_cascade': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ],
    '1536_cascade': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ],
}


class DownloadAndLoadTrellis2Model:
    """Load TRELLIS.2 Image-to-3D pipeline from HuggingFace."""

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

    RETURN_TYPES = ("TRELLIS2_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Load TRELLIS.2 4B Image-to-3D pipeline from HuggingFace.

Resolution modes (only downloads models needed):
- 512: Fast, lower quality (downloads ~4GB)
- 1024: Higher quality (downloads ~5GB)
- 1024_cascade: Best quality, uses 512->1024 cascade (downloads ~7GB)
- 1536_cascade: Highest resolution output (downloads ~7GB)

Parameters:
- resolution: Output resolution mode
- low_vram: Enable low-VRAM mode (recommended)
"""

    def loadmodel(self, resolution='1024_cascade', low_vram=True):
        device = mm.get_torch_device()
        model = 'microsoft/TRELLIS.2-4B'

        logger.info(f"Loading TRELLIS.2 pipeline (resolution={resolution})")

        from ..trellis2.pipelines import Trellis2ImageTo3DPipeline

        # Get list of models needed for this resolution
        needed_models = MODELS_BY_RESOLUTION.get(resolution, MODELS_BY_RESOLUTION['1024_cascade'])

        # Load pipeline with only needed models
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

        logger.info(f"TRELLIS.2 pipeline loaded successfully (resolution={resolution}, low_vram={low_vram})")

        trellis_model = {
            "pipeline": pipeline,
            "model_name": model,
            "resolution": resolution,
            "device": device,
        }

        return (trellis_model,)


class Trellis2SetSamplerParams:
    """Configure sampling parameters for TRELLIS.2 pipeline."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                # Sparse Structure Sampler
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "ss_guidance_rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "ss_rescale_t": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                # Shape SLat Sampler
                "shape_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shape_guidance_rescale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shape_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "shape_rescale_t": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                # Texture SLat Sampler
                "tex_guidance_strength": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "tex_guidance_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tex_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "tex_rescale_t": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_SAMPLER_PARAMS",)
    RETURN_NAMES = ("sampler_params",)
    FUNCTION = "configure"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Configure sampling parameters for the TRELLIS.2 pipeline.

Three stages:
1. Sparse Structure Generation (ss_*) - Generates the 3D structure
2. Shape Generation (shape_*) - Generates detailed shape
3. Material Generation (tex_*) - Generates PBR materials/textures
"""

    def configure(
        self,
        ss_guidance_strength=7.5,
        ss_guidance_rescale=0.7,
        ss_sampling_steps=12,
        ss_rescale_t=5.0,
        shape_guidance_strength=7.5,
        shape_guidance_rescale=0.5,
        shape_sampling_steps=12,
        shape_rescale_t=3.0,
        tex_guidance_strength=1.0,
        tex_guidance_rescale=0.0,
        tex_sampling_steps=12,
        tex_rescale_t=3.0,
    ):
        sampler_params = {
            "sparse_structure_sampler_params": {
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            "shape_slat_sampler_params": {
                "steps": shape_sampling_steps,
                "guidance_strength": shape_guidance_strength,
                "guidance_rescale": shape_guidance_rescale,
                "rescale_t": shape_rescale_t,
            },
            "tex_slat_sampler_params": {
                "steps": tex_sampling_steps,
                "guidance_strength": tex_guidance_strength,
                "guidance_rescale": tex_guidance_rescale,
                "rescale_t": tex_rescale_t,
            },
        }

        return (sampler_params,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadTrellis2Model": DownloadAndLoadTrellis2Model,
    "Trellis2SetSamplerParams": Trellis2SetSamplerParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadTrellis2Model": "(Down)Load TRELLIS.2 Model",
    "Trellis2SetSamplerParams": "TRELLIS.2 Sampler Parameters",
}

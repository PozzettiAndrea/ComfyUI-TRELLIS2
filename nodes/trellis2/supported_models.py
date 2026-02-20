"""
TRELLIS2 supported model configurations.

Follows the pattern in comfy/supported_models.py.
These configs are registered with ComfyUI at import time
so that TRELLIS2 checkpoints can be auto-detected and loaded.

Uses lazy imports to avoid loading heavy model/sparse deps at registration time.
"""
import torch
import comfy.supported_models_base
from . import latent_formats as trellis2_latent_formats


class TRELLIS2SparseStructure(comfy.supported_models_base.BASE):
    """
    Config for TRELLIS2 SparseStructureFlowModel.

    Dense 3D flow model that generates binary voxel occupancy.
    Uses ModulatedTransformerCrossBlock with AdaLN conditioning.
    """
    unet_config = {
        "image_model": "trellis2_sparse_structure",
    }

    sampling_settings = {
        "shift": 1.0,
        "multiplier": 1000,
    }

    unet_extra_config = {}
    latent_format = trellis2_latent_formats.TRELLIS2SparseStructure

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    # No text encoder or VAE in the same checkpoint
    vae_key_prefix = []
    text_encoder_key_prefix = []

    def get_model(self, state_dict, prefix="", device=None):
        from . import model_base as trellis2_model_base
        out = trellis2_model_base.TRELLIS2SparseStructure(self, device=device)
        return out

    def clip_target(self, state_dict={}):
        return None


class TRELLIS2SLat(comfy.supported_models_base.BASE):
    """
    Config for TRELLIS2 SLatFlowModel.

    Sparse 3D flow model for shape/texture structured latent generation.
    Uses ModulatedSparseTransformerCrossBlock with SparseLinear layers.
    """
    unet_config = {
        "image_model": "trellis2_slat",
    }

    sampling_settings = {
        "shift": 1.0,
        "multiplier": 1000,
    }

    unet_extra_config = {}
    latent_format = trellis2_latent_formats.TRELLIS2ShapeSLat

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = []
    text_encoder_key_prefix = []

    def get_model(self, state_dict, prefix="", device=None):
        # SLat BaseModel subclass not yet implemented (needs SparseTensor support)
        return None

    def clip_target(self, state_dict={}):
        return None

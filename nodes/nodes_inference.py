"""Inference nodes for TRELLIS.2 Image-to-3D generation."""
import torch
import numpy as np
from PIL import Image

import comfy.model_management as mm

from .utils import logger, tensor_to_pil, pil_to_tensor


class Trellis2PreprocessImage:
    """Preprocess image for TRELLIS.2 (background removal and cropping)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2_PIPELINE",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preprocessed_image",)
    FUNCTION = "preprocess"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Preprocess image for TRELLIS.2 generation.

This node:
- Removes background if no alpha channel present (uses BiRefNet)
- Crops to object bounding box
- Prepares image for 3D generation

Input should ideally be an RGBA image with transparent background,
or an RGB image (background will be removed automatically).
"""

    def preprocess(self, pipeline, image):
        pipe = pipeline["pipeline"]

        # Convert ComfyUI tensor to PIL
        pil_image = tensor_to_pil(image)

        # Use pipeline's preprocessing
        processed = pipe.preprocess_image(pil_image)

        # Convert back to tensor
        output_tensor = pil_to_tensor(processed)

        return (output_tensor,)


class Trellis2ImageTo3D:
    """Generate 3D mesh from image using TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2_PIPELINE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "sampler_params": ("TRELLIS2_SAMPLER_PARAMS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "resolution": (["512", "1024", "1536"], {"default": "1024"}),
                "preprocess_image": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_MESH", "TRELLIS2_LATENT")
    RETURN_NAMES = ("mesh", "latent")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate a 3D mesh with PBR materials from a single image.

Parameters:
- pipeline: The loaded TRELLIS.2 pipeline
- image: Input image (RGB or RGBA)
- sampler_params: Optional custom sampling parameters
- seed: Random seed for reproducibility
- resolution: Output resolution (512, 1024, or 1536)
- preprocess_image: Whether to preprocess (remove background, crop)

Returns:
- mesh: The generated 3D mesh with PBR materials
- latent: Latent codes for further manipulation
"""

    def generate(
        self,
        pipeline,
        image,
        sampler_params=None,
        seed=0,
        resolution="1024",
        preprocess_image=True,
    ):
        pipe = pipeline["pipeline"]

        # Convert ComfyUI tensor to PIL
        pil_image = tensor_to_pil(image)

        # Determine pipeline type based on resolution
        pipeline_type = {
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution]

        # Build kwargs
        kwargs = {
            "seed": seed,
            "preprocess_image": preprocess_image,
            "pipeline_type": pipeline_type,
            "return_latent": True,
        }

        # Add sampler params if provided
        if sampler_params is not None:
            kwargs.update(sampler_params)

        logger.info(f"Generating 3D mesh (resolution={resolution}, seed={seed})")

        # Run generation
        outputs, latents = pipe.run(pil_image, **kwargs)
        mesh = outputs[0]

        # Simplify mesh (nvdiffrast limit)
        mesh.simplify(16777216)

        logger.info("3D mesh generated successfully")

        # Pack latent for potential reuse
        shape_slat, tex_slat, res = latents
        latent_dict = {
            'shape_slat_feats': shape_slat.feats.cpu().numpy(),
            'tex_slat_feats': tex_slat.feats.cpu().numpy(),
            'coords': shape_slat.coords.cpu().numpy(),
            'res': res,
        }

        # Pack mesh
        mesh_dict = {
            "mesh": mesh,
            "pipeline": pipe,
        }

        torch.cuda.empty_cache()

        return (mesh_dict, latent_dict)


class Trellis2DecodeLatent:
    """Decode latent codes back to mesh."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2_PIPELINE",),
                "latent": ("TRELLIS2_LATENT",),
            },
        }

    RETURN_TYPES = ("TRELLIS2_MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "decode"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Decode latent codes back to a 3D mesh.

Useful for regenerating mesh from saved latents
or for experimenting with latent manipulation.
"""

    def decode(self, pipeline, latent):
        pipe = pipeline["pipeline"]

        from ..trellis2.modules.sparse import SparseTensor

        # Unpack latent
        shape_slat = SparseTensor(
            feats=torch.from_numpy(latent['shape_slat_feats']).cuda(),
            coords=torch.from_numpy(latent['coords']).cuda(),
        )
        tex_slat = shape_slat.replace(
            torch.from_numpy(latent['tex_slat_feats']).cuda()
        )
        res = latent['res']

        # Decode
        mesh = pipe.decode_latent(shape_slat, tex_slat, res)[0]
        mesh.simplify(16777216)

        mesh_dict = {
            "mesh": mesh,
            "pipeline": pipe,
        }

        torch.cuda.empty_cache()

        return (mesh_dict,)


NODE_CLASS_MAPPINGS = {
    "Trellis2PreprocessImage": Trellis2PreprocessImage,
    "Trellis2ImageTo3D": Trellis2ImageTo3D,
    "Trellis2DecodeLatent": Trellis2DecodeLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2PreprocessImage": "TRELLIS.2 Preprocess Image",
    "Trellis2ImageTo3D": "TRELLIS.2 Image to 3D",
    "Trellis2DecodeLatent": "TRELLIS.2 Decode Latent",
}

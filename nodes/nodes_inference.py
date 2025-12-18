"""Inference nodes for TRELLIS.2 Image-to-3D generation."""
import torch
import numpy as np
from PIL import Image
import trimesh as Trimesh
from trimesh.voxel.base import VoxelGrid

import comfy.model_management as mm

from .utils import logger, tensor_to_pil, pil_to_tensor


def mesh_with_voxel_to_outputs(mesh_obj, pipeline):
    """
    Convert TRELLIS MeshWithVoxel to separate TRIMESH and VOXELGRID outputs.

    Returns:
        trimesh: trimesh.Trimesh geometry (GeometryPack compatible)
        voxelgrid: trimesh.voxel.VoxelGrid with PBR attributes attached
    """
    # Extract geometry as trimesh
    vertices = mesh_obj.vertices.cpu().numpy()
    faces = mesh_obj.faces.cpu().numpy()

    # Coordinate system conversion (Y-up to Z-up for compatibility)
    vertices_converted = vertices.copy()
    vertices_converted[:, 1], vertices_converted[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

    tri_mesh = Trimesh.Trimesh(
        vertices=vertices_converted,
        faces=faces,
        process=False
    )

    # Create VoxelGrid with PBR attributes
    # Use sparse coordinates to determine grid shape
    coords = mesh_obj.coords.cpu()
    grid_shape = tuple((coords.max(dim=0).values + 1).int().tolist())

    # Create boolean encoding from sparse coords
    encoding = np.zeros(grid_shape, dtype=bool)
    coords_np = coords.numpy().astype(int)
    encoding[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]] = True

    voxel_grid = VoxelGrid(encoding)

    # Attach PBR attributes
    voxel_grid.pbr_attrs = mesh_obj.attrs  # Sparse tensor features
    voxel_grid.pbr_coords = mesh_obj.coords  # Sparse coordinates
    voxel_grid.pbr_layout = pipeline.pbr_attr_layout  # {'base_color': slice(0,3), ...}
    voxel_grid.pbr_voxel_size = mesh_obj.voxel_size

    # Store original vertices for BVH lookup during texture baking
    voxel_grid.original_vertices = mesh_obj.vertices
    voxel_grid.original_faces = mesh_obj.faces

    return tri_mesh, voxel_grid


class Trellis2PreprocessImage:
    """Preprocess image for TRELLIS.2 (background removal and cropping)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("TRELLIS2_PIPELINE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preprocessed_image",)
    FUNCTION = "preprocess"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Preprocess image for TRELLIS.2 generation.

This node:
- If mask provided: uses mask as alpha (skips BiRefNet)
- If no mask: removes background automatically (uses BiRefNet)
- Crops to object bounding box
- Premultiplies alpha (RGB * alpha)

Output is an RGB image ready for 3D generation.
"""

    def preprocess(self, pipeline, image, mask=None):
        pipe = pipeline["pipeline"]

        # Convert ComfyUI tensor to PIL
        pil_image = tensor_to_pil(image)

        # If mask provided, apply it as alpha channel
        if mask is not None:
            # Convert mask tensor to numpy
            if mask.dim() == 3:
                mask_np = mask[0].cpu().numpy()
            else:
                mask_np = mask.cpu().numpy()

            # Resize mask to match image if needed
            if mask_np.shape[:2] != (pil_image.height, pil_image.width):
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((pil_image.width, pil_image.height), Image.LANCZOS)
                mask_np = np.array(mask_pil) / 255.0

            # Apply mask as alpha channel
            pil_image = pil_image.convert('RGB')
            img_np = np.array(pil_image)
            alpha = (mask_np * 255).astype(np.uint8)
            rgba = np.dstack([img_np, alpha])
            pil_image = Image.fromarray(rgba, 'RGBA')
            logger.info("Applied mask as alpha channel")

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
                "mask": ("MASK",),
                "sampler_params": ("TRELLIS2_SAMPLER_PARAMS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "resolution": (["512", "1024", "1536"], {"default": "1024"}),
                "preprocess_image": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "VOXELGRID", "TRELLIS2_LATENT")
    RETURN_NAMES = ("trimesh", "voxelgrid", "latent")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate a 3D mesh with PBR materials from a single image.

Parameters:
- pipeline: The loaded TRELLIS.2 pipeline
- image: Input image (RGB or RGBA)
- mask: Optional mask (white=foreground, black=background). If provided, skips auto background removal.
- sampler_params: Optional custom sampling parameters
- seed: Random seed for reproducibility
- resolution: Output resolution (512, 1024, or 1536)
- preprocess_image: Whether to preprocess (crop to object). Auto bg removal skipped if mask provided.

Returns:
- trimesh: The generated 3D mesh geometry (GeometryPack compatible)
- voxelgrid: VoxelGrid with PBR attributes for texture baking
- latent: Latent codes for further manipulation
"""

    def generate(
        self,
        pipeline,
        image,
        mask=None,
        sampler_params=None,
        seed=0,
        resolution="1024",
        preprocess_image=True,
    ):
        pipe = pipeline["pipeline"]

        # Convert ComfyUI tensor to PIL
        pil_image = tensor_to_pil(image)

        # If mask provided, apply it as alpha channel
        if mask is not None:
            # Convert mask tensor to numpy (ComfyUI masks are [H,W] or [B,H,W])
            if mask.dim() == 3:
                mask_np = mask[0].cpu().numpy()
            else:
                mask_np = mask.cpu().numpy()

            # Resize mask to match image if needed
            if mask_np.shape[:2] != (pil_image.height, pil_image.width):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((pil_image.width, pil_image.height), PILImage.LANCZOS)
                mask_np = np.array(mask_pil) / 255.0

            # Apply mask as alpha channel
            pil_image = pil_image.convert('RGB')
            img_np = np.array(pil_image)
            alpha = (mask_np * 255).astype(np.uint8)
            rgba = np.dstack([img_np, alpha])
            pil_image = Image.fromarray(rgba, 'RGBA')
            logger.info("Applied mask as alpha channel")

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

        # Convert to TRIMESH + VOXELGRID outputs
        tri_mesh, voxel_grid = mesh_with_voxel_to_outputs(mesh, pipe)

        # Pack latent for potential reuse
        shape_slat, tex_slat, res = latents
        latent_dict = {
            'shape_slat_feats': shape_slat.feats.cpu().numpy(),
            'tex_slat_feats': tex_slat.feats.cpu().numpy(),
            'coords': shape_slat.coords.cpu().numpy(),
            'res': res,
        }

        torch.cuda.empty_cache()

        return (tri_mesh, voxel_grid, latent_dict)


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

    RETURN_TYPES = ("TRIMESH", "VOXELGRID")
    RETURN_NAMES = ("trimesh", "voxelgrid")
    FUNCTION = "decode"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Decode latent codes back to a 3D mesh.

Useful for regenerating mesh from saved latents
or for experimenting with latent manipulation.

Returns:
- trimesh: The decoded mesh geometry (GeometryPack compatible)
- voxelgrid: VoxelGrid with PBR attributes for texture baking
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

        # Convert to TRIMESH + VOXELGRID outputs
        tri_mesh, voxel_grid = mesh_with_voxel_to_outputs(mesh, pipe)

        torch.cuda.empty_cache()

        return (tri_mesh, voxel_grid)


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

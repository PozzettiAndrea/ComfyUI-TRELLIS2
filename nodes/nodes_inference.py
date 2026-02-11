"""Inference nodes for TRELLIS.2 Image-to-3D generation."""
import gc
import torch
import numpy as np
from PIL import Image
import trimesh as Trimesh
from trimesh.voxel.base import VoxelGrid

import comfy.model_management as mm

from .utils import logger, tensor_to_pil


def mesh_to_trimesh(mesh_obj):
    """
    Convert TRELLIS Mesh to trimesh.Trimesh.

    Returns:
        trimesh: trimesh.Trimesh geometry (GeometryPack compatible)
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

    return tri_mesh


def mesh_with_voxel_to_outputs(mesh_obj, pbr_layout):
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

    # Attach PBR attributes - move to CPU to free GPU memory
    # They'll be moved back to GPU during export if needed
    voxel_grid.pbr_attrs = mesh_obj.attrs.cpu() if hasattr(mesh_obj.attrs, 'cpu') else mesh_obj.attrs
    voxel_grid.pbr_coords = mesh_obj.coords.cpu() if hasattr(mesh_obj.coords, 'cpu') else mesh_obj.coords
    voxel_grid.pbr_layout = pbr_layout  # {'base_color': slice(0,3), ...}
    voxel_grid.pbr_voxel_size = mesh_obj.voxel_size

    # Store original vertices for BVH lookup during texture baking - move to CPU
    voxel_grid.original_vertices = mesh_obj.vertices.cpu() if hasattr(mesh_obj.vertices, 'cpu') else mesh_obj.vertices
    voxel_grid.original_faces = mesh_obj.faces.cpu() if hasattr(mesh_obj.faces, 'cpu') else mesh_obj.faces

    # Clear GPU cache after moving tensors to CPU
    torch.cuda.empty_cache()

    return tri_mesh, voxel_grid


class Trellis2GetConditioning:
    """Extract image conditioning using DinoV3 for TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dinov3": ("TRELLIS2_DINOV3",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "include_1024": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TRELLIS2_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "get_conditioning"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Preprocess image and extract visual features using DinoV3.

This node handles:
1. Applying mask as alpha channel
2. Cropping to object bounding box
3. Alpha premultiplication
4. DinoV3 feature extraction

Parameters:
- dinov3: The loaded DinoV3 model
- image: Input image (RGB)
- mask: Foreground mask (white=object, black=background)
- include_1024: Also extract 1024px features (needed for cascade modes)

Use any background removal node (BiRefNet, rembg, etc.) to generate the mask.
"""

    def get_conditioning(self, dinov3, image, mask, include_1024=True):
        device = dinov3["device"]
        model = dinov3["model"]
        low_vram = dinov3["low_vram"]

        # Convert ComfyUI tensor to PIL
        pil_image = tensor_to_pil(image)

        # Apply mask as alpha channel
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
        alpha_np = (mask_np * 255).astype(np.uint8)
        rgba = np.dstack([img_np, alpha_np])
        pil_image = Image.fromarray(rgba, 'RGBA')

        # Crop to bounding box
        bbox = np.argwhere(alpha_np > 0.8 * 255)
        if len(bbox) > 0:
            bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
            center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            size = int(size * 1)
            bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
            pil_image = pil_image.crop(bbox)

        # Premultiply alpha
        output_np = np.array(pil_image.convert('RGBA')).astype(np.float32) / 255
        output_np = output_np[:, :, :3] * output_np[:, :, 3:4]
        pil_image = Image.fromarray((output_np * 255).astype(np.uint8))

        logger.info("Extracting DinoV3 conditioning...")

        # Get 512px conditioning
        model.image_size = 512
        if low_vram:
            model.to(device)
        cond_512 = model([pil_image])

        # Get 1024px conditioning if requested
        cond_1024 = None
        if include_1024:
            model.image_size = 1024
            cond_1024 = model([pil_image])

        if low_vram:
            model.cpu()

        # Create negative conditioning
        neg_cond = torch.zeros_like(cond_512)

        conditioning = {
            'cond_512': cond_512,
            'neg_cond': neg_cond,
        }
        if cond_1024 is not None:
            conditioning['cond_1024'] = cond_1024

        logger.info("DinoV3 conditioning extracted successfully")

        # Clean up intermediate tensors
        gc.collect()
        torch.cuda.empty_cache()

        return (conditioning,)


class Trellis2ImageToShape:
    """Generate 3D shape from conditioning using TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape_pipeline": ("TRELLIS2_SHAPE_PIPELINE", {"tooltip": "Shape generation models from Load Shape Model node"}),
                "conditioning": ("TRELLIS2_CONDITIONING", {"tooltip": "Image conditioning from Get Conditioning node"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Random seed for reproducible generation"}),
                "resolution": (["512", "1024", "1536"], {"default": "1024", "tooltip": "Output mesh resolution. Higher = more detail but slower"}),
                # Sparse Structure Sampler
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Sparse structure CFG scale. Higher = stronger adherence to input image"}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1, "tooltip": "Sparse structure sampling steps. More steps = better quality but slower"}),
                # Shape SLat Sampler
                "shape_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Shape CFG scale. Higher = stronger adherence to input image"}),
                "shape_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1, "tooltip": "Shape sampling steps. More steps = better quality but slower"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "TRELLIS2_SHAPE_SLAT")
    RETURN_NAMES = ("trimesh", "shape_slat")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate 3D geometry from image conditioning.

This node generates shape only (no texture/PBR).
Connect shape_slat output to "Shape to Texture" for PBR materials.

Parameters:
- shape_pipeline: The loaded shape models
- conditioning: DinoV3 conditioning from "Get Conditioning" node
- seed: Random seed for reproducibility
- resolution: Output resolution (512, 1024, or 1536)
- ss_*: Sparse structure sampling parameters
- shape_*: Shape latent sampling parameters

Returns:
- trimesh: The generated 3D mesh geometry
- shape_slat: Shape latent for texture generation
"""

    def generate(
        self,
        shape_pipeline,
        conditioning,
        seed=0,
        resolution="1024",
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        shape_guidance_strength=7.5,
        shape_sampling_steps=12,
    ):
        pipe = shape_pipeline["pipeline"]

        # Determine pipeline type based on resolution
        pipeline_type = {
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution]

        # Build sampler params
        sampler_params = {
            "sparse_structure_sampler_params": {
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
            },
            "shape_slat_sampler_params": {
                "steps": shape_sampling_steps,
                "guidance_strength": shape_guidance_strength,
            },
        }

        logger.info(f"Generating 3D shape (resolution={resolution}, seed={seed})")

        # Run shape generation
        meshes, shape_slat, res = pipe.run_shape(
            conditioning,
            seed=seed,
            pipeline_type=pipeline_type,
            **sampler_params
        )
        mesh = meshes[0]

        # Fill holes in mesh
        mesh.fill_holes()

        logger.info("3D shape generated successfully")

        # Convert to trimesh
        tri_mesh = mesh_to_trimesh(mesh)

        # Pack shape_slat for texture node
        shape_slat_dict = {
            'shape_slat_feats': shape_slat.feats.cpu(),
            'shape_slat_coords': shape_slat.coords.cpu(),
            'resolution': res,
            'pipeline_type': pipeline_type,
        }

        # Clean up GPU tensors
        del shape_slat, meshes, mesh
        gc.collect()
        torch.cuda.empty_cache()

        return (tri_mesh, shape_slat_dict)


class Trellis2ShapeToTexture:
    """Generate PBR textures for existing shape using TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture_pipeline": ("TRELLIS2_TEXTURE_PIPELINE", {"tooltip": "Texture generation models from Load Texture Model node"}),
                "conditioning": ("TRELLIS2_CONDITIONING", {"tooltip": "Image conditioning from Get Conditioning node (same as used for shape)"}),
                "shape_slat": ("TRELLIS2_SHAPE_SLAT", {"tooltip": "Shape latent from Image to Shape node"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Random seed for texture variation"}),
                "tex_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Texture CFG scale. Higher = stronger adherence to input image"}),
                "tex_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1, "tooltip": "Texture sampling steps. More steps = better quality but slower"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "VOXELGRID")
    RETURN_NAMES = ("trimesh", "voxelgrid")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate PBR materials for an existing shape.

Takes shape_slat from "Image to Shape" node and generates:
- base_color (RGB)
- metallic
- roughness
- alpha

Parameters:
- texture_pipeline: The loaded texture models
- conditioning: DinoV3 conditioning (same as used for shape)
- shape_slat: Shape latent from "Image to Shape" node
- seed: Random seed for texture variation
- tex_*: Texture sampling parameters

Returns:
- trimesh: Mesh geometry (same as input shape)
- voxelgrid: VoxelGrid with PBR attributes for texture baking
"""

    def generate(
        self,
        texture_pipeline,
        conditioning,
        shape_slat,
        seed=0,
        tex_guidance_strength=7.5,
        tex_sampling_steps=12,
    ):
        pipe = texture_pipeline["pipeline"]

        from ..trellis2.modules.sparse import SparseTensor

        # Reconstruct shape_slat SparseTensor
        shape_slat_tensor = SparseTensor(
            feats=shape_slat['shape_slat_feats'].cuda(),
            coords=shape_slat['shape_slat_coords'].cuda(),
        )
        resolution = shape_slat['resolution']
        pipeline_type = shape_slat['pipeline_type']

        # Build sampler params
        sampler_params = {
            "tex_slat_sampler_params": {
                "steps": tex_sampling_steps,
                "guidance_strength": tex_guidance_strength,
            },
        }

        logger.info(f"Generating PBR textures (seed={seed})")

        # Run texture generation
        meshes = pipe.run_texture(
            conditioning,
            shape_slat_tensor,
            resolution,
            seed=seed,
            pipeline_type=pipeline_type,
            **sampler_params
        )

        # Clean up shape_slat_tensor immediately after use
        del shape_slat_tensor
        torch.cuda.empty_cache()

        mesh = meshes[0]

        # Simplify mesh (nvdiffrast limit)
        mesh.simplify(16777216)

        # Clear GPU cache after mesh simplification
        torch.cuda.empty_cache()

        logger.info("PBR textures generated successfully")

        # Convert to TRIMESH + VOXELGRID outputs
        tri_mesh, voxel_grid = mesh_with_voxel_to_outputs(mesh, pipe.pbr_attr_layout)

        # Final cleanup
        del meshes, mesh
        gc.collect()
        torch.cuda.empty_cache()

        return (tri_mesh, voxel_grid)


class Trellis2DecodeLatent:
    """Decode latent codes back to mesh (legacy support)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape_pipeline": ("TRELLIS2_SHAPE_PIPELINE",),
                "texture_pipeline": ("TRELLIS2_TEXTURE_PIPELINE",),
                "latent": ("TRELLIS2_LATENT",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "VOXELGRID")
    RETURN_NAMES = ("trimesh", "voxelgrid")
    FUNCTION = "decode"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Decode latent codes back to a 3D mesh with textures.

Useful for regenerating mesh from saved latents.

Returns:
- trimesh: The decoded mesh geometry
- voxelgrid: VoxelGrid with PBR attributes for texture baking
"""

    def decode(self, shape_pipeline, texture_pipeline, latent):
        shape_pipe = shape_pipeline["pipeline"]
        tex_pipe = texture_pipeline["pipeline"]

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

        # Use shape pipeline for decoding (it has the decoders)
        mesh = shape_pipe.decode_latent(shape_slat, tex_slat, res)[0]

        # Clean up latent tensors
        del shape_slat, tex_slat
        torch.cuda.empty_cache()

        mesh.simplify(16777216)

        # Convert to TRIMESH + VOXELGRID outputs
        tri_mesh, voxel_grid = mesh_with_voxel_to_outputs(mesh, shape_pipe.pbr_attr_layout)

        # Final cleanup
        del mesh
        gc.collect()
        torch.cuda.empty_cache()

        return (tri_mesh, voxel_grid)


NODE_CLASS_MAPPINGS = {
    "Trellis2GetConditioning": Trellis2GetConditioning,
    "Trellis2ImageToShape": Trellis2ImageToShape,
    "Trellis2ShapeToTexture": Trellis2ShapeToTexture,
    "Trellis2DecodeLatent": Trellis2DecodeLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2GetConditioning": "TRELLIS.2 Get Conditioning",
    "Trellis2ImageToShape": "TRELLIS.2 Image to Shape",
    "Trellis2ShapeToTexture": "TRELLIS.2 Shape to Texture",
    "Trellis2DecodeLatent": "TRELLIS.2 Decode Latent",
}

"""Inference nodes for TRELLIS.2 Image-to-3D generation."""
import gc
import torch
import numpy as np
from PIL import Image
import trimesh as Trimesh
import cumesh as CuMesh

import comfy.model_management as mm

from .utils import logger, tensor_to_pil


def mesh_to_trimesh(mesh_obj):
    """
    Convert TRELLIS Mesh to trimesh.Trimesh (untextured).

    Used by Image to Shape node to output untextured mesh for preview/export.
    """
    # Unify face orientations using CuMesh
    cumesh = CuMesh.CuMesh()
    cumesh.init(mesh_obj.vertices, mesh_obj.faces.int())
    cumesh.unify_face_orientations()
    unified_verts, unified_faces = cumesh.read()

    vertices = unified_verts.cpu().numpy().astype(np.float32)
    faces = unified_faces.cpu().numpy()
    del cumesh, unified_verts, unified_faces

    # Coordinate system conversion (Y-up to Z-up)
    vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

    return Trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def mesh_with_voxel_to_outputs(mesh_obj, pbr_layout):
    """
    Convert TRELLIS MeshWithVoxel to separate TRIMESH, VOXELGRID, and debug POINTCLOUD.

    Returns:
        trimesh: trimesh.Trimesh geometry for preview/remeshing
        voxelgrid: dict with sparse PBR data on GPU for Rasterize PBR node
        pointcloud: trimesh.PointCloud with all 6 PBR channels on CPU for debugging
    """
    # === TRIMESH OUTPUT ===
    # Unify face orientations using CuMesh (fixes inconsistent winding from dual-grid extraction)
    cumesh = CuMesh.CuMesh()
    cumesh.init(mesh_obj.vertices, mesh_obj.faces.int())
    cumesh.unify_face_orientations()
    unified_verts, unified_faces = cumesh.read()
    logger.info(f"Unified face orientations: {unified_faces.shape[0]} faces")

    vertices = unified_verts.cpu().numpy().astype(np.float32)
    faces = unified_faces.cpu().numpy()
    del cumesh, unified_verts, unified_faces

    # Coordinate system conversion (Y-up to Z-up)
    vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

    tri_mesh = Trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )

    # === VOXELGRID OUTPUT (lightweight, GPU) ===
    voxel_grid = {
        'coords': mesh_obj.coords,       # (L, 3) GPU tensor
        'attrs': mesh_obj.attrs,         # (L, 6) GPU tensor
        'voxel_size': mesh_obj.voxel_size,
        'layout': pbr_layout,
        # Original mesh needed for BVH in Rasterize PBR (maps new verts to voxel field)
        'original_vertices': mesh_obj.vertices,  # GPU tensor
        'original_faces': mesh_obj.faces,        # GPU tensor
    }

    # === POINTCLOUD OUTPUT (CPU, all 6 channels) ===
    coords_np = mesh_obj.coords.cpu().numpy().astype(np.float32)
    attrs_np = mesh_obj.attrs.cpu().numpy()  # (L, 6) in [-1, 1]

    # Random subsample to 5% for visualization performance
    num_points = coords_np.shape[0]
    subsample_ratio = 0.05
    num_keep = int(num_points * subsample_ratio)
    indices = np.random.choice(num_points, size=num_keep, replace=False)
    coords_np = coords_np[indices]
    attrs_np = attrs_np[indices]
    logger.info(f"[DEBUG] Subsampled pointcloud: {num_points} -> {num_keep} points ({subsample_ratio*100:.0f}%)")

    # DEBUG: Print attr statistics
    logger.info(f"[DEBUG] attrs shape: {attrs_np.shape}, dtype: {attrs_np.dtype}")
    logger.info(f"[DEBUG] attrs range: [{attrs_np.min():.3f}, {attrs_np.max():.3f}]")
    for name, slc in pbr_layout.items():
        channel = attrs_np[:, slc]
        logger.info(f"[DEBUG] {name}: min={channel.min():.3f}, max={channel.max():.3f}, mean={channel.mean():.3f}")

    # Check alpha (occupancy) distribution
    alpha_slice = pbr_layout.get('alpha', slice(5, 6))
    alpha_raw = attrs_np[:, alpha_slice].flatten()  # in [-1, 1]
    alpha_norm = (alpha_raw + 1) * 0.5  # convert to [0, 1]

    low_alpha = (alpha_norm < 0.5).sum()
    very_low_alpha = (alpha_norm < 0.1).sum()
    high_alpha = (alpha_norm > 0.9).sum()
    logger.info(f"[DEBUG] Alpha distribution (out of {len(alpha_norm)} voxels):")
    logger.info(f"[DEBUG]   alpha < 0.1 (nearly transparent): {very_low_alpha} ({100*very_low_alpha/len(alpha_norm):.1f}%)")
    logger.info(f"[DEBUG]   alpha < 0.5 (semi-transparent):   {low_alpha} ({100*low_alpha/len(alpha_norm):.1f}%)")
    logger.info(f"[DEBUG]   alpha > 0.9 (nearly opaque):      {high_alpha} ({100*high_alpha/len(alpha_norm):.1f}%)")

    # Convert voxel indices to world positions
    voxel_size = mesh_obj.voxel_size
    point_positions = coords_np * voxel_size

    # Apply same Y-up to Z-up conversion
    point_positions[:, 1], point_positions[:, 2] = (
        point_positions[:, 2].copy(),
        -point_positions[:, 1].copy()
    )

    # Convert attrs from [-1, 1] to [0, 1] for storage
    attrs_normalized = (attrs_np + 1.0) * 0.5  # (L, 6) in [0, 1]

    # For trimesh.PointCloud colors, use base_color RGB + alpha from attrs
    base_color_slice = pbr_layout.get('base_color', slice(0, 3))
    alpha_slice = pbr_layout.get('alpha', slice(5, 6))

    colors_rgb = (attrs_normalized[:, base_color_slice] * 255).clip(0, 255).astype(np.uint8)
    colors_alpha = (attrs_normalized[:, alpha_slice] * 255).clip(0, 255).astype(np.uint8)

    colors_rgba = np.concatenate([colors_rgb, colors_alpha], axis=1)

    pointcloud = Trimesh.PointCloud(
        vertices=point_positions,
        colors=colors_rgba
    )

    # Attach PBR channels as vertex_attributes for field visualization
    pointcloud.vertex_attributes = {}
    for attr_name, attr_slice in pbr_layout.items():
        values = attrs_normalized[:, attr_slice]
        if values.shape[1] == 1:
            # Scalar field (metallic, roughness, alpha)
            pointcloud.vertex_attributes[attr_name] = values[:, 0].astype(np.float32)
        else:
            # Vector field (base_color RGB) - store as separate channels
            pointcloud.vertex_attributes[f'{attr_name}_r'] = values[:, 0].astype(np.float32)
            pointcloud.vertex_attributes[f'{attr_name}_g'] = values[:, 1].astype(np.float32)
            pointcloud.vertex_attributes[f'{attr_name}_b'] = values[:, 2].astype(np.float32)

    # Also keep in metadata for full access
    pointcloud.metadata['pbr_attrs'] = attrs_normalized  # (L, 6) numpy
    pointcloud.metadata['pbr_layout'] = pbr_layout

    logger.info(f"Created outputs: mesh={len(vertices)} verts, voxels={len(coords_np)} points, fields={list(pointcloud.vertex_attributes.keys())}")

    return tri_mesh, voxel_grid, pointcloud


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

    RETURN_TYPES = ("TRELLIS2_SHAPE_SLAT", "TRELLIS2_SUBS", "TRIMESH")
    RETURN_NAMES = ("shape_slat", "subs", "mesh")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate 3D shape from image conditioning.

This node generates shape geometry (no texture/PBR).
Connect shape_slat and subs to "Shape to Textured Mesh" for PBR materials.

Parameters:
- shape_pipeline: The loaded shape models
- conditioning: DinoV3 conditioning from "Get Conditioning" node
- seed: Random seed for reproducibility
- resolution: Output resolution (512, 1024, or 1536)
- ss_*: Sparse structure sampling parameters
- shape_*: Shape latent sampling parameters

Returns:
- shape_slat: Shape latent for texture generation (GPU)
- subs: Substructures for texture generation (GPU)
- mesh: Untextured mesh for preview/export
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

        # Run shape generation (returns mesh, shape_slat, subs, resolution)
        meshes, shape_slat, subs, res = pipe.run_shape(
            conditioning,
            seed=seed,
            pipeline_type=pipeline_type,
            **sampler_params
        )
        mesh = meshes[0]

        # Fill holes in mesh
        mesh.fill_holes()

        logger.info("3D shape generated successfully")

        # Convert to trimesh for preview/export (untextured)
        tri_mesh = mesh_to_trimesh(mesh)
        logger.info(f"Untextured mesh: {len(tri_mesh.vertices)} vertices, {len(tri_mesh.faces)} faces")

        # Pack shape_slat for texture node (stays on GPU)
        shape_slat_dict = {
            'tensor': shape_slat,  # Keep on GPU
            'meshes': meshes,      # Keep on GPU for texture node
            'resolution': res,
            'pipeline_type': pipeline_type,
        }

        # subs stays on GPU as-is (list of SparseTensors)
        # Don't clean up - texture node needs these!

        return (shape_slat_dict, subs, tri_mesh)


class Trellis2ShapeToTexturedMesh:
    """Generate PBR textured mesh from shape using TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture_pipeline": ("TRELLIS2_TEXTURE_PIPELINE", {"tooltip": "Texture generation models from Load Texture Model node"}),
                "conditioning": ("TRELLIS2_CONDITIONING", {"tooltip": "Image conditioning from Get Conditioning node (same as used for shape)"}),
                "shape_slat": ("TRELLIS2_SHAPE_SLAT", {"tooltip": "Shape latent from Image to Shape node"}),
                "subs": ("TRELLIS2_SUBS", {"tooltip": "Substructures from Image to Shape node"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Random seed for texture variation"}),
                "tex_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Texture CFG scale. Higher = stronger adherence to input image"}),
                "tex_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1, "tooltip": "Texture sampling steps. More steps = better quality but slower"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "TRELLIS2_VOXELGRID", "TRIMESH")
    RETURN_NAMES = ("trimesh", "voxelgrid", "pbr_pointcloud")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate PBR textured mesh from shape.

Takes shape_slat and subs from "Image to Shape" node and generates PBR materials:
- base_color (RGB)
- metallic
- roughness
- alpha

This node only runs texture inference (shape decoder is skipped).

Parameters:
- texture_pipeline: The loaded texture models
- conditioning: DinoV3 conditioning (same as used for shape)
- shape_slat: Shape latent from "Image to Shape" node
- subs: Substructures from "Image to Shape" node
- seed: Random seed for texture variation
- tex_*: Texture sampling parameters

Returns:
- trimesh: The 3D mesh with PBR vertex attributes
- voxelgrid: Sparse PBR voxel data (GPU) for Rasterize PBR node
- pbr_pointcloud: Debug point cloud with all 6 PBR channels (CPU)
"""

    def generate(
        self,
        texture_pipeline,
        conditioning,
        shape_slat,
        subs,
        seed=0,
        tex_guidance_strength=7.5,
        tex_sampling_steps=12,
    ):
        # Lazy load texture pipeline (deferred from loader node for optimal VRAM)
        if texture_pipeline["pipeline"] is None:
            from ..trellis2.pipelines import Trellis2ImageTo3DPipeline

            keep_model_loaded = texture_pipeline["keep_model_loaded"]
            device = texture_pipeline["device"]

            logger.info(f"Loading texture models (deferred from loader node)...")
            pipe = Trellis2ImageTo3DPipeline.from_pretrained(
                texture_pipeline["model_name"],
                models_to_load=texture_pipeline["models_to_load"],
                enable_disk_offload=not keep_model_loaded
            )
            pipe.keep_model_loaded = keep_model_loaded
            pipe.default_pipeline_type = texture_pipeline["resolution"]

            # Move to device (only if keeping models loaded)
            if keep_model_loaded:
                if device.type == 'cuda':
                    pipe.cuda()
                else:
                    pipe.to(device)
            else:
                pipe._device = device

            # Store for potential reuse within same workflow execution
            texture_pipeline["pipeline"] = pipe
            logger.info("Texture models loaded successfully")
        else:
            pipe = texture_pipeline["pipeline"]

        # Extract shape data (already on GPU)
        shape_slat_tensor = shape_slat['tensor']
        meshes = shape_slat['meshes']
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

        # Run texture generation with pre-computed subs (skips shape decoder!)
        textured_meshes = pipe.run_texture_with_subs(
            conditioning,
            shape_slat_tensor,
            subs,
            meshes,
            resolution,
            seed=seed,
            pipeline_type=pipeline_type,
            **sampler_params
        )

        mesh = textured_meshes[0]

        # Simplify mesh (nvdiffrast limit)
        mesh.simplify(16777216)

        # Clear GPU cache after mesh simplification
        torch.cuda.empty_cache()

        logger.info("PBR textures generated successfully")

        # Convert to TRIMESH + VOXELGRID + POINTCLOUD
        tri_mesh, voxel_grid, pointcloud = mesh_with_voxel_to_outputs(mesh, pipe.pbr_attr_layout)

        # Cleanup shape data now that we're done
        del shape_slat_tensor, meshes, subs, textured_meshes
        gc.collect()
        torch.cuda.empty_cache()

        return (tri_mesh, voxel_grid, pointcloud)


class Trellis2RemoveBackground:
    """Remove background from image using BiRefNet (TRELLIS rembg)."""

    _model = None  # Class-level cache

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "low_vram": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Remove background from image using BiRefNet (same as TRELLIS rembg).

This node extracts a foreground mask using the BiRefNet segmentation model.
The mask can be used with the "Get Conditioning" node.

Parameters:
- image: Input RGB image
- low_vram: Move model to CPU when not in use (slower but saves VRAM)

Returns:
- image: Original image (unchanged)
- mask: Foreground mask (white=object, black=background)
"""

    def remove_background(self, image, low_vram=True):
        from ..trellis2.pipelines import rembg

        device = mm.get_torch_device()

        # Load or reuse cached model
        if Trellis2RemoveBackground._model is None:
            logger.info("Loading BiRefNet model for background removal...")
            Trellis2RemoveBackground._model = rembg.BiRefNet(model_name="briaai/RMBG-2.0")
            if not low_vram:
                Trellis2RemoveBackground._model.to(device)

        model = Trellis2RemoveBackground._model

        # Convert ComfyUI tensor to PIL
        pil_image = tensor_to_pil(image)

        logger.info("Removing background...")

        if low_vram:
            model.to(device)

        # Run BiRefNet - returns RGBA image
        output = model(pil_image)

        if low_vram:
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        # Extract mask from alpha channel
        output_np = np.array(output)
        mask_np = output_np[:, :, 3].astype(np.float32) / 255.0

        # Convert mask to ComfyUI format (B, H, W)
        mask_tensor = torch.tensor(mask_np).unsqueeze(0)

        logger.info("Background removed successfully")

        # Return original image + mask
        return (image, mask_tensor)


NODE_CLASS_MAPPINGS = {
    "Trellis2RemoveBackground": Trellis2RemoveBackground,
    "Trellis2GetConditioning": Trellis2GetConditioning,
    "Trellis2ImageToShape": Trellis2ImageToShape,
    "Trellis2ShapeToTexturedMesh": Trellis2ShapeToTexturedMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2RemoveBackground": "TRELLIS.2 Remove Background",
    "Trellis2GetConditioning": "TRELLIS.2 Get Conditioning",
    "Trellis2ImageToShape": "TRELLIS.2 Image to Shape",
    "Trellis2ShapeToTexturedMesh": "TRELLIS.2 Shape to Textured Mesh",
}

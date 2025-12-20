"""Modular mesh processing nodes for TRELLIS.2."""
import gc
import os
import torch
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
from pathlib import Path

import folder_paths

from .utils import logger


class Trellis2Simplify:
    """Simplify mesh to target face count using CuMesh."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_face_count": ("INT", {"default": 500000, "min": 1000, "max": 5000000, "step": 1000}),
            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": True}),
                "fill_holes_perimeter": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001}),
                "remesh": ("BOOLEAN", {"default": False}),
                "remesh_band": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "simplify"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Simplify mesh to target face count.

Uses CuMesh for GPU-accelerated simplification.

Parameters:
- target_face_count: Target number of faces
- fill_holes: Fill small holes before simplifying
- fill_holes_perimeter: Max hole perimeter to fill
- remesh: Apply dual-contouring remesh for cleaner topology
- remesh_band: Remesh band width
"""

    def simplify(
        self,
        trimesh,
        target_face_count=500000,
        fill_holes=True,
        fill_holes_perimeter=0.03,
        remesh=False,
        remesh_band=1.0,
    ):
        import cumesh as CuMesh
        import trimesh as Trimesh

        logger.info(f"Simplify: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces -> {target_face_count} target")

        # Convert to torch tensors
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).cuda()

        # Undo coordinate conversion if needed (Z-up back to Y-up)
        vertices_orig = vertices.clone()
        vertices_orig[:, 1], vertices_orig[:, 2] = -vertices[:, 2].clone(), vertices[:, 1].clone()

        # Initialize CuMesh
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices_orig, faces)
        logger.info(f"Initial: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")

        # Fill holes
        if fill_holes:
            cumesh.fill_holes(max_hole_perimeter=fill_holes_perimeter)
            logger.info(f"After fill holes: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")

        # Optional remesh
        if remesh:
            curr_verts, curr_faces = cumesh.read()
            bvh = CuMesh.cuBVH(curr_verts, curr_faces)

            # Estimate grid parameters
            aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda')
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            resolution = 512  # Default resolution for remeshing

            cumesh.init(*CuMesh.remeshing.remesh_narrow_band_dc(
                curr_verts, curr_faces,
                center=center,
                scale=(resolution + 3 * remesh_band) / resolution * scale,
                resolution=resolution,
                band=remesh_band,
                project_back=0.0,
                verbose=True,
                bvh=bvh,
            ))
            logger.info(f"After remesh: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
            # Clean up BVH after remesh
            del bvh, curr_verts, curr_faces

        # Unify face orientations before simplify
        cumesh.unify_face_orientations()
        logger.info("Unified face orientations (pre-simplify)")

        # Simplify
        cumesh.simplify(target_face_count, verbose=True)
        logger.info(f"After simplify: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")

        # Unify face orientations again after simplify (simplify can break it)
        cumesh.unify_face_orientations()
        logger.info("Unified face orientations (post-simplify)")

        # Read result
        out_vertices, out_faces = cumesh.read()
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()

        # Convert back to Z-up
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()

        # Build new trimesh
        result = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            process=False
        )

        logger.info(f"Simplify complete: {len(result.vertices)} vertices, {len(result.faces)} faces")

        # Clean up GPU memory
        del vertices, faces, vertices_orig, out_vertices, out_faces, cumesh
        gc.collect()
        torch.cuda.empty_cache()

        return (result,)


class Trellis2UVUnwrap:
    """UV unwrap mesh using CuMesh/xatlas. No texture baking."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "chart_cone_angle": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 359.9, "step": 1.0}),
                "chart_refine_iterations": ("INT", {"default": 0, "min": 0, "max": 10}),
                "chart_global_iterations": ("INT", {"default": 1, "min": 0, "max": 10}),
                "chart_smooth_strength": ("INT", {"default": 1, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "unwrap"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
UV unwrap mesh using xatlas.

Just creates UVs - no texture baking. Use Rasterize PBR node after this.

Parameters:
- chart_cone_angle: UV chart clustering threshold (degrees)
- chart_refine_iterations: Refine UV charts
- chart_global_iterations: Global UV optimization passes
- chart_smooth_strength: UV smoothing strength

TIP: Simplify mesh first! UV unwrapping 10M faces takes forever.
"""

    def unwrap(
        self,
        trimesh,
        chart_cone_angle=90.0,
        chart_refine_iterations=0,
        chart_global_iterations=1,
        chart_smooth_strength=1,
    ):
        import cumesh as CuMesh
        import trimesh as Trimesh

        logger.info(f"UV Unwrap: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Convert to torch
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).cuda()

        # Undo coord conversion
        vertices_orig = vertices.clone()
        vertices_orig[:, 1], vertices_orig[:, 2] = -vertices[:, 2].clone(), vertices[:, 1].clone()

        chart_cone_angle_rad = np.radians(chart_cone_angle)

        # Initialize CuMesh
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices_orig, faces)

        # UV Unwrap
        logger.info("Unwrapping UVs...")
        out_vertices, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": chart_cone_angle_rad,
                "refine_iterations": chart_refine_iterations,
                "global_iterations": chart_global_iterations,
                "smooth_strength": chart_smooth_strength,
            },
            return_vmaps=True,
            verbose=True,
        )

        out_vertices = out_vertices.cpu().numpy()
        out_faces = out_faces.cpu().numpy()
        out_uvs = out_uvs.cpu().numpy()

        # Compute normals
        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps.cuda()].cpu().numpy()

        # Convert to Z-up
        out_vertices[:, 1], out_vertices[:, 2] = out_vertices[:, 2].copy(), -out_vertices[:, 1].copy()
        out_normals[:, 1], out_normals[:, 2] = out_normals[:, 2].copy(), -out_normals[:, 1].copy()
        out_uvs[:, 1] = 1 - out_uvs[:, 1]

        # Build trimesh with UVs
        result = Trimesh.Trimesh(
            vertices=out_vertices,
            faces=out_faces,
            vertex_normals=out_normals,
            process=False,
        )
        # Attach UVs as visual
        result.visual = Trimesh.visual.TextureVisuals(uv=out_uvs)

        logger.info(f"UV Unwrap complete: {len(result.vertices)} vertices, {len(result.faces)} faces")

        # Clean up GPU memory
        del vertices, faces, vertices_orig, cumesh
        gc.collect()
        torch.cuda.empty_cache()

        return (result,)


class Trellis2RasterizePBR:
    """Rasterize PBR textures from voxel data onto UV-mapped mesh."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "voxelgrid": ("TRELLIS2_VOXELGRID",),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 16384, "step": 512}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "rasterize"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Bake PBR textures from voxel data onto UV-mapped mesh.

Takes a mesh WITH UVs and bakes color/metallic/roughness from the VOXELGRID.

Input mesh MUST have UVs (use UV Unwrap node first).

Parameters:
- texture_size: Resolution of baked textures (512-16384px)
"""

    def rasterize(
        self,
        trimesh,
        voxelgrid,
        texture_size=2048,
    ):
        import cumesh as CuMesh
        import nvdiffrast.torch as dr
        from flex_gemm.ops.grid_sample import grid_sample_3d
        import trimesh as Trimesh

        # Check for UVs
        if not hasattr(trimesh.visual, 'uv') or trimesh.visual.uv is None:
            raise ValueError("Input mesh has no UVs! Use UV Unwrap node first.")

        # Check for voxel data
        if 'attrs' not in voxelgrid:
            raise ValueError("VoxelGrid has no PBR attributes.")

        logger.info(f"Rasterize PBR: {len(trimesh.vertices)} vertices, texture {texture_size}px")

        # Get mesh data
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).cuda()
        uvs = torch.tensor(trimesh.visual.uv, dtype=torch.float32).cuda()

        # NOTE: Don't call unify_face_orientations here - it would break UV mapping!
        # Face orientation should be fixed BEFORE UV unwrapping (in Simplify or earlier)

        # Undo Z-up to Y-up for voxel sampling
        vertices_yup = vertices.clone()
        vertices_yup[:, 1], vertices_yup[:, 2] = -vertices[:, 2].clone(), vertices[:, 1].clone()

        # Get voxel data from dict (already on GPU)
        attr_volume = voxelgrid['attrs'].cuda()
        coords = voxelgrid['coords'].cuda()
        voxel_size = voxelgrid['voxel_size']
        attr_layout = voxelgrid['layout']
        orig_vertices = voxelgrid['original_vertices'].cuda()
        orig_faces = voxelgrid['original_faces'].cuda()

        # DEBUG: Print raw voxel attribute statistics
        logger.info(f"[DEBUG] attr_volume shape: {attr_volume.shape}")
        logger.info(f"[DEBUG] attr_layout: {attr_layout}")
        logger.info(f"[DEBUG] coords shape: {coords.shape}, min: {coords.min(dim=0).values.tolist()}, max: {coords.max(dim=0).values.tolist()}")

        # Check raw attribute ranges
        for name, slc in attr_layout.items():
            channel_data = attr_volume[:, slc]
            logger.info(f"[DEBUG] Raw {name}: min={channel_data.min().item():.4f}, max={channel_data.max().item():.4f}, mean={channel_data.mean().item():.4f}")

        # Count low alpha voxels
        alpha_slc = attr_layout['alpha']
        alpha_raw = attr_volume[:, alpha_slc]
        low_alpha_count = (alpha_raw < 0.5).sum().item()
        zero_alpha_count = (alpha_raw < 0.01).sum().item()
        logger.info(f"[DEBUG] Voxels with alpha < 0.5: {low_alpha_count}/{attr_volume.shape[0]}")
        logger.info(f"[DEBUG] Voxels with alpha < 0.01: {zero_alpha_count}/{attr_volume.shape[0]}")

        # AABB
        aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device='cuda')

        # Grid size
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = torch.tensor([voxel_size] * 3, device='cuda')
            elif isinstance(voxel_size, (list, tuple, np.ndarray)):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device='cuda')
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            grid_size = torch.tensor([1024, 1024, 1024], dtype=torch.int32, device='cuda')
            voxel_size = (aabb[1] - aabb[0]) / grid_size

        # Build BVH from original mesh for accurate attribute lookup
        logger.info("Building BVH...")
        bvh = CuMesh.cuBVH(orig_vertices, orig_faces)

        logger.info("Rasterizing in UV space...")

        # Setup nvdiffrast
        ctx = dr.RasterizeCudaContext()

        # Prepare UVs for rasterization
        uvs_rast = torch.cat([
            uvs * 2 - 1,
            torch.zeros_like(uvs[:, :1]),
            torch.ones_like(uvs[:, :1])
        ], dim=-1).unsqueeze(0)

        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

        # Rasterize in chunks
        chunk_size = 100000
        for i in range(0, faces.shape[0], chunk_size):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, faces[i:i+chunk_size],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i
            rast = torch.where(mask_chunk, rast_chunk, rast)
            # Clean up chunk tensors to prevent memory accumulation
            del rast_chunk, mask_chunk

        # Clean up after rasterization loop
        del ctx, uvs_rast
        torch.cuda.empty_cache()

        mask = rast[0, ..., 3] > 0

        # Interpolate 3D positions
        pos = dr.interpolate(vertices_yup.unsqueeze(0), rast, faces)[0][0]
        valid_pos = pos[mask]

        # Map to original mesh
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

        # Map vertex positions to original mesh for accurate field sampling
        # (UV unwrap splits vertices at seams, need to map back to original positions)
        logger.info("Mapping vertices to original mesh...")
        _, vert_face_id, vert_uvw = bvh.unsigned_distance(vertices_yup, return_uvw=True)
        vert_orig_tris = orig_vertices[orig_faces[vert_face_id.long()]]
        vertices_mapped = (vert_orig_tris * vert_uvw.unsqueeze(-1)).sum(dim=1)

        # Clean up BVH and intermediate tensors
        del bvh, face_id, uvw, orig_tri_verts, vert_face_id, vert_uvw, vert_orig_tris, pos, rast, vertices_yup
        torch.cuda.empty_cache()

        # Sample voxel attributes for texture
        logger.info("Sampling voxel attributes...")
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )

        # Sample PBR attributes at mapped vertex positions for GeometryPack fields mode
        logger.info("Sampling vertex PBR attributes...")
        vertex_pbr_attrs = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((vertices_mapped - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )[0]  # [N_verts, 6]

        logger.info("Building PBR textures...")

        # Clean up voxel sampling intermediates
        del valid_pos, attr_volume, coords, vertices_mapped
        torch.cuda.empty_cache()

        mask_np = mask.cpu().numpy()

        # Extract PBR channels
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # DEBUG: Print texture statistics after sampling (before inpaint)
        logger.info(f"[DEBUG] Texture base_color: min={base_color.min()}, max={base_color.max()}, mean={base_color.mean():.2f}")
        logger.info(f"[DEBUG] Texture metallic: min={metallic.min()}, max={metallic.max()}, mean={metallic.mean():.2f}")
        logger.info(f"[DEBUG] Texture roughness: min={roughness.min()}, max={roughness.max()}, mean={roughness.mean():.2f}")
        logger.info(f"[DEBUG] Texture alpha (before inpaint): min={alpha.min()}, max={alpha.max()}, mean={alpha.mean():.2f}")

        # In-mask statistics (actual texture values, excluding masked-out pixels)
        alpha_in_mask = alpha[mask_np]
        metallic_in_mask = metallic[mask_np]
        roughness_in_mask = roughness[mask_np]
        bc_in_mask = base_color[mask_np]
        logger.info(f"[DEBUG] In-mask base_color: min={bc_in_mask.min()}, max={bc_in_mask.max()}, mean={bc_in_mask.mean():.2f}")
        logger.info(f"[DEBUG] In-mask metallic: min={metallic_in_mask.min()}, max={metallic_in_mask.max()}, mean={metallic_in_mask.mean():.2f}")
        logger.info(f"[DEBUG] In-mask roughness: min={roughness_in_mask.min()}, max={roughness_in_mask.max()}, mean={roughness_in_mask.mean():.2f}")
        logger.info(f"[DEBUG] In-mask alpha: min={alpha_in_mask.min()}, max={alpha_in_mask.max()}, mean={alpha_in_mask.mean():.2f}")
        low_alpha_pixels = np.sum(alpha_in_mask < 200)
        zero_alpha_pixels = np.sum(alpha_in_mask == 0)
        logger.info(f"[DEBUG] Pixels with alpha < 200 (in valid mask): {low_alpha_pixels}/{alpha_in_mask.size}")
        logger.info(f"[DEBUG] Pixels with alpha == 0 (in valid mask): {zero_alpha_pixels}/{alpha_in_mask.size}")

        # Clean up attrs tensor after extraction to CPU
        del attrs, mask
        gc.collect()
        torch.cuda.empty_cache()

        # Inpaint UV seams
        mask_inv = (~mask_np).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

        # DEBUG: Print alpha statistics after inpainting
        logger.info(f"[DEBUG] Texture alpha (after inpaint): min={alpha.min()}, max={alpha.max()}, mean={alpha.mean():.2f}")
        low_alpha_final = np.sum(alpha < 200)
        zero_alpha_final = np.sum(alpha == 0)
        logger.info(f"[DEBUG] Final texture pixels with alpha < 200: {low_alpha_final}/{alpha.size}")
        logger.info(f"[DEBUG] Final texture pixels with alpha == 0: {zero_alpha_final}/{alpha.size}")

        # Create PBR material (double-sided to avoid backface culling issues)
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([
                np.zeros_like(metallic),
                roughness,
                metallic
            ], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='BLEND',
            doubleSided=True,
        )

        # Build result
        result = Trimesh.Trimesh(
            vertices=trimesh.vertices,
            faces=trimesh.faces,
            vertex_normals=trimesh.vertex_normals if hasattr(trimesh, 'vertex_normals') else None,
            process=False,
            visual=Trimesh.visual.TextureVisuals(uv=trimesh.visual.uv, material=material)
        )

        # Attach PBR vertex attributes for GeometryPack fields visualization
        result.vertex_attributes = {}
        for attr_name, attr_slice in attr_layout.items():
            values = vertex_pbr_attrs[:, attr_slice].clamp(0, 1).cpu().numpy()
            if values.shape[1] == 1:
                # Scalar field (metallic, roughness, alpha)
                result.vertex_attributes[attr_name] = values[:, 0].astype(np.float32)
            else:
                # Vector field (base_color RGB) - store as separate channels
                result.vertex_attributes[f'{attr_name}_r'] = values[:, 0].astype(np.float32)
                result.vertex_attributes[f'{attr_name}_g'] = values[:, 1].astype(np.float32)
                result.vertex_attributes[f'{attr_name}_b'] = values[:, 2].astype(np.float32)

        logger.info(f"Rasterize complete: {texture_size}x{texture_size} PBR textures, {len(result.vertex_attributes)} vertex fields")

        # Final cleanup
        del vertices, faces, uvs, orig_vertices, orig_faces, vertex_pbr_attrs
        gc.collect()
        torch.cuda.empty_cache()

        return (result,)


class Trellis2ExportTrimesh:
    """Export trimesh to file (GLB, OBJ, PLY, etc.)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "trellis2"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"], {"default": "glb"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "TRELLIS2"
    OUTPUT_NODE = True
    DESCRIPTION = """
Export trimesh to various 3D file formats.

Supports: GLB, OBJ, PLY, STL, 3MF, DAE
"""

    def export(self, trimesh, filename_prefix="trellis2", file_format="glb"):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.{file_format}"

        output_dir = folder_paths.get_output_directory()
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(exist_ok=True)

        trimesh.export(str(output_path), file_type=file_format)

        logger.info(f"Exported to: {output_path}")

        return (str(output_path),)


NODE_CLASS_MAPPINGS = {
    "Trellis2Simplify": Trellis2Simplify,
    "Trellis2UVUnwrap": Trellis2UVUnwrap,
    "Trellis2RasterizePBR": Trellis2RasterizePBR,
    "Trellis2ExportTrimesh": Trellis2ExportTrimesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2Simplify": "TRELLIS.2 Simplify Mesh",
    "Trellis2UVUnwrap": "TRELLIS.2 UV Unwrap",
    "Trellis2RasterizePBR": "TRELLIS.2 Rasterize PBR",
    "Trellis2ExportTrimesh": "TRELLIS.2 Export Trimesh",
}

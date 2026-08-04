"""Modular mesh processing nodes for TRELLIS.2."""
import gc
import os
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .utils import logger
import comfy.model_management


def _log_vram(label):
    import sys
    import torch
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        msg = f"[VRAM] {label}: alloc={alloc:.0f}MB reserved={reserved:.0f}MB peak={peak:.0f}MB"
        logger.info(msg)
        print(msg, file=sys.stderr, flush=True)



class Trellis2Simplify(io.ComfyNode):
    """Simplify mesh to target face count using CuMesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2Simplify",
            display_name="TRELLIS.2 Simplify Mesh",
            category="TRELLIS2",
            description="""Simplify mesh to target face count.

Uses CuMesh for GPU-accelerated simplification.

Parameters:
- target_face_count: Target number of faces
- fill_holes: Fill small holes before simplifying
- fill_holes_perimeter: Max hole perimeter to fill
- remesh: Apply dual-contouring remesh for cleaner topology
- remesh_band: Remesh band width
- project_back: How strongly remeshed vertices snap back to the original surface (0.0 = smooth, 1.0 = sharp)
""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("target_face_count", default=500000, min=1000, max=5000000, step=1000),
                io.Boolean.Input("fill_holes", default=True, optional=True),
                io.Float.Input("fill_holes_perimeter", default=0.03, min=0.001, max=0.5, step=0.001, optional=True),
                io.Boolean.Input("remesh", default=False, optional=True),
                io.Float.Input("remesh_band", default=1.0, min=0.1, max=5.0, step=0.1, optional=True),
                io.Float.Input("project_back", default=0.0, min=0.0, max=1.0, step=0.05, optional=True,
                               tooltip="How strongly remeshed vertices snap back to the original surface (0.0 = smooth, 1.0 = sharp)."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        target_face_count=500000,
        fill_holes=True,
        fill_holes_perimeter=0.03,
        remesh=False,
        remesh_band=1.0,
        project_back=0.0,
    ):
        import torch
        import cumesh_vb as CuMesh
        import trimesh as Trimesh

        logger.info(f"Simplify: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces -> {target_face_count} target")

        comfy.model_management.throw_exception_if_processing_interrupted()

        device = comfy.model_management.get_torch_device()

        torch.cuda.reset_peak_memory_stats()
        _log_vram("Simplify Start")

        # Convert to torch tensors (TRIMESH is already in internal Z-up)
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).to(device)

        # Initialize CuMesh
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        logger.info(f"Initial: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
        _log_vram("After CuMesh.init")

        # Fill holes
        if fill_holes:
            cumesh.fill_holes(max_hole_perimeter=fill_holes_perimeter)
            logger.info(f"After fill holes: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
            _log_vram("After fill_holes")

        # Optional remesh
        if remesh:
            curr_verts, curr_faces = cumesh.read()
            bvh = CuMesh.cuBVH(curr_verts, curr_faces)
            _log_vram("After BVH build")

            # Estimate grid parameters
            aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device=device)
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            resolution = 512  # Default resolution for remeshing

            cumesh.init(*CuMesh.remeshing.remesh_narrow_band_dc(
                curr_verts, curr_faces,
                center=center,
                scale=(resolution + 3 * remesh_band) / resolution * scale,
                resolution=resolution,
                band=remesh_band,
                project_back=project_back,
                verbose=True,
                bvh=bvh,
            ))
            logger.info(f"After remesh: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
            _log_vram("After DC remesh")
            # Clean up BVH after remesh
            del bvh, curr_verts, curr_faces
            _log_vram("After BVH cleanup")

        # Unify face orientations before simplify
        cumesh.unify_face_orientations()
        logger.info("Unified face orientations (pre-simplify)")
        _log_vram("After unify (pre-simplify)")

        # Simplify
        cumesh.simplify(target_face_count, verbose=True)
        logger.info(f"After simplify: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
        _log_vram("After simplify")

        # Unify face orientations again after simplify (simplify can break it)
        cumesh.unify_face_orientations()
        logger.info("Unified face orientations (post-simplify)")
        _log_vram("After unify (post-simplify)")

        # Read result
        out_vertices, out_faces = cumesh.read()
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()

        # Build new trimesh (stays in internal Z-up)
        result = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            process=False
        )

        logger.info(f"Simplify complete: {len(result.vertices)} vertices, {len(result.faces)} faces")

        # Clean up GPU memory
        del vertices, faces, out_vertices, out_faces, cumesh
        gc.collect()
        comfy.model_management.soft_empty_cache()
        _log_vram("After cleanup")

        return io.NodeOutput(result)


class Trellis2UVUnwrap(io.ComfyNode):
    """UV unwrap mesh using CuMesh/xatlas. No texture baking."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2UVUnwrap",
            display_name="TRELLIS.2 UV Unwrap",
            category="TRELLIS2",
            description="""UV unwrap mesh using xatlas.

Just creates UVs - no texture baking. Use Rasterize PBR node after this.

Parameters:
- chart_cone_angle: UV chart clustering threshold (degrees)
- chart_refine_iterations: Refine UV charts
- chart_global_iterations: Global UV optimization passes
- chart_smooth_strength: UV smoothing strength

TIP: Simplify mesh first! UV unwrapping 10M faces takes forever.""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("chart_cone_angle", default=90.0, min=0.0, max=359.9, step=1.0, optional=True),
                io.Int.Input("chart_refine_iterations", default=0, min=0, max=10, optional=True),
                io.Int.Input("chart_global_iterations", default=1, min=0, max=10, optional=True),
                io.Int.Input("chart_smooth_strength", default=1, min=0, max=10, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        chart_cone_angle=90.0,
        chart_refine_iterations=0,
        chart_global_iterations=1,
        chart_smooth_strength=1,
    ):
        import torch
        import cumesh_vb as CuMesh
        import trimesh as Trimesh

        logger.info(f"UV Unwrap: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        comfy.model_management.throw_exception_if_processing_interrupted()

        device = comfy.model_management.get_torch_device()

        # Convert to torch
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).to(device)

        chart_cone_angle_rad = np.radians(chart_cone_angle)

        # Initialize CuMesh (TRIMESH is already in internal Z-up)
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)

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
        out_normals = cumesh.read_vertex_normals()[out_vmaps.to(device)].cpu().numpy()

        # Build trimesh with UVs (stays in internal Z-up)
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
        del vertices, faces, cumesh
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(result)


class Trellis2ProcessMesh(io.ComfyNode):
    """Combined mesh processing: fill holes, remesh, simplify, cleanup, UV unwrap."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ProcessMesh",
            display_name="TRELLIS.2 Process Mesh",
            category="TRELLIS2",
            description="""All-in-one mesh processing pipeline.

Combines fill holes -> remesh -> simplify -> cleanup -> UV unwrap in a single
CuMesh session for efficiency.

Output mesh has UVs and normals ready for Rasterize PBR.""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                # 1. Remesh or simplify
                io.DynamicCombo.Input("remesh", options=[
                    io.DynamicCombo.Option("off", [
                        io.Boolean.Input("fill_holes", default=True),
                        io.Float.Input("fill_holes_perimeter", default=0.03, min=0.001, max=0.5, step=0.001),
                    ]),
                    io.DynamicCombo.Option("on", [
                        io.Float.Input("remesh_band", default=1.0, min=0.1, max=5.0, step=0.1),
                        io.Boolean.Input("remove_inner_faces", default=False),
                    ]),
                ]),
                # 3. Floater removal
                io.Float.Input("floater_threshold", default=1e-3, min=0.0, max=0.1, step=0.001, optional=True,
                    tooltip="Area threshold for removing small disconnected components. 0 = disabled."),
                # 4. Simplification
                io.Int.Input("target_face_count", default=500000, min=1000, max=5000000, step=1000),
                # 5. Weld vertices
                io.Boolean.Input("weld_vertices", default=True, optional=True),
                io.Int.Input("weld_digits", default=4, min=1, max=8, optional=True),
                # 6. UV unwrap
                io.Float.Input("chart_cone_angle", default=90.0, min=0.0, max=359.9, step=1.0, optional=True),
                io.Int.Input("chart_refine_iterations", default=0, min=0, max=10, optional=True),
                io.Int.Input("chart_global_iterations", default=1, min=0, max=10, optional=True),
                io.Int.Input("chart_smooth_strength", default=1, min=0, max=10, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        remesh=None,
        floater_threshold=1e-3,
        target_face_count=500000,
        weld_vertices=True,
        weld_digits=4,
        chart_cone_angle=90.0,
        chart_refine_iterations=0,
        chart_global_iterations=1,
        chart_smooth_strength=1,
    ):
        import torch
        import cumesh_vb as CuMesh
        import trimesh as Trimesh

        # Extract remesh toggle parameters
        if remesh is None:
            remesh = {"remesh": "off"}
        do_remesh = remesh.get("remesh", "off") == "on"
        remesh_band = remesh.get("remesh_band", 1.0)
        remove_inner_faces = remesh.get("remove_inner_faces", False)
        fill_holes = remesh.get("fill_holes", True)
        fill_holes_perimeter = remesh.get("fill_holes_perimeter", 0.03)

        mode_str = "remesh" if do_remesh else "simplify"
        logger.info(f"ProcessMesh ({mode_str}): {len(trimesh.vertices)} verts, {len(trimesh.faces)} faces -> target {target_face_count}")

        comfy.model_management.throw_exception_if_processing_interrupted()
        device = comfy.model_management.get_torch_device()
        torch.cuda.reset_peak_memory_stats()
        _log_vram("ProcessMesh Start")

        # TRIMESH is already in internal Z-up
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).to(device)

        import sys
        _ma = torch.cuda.memory_allocated
        _v = lambda: _ma() // 1048576
        def _print(msg):
            print(f"[ProcessMesh] {msg} (alloc={_v()}MB)", file=sys.stderr, flush=True)

        _print(f"Input: {vertices.shape[0]} verts, {faces.shape[0]} faces")

        # 1. Init cumesh
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices, faces)
        del vertices, faces

        # 2. Remesh first (reduces 20M+ faces to ~500K) — do this BEFORE
        #    floater removal so we don't OOM building adjacency on the full mesh
        if do_remesh:
            curr_verts, curr_faces = cumesh.read()

            aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device=device)
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            resolution = 512

            _print(f"Remeshing (quad DC, resolution={resolution}, band={remesh_band}, remove_inner_faces={remove_inner_faces})...")
            cumesh.init(*CuMesh.remeshing.remesh_narrow_band_dc_quad(
                curr_verts, curr_faces,
                center=center,
                scale=scale * 1.1,
                resolution=resolution,
                band=remesh_band,
                project_back=0.0,
                verbose=True,
                remove_inner_faces=remove_inner_faces,
            ))
            _print(f"After remesh: {cumesh.num_vertices} verts, {cumesh.num_faces} faces")
            del curr_verts, curr_faces

        # 3. Remove floaters
        if floater_threshold > 0:
            _print(f"Removing floaters (area_threshold={floater_threshold})...")
            cumesh.remove_small_connected_components(floater_threshold)
            _print(f"After floater removal: {cumesh.num_vertices} verts, {cumesh.num_faces} faces")

        comfy.model_management.throw_exception_if_processing_interrupted()

        # 4. Cleanup + simplify (matches original to_glb pipeline)
        if not do_remesh:
            # Branch A: 2-pass simplify (original pattern)
            _print("Cleaning up mesh...")
            cumesh.remove_duplicate_faces()
            cumesh.repair_non_manifold_edges()
            if floater_threshold > 0:
                cumesh.remove_small_connected_components(floater_threshold)
            if fill_holes:
                cumesh.fill_holes(max_hole_perimeter=fill_holes_perimeter)

            _print(f"Simplifying to {target_face_count * 3} faces (pass 1/2)...")
            cumesh.simplify(target_face_count * 3, verbose=True)
            _print(f"After pass 1: {cumesh.num_vertices} verts, {cumesh.num_faces} faces")

            cumesh.remove_duplicate_faces()
            cumesh.repair_non_manifold_edges()
            if floater_threshold > 0:
                cumesh.remove_small_connected_components(floater_threshold)
            if fill_holes:
                cumesh.fill_holes(max_hole_perimeter=fill_holes_perimeter)

            _print(f"Simplifying to {target_face_count} faces (pass 2/2)...")
            cumesh.simplify(target_face_count, verbose=True)
            _print(f"After pass 2: {cumesh.num_vertices} verts, {cumesh.num_faces} faces")

            cumesh.remove_duplicate_faces()
            cumesh.repair_non_manifold_edges()
            if floater_threshold > 0:
                cumesh.remove_small_connected_components(floater_threshold)
            if fill_holes:
                cumesh.fill_holes(max_hole_perimeter=fill_holes_perimeter)
            cumesh.unify_face_orientations()
        else:
            # Branch B: Single simplify after remesh
            _print(f"Simplifying to {target_face_count} faces...")
            cumesh.simplify(target_face_count, verbose=True)
            _print(f"After simplify: {cumesh.num_vertices} verts, {cumesh.num_faces} faces")

        comfy.model_management.throw_exception_if_processing_interrupted()

        # 5. Weld vertices (before UV unwrap — welding after UV corrupts seams)
        if weld_vertices:
            _print(f"Welding vertices (digits={weld_digits})...")
            pre_verts = cumesh.num_vertices
            weld_verts, weld_faces = cumesh.read()
            weld_mesh = Trimesh.Trimesh(
                vertices=weld_verts.cpu().numpy(),
                faces=weld_faces.cpu().numpy(),
                process=False,
            )
            weld_mesh.merge_vertices(digits_vertex=weld_digits)
            weld_mesh.remove_unreferenced_vertices()
            weld_mesh.update_faces(weld_mesh.nondegenerate_faces())
            cumesh.init(
                torch.tensor(weld_mesh.vertices, dtype=torch.float32).to(device),
                torch.tensor(weld_mesh.faces, dtype=torch.int32).to(device),
            )
            _print(f"After weld: {pre_verts} -> {cumesh.num_vertices} verts, {cumesh.num_faces} faces")
            del weld_verts, weld_faces, weld_mesh

        # 6. UV unwrap (last step — produces split vertices at seams)
        _print("UV unwrapping...")
        chart_cone_angle_rad = np.radians(chart_cone_angle)
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

        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps.to(device)].cpu().numpy()
        _print(f"After UV unwrap: {out_vertices.shape[0]} verts, {out_faces.shape[0]} faces")

        # Output stays in internal Z-up (conversion to Y-up happens only at export)
        result = Trimesh.Trimesh(
            vertices=out_vertices,
            faces=out_faces,
            vertex_normals=out_normals,
            process=False,
        )
        result.visual = Trimesh.visual.TextureVisuals(uv=out_uvs)

        _print(f"Done: {len(result.vertices)} verts, {len(result.faces)} faces")

        del cumesh, out_vmaps
        gc.collect()
        comfy.model_management.soft_empty_cache()
        _log_vram("ProcessMesh done")

        return io.NodeOutput(result)


class Trellis2RasterizePBR(io.ComfyNode):
    """Rasterize PBR textures from voxel data onto UV-mapped mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2RasterizePBR",
            display_name="TRELLIS.2 Rasterize PBR",
            category="TRELLIS2",
            description="""Bake PBR textures from voxel data onto UV-mapped mesh.

Takes a mesh WITH UVs and bakes color/metallic/roughness from the VOXELGRID.

Input mesh MUST have UVs (use UV Unwrap node first).

Parameters:
- texture_size: Resolution of baked textures (512-16384px)
- pixel_expansion: Base Color map padding/dilation to prevent seams (0-64px)
- pbr_pixel_expansion: Metallic, Roughness, and Alpha map padding/dilation to prevent seams (0-64px)
- double_sided: Enable double-sided material rendering (prevents transparent backfaces)
- force_opaque: Force the texture to be opaque. Eliminates transparent pixel gaps and reduces file size.
""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Custom("TRELLIS2_VOXELGRID").Input("voxelgrid"),
                io.Int.Input("texture_size", default=2048, min=512, max=16384, step=512),
                io.Custom("TRIMESH").Input("original_mesh", optional=True,
                    tooltip="Original mesh (pre-simplification) for BVH projection. Improves texture accuracy by projecting texel positions back to the original surface."),
                io.Int.Input("pixel_expansion", default=3, min=0, max=64, step=1, optional=True,
                             tooltip="Pixel expansion (inpaint radius) for the Base Color map to prevent seam bleeding."),
                io.Int.Input("pbr_pixel_expansion", default=1, min=0, max=64, step=1, optional=True,
                             tooltip="Pixel expansion (inpaint radius) for Metallic, Roughness, and Alpha maps."),
                io.Boolean.Input("double_sided", default=True, optional=True,
                                 tooltip="Enable double-sided material rendering (prevents transparent backfaces)."),
                io.Boolean.Input("base_force_opaque", default=False, optional=True,
                                 tooltip="Strip alpha channel from the base_color texture if it exists."),
                io.Combo.Input("expansion_mode", options=["hybrid", "fast_dilate", "cv2_inpaint"], default="hybrid", optional=True,
                               tooltip="Seam prevention algorithm: 'hybrid' is fast and smooth (combines dilation and inpainting), 'fast_dilate' is near-instant, 'cv2_inpaint' is smooth but very slow."),
                io.Combo.Input("compression_mode", options=["WEBP", "PNG", "JPEG"], default="WEBP", optional=True,
                               tooltip="Texture compression format inside GLB: 'WEBP' supports transparency and is highly compressed, 'PNG' is lossless and supports transparency, 'JPEG' does NOT support alpha/transparency."),
                io.Int.Input("texture_quality", default=90, min=10, max=100, step=1, optional=True,
                             tooltip="Compression quality level (10-100). 100 is fully lossless."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
                io.Image.Output(display_name="base_color"),
                io.Image.Output(display_name="normal_map"),
                io.Image.Output(display_name="metallic_roughness"),
                io.Image.Output(display_name="alpha_map"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        voxelgrid,
        texture_size=2048,
        original_mesh=None,
        pixel_expansion=3,
        pbr_pixel_expansion=1,
        double_sided=True,
        base_force_opaque=False,
        expansion_mode="hybrid",
        compression_mode="WEBP",
        texture_quality=90,
    ):
        import torch
        import cv2
        import cumesh_vb as CuMesh
        from flex_gemm_ap.ops.grid_sample import grid_sample_3d
        import trimesh as Trimesh

        # Check for UVs
        if not hasattr(trimesh.visual, 'uv') or trimesh.visual.uv is None:
            raise ValueError("Input mesh has no UVs! Use UV Unwrap node first.")

        # Check for voxel data
        if 'attrs' not in voxelgrid:
            raise ValueError("VoxelGrid has no PBR attributes.")

        logger.info(f"Rasterize PBR: {len(trimesh.vertices)} vertices, texture {texture_size}px")

        comfy.model_management.throw_exception_if_processing_interrupted()

        device = comfy.model_management.get_torch_device()

        # Get mesh data
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).to(device)
        uvs = torch.tensor(trimesh.visual.uv, dtype=torch.float32).to(device)

        # Get voxel data from dict
        attr_volume = voxelgrid['attrs']
        if isinstance(attr_volume, np.ndarray):
            attr_volume = torch.from_numpy(attr_volume)
        attr_volume = attr_volume.to(device)

        coords = voxelgrid['coords']
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
        coords = coords.to(device)

        voxel_size = voxelgrid['voxel_size']
        # AABB
        aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device=device)

        # Grid size
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = torch.tensor([voxel_size] * 3, device=device)
            elif isinstance(voxel_size, (list, tuple, np.ndarray)):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=device)
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            grid_size = torch.tensor([1024, 1024, 1024], dtype=torch.int32, device=device)
            voxel_size = (aabb[1] - aabb[0]) / grid_size

        mask, valid_pos = _rasterize_uv(vertices, faces, uvs, texture_size, device)

        # BVH projection: snap texel positions back to original mesh surface
        if original_mesh is not None:
            import cumesh_vb as CuMesh
            orig_verts = torch.tensor(original_mesh.vertices, dtype=torch.float32).to(device)
            orig_faces = torch.tensor(original_mesh.faces, dtype=torch.int32).to(device)
            bvh = CuMesh.cuBVH(orig_verts, orig_faces)
            _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
            orig_tri_verts = orig_verts[orig_faces[face_id.long()]]
            valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
            del bvh, orig_verts, orig_faces, face_id, uvw, orig_tri_verts

        comfy.model_management.soft_empty_cache()

        # Sample voxel attributes for texture pixels
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )

        # Sample PBR attributes at vertex positions (already in internal Z-up)
        logger.info("Sampling vertex PBR attributes...")
        vertex_pbr_attrs = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((vertices - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )[0]

        logger.info("Building PBR textures...")

        del valid_pos, attr_volume, coords
        comfy.model_management.soft_empty_cache()

        mask_np = mask.cpu().numpy()

        # Extract PBR channels
        base_color = np.clip(attrs[..., 0:3].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., 3:4].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., 4:5].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., 5:6].cpu().numpy() * 255, 0, 255).astype(np.uint8)

        del attrs, mask
        gc.collect()
        comfy.model_management.soft_empty_cache()

        # Inpaint UV seams (pixel expansion)
        mask_inv = (~mask_np).astype(np.uint8)

        def _fast_dilate_fill(img, mask_i, iterations):
            dilated = img.copy()
            kernel = np.ones((3, 3), np.uint8)
            is_3d = img.ndim == 3
            mask_cond = mask_i[:, :, np.newaxis] > 0 if is_3d else mask_i > 0
            for _ in range(iterations):
                dilated_step = cv2.dilate(dilated, kernel)
                dilated = np.where(mask_cond, dilated_step, dilated)
            return dilated

        def _apply_expansion(img, mask_i, radius, is_pbr=False):
            if radius <= 0:
                return img
            
            # Extract channel if 3D single-channel
            if is_pbr and img.ndim == 3 and img.shape[-1] == 1:
                img_2d = img[..., 0]
                use_pbr_wrap = True
            else:
                img_2d = img
                use_pbr_wrap = False

            if expansion_mode == "fast_dilate":
                res = _fast_dilate_fill(img_2d, mask_i, radius)
            elif expansion_mode == "cv2_inpaint":
                res = cv2.inpaint(img_2d, mask_i, radius, cv2.INPAINT_TELEA)
            else:  # hybrid
                if radius <= 4:
                    res = cv2.inpaint(img_2d, mask_i, radius, cv2.INPAINT_TELEA)
                else:
                    res = _fast_dilate_fill(img_2d, mask_i, radius)

            if use_pbr_wrap:
                return res[..., None]
            return res

        base_color = _apply_expansion(base_color, mask_inv, pixel_expansion)
        metallic = _apply_expansion(metallic, mask_inv, pbr_pixel_expansion, is_pbr=True)
        roughness = _apply_expansion(roughness, mask_inv, pbr_pixel_expansion, is_pbr=True)
        alpha = _apply_expansion(alpha, mask_inv, pbr_pixel_expansion, is_pbr=True)

        # Compress textures for GLB
        def _compress_texture(pil_img, format_mode, q):
            if format_mode == "PNG" or q >= 100:
                return pil_img
            import io
            out = io.BytesIO()
            if format_mode == "JPEG":
                pil_img.convert('RGB').save(out, format='JPEG', quality=q)
            else: # WEBP
                pil_img.save(out, format='WEBP', quality=q)
            out.seek(0)
            return Image.open(out)

        # Create PBR material
        base_texture = Image.fromarray(base_color) if base_force_opaque else Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        base_texture = _compress_texture(base_texture, compression_mode, texture_quality)

        mr_texture = Image.fromarray(np.concatenate([
            np.zeros_like(metallic),
            roughness,
            metallic
        ], axis=-1))
        mr_texture = _compress_texture(mr_texture, compression_mode, texture_quality)

        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=base_texture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=mr_texture,
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE' if base_force_opaque else 'BLEND',
            doubleSided=double_sided,
        )

        # Build result
        result = Trimesh.Trimesh(
            vertices=trimesh.vertices,
            faces=trimesh.faces,
            vertex_normals=trimesh.vertex_normals if hasattr(trimesh, 'vertex_normals') else None,
            process=False,
            visual=Trimesh.visual.TextureVisuals(uv=trimesh.visual.uv, material=material)
        )

        # Attach PBR vertex attributes
        result.vertex_attributes = {}
        for attr_name, attr_slice in [('base_color', slice(0,3)), ('metallic', slice(3,4)), ('roughness', slice(4,5)), ('alpha', slice(5,6))]:
            values = vertex_pbr_attrs[:, attr_slice].clamp(0, 1).cpu().numpy()
            if values.shape[1] == 1:
                result.vertex_attributes[attr_name] = values[:, 0].astype(np.float32)
            else:
                result.vertex_attributes[f'{attr_name}_r'] = values[:, 0].astype(np.float32)
                result.vertex_attributes[f'{attr_name}_g'] = values[:, 1].astype(np.float32)
                result.vertex_attributes[f'{attr_name}_b'] = values[:, 2].astype(np.float32)

        logger.info(f"Rasterize complete: {texture_size}x{texture_size} PBR textures")

        # Convert PBR textures to ComfyUI IMAGE tensors
        from .image_utils import numpy_to_comfy
        base_color_tensor = numpy_to_comfy(base_color)

        flat_normal = np.full((texture_size, texture_size, 3), [128, 128, 255], dtype=np.uint8)
        normal_map_tensor = numpy_to_comfy(flat_normal)

        metallic_roughness_np = np.concatenate([
            np.zeros_like(metallic),
            roughness,
            metallic
        ], axis=-1)
        metallic_roughness_tensor = numpy_to_comfy(metallic_roughness_np)

        alpha_3ch = np.concatenate([alpha, alpha, alpha], axis=-1)
        alpha_map_tensor = numpy_to_comfy(alpha_3ch)

        del vertices, faces, uvs, vertex_pbr_attrs
        gc.collect()
        comfy.model_management.soft_empty_cache()

        # Debug: log trimesh output attributes
        import sys
        print(f"[RasterizePBR] Output trimesh:", file=sys.stderr, flush=True)
        print(f"  vertices: {result.vertices.shape}", file=sys.stderr, flush=True)
        print(f"  faces: {result.faces.shape}", file=sys.stderr, flush=True)
        print(f"  visual type: {type(result.visual).__name__}", file=sys.stderr, flush=True)
        if hasattr(result.visual, 'uv') and result.visual.uv is not None:
            print(f"  UVs: {result.visual.uv.shape}", file=sys.stderr, flush=True)
        else:
            print(f"  UVs: None", file=sys.stderr, flush=True)
        if hasattr(result.visual, 'material') and result.visual.material is not None:
            mat = result.visual.material
            print(f"  material type: {type(mat).__name__}", file=sys.stderr, flush=True)
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                print(f"  baseColorTexture: {mat.baseColorTexture.size} mode={mat.baseColorTexture.mode}", file=sys.stderr, flush=True)
            else:
                print(f"  baseColorTexture: None", file=sys.stderr, flush=True)
            if hasattr(mat, 'metallicRoughnessTexture') and mat.metallicRoughnessTexture is not None:
                print(f"  metallicRoughnessTexture: {mat.metallicRoughnessTexture.size} mode={mat.metallicRoughnessTexture.mode}", file=sys.stderr, flush=True)
            else:
                print(f"  metallicRoughnessTexture: None", file=sys.stderr, flush=True)
        else:
            print(f"  material: None", file=sys.stderr, flush=True)
        if hasattr(result, 'vertex_normals') and result.vertex_normals is not None:
            print(f"  vertex_normals: {result.vertex_normals.shape}", file=sys.stderr, flush=True)
        else:
            print(f"  vertex_normals: None", file=sys.stderr, flush=True)
        print(f"  vertex_attributes: {list(result.vertex_attributes.keys()) if hasattr(result, 'vertex_attributes') else 'None'}", file=sys.stderr, flush=True)

        return io.NodeOutput(result, base_color_tensor, normal_map_tensor, metallic_roughness_tensor, alpha_map_tensor)


def remesh_narrow_band_dc_lowmem(
    vertices, faces, center, scale, resolution,
    band=1, project_back=0, verbose=False, bvh=None,
    topo_chunk=500_000, tri_chunk=500_000,
    remove_inner_faces=False,
):
    """Low-memory version of cumesh_vb.remeshing.remesh_narrow_band_dc.

    Same algorithm but chunks the topology generation and triangle splitting
    steps to avoid materializing huge intermediate tensors.
    """
    import torch
    from cumesh_vb import _C
    from cumesh_vb.bvh import cuBVH
    from cumesh_vb.remeshing import _init_hashmap
    from tqdm import tqdm

    device = vertices.device

    # --- Constants ---
    edge_neighbor_voxel_offset = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ], dtype=torch.int32, device=device).unsqueeze(0)  # (1, 3, 4, 3)

    quad_split_1_n = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device)
    quad_split_1_p = torch.tensor([0, 2, 1, 0, 3, 2], dtype=torch.long, device=device)
    quad_split_2_n = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device)
    quad_split_2_p = torch.tensor([0, 3, 1, 3, 2, 1], dtype=torch.long, device=device)

    OFFSETS = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.int32, device=device)

    # --- 1. Build BVH ---
    if bvh is None:
        if verbose:
            print("Building BVH...")
        bvh = cuBVH(vertices, faces)

    eps = band * scale / resolution

    # --- 2. Sparse Grid Construction ---
    base_resolution = resolution
    while base_resolution > 32:
        assert base_resolution % 2 == 0
        base_resolution //= 2

    coords = torch.stack(torch.meshgrid(
        torch.arange(base_resolution, device=device),
        torch.arange(base_resolution, device=device),
        torch.arange(base_resolution, device=device),
        indexing='ij',
    ), dim=-1).int().reshape(-1, 3)

    pbar = tqdm(
        total=int(torch.log2(torch.tensor(resolution // base_resolution)).item()) + 1,
        desc="Building Sparse Grid", disable=not verbose,
    )

    while True:
        cell_size = scale / base_resolution
        pts = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center
        distances = bvh.unsigned_distance(pts)[0]
        distances -= eps
        distances = torch.abs(distances)
        subdiv_mask = distances < 0.87 * cell_size
        coords = coords[subdiv_mask]
        if base_resolution >= resolution:
            break
        base_resolution *= 2
        coords *= 2
        coords = (coords.unsqueeze(1) + OFFSETS.unsqueeze(0)).reshape(-1, 3)
        pbar.update(1)

    Nvox = coords.shape[0]
    if verbose:
        print(f"Sparse grid: {Nvox:,} voxels")

    # --- 3. Hashmaps + DC vertices ---
    hashmap_vox = _init_hashmap(resolution, 2 * Nvox, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap_vox,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1),
        resolution, resolution, resolution,
    )

    coords = coords.contiguous()
    grid_verts = _C.get_sparse_voxel_grid_active_vertices(
        *hashmap_vox, coords, resolution, resolution, resolution,
    )
    Nvert = grid_verts.shape[0]

    pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center
    distances_vert = bvh.unsigned_distance(pts_vert)[0]
    distances_vert -= eps

    pbar.update(1)
    pbar.close()

    if verbose:
        print("Running Dual Contouring...")

    hashmap_vert = _init_hashmap(resolution + 1, 2 * Nvert, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap_vert,
        torch.cat([torch.zeros_like(grid_verts[:, :1]), grid_verts], dim=1),
        resolution + 1, resolution + 1, resolution + 1,
    )

    dual_verts, intersected = _C.simple_dual_contour(
        *hashmap_vert, coords, distances_vert,
        resolution + 1, resolution + 1, resolution + 1,
    )

    # Free hashmap_vert — no longer needed
    del hashmap_vert, grid_verts, distances_vert, pts_vert
    torch.cuda.empty_cache()

    # --- 4. Chunked Topology Generation ---
    if verbose:
        print(f"Topology generation (chunked, {topo_chunk:,} voxels/chunk)...")

    R = resolution
    all_quad_indices = []
    all_intersected_dirs = []
    for start in range(0, Nvox, topo_chunk):
        end = min(start + topo_chunk, Nvox)
        c = coords[start:end]
        inter = intersected[start:end]
        chunk_n = c.shape[0]

        # (chunk, 3, 4, 3)
        neighbors = c.reshape(chunk_n, 1, 1, 3) + edge_neighbor_voxel_offset
        mask = inter != 0
        connected = neighbors[mask]  # (M, 4, 3)
        dirs = inter[mask]           # (M,)
        M = connected.shape[0]
        if M == 0:
            del neighbors, connected, dirs
            continue

        hash_key = torch.cat([
            torch.zeros((M * 4, 1), dtype=torch.int, device=device),
            connected.reshape(-1, 3),
        ], dim=1)
        indices = _C.hashmap_lookup_3d_cuda(
            *hashmap_vox, hash_key, R, R, R,
        ).reshape(M, 4).int()
        valid = (indices != 0xffffffff).all(dim=1)
        if valid.any():
            all_quad_indices.append(indices[valid])
            all_intersected_dirs.append(dirs[valid].int())

        del neighbors, connected, dirs, hash_key, indices, valid

    quad_indices = torch.cat(all_quad_indices)
    intersected_dir = torch.cat(all_intersected_dirs)
    del all_quad_indices, all_intersected_dirs, intersected
    L = quad_indices.shape[0]

    if verbose:
        print(f"  {L:,} quads")

    # --- 5. Remove unreferenced vertices ---
    unique_verts = torch.unique(quad_indices.reshape(-1))
    dual_verts = dual_verts[unique_verts]
    vert_map = torch.zeros((Nvox,), dtype=torch.int32, device=device)
    vert_map[unique_verts] = torch.arange(unique_verts.shape[0], dtype=torch.int32, device=device)
    quad_indices = vert_map[quad_indices]
    del vert_map, unique_verts

    mesh_vertices = (dual_verts / resolution - 0.5) * scale + center

    # --- 5b. Quad-level inner face removal ---
    if remove_inner_faces:
        if verbose:
            print(f"Removing inner quads from {quad_indices.shape[0]:,} quads...")
        inner_chunk = 524_288
        quad_centers = torch.empty((quad_indices.shape[0], 3), dtype=torch.float32, device=device)
        for i in range(0, quad_indices.shape[0], inner_chunk):
            end = min(i + inner_chunk, quad_indices.shape[0])
            q = quad_indices[i:end].long()
            quad_centers[i:end] = (
                mesh_vertices[q[:, 0]] + mesh_vertices[q[:, 1]] +
                mesh_vertices[q[:, 2]] + mesh_vertices[q[:, 3]]
            ) * 0.25

        sdf = torch.empty(quad_centers.shape[0], dtype=torch.float32, device=device)
        for i in range(0, quad_centers.shape[0], inner_chunk):
            end = min(i + inner_chunk, quad_centers.shape[0])
            sdf[i:end] = bvh.signed_distance(quad_centers[i:end], mode='raystab')[0]

        is_outer = sdf >= -eps * 0.1
        n_removed = (~is_outer).sum().item()
        if verbose:
            print(f"  SDF stats: min={sdf.min().item():.6f} max={sdf.max().item():.6f} eps={eps:.6f} threshold={-eps*0.1:.6f}")
            print(f"  outer={is_outer.sum().item():,} inner={n_removed:,}")
        quad_indices = quad_indices[is_outer]
        intersected_dir = intersected_dir[is_outer]

        # Re-index vertices to remove unused
        used = torch.zeros(mesh_vertices.shape[0], dtype=torch.bool, device=device)
        used[quad_indices.flatten()] = True
        new_idx = torch.full((mesh_vertices.shape[0],), -1, dtype=torch.int32, device=device)
        new_idx[used] = torch.arange(used.sum(), dtype=torch.int32, device=device)
        mesh_vertices = mesh_vertices[used]
        for i in range(0, quad_indices.shape[0], inner_chunk):
            end = min(i + inner_chunk, quad_indices.shape[0])
            quad_indices[i:end] = new_idx[quad_indices[i:end]]
        L = quad_indices.shape[0]

        if verbose:
            print(f"  Removed {n_removed:,} inner quads, {L:,} remaining")
            print(f"  mesh_vertices: {mesh_vertices.shape}, has_nan={torch.isnan(mesh_vertices).any().item()}")
            print(f"  quad_indices: {quad_indices.shape}, min={quad_indices.min().item()}, max={quad_indices.max().item()}, has_neg={((quad_indices < 0).any()).item()}")
        del quad_centers, sdf, is_outer, used, new_idx

    # --- 6. Chunked Triangle Splitting ---
    if verbose:
        print(f"Triangle splitting (chunked, {tri_chunk:,} quads/chunk)...")

    all_triangles = []
    for start in range(0, L, tri_chunk):
        end = min(start + tri_chunk, L)
        qi = quad_indices[start:end]
        idir = intersected_dir[start:end]
        is_pos = (idir == 1).unsqueeze(1)

        # Split 1
        t0 = torch.where(is_pos, qi[:, quad_split_1_p], qi[:, quad_split_1_n])
        n0a = torch.linalg.cross(
            mesh_vertices[t0[:, 1]] - mesh_vertices[t0[:, 0]],
            mesh_vertices[t0[:, 2]] - mesh_vertices[t0[:, 0]],
        )
        n0b = torch.linalg.cross(
            mesh_vertices[t0[:, 2]] - mesh_vertices[t0[:, 1]],
            mesh_vertices[t0[:, 3]] - mesh_vertices[t0[:, 1]],
        )
        align0 = (n0a * n0b).sum(dim=1).abs()

        # Split 2
        t1 = torch.where(is_pos, qi[:, quad_split_2_p], qi[:, quad_split_2_n])
        n1a = torch.linalg.cross(
            mesh_vertices[t1[:, 1]] - mesh_vertices[t1[:, 0]],
            mesh_vertices[t1[:, 2]] - mesh_vertices[t1[:, 0]],
        )
        n1b = torch.linalg.cross(
            mesh_vertices[t1[:, 2]] - mesh_vertices[t1[:, 1]],
            mesh_vertices[t1[:, 3]] - mesh_vertices[t1[:, 1]],
        )
        align1 = (n1a * n1b).sum(dim=1).abs()

        selected = torch.where((align0 > align1).unsqueeze(1), t0, t1)
        all_triangles.append(selected)
        del qi, idir, t0, t1, n0a, n0b, n1a, n1b, align0, align1, selected

    mesh_triangles = torch.cat(all_triangles).reshape(-1, 3)
    del all_triangles, quad_indices, intersected_dir

    if verbose:
        print(f"After triangle split: verts={mesh_vertices.shape[0]:,} faces={mesh_triangles.shape[0]:,}")
        print(f"  tri min={mesh_triangles.min().item()} max={mesh_triangles.max().item()} n_verts={mesh_vertices.shape[0]}")
        has_invalid = (mesh_triangles < 0).any().item() or (mesh_triangles >= mesh_vertices.shape[0]).any().item()
        print(f"  has_invalid_indices={has_invalid}")
        print(f"  has_nan_verts={torch.isnan(mesh_vertices).any().item()}")

    # --- 7. Project back ---
    if project_back > 0:
        if verbose:
            print("Projecting back to original mesh...")
        _, face_id, uvw = bvh.unsigned_distance(mesh_vertices, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]]
        projected_verts = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        mesh_vertices -= project_back * (mesh_vertices - projected_verts)

    if verbose:
        print(f"  {mesh_vertices.shape[0]:,} vertices, {mesh_triangles.shape[0]:,} faces")

    return mesh_vertices, mesh_triangles.int()


def _batched_unsigned_distance(bvh, positions, batch_size=500_000, return_uvw=False):
    """Batch unsigned_distance queries to avoid GPU kernel timeout on large meshes."""
    import torch
    N = positions.shape[0]
    if N <= batch_size:
        return bvh.unsigned_distance(positions, return_uvw=return_uvw)
    distances_list, face_id_list, uvw_list = [], [], []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        d, f, u = bvh.unsigned_distance(positions[i:end], return_uvw=return_uvw)
        distances_list.append(d)
        face_id_list.append(f)
        if return_uvw:
            uvw_list.append(u)
    return (
        torch.cat(distances_list),
        torch.cat(face_id_list),
        torch.cat(uvw_list) if return_uvw else None,
    )


def _rasterize_uv(vertices, faces, uvs, texture_size, device):
    """Rasterize mesh in UV space using DRTK and return (mask, valid_pos).

    Args:
        vertices: [V, 3] vertex positions
        faces: [F, 3] face indices (int32)
        uvs: [V, 2] UV coordinates in [0, 1]
        texture_size: output texture resolution
        device: torch device

    Returns:
        mask: [H, W] bool tensor — which pixels are covered
        valid_pos: [N, 3] 3D positions for covered pixels
    """
    import torch
    import drtk

    chunk_size = 100_000
    S = texture_size

    verts_uv = torch.stack([
        uvs[:, 0] * S - 0.5,
        uvs[:, 1] * S - 0.5,
        torch.ones(uvs.shape[0], device=device),
    ], dim=-1).float().unsqueeze(0)  # [1, V, 3]

    rast_face_ids = torch.full((S, S), -1, dtype=torch.int32, device=device)
    for i in range(0, faces.shape[0], chunk_size):
        import comfy.model_management
        comfy.model_management.throw_exception_if_processing_interrupted()
        chunk_vi = faces[i:i+chunk_size].int()
        index_img = drtk.rasterize(verts_uv, chunk_vi, height=S, width=S)
        chunk_hit = index_img[0] >= 0
        rast_face_ids[chunk_hit] = (index_img[0][chunk_hit] + i).int()
        del index_img, chunk_hit

    mask = rast_face_ids >= 0

    _, bary_img = drtk.render(verts_uv, faces.int(), rast_face_ids.unsqueeze(0))
    # bary_img: [N, 3, H, W] -> [H, W, 3]
    bary = bary_img[0].permute(1, 2, 0)

    bary_masked = bary[mask]
    face_ids = rast_face_ids[mask].long()
    face_verts = vertices[faces[face_ids].long()]
    valid_pos = (face_verts * bary_masked.unsqueeze(-1)).sum(dim=1)
    del verts_uv, rast_face_ids, bary_img, bary, face_verts, bary_masked, face_ids

    import comfy.model_management
    comfy.model_management.soft_empty_cache()
    return mask, valid_pos


class Trellis2ExportGLB(io.ComfyNode):
    """All-in-one: load voxelgrid NPZ -> simplify -> UV unwrap -> bake PBR -> export GLB."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ExportGLB",
            display_name="TRELLIS.2 Export GLB",
            category="TRELLIS2",
            is_output_node=True,
            description="""All-in-one textured GLB export from voxelgrid data.

Takes the voxelgrid_npz_path from "Shape to Textured Mesh" and:
1. Optionally remeshes (Dual Contouring)
2. Simplifies to decimation_target faces
3. UV unwraps and bakes PBR textures
4. Exports textured GLB""",
            inputs=[
                io.String.Input("voxelgrid_path"),
                io.Int.Input("decimation_target", default=500000, min=1000, max=5000000, step=1000, optional=True),
                io.Int.Input("texture_size", default=2048, min=512, max=8192, step=512, optional=True),
                io.Boolean.Input("remesh", default=True, optional=True),
                io.Boolean.Input("pre_simplify", default=True, optional=True,
                    tooltip="Pre-simplify mesh before remesh to massively reduce VRAM. May lose very thin features."),
                io.Boolean.Input("remove_inner_faces", default=False, optional=True,
                    tooltip="Remove faces whose centers are inside the original mesh (removes internal geometry artifacts)."),
                io.Boolean.Input("double_sided", default=False, optional=True,
                    tooltip="Mark material as double-sided in GLB (renders both front and back faces)."),
                io.String.Input("filename_prefix", default="trellis2", optional=True),
                io.Int.Input("pixel_expansion", default=3, min=0, max=64, step=1, optional=True,
                             tooltip="Pixel expansion (inpaint radius) for the Base Color map to prevent seam bleeding."),
                io.Int.Input("pbr_pixel_expansion", default=1, min=0, max=64, step=1, optional=True,
                             tooltip="Pixel expansion (inpaint radius) for Metallic, Roughness, and Alpha maps."),
                io.Boolean.Input("force_opaque", default=True, optional=True,
                                 tooltip="Force the texture to be opaque. Eliminates transparent pixel gaps and reduces file size."),
            ],
            outputs=[
                io.String.Output(display_name="glb_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        voxelgrid_path,
        decimation_target=500000,
        texture_size=2048,
        remesh=True,
        pre_simplify=True,
        remove_inner_faces=False,
        double_sided=False,
        filename_prefix="trellis2",
        pixel_expansion=3,
        pbr_pixel_expansion=1,
        force_opaque=True,
    ):
        import json
        import torch
        import cv2
        import cumesh_vb as CuMesh
        from flex_gemm_ap.ops.grid_sample import grid_sample_3d
        import trimesh as Trimesh

        torch.cuda.empty_cache()

        # --- 1. Load voxelgrid NPZ ---
        logger.info(f"ExportGLB: loading {voxelgrid_path}")
        data = np.load(voxelgrid_path, allow_pickle=True)

        vertices = torch.tensor(data['vertices'].astype(np.float32))
        faces = torch.tensor(data['faces'].astype(np.int32))
        coords = torch.tensor(data['coords'].astype(np.float32))
        attr_volume = torch.tensor(data['attrs'].astype(np.float32))
        voxel_size_raw = data['voxel_size']
        voxel_size_f = float(voxel_size_raw[0]) if hasattr(voxel_size_raw, '__len__') else float(voxel_size_raw)

        logger.info(f"ExportGLB: {vertices.shape[0]} verts, {faces.shape[0]} faces, {coords.shape[0]} voxels")

        comfy.model_management.throw_exception_if_processing_interrupted()
        device = comfy.model_management.get_torch_device()

        # Compute grid size from voxel_size
        aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device=device)
        voxel_size = torch.tensor([voxel_size_f] * 3, dtype=torch.float32, device=device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

        # --- 2. Pre-simplify before remesh ---
        if pre_simplify and remesh and faces.shape[0] > 2_000_000:
            logger.info(f"Pre-simplifying {faces.shape[0]} faces -> 2M before remesh")
            premesh = CuMesh.CuMesh()
            premesh.init(vertices.to(device), faces.to(device))
            premesh.simplify(2_000_000, verbose=True)
            vertices, faces = premesh.read()
            vertices, faces = vertices.cpu(), faces.cpu()
            del premesh
            torch.cuda.empty_cache()
            logger.info(f"Pre-simplified: {vertices.shape[0]} verts, {faces.shape[0]} faces")

        # --- 3. DC remesh ---
        if remesh:
            dc_resolution = grid_size.max().item()
            dc_center = aabb.mean(dim=0)
            dc_scale = (aabb[1] - aabb[0]).max().item()
            remesh_band = 1.0

            logger.info(f"Running low-mem DC: resolution={dc_resolution}")
            new_verts, new_faces = remesh_narrow_band_dc_lowmem(
                vertices.to(device), faces.to(device),
                center=dc_center,
                scale=(dc_resolution + 3 * remesh_band) / dc_resolution * dc_scale,
                resolution=dc_resolution,
                band=remesh_band,
                project_back=0.9,
                verbose=True,
                remove_inner_faces=remove_inner_faces,
            )
            vertices = new_verts.cpu()
            faces = new_faces.cpu()
            del new_verts, new_faces
            torch.cuda.empty_cache()
            logger.info(f"Remeshed: {vertices.shape[0]} verts, {faces.shape[0]} faces")
            logger.info(f"  faces min={faces.min().item()} max={faces.max().item()} n_verts={vertices.shape[0]}")
            has_invalid = (faces < 0).any().item() or (faces >= vertices.shape[0]).any().item()
            logger.info(f"  has_invalid_indices={has_invalid} has_nan_verts={torch.isnan(vertices).any().item()}")

        # --- 4. Fill holes + Build BVH from current mesh ---
        vertices = vertices.to(device)
        faces = faces.to(device)

        mesh = CuMesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        vertices, faces = mesh.read()
        logger.info(f"After fill holes: {mesh.num_vertices} verts, {mesh.num_faces} faces")

        # --- 5. Build BVH from original (pre-simplify) mesh for accurate attribute lookup ---
        orig_vertices = vertices.clone()
        orig_faces = faces.clone()
        bvh = CuMesh.cuBVH(orig_vertices, orig_faces)

        comfy.model_management.throw_exception_if_processing_interrupted()

        # --- 6. Simplify + cleanup ---
        if not remesh:
            # No-remesh: aggressive simplify -> cleanup -> final simplify -> cleanup
            mesh.simplify(decimation_target * 3, verbose=True)
            logger.info(f"After initial simplify: {mesh.num_vertices} verts, {mesh.num_faces} faces")
            mesh.remove_duplicate_faces()
            mesh.repair_non_manifold_edges()
            mesh.remove_small_connected_components(1e-5)
            mesh.fill_holes(max_hole_perimeter=3e-2)
            mesh.simplify(decimation_target, verbose=True)
            logger.info(f"After final simplify: {mesh.num_vertices} verts, {mesh.num_faces} faces")
            mesh.remove_duplicate_faces()
            mesh.repair_non_manifold_edges()
            mesh.remove_small_connected_components(1e-5)
            mesh.fill_holes(max_hole_perimeter=3e-2)
            mesh.unify_face_orientations()
        else:
            # Remesh: just simplify (DC already cleaned topology)
            mesh.simplify(decimation_target, verbose=True)
            logger.info(f"After simplify: {mesh.num_vertices} verts, {mesh.num_faces} faces")

        comfy.model_management.throw_exception_if_processing_interrupted()

        # --- 8. UV unwrap ---
        logger.info("UV unwrapping...")
        out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": np.radians(90.0),
                "refine_iterations": 0,
                "global_iterations": 1,
                "smooth_strength": 1,
            },
            return_vmaps=True,
            verbose=True,
        )
        out_vertices = out_vertices.to(device)
        out_faces = out_faces.to(device)
        out_uvs = out_uvs.to(device)
        out_vmaps = out_vmaps.to(device)
        mesh.compute_vertex_normals()
        out_normals = mesh.read_vertex_normals()[out_vmaps]
        logger.info(f"UV unwrap done: {out_vertices.shape[0]} verts, {out_faces.shape[0]} faces")

        comfy.model_management.throw_exception_if_processing_interrupted()

        # --- 9. Rasterize in UV space ---
        logger.info("Rasterizing UV space...")
        mask, valid_pos = _rasterize_uv(
            out_vertices, out_faces, out_uvs, texture_size, device,
        )

        comfy.model_management.soft_empty_cache()

        # --- 10. BVH map to original surface ---
        logger.info("Mapping to original surface...")
        _, face_id, uvw = _batched_unsigned_distance(bvh, valid_pos, return_uvw=True)
        orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        del face_id, uvw, orig_tri_verts

        # --- 11. Sample voxel attributes ---
        logger.info("Sampling PBR attributes...")
        coords = coords.to(device)
        attr_volume = attr_volume.to(device)

        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )

        del valid_pos, bvh, orig_vertices, orig_faces, attr_volume, coords
        comfy.model_management.soft_empty_cache()

        # --- 12. Inpaint UV seams ---
        logger.info("Inpainting UV seams...")
        mask_np = mask.cpu().numpy()
        mask_inv = (~mask_np).astype(np.uint8)

        base_color = np.clip(attrs[..., 0:3].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., 3:4].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., 4:5].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., 5:6].cpu().numpy() * 255, 0, 255).astype(np.uint8)

        del attrs, mask
        gc.collect()
        comfy.model_management.soft_empty_cache()

        if pixel_expansion > 0:
            base_color = cv2.inpaint(base_color, mask_inv, pixel_expansion, cv2.INPAINT_TELEA)
        if pbr_pixel_expansion > 0:
            metallic = cv2.inpaint(metallic, mask_inv, pbr_pixel_expansion, cv2.INPAINT_TELEA)[..., None]
            roughness = cv2.inpaint(roughness, mask_inv, pbr_pixel_expansion, cv2.INPAINT_TELEA)[..., None]
            alpha = cv2.inpaint(alpha, mask_inv, pbr_pixel_expansion, cv2.INPAINT_TELEA)[..., None]

        # --- 13. Build PBR material ---
        base_texture = Image.fromarray(base_color) if force_opaque else Image.fromarray(np.concatenate([base_color, alpha], axis=-1))

        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=base_texture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            doubleSided=double_sided,
        )

        # --- 14. Coordinate conversion Y-up -> Z-up ---
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        uvs_np = out_uvs.cpu().numpy()
        normals_np = out_normals.cpu().numpy()

        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
        uvs_np[:, 1] = 1 - uvs_np[:, 1]

        # --- 15. Assemble trimesh ---
        textured_mesh = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=Trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
        )

        # --- 16. Export GLB ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.glb"
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        logger.info(f"Pre-export: verts={vertices_np.shape} faces={faces_np.shape} uvs={uvs_np.shape}")
        logger.info(f"  verts has_nan={np.isnan(vertices_np).any()} faces min={faces_np.min()} max={faces_np.max()}")
        logger.info(f"  uvs has_nan={np.isnan(uvs_np).any()} range=[{uvs_np.min():.3f}, {uvs_np.max():.3f}]")
        logger.info(f"  trimesh valid={textured_mesh.is_volume if hasattr(textured_mesh, 'is_volume') else 'N/A'}")
        textured_mesh.export(output_path, file_type='glb')
        logger.info(f"GLB exported: {output_path} size={os.path.getsize(output_path)} bytes")

        del textured_mesh, out_vertices, out_faces, out_uvs, out_normals, mesh
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(output_path)


class Trellis2ExportTrimesh(io.ComfyNode):
    """Export trimesh to file (GLB, OBJ, PLY, etc.).

    Note: This is NOT isolated because it's just disk I/O.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ExportTrimesh",
            display_name="TRELLIS.2 Export Trimesh",
            category="TRELLIS2",
            is_output_node=True,
            description="""Export trimesh to various 3D file formats.

Supports: GLB, OBJ, PLY, STL, 3MF, DAE""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.String.Input("filename_prefix", default="trellis2", optional=True),
                io.Combo.Input("file_format", options=["glb", "obj", "ply", "stl", "3mf", "dae"],
                               default="glb", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, filename_prefix="trellis2", file_format="glb"):
        import copy

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.{file_format}"

        output_dir = folder_paths.get_output_directory()
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(exist_ok=True)

        # For glTF/GLB formats, convert internal Z-up -> Y-up at export time
        if file_format in ('glb', 'gltf'):
            export_mesh = copy.deepcopy(trimesh)
            verts = export_mesh.vertices.copy()
            verts[:, 1], verts[:, 2] = verts[:, 2].copy(), -verts[:, 1].copy()
            export_mesh.vertices = verts
            if hasattr(export_mesh, 'vertex_normals') and export_mesh.vertex_normals is not None and len(export_mesh.vertex_normals) > 0:
                normals = export_mesh.vertex_normals.copy()
                normals[:, 1], normals[:, 2] = normals[:, 2].copy(), -normals[:, 1].copy()
                export_mesh.vertex_normals = normals
            if hasattr(export_mesh.visual, 'uv') and export_mesh.visual.uv is not None:
                uvs = export_mesh.visual.uv.copy()
                uvs[:, 1] = 1 - uvs[:, 1]
                export_mesh.visual.uv = uvs
            export_mesh.export(str(output_path), file_type=file_format)
        else:
            trimesh.export(str(output_path), file_type=file_format)

        logger.info(f"Exported to: {output_path}")

        return io.NodeOutput(str(output_path))


class Trellis2SetMeshTextures(io.ComfyNode):
    """Pack custom (or upscaled) PBR textures directly back into a 3D mesh's material."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2SetMeshTextures",
            display_name="TRELLIS.2 Set Mesh Textures",
            category="TRELLIS2",
            description="""Pack custom (or upscaled) PBR textures directly back into a 3D mesh's material.

If the input textures are not connected, the original textures on the mesh are fully preserved.""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                # Base Color & Geometry controls
                io.Image.Input("base_color", optional=True,
                               tooltip="Base Color texture map."),
                io.Image.Input("alpha_map", optional=True,
                               tooltip="Alpha/Transparency map."),
                io.Boolean.Input("base_force_opaque", default=False, optional=True,
                                 tooltip="Strip alpha channel from the base_color texture if it exists."),
                io.Boolean.Input("double_sided", default=True, optional=True,
                                 tooltip="Enable double-sided material rendering (prevents transparent backfaces)."),
                # Normal details
                io.Image.Input("normal_map", optional=True,
                               tooltip="Tangent Space Normal map."),
                io.Float.Input("normal_scale", default=1.0, min=0.0, max=5.0, step=0.01, optional=True,
                               tooltip="Normal map strength/intensity multiplier."),
                # Metallic & Roughness
                io.Image.Input("metallic_roughness", optional=True,
                               tooltip="Metallic/Roughness combined texture map (Green=Roughness, Blue=Metallic)."),
                io.Float.Input("metallic_factor", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Metallic multiplier or uniform constant (if no map is connected)."),
                io.Float.Input("roughness_factor", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Roughness multiplier or uniform constant (if no map is connected)."),
                # Emissive & Occlusion
                io.Image.Input("emissive_map", optional=True,
                               tooltip="Emissive / Glow texture map."),
                io.Float.Input("emissive_strength", default=1.0, min=0.0, max=10.0, step=0.1, optional=True,
                               tooltip="Emissive glow multiplier/strength."),
                io.Image.Input("occlusion_map", optional=True,
                               tooltip="Custom Ambient Occlusion (AO) texture map."),
                io.Float.Input("occlusion_strength", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Ambient Occlusion shadow strength multiplier."),
                # Compression settings
                io.Combo.Input("compression_mode", options=["PNG", "JPEG", "WEBP"], default="WEBP", optional=True,
                               tooltip="Texture compression format inside GLB: 'WEBP' is state-of-the-art with transparency, 'JPEG' is legacy (does NOT support alpha/transparency), 'PNG' is lossless."),
                io.Int.Input("texture_quality", default=90, min=10, max=100, step=1, optional=True,
                             tooltip="Compression quality level (10-100). 100 is fully lossless."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        base_color=None,
        alpha_map=None,
        base_force_opaque=False,
        double_sided=True,
        normal_map=None,
        normal_scale=1.0,
        metallic_roughness=None,
        metallic_factor=1.0,
        roughness_factor=1.0,
        emissive_map=None,
        emissive_strength=1.0,
        occlusion_map=None,
        occlusion_strength=1.0,
        compression_mode="WEBP",
        texture_quality=90,
    ):
        from PIL import Image
        import numpy as np
        import trimesh as Trimesh
        from .image_utils import comfy_to_pil

        # Always instantiate a brand new, empty PBR material to ignore existing textures
        material = Trimesh.visual.material.PBRMaterial(
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicFactor=metallic_factor,
            roughnessFactor=roughness_factor,
            normalScale=normal_scale,
            occlusionStrength=occlusion_strength,
            alphaMode='OPAQUE' if base_force_opaque else 'BLEND',
            doubleSided=double_sided,
        )

        # Compress helper for override textures
        def _compress(pil_img):
            if compression_mode == "PNG" or texture_quality >= 100:
                return pil_img
            import io
            out = io.BytesIO()
            if compression_mode == "JPEG":
                pil_img.convert('RGB').save(out, format='JPEG', quality=texture_quality)
            else: # WEBP
                pil_img.save(out, format='WEBP', quality=texture_quality)
            out.seek(0)
            return Image.open(out)

        # 1. Update Base Color / Albedo texture
        if base_color is not None or alpha_map is not None:
            if base_color is not None:
                pil_color = comfy_to_pil(base_color).convert('RGBA')
            else:
                pil_color = Image.new('RGBA', (2048, 2048), (255, 255, 255, 255))

            if alpha_map is not None and not base_force_opaque:
                pil_alpha = comfy_to_pil(alpha_map).convert('L')
                if pil_alpha.size != pil_color.size:
                    pil_alpha = pil_alpha.resize(pil_color.size, Image.Resampling.BILINEAR)
                pil_color.putalpha(pil_alpha)

            if base_force_opaque:
                pil_color = pil_color.convert('RGB')

            pil_color = _compress(pil_color)
            material.baseColorTexture = pil_color

        # 2. Update Tangent Space Normal texture
        if normal_map is not None:
            pil_normal = comfy_to_pil(normal_map)
            material.normalTexture = pil_normal

        # 3. Update Metallic Roughness texture
        if metallic_roughness is not None:
            pil_mr = comfy_to_pil(metallic_roughness)
            pil_mr = _compress(pil_mr)
            material.metallicRoughnessTexture = pil_mr

        # 4. Update Emissive (Glow) texture
        if emissive_map is not None:
            pil_emissive = comfy_to_pil(emissive_map)
            pil_emissive = _compress(pil_emissive)
            material.emissiveTexture = pil_emissive
            material.emissiveFactor = np.array([emissive_strength, emissive_strength, emissive_strength], dtype=np.float32)

        # 5. Update Ambient Occlusion texture
        if occlusion_map is not None:
            pil_occlusion = comfy_to_pil(occlusion_map)
            pil_occlusion = _compress(pil_occlusion)
            material.occlusionTexture = pil_occlusion

        # Rebuild visual visuals to ensure correct propagation
        result = Trimesh.Trimesh(
            vertices=trimesh.vertices,
            faces=trimesh.faces,
            vertex_normals=trimesh.vertex_normals if hasattr(trimesh, 'vertex_normals') else None,
            process=False,
            visual=Trimesh.visual.TextureVisuals(uv=trimesh.visual.uv, material=material)
        )
        if hasattr(trimesh, 'vertex_attributes'):
            result.vertex_attributes = trimesh.vertex_attributes

        return io.NodeOutput(result)


NODE_CLASS_MAPPINGS = {
    "Trellis2Simplify": Trellis2Simplify,
    "Trellis2UVUnwrap": Trellis2UVUnwrap,
    "Trellis2ProcessMesh": Trellis2ProcessMesh,
    "Trellis2RasterizePBR": Trellis2RasterizePBR,
    "Trellis2ExportGLB": Trellis2ExportGLB,
    "Trellis2ExportTrimesh": Trellis2ExportTrimesh,
    "Trellis2SetMeshTextures": Trellis2SetMeshTextures,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2Simplify": "TRELLIS.2 Simplify Mesh",
    "Trellis2UVUnwrap": "TRELLIS.2 UV Unwrap",
    "Trellis2ProcessMesh": "TRELLIS.2 Process Mesh",
    "Trellis2RasterizePBR": "TRELLIS.2 Rasterize PBR",
    "Trellis2ExportGLB": "TRELLIS.2 Export GLB",
    "Trellis2ExportTrimesh": "TRELLIS.2 Export Trimesh",
    "Trellis2SetMeshTextures": "TRELLIS.2 Set Mesh Textures",
}

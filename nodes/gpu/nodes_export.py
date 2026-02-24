"""Export nodes for TRELLIS.2 3D meshes - runs in GPU subprocess."""
import os
from datetime import datetime

import folder_paths


class Trellis2ExportGLB:
    """Export TRELLIS.2 mesh to GLB format with PBR textures from voxel data."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "voxelgrid": ("TRELLIS2_VOXELGRID",),
            },
            "optional": {
                "decimation_target": ("INT", {"default": 500000, "min": 10000, "max": 2000000, "step": 10000}),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 512}),
                "remesh": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "trellis2"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "export"
    CATEGORY = "TRELLIS2"
    OUTPUT_NODE = True
    DESCRIPTION = """
Export mesh to GLB format with PBR materials baked from voxel data.

Parameters:
- trimesh: The 3D mesh geometry
- voxelgrid: VoxelGrid with PBR attributes
- decimation_target: Target face count for mesh simplification
- texture_size: Resolution of baked textures (512-4096)
- remesh: Enable mesh cleaning/remeshing
- filename_prefix: Prefix for output filename

Output GLB is saved to ComfyUI output folder.
"""

    def export(self, trimesh, voxelgrid, decimation_target=500000, texture_size=2048, remesh=True, filename_prefix="trellis2"):
        import torch
        import numpy as np
        import o_voxel

        # Check if voxelgrid has PBR attributes
        if 'attrs' not in voxelgrid:
            raise ValueError("VoxelGrid does not have PBR attributes. Use a voxelgrid from TRELLIS.2 generation.")

        print(f"[TRELLIS2] Exporting GLB (decimation={decimation_target}, texture={texture_size}, remesh={remesh})")

        # Get tensors from dict
        device = torch.device('cuda')
        vertices = voxelgrid['original_vertices']
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices)
        vertices = vertices.to(device)

        faces = voxelgrid['original_faces']
        if isinstance(faces, np.ndarray):
            faces = torch.from_numpy(faces)
        faces = faces.to(device)

        attr_volume = voxelgrid['attrs']
        if isinstance(attr_volume, np.ndarray):
            attr_volume = torch.from_numpy(attr_volume)
        attr_volume = attr_volume.to(device)

        coords = voxelgrid['coords']
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
        coords = coords.to(device)

        # Generate GLB using o_voxel
        glb = o_voxel.postprocess.to_glb(
            vertices=vertices,
            faces=faces,
            attr_volume=attr_volume,
            coords=coords,
            attr_layout=voxelgrid['layout'],
            voxel_size=voxelgrid['voxel_size'],
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )

        # Clean up GPU tensors
        del vertices, faces, attr_volume, coords
        torch.cuda.empty_cache()

        # Generate filename with timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.glb"

        # Save to output folder
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        glb.export(output_path, extension_webp=False)

        print(f"[TRELLIS2] GLB exported to: {output_path}")

        torch.cuda.empty_cache()

        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "Trellis2ExportGLB": Trellis2ExportGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2ExportGLB": "TRELLIS.2 Export GLB",
}

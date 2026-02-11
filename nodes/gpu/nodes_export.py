"""Export nodes for TRELLIS.2 3D meshes - runs in GPU subprocess."""
import json
import os
from datetime import datetime

import folder_paths


class Trellis2ExportGLB:
    """Export TRELLIS.2 mesh to GLB format with PBR textures from voxel data."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "voxelgrid_path": ("STRING", {"forceInput": True}),
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
- voxelgrid_path: Path to .npz file containing voxelgrid data
- decimation_target: Target face count for mesh simplification
- texture_size: Resolution of baked textures (512-4096)
- remesh: Enable mesh cleaning/remeshing
- filename_prefix: Prefix for output filename

Output GLB is saved to ComfyUI output folder.
"""

    def export(self, voxelgrid_path, decimation_target=500000, texture_size=2048, remesh=True, filename_prefix="trellis2"):
        import torch
        import numpy as np
        import o_voxel

        # Load voxelgrid from .npz file
        print(f"[TRELLIS2] Loading voxelgrid from: {voxelgrid_path}")
        data = np.load(voxelgrid_path, allow_pickle=False)

        coords = data['coords']
        attrs = data['attrs']
        voxel_size = float(data['voxel_size'][0])
        vertices = data['vertices']
        faces = data['faces']
        layout = json.loads(str(data['layout']))

        # Convert list [start, end] back to slice objects (JSON can't serialize slices)
        for key in layout:
            if isinstance(layout[key], list) and len(layout[key]) == 2:
                layout[key] = slice(layout[key][0], layout[key][1])

        print(f"[TRELLIS2] Exporting GLB (decimation={decimation_target}, texture={texture_size}, remesh={remesh})")

        # Move to GPU
        device = torch.device('cuda')
        vertices = torch.from_numpy(vertices).to(device)
        faces = torch.from_numpy(faces).to(device).int()
        attrs = torch.from_numpy(attrs).to(device)
        coords = torch.from_numpy(coords).to(device)

        # Generate GLB using o_voxel
        glb = o_voxel.postprocess.to_glb(
            vertices=vertices,
            faces=faces,
            attr_volume=attrs,
            coords=coords,
            attr_layout=layout,
            voxel_size=voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )

        # Clean up GPU tensors
        del vertices, faces, attrs, coords
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

"""Export nodes for TRELLIS.2 3D meshes."""
import os
import torch
import tempfile
from datetime import datetime

import folder_paths

from .utils import logger


class Trellis2ExportGLB:
    """Export TRELLIS.2 mesh to GLB format."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRELLIS2_MESH",),
            },
            "optional": {
                "decimation_target": ("INT", {"default": 500000, "min": 10000, "max": 2000000, "step": 10000}),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 512}),
                "filename_prefix": ("STRING", {"default": "trellis2"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "export"
    CATEGORY = "TRELLIS2"
    OUTPUT_NODE = True
    DESCRIPTION = """
Export TRELLIS.2 mesh to GLB format with PBR materials.

Parameters:
- mesh: The generated 3D mesh
- decimation_target: Target face count for mesh simplification
- texture_size: Resolution of baked textures (512-4096)
- filename_prefix: Prefix for output filename

Output GLB is saved to ComfyUI output folder.
"""

    def export(self, mesh, decimation_target=500000, texture_size=2048, filename_prefix="trellis2"):
        try:
            import o_voxel
        except ImportError:
            raise ImportError(
                "Could not import o_voxel. Please ensure TRELLIS.2 dependencies are installed."
            )

        mesh_obj = mesh["mesh"]
        pipeline = mesh["pipeline"]

        logger.info(f"Exporting GLB (decimation={decimation_target}, texture={texture_size})")

        # Generate GLB
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh_obj.vertices,
            faces=mesh_obj.faces,
            attr_volume=mesh_obj.attrs,
            coords=mesh_obj.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=mesh_obj.voxel_shape[0] if hasattr(mesh_obj, 'voxel_shape') else 1024,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )

        # Generate filename with timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.glb"

        # Save to output folder
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        glb.export(output_path, extension_webp=True)

        logger.info(f"GLB exported to: {output_path}")

        torch.cuda.empty_cache()

        return (output_path,)


class Trellis2RenderPreview:
    """Render preview images of TRELLIS.2 mesh."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRELLIS2_MESH",),
            },
            "optional": {
                "num_views": ("INT", {"default": 8, "min": 1, "max": 36, "step": 1}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 128}),
                "render_mode": (["normal", "clay", "base_color"], {"default": "normal"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview_images",)
    FUNCTION = "render"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Render preview images of the generated 3D mesh.

Parameters:
- mesh: The generated 3D mesh
- num_views: Number of views to render (rotating around object)
- resolution: Render resolution
- render_mode: Rendering style (normal, clay, base_color)
"""

    def render(self, mesh, num_views=8, resolution=512, render_mode="normal"):
        from ..trellis2.utils import render_utils

        mesh_obj = mesh["mesh"]

        logger.info(f"Rendering {num_views} preview images at {resolution}px")

        # Render snapshots
        images = render_utils.render_snapshot(
            mesh_obj,
            resolution=resolution,
            r=2,
            fov=36,
            nviews=num_views,
        )

        # Get the requested render mode
        if render_mode in images:
            frames = images[render_mode]
        else:
            # Fallback to normal
            frames = images.get("normal", list(images.values())[0])

        # Convert to tensor batch [N, H, W, C]
        import numpy as np
        frames_np = np.stack([np.array(f) for f in frames], axis=0)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0

        return (frames_tensor,)


class Trellis2RenderVideo:
    """Render a rotating video of the TRELLIS.2 mesh."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRELLIS2_MESH",),
            },
            "optional": {
                "fps": ("INT", {"default": 15, "min": 1, "max": 60, "step": 1}),
                "filename_prefix": ("STRING", {"default": "trellis2_video"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "render_video"
    CATEGORY = "TRELLIS2"
    OUTPUT_NODE = True
    DESCRIPTION = """
Render a rotating video of the 3D mesh with PBR materials.

Requires HDRI environment maps in TRELLIS.2 assets folder.
"""

    def render_video(self, mesh, fps=15, filename_prefix="trellis2_video"):
        from ..trellis2.utils import render_utils
        import imageio

        mesh_obj = mesh["mesh"]

        logger.info("Rendering video...")

        # Render video frames
        video_frames = render_utils.render_video(mesh_obj)
        vis_frames = render_utils.make_pbr_vis_frames(video_frames)

        # Generate filename
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.mp4"

        # Save to output folder
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        imageio.mimsave(output_path, vis_frames, fps=fps)

        logger.info(f"Video saved to: {output_path}")

        torch.cuda.empty_cache()

        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "Trellis2ExportGLB": Trellis2ExportGLB,
    "Trellis2RenderPreview": Trellis2RenderPreview,
    "Trellis2RenderVideo": Trellis2RenderVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2ExportGLB": "TRELLIS.2 Export GLB",
    "Trellis2RenderPreview": "TRELLIS.2 Render Preview",
    "Trellis2RenderVideo": "TRELLIS.2 Render Video",
}

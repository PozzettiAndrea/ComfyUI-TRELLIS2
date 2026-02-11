"""Export nodes for TRELLIS.2 3D meshes."""
import os
import torch
import numpy as np
from datetime import datetime

import folder_paths

from .utils import logger


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
        try:
            import o_voxel
        except ImportError:
            raise ImportError(
                "Could not import o_voxel. Please ensure TRELLIS.2 dependencies are installed."
            )

        # Check if voxelgrid has PBR attributes
        if 'attrs' not in voxelgrid:
            raise ValueError("VoxelGrid does not have PBR attributes. Use a voxelgrid from TRELLIS.2 generation.")

        logger.info(f"Exporting GLB (decimation={decimation_target}, texture={texture_size}, remesh={remesh})")

        # Get tensors from dict (already on GPU)
        device = torch.device('cuda')
        vertices = voxelgrid['original_vertices'].to(device)
        faces = voxelgrid['original_faces'].to(device)
        attr_volume = voxelgrid['attrs'].to(device)
        coords = voxelgrid['coords'].to(device)

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

        logger.info(f"GLB exported to: {output_path}")

        torch.cuda.empty_cache()

        return (output_path,)


class Trellis2RenderPreview:
    """Render preview images of a mesh."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
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
Render preview images of the 3D mesh.

Parameters:
- trimesh: The 3D mesh geometry
- num_views: Number of views to render (rotating around object)
- resolution: Render resolution
- render_mode: Rendering style (normal, clay, base_color)
"""

    def render(self, trimesh, num_views=8, resolution=512, render_mode="normal"):
        import pyrender
        import math

        logger.info(f"Rendering {num_views} preview images at {resolution}px")

        # Create pyrender scene
        scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])

        # Create mesh for pyrender
        mesh = pyrender.Mesh.from_trimesh(trimesh)
        scene.add(mesh)

        # Setup camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)

        # Calculate camera distance based on mesh bounds
        bounds = trimesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        extent = np.linalg.norm(bounds[1] - bounds[0])
        distance = extent * 2.0

        # Render from multiple views
        frames = []
        renderer = pyrender.OffscreenRenderer(resolution, resolution)

        for i in range(num_views):
            angle = 2 * math.pi * i / num_views

            # Camera position
            cam_pos = np.array([
                center[0] + distance * math.sin(angle),
                center[1] + 0.3 * distance,
                center[2] + distance * math.cos(angle)
            ])

            # Look at center
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0]))
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            camera_pose = np.eye(4)
            camera_pose[:3, 0] = right
            camera_pose[:3, 1] = up
            camera_pose[:3, 2] = -forward
            camera_pose[:3, 3] = cam_pos

            # Add camera to scene
            cam_node = scene.add(camera, pose=camera_pose)

            # Add light
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            light_node = scene.add(light, pose=camera_pose)

            # Render
            color, _ = renderer.render(scene)
            frames.append(color)

            # Remove camera and light for next view
            scene.remove_node(cam_node)
            scene.remove_node(light_node)

        renderer.delete()

        # Convert to tensor batch [N, H, W, C]
        frames_np = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0

        return (frames_tensor,)


class Trellis2RenderVideo:
    """Render a rotating video of the mesh."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "num_frames": ("INT", {"default": 60, "min": 10, "max": 360, "step": 10}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 60, "step": 1}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 128}),
                "filename_prefix": ("STRING", {"default": "trellis2_video"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "render_video"
    CATEGORY = "TRELLIS2"
    OUTPUT_NODE = True
    DESCRIPTION = """
Render a rotating video of the 3D mesh.

Parameters:
- trimesh: The 3D mesh geometry
- num_frames: Number of frames in the video
- fps: Frames per second
- resolution: Render resolution
- filename_prefix: Prefix for output filename
"""

    def render_video(self, trimesh, num_frames=60, fps=15, resolution=512, filename_prefix="trellis2_video"):
        import pyrender
        import imageio
        import math

        logger.info(f"Rendering video ({num_frames} frames at {fps}fps)...")

        # Create pyrender scene
        scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])

        # Create mesh for pyrender
        pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh)
        scene.add(pyrender_mesh)

        # Setup camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)

        # Calculate camera distance
        bounds = trimesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        extent = np.linalg.norm(bounds[1] - bounds[0])
        distance = extent * 2.0

        # Render frames
        frames = []
        renderer = pyrender.OffscreenRenderer(resolution, resolution)

        for i in range(num_frames):
            angle = 2 * math.pi * i / num_frames

            cam_pos = np.array([
                center[0] + distance * math.sin(angle),
                center[1] + 0.3 * distance,
                center[2] + distance * math.cos(angle)
            ])

            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0]))
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            camera_pose = np.eye(4)
            camera_pose[:3, 0] = right
            camera_pose[:3, 1] = up
            camera_pose[:3, 2] = -forward
            camera_pose[:3, 3] = cam_pos

            cam_node = scene.add(camera, pose=camera_pose)
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            light_node = scene.add(light, pose=camera_pose)

            color, _ = renderer.render(scene)
            frames.append(color)

            scene.remove_node(cam_node)
            scene.remove_node(light_node)

        renderer.delete()

        # Generate filename
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.mp4"

        # Save to output folder
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        imageio.mimsave(output_path, frames, fps=fps)

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

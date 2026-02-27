"""Render/preview nodes for TRELLIS.2 3D meshes (CPU-based)."""
import os
import logging
import numpy as np
from datetime import datetime

import folder_paths
import comfy.model_management
from comfy_api.latest import io

# Create logger for non-isolated classes
logger = logging.getLogger("trellis2")


class Trellis2RenderPreview(io.ComfyNode):
    """Render preview images of a mesh.

    Note: This is NOT isolated because pyrender runs on CPU.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2RenderPreview",
            display_name="TRELLIS.2 Render Preview",
            category="TRELLIS2",
            description="""Render preview images of the 3D mesh.

Parameters:
- trimesh: The 3D mesh geometry
- num_views: Number of views to render (rotating around object)
- resolution: Render resolution
- render_mode: Rendering style (normal, clay, base_color)""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("num_views", default=8, min=1, max=36, step=1, optional=True),
                io.Int.Input("resolution", default=512, min=256, max=2048, step=128, optional=True),
                io.Combo.Input("render_mode", options=["normal", "clay", "base_color"],
                               default="normal", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="preview_images"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, num_views=8, resolution=512, render_mode="normal"):
        import torch
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
            comfy.model_management.throw_exception_if_processing_interrupted()
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

        return io.NodeOutput(frames_tensor)


class Trellis2RenderVideo(io.ComfyNode):
    """Render a rotating video of the mesh.

    Note: This is NOT isolated because pyrender runs on CPU.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2RenderVideo",
            display_name="TRELLIS.2 Render Video",
            category="TRELLIS2",
            is_output_node=True,
            description="""Render a rotating video of the 3D mesh.

Parameters:
- trimesh: The 3D mesh geometry
- num_frames: Number of frames in the video
- fps: Frames per second
- resolution: Render resolution
- filename_prefix: Prefix for output filename""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("num_frames", default=60, min=10, max=360, step=10, optional=True),
                io.Int.Input("fps", default=15, min=1, max=60, step=1, optional=True),
                io.Int.Input("resolution", default=512, min=256, max=2048, step=128, optional=True),
                io.String.Input("filename_prefix", default="trellis2_video", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="video_path"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, num_frames=60, fps=15, resolution=512, filename_prefix="trellis2_video"):
        import torch
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
            comfy.model_management.throw_exception_if_processing_interrupted()
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

        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(output_path)


NODE_CLASS_MAPPINGS = {
    "Trellis2RenderPreview": Trellis2RenderPreview,
    "Trellis2RenderVideo": Trellis2RenderVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2RenderPreview": "TRELLIS.2 Render Preview",
    "Trellis2RenderVideo": "TRELLIS.2 Render Video",
}

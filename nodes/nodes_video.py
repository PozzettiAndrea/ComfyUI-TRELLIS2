"""Video-to-Animation nodes for TRELLIS.2.

These nodes enable generating animated 3D from video frame sequences.
"""

from comfy_env import isolated


@isolated(env="trellis2", import_paths=[".", ".."])
class Trellis2VideoConditioning:
    """Extract image conditioning from multiple video frames using DinoV3."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "images": ("IMAGE", {"tooltip": "Batch of video frames [B, H, W, C] where B = number of frames"}),
                "mask": ("MASK", {"tooltip": "Single mask for all frames [H,W] or per-frame masks [B, H, W]"}),
            },
            "optional": {
                "include_1024": ("BOOLEAN", {"default": True}),
                "background_color": (["black", "gray", "white"], {"default": "black"}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 32, "tooltip": "Frames to process at once (lower = less VRAM)"}),
            },
        }

    RETURN_TYPES = ("TRELLIS2_VIDEO_CONDITIONING",)
    RETURN_NAMES = ("video_conditioning",)
    FUNCTION = "get_conditioning"
    CATEGORY = "TRELLIS2/Video"
    DESCRIPTION = """
Extract DinoV3 conditioning from multiple video frames.

This node processes a batch of video frames and extracts visual features
for each frame, which can then be used for temporal 3D generation.

Parameters:
- model_config: The loaded TRELLIS.2 config
- images: Batch of video frames (B, H, W, C) where B = num_frames
- mask: Foreground masks - supports two modes:
    * Single mask (H, W) or (1, H, W): Applied to ALL frames
    * Per-frame masks (B, H, W): Each frame gets its own mask
- include_1024: Also extract 1024px features (for cascade modes)
- batch_size: Number of frames to process at once (lower = less VRAM)

The output contains per-frame conditioning that can be aggregated
temporally for animated 3D generation.
"""

    def get_conditioning(self, model_config, images, mask, include_1024=True, background_color="black", batch_size=8):
        from utils.video_stages import run_video_conditioning

        video_conditioning = run_video_conditioning(
            model_config=model_config,
            images=images,
            masks=mask,
            include_1024=include_1024,
            background_color=background_color,
            batch_size=batch_size,
        )

        return (video_conditioning,)


@isolated(env="trellis2", import_paths=[".", ".."])
class Trellis2VideoToShape:
    """Generate 3D shape from temporal video conditioning."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "video_conditioning": ("TRELLIS2_VIDEO_CONDITIONING",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "aggregation_mode": (["concat", "mean", "keyframe_middle"], {
                    "default": "concat",
                    "tooltip": "concat: full temporal attention, mean: average features, keyframe_middle: use middle frame"
                }),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "shape_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shape_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "max_tokens": ("INT", {"default": 49152, "min": 16384, "max": 65536, "step": 4096}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_SHAPE_RESULT", "TRIMESH")
    RETURN_NAMES = ("shape_result", "mesh")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2/Video"
    DESCRIPTION = """
Generate 3D shape from temporal video conditioning.

This node aggregates conditioning from all video frames to generate
a single 3D shape that represents the "consensus" structure across
the entire video sequence.

Aggregation modes:
- concat: Concatenates all frame features - cross-attention sees full temporal context
- mean: Averages all frame features - faster, but loses temporal detail
- keyframe_middle: Uses only the middle frame - simplest approach

The generated shape provides a consistent topology for animated texture generation.
"""

    def generate(
        self,
        model_config,
        video_conditioning,
        seed=0,
        aggregation_mode="concat",
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        shape_guidance_strength=7.5,
        shape_sampling_steps=12,
        max_tokens=49152,
    ):
        import trimesh as Trimesh
        from utils.video_stages import run_temporal_shape_generation

        shape_result = run_temporal_shape_generation(
            model_config=model_config,
            video_conditioning=video_conditioning,
            seed=seed,
            aggregation_mode=aggregation_mode,
            ss_guidance_strength=ss_guidance_strength,
            ss_sampling_steps=ss_sampling_steps,
            shape_guidance_strength=shape_guidance_strength,
            shape_sampling_steps=shape_sampling_steps,
            max_num_tokens=max_tokens,
        )

        # Create trimesh
        tri_mesh = Trimesh.Trimesh(
            vertices=shape_result['mesh_vertices'],
            faces=shape_result['mesh_faces'],
            process=False
        )

        return (shape_result, tri_mesh)


@isolated(env="trellis2", import_paths=[".", ".."])
class Trellis2AnimatedTexture:
    """Generate animated PBR textures from video conditioning on shared shape."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "video_conditioning": ("TRELLIS2_VIDEO_CONDITIONING",),
                "shape_result": ("TRELLIS2_SHAPE_RESULT",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "tex_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "tex_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "keyframe_interval": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 81,
                    "step": 1,
                    "tooltip": "Generate texture every N frames (1=all frames, higher=faster with interpolation)"
                }),
                "interpolation_mode": (["slerp", "linear"], {
                    "default": "slerp",
                    "tooltip": "slerp: spherical interpolation (smoother), linear: direct interpolation"
                }),
            }
        }

    RETURN_TYPES = ("TRELLIS2_ANIMATION",)
    RETURN_NAMES = ("animation",)
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2/Video"
    DESCRIPTION = """
Generate animated PBR textures on a shared shape structure.

This node generates per-frame texture attributes (base_color, metallic,
roughness, alpha) on the shared voxel coordinates from the shape.

The keyframe_interval controls the speed vs quality tradeoff:
- 1: Generate unique texture for every frame (slowest, most accurate)
- 10: Generate every 10th frame, interpolate between (9x faster)
- Higher values = faster but less accurate

Interpolation modes:
- slerp: Spherical linear interpolation - smoother, preserves magnitude
- linear: Direct linear interpolation - faster, may have artifacts

Output is an animation sequence that can be exported to GLB/OBJ sequence.
"""

    def generate(
        self,
        model_config,
        video_conditioning,
        shape_result,
        seed=0,
        tex_guidance_strength=7.5,
        tex_sampling_steps=12,
        keyframe_interval=10,
        interpolation_mode="slerp",
    ):
        from utils.video_stages import run_animated_texture_generation

        animation = run_animated_texture_generation(
            model_config=model_config,
            video_conditioning=video_conditioning,
            shape_result=shape_result,
            seed=seed,
            tex_guidance_strength=tex_guidance_strength,
            tex_sampling_steps=tex_sampling_steps,
            keyframe_interval=keyframe_interval,
            interpolation_mode=interpolation_mode,
        )

        return (animation,)


class Trellis2ExportAnimation:
    """Export animated 3D sequence to files."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "animation": ("TRELLIS2_ANIMATION",),
                "output_path": ("STRING", {"default": "output/animation"}),
                "format": (["glb_sequence", "obj_sequence", "vxz_sequence"], {"default": "glb_sequence"}),
            },
            "optional": {
                "decimation_target": ("INT", {"default": 100000, "min": 10000, "max": 1000000, "step": 10000}),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 512}),
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "frame_end": ("INT", {"default": -1, "min": -1, "max": 1000, "tooltip": "-1 = all frames"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_directory",)
    FUNCTION = "export"
    CATEGORY = "TRELLIS2/Video"
    OUTPUT_NODE = True
    DESCRIPTION = """
Export animated 3D sequence to disk.

Formats:
- glb_sequence: Individual GLB files per frame (frame_0000.glb, ...)
- obj_sequence: Individual OBJ files per frame
- vxz_sequence: O-Voxel compressed format (smallest)

Parameters:
- output_path: Base path for output (frame number appended)
- decimation_target: Target triangle count per frame
- texture_size: Texture resolution for GLB export
- frame_start/end: Range of frames to export (-1 = all)
"""

    def export(
        self,
        animation,
        output_path,
        format="glb_sequence",
        decimation_target=100000,
        texture_size=2048,
        frame_start=0,
        frame_end=-1,
    ):
        import os
        import numpy as np

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        coords = animation['coords']
        attrs_sequence = animation['attrs_sequence']
        voxel_size = animation['voxel_size']
        layout = animation['layout']
        num_frames = animation['num_frames']
        mesh_vertices = animation['mesh_vertices']
        mesh_faces = animation['mesh_faces']

        # Determine frame range
        if frame_end == -1:
            frame_end = num_frames
        frame_end = min(frame_end, num_frames)

        print(f"[TRELLIS2] Exporting frames {frame_start} to {frame_end-1}...")

        for frame_idx in range(frame_start, frame_end):
            attrs = attrs_sequence[frame_idx]
            frame_name = f"frame_{frame_idx:04d}"

            if format == "glb_sequence":
                try:
                    import o_voxel

                    glb = o_voxel.postprocess.to_glb(
                        vertices=mesh_vertices,
                        faces=mesh_faces,
                        attr_volume=attrs,
                        coords=coords,
                        attr_layout=layout,
                        voxel_size=voxel_size,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        decimation_target=decimation_target,
                        texture_size=texture_size,
                        remesh=True,
                        remesh_band=1,
                        remesh_project=0,
                        verbose=False
                    )
                    glb.export(os.path.join(output_path, f"{frame_name}.glb"), extension_webp=True)
                except ImportError:
                    print("[TRELLIS2] o_voxel not available, falling back to OBJ export")
                    format = "obj_sequence"

            if format == "obj_sequence":
                import trimesh
                mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)

                # Apply vertex colors from attrs (base_color)
                base_color_slice = layout.get('base_color', slice(0, 3))
                # Note: This is simplified - proper implementation would sample attrs at vertices
                mesh.export(os.path.join(output_path, f"{frame_name}.obj"))

            elif format == "vxz_sequence":
                try:
                    import o_voxel
                    import torch

                    coords_tensor = torch.from_numpy(coords).long()
                    attrs_dict = {
                        'base_color': (attrs[:, layout['base_color']] * 255).astype(np.uint8),
                        'metallic': (attrs[:, layout['metallic']] * 255).astype(np.uint8),
                        'roughness': (attrs[:, layout['roughness']] * 255).astype(np.uint8),
                        'alpha': (attrs[:, layout['alpha']] * 255).astype(np.uint8),
                    }
                    o_voxel.io.write(os.path.join(output_path, f"{frame_name}.vxz"), coords_tensor, attrs_dict)
                except ImportError:
                    print("[TRELLIS2] o_voxel not available for vxz export")

            if (frame_idx - frame_start) % 10 == 0:
                print(f"[TRELLIS2] Exported frame {frame_idx}")

        print(f"[TRELLIS2] Export complete: {frame_end - frame_start} frames to {output_path}")
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "Trellis2VideoConditioning": Trellis2VideoConditioning,
    "Trellis2VideoToShape": Trellis2VideoToShape,
    "Trellis2AnimatedTexture": Trellis2AnimatedTexture,
    "Trellis2ExportAnimation": Trellis2ExportAnimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2VideoConditioning": "TRELLIS.2 Video Conditioning",
    "Trellis2VideoToShape": "TRELLIS.2 Video to Shape",
    "Trellis2AnimatedTexture": "TRELLIS.2 Animated Texture",
    "Trellis2ExportAnimation": "TRELLIS.2 Export Animation",
}

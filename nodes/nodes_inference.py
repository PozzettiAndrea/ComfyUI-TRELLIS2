"""Inference nodes for TRELLIS.2 Image-to-3D generation."""
import logging
log = logging.getLogger("trellis2")


class Trellis2GetConditioning:
    """Extract image conditioning using DinoV3 for TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "background_color": (["black", "gray", "white"], {"default": "black"}),
            },
        }

    RETURN_TYPES = ("TRELLIS2_CONDITIONING", "IMAGE")
    RETURN_NAMES = ("conditioning", "preprocessed_image")
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
- model_config: The loaded TRELLIS.2 config
- image: Input image (RGB)
- mask: Foreground mask (white=object, black=background)
Use any background removal node (BiRefNet, rembg, etc.) to generate the mask.
"""

    def get_conditioning(self, model_config, image, mask, background_color="black"):
        # All heavy imports happen inside subprocess
        from .trellis_utils import run_conditioning

        # Auto-detect whether 1024 features are needed from resolution mode
        resolution = model_config.get("resolution", "1024_cascade")
        include_1024 = resolution in ("1024_cascade", "1536_cascade", "1024")

        conditioning, preprocessed_image = run_conditioning(
            model_config=model_config,
            image=image,
            mask=mask,
            include_1024=include_1024,
            background_color=background_color,
        )

        return (conditioning, preprocessed_image)


class Trellis2ImageToShape:
    """Generate 3D shape from conditioning using TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG", {"tooltip": "Model config from Load TRELLIS.2 Models node"}),
                "conditioning": ("TRELLIS2_CONDITIONING", {"tooltip": "Image conditioning from Get Conditioning node"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Random seed for reproducible generation"}),
                # Sparse Structure Sampler
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Sparse structure CFG scale. Higher = stronger adherence to input image"}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1, "tooltip": "Sparse structure sampling steps. More steps = better quality but slower"}),
                # Shape SLat Sampler
                "shape_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Shape CFG scale. Higher = stronger adherence to input image"}),
                "shape_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1, "tooltip": "Shape sampling steps. More steps = better quality but slower"}),
                # VRAM Control
                "max_tokens": ("INT", {"default": 49152, "min": 16384, "max": 65536, "step": 4096, "tooltip": "Max tokens for 1024 cascade. Lower = less VRAM but potentially lower quality. Default 49152 (~9GB), try 32768 (~7GB) or 24576 (~6GB) for lower VRAM."}),
            }
        }

    RETURN_TYPES = ("TRELLIS2_SHAPE_RESULT", "TRIMESH")
    RETURN_NAMES = ("shape_result", "mesh")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate 3D shape from image conditioning.

This node generates shape geometry (no texture/PBR).
Connect shape_result to "Shape to Textured Mesh" for PBR materials.

Parameters:
- model_config: The loaded model config (resolution is set in Load Models node)
- conditioning: DinoV3 conditioning from "Get Conditioning" node
- seed: Random seed for reproducibility
- ss_*: Sparse structure sampling parameters
- shape_*: Shape latent sampling parameters

Returns:
- shape_result: Shape data for texture generation
- mesh: Untextured mesh for preview/export
"""

    def generate(
        self,
        model_config,
        conditioning,
        seed=0,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        shape_guidance_strength=7.5,
        shape_sampling_steps=12,
        max_tokens=49152,
    ):
        # All heavy imports happen inside subprocess
        import trimesh as Trimesh
        from .trellis_utils import run_shape_generation

        # run_shape_generation returns (file_ref, vertices, faces)
        # file_ref is passed to downstream nodes, vertices/faces used for Trimesh
        shape_result, vertices, faces = run_shape_generation(
            model_config=model_config,
            conditioning=conditioning,
            seed=seed,
            ss_guidance_strength=ss_guidance_strength,
            ss_sampling_steps=ss_sampling_steps,
            shape_guidance_strength=shape_guidance_strength,
            shape_sampling_steps=shape_sampling_steps,
            max_num_tokens=max_tokens,
        )

        # Create trimesh from vertices/faces
        tri_mesh = Trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )

        return (shape_result, tri_mesh)


class Trellis2ShapeToTexturedMesh:
    """Generate PBR textured mesh from shape using TRELLIS.2."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG", {"tooltip": "Model config from Load TRELLIS.2 Models node"}),
                "conditioning": ("TRELLIS2_CONDITIONING", {"tooltip": "Image conditioning from Get Conditioning node (same as used for shape)"}),
                "shape_result": ("TRELLIS2_SHAPE_RESULT", {"tooltip": "Shape result from Image to Shape node"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Random seed for texture variation"}),
                "tex_guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Texture CFG scale. Higher = stronger adherence to input image"}),
                "tex_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1, "tooltip": "Texture sampling steps. More steps = better quality but slower"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mesh_glb_path", "voxelgrid_npz_path")
    FUNCTION = "generate"
    CATEGORY = "TRELLIS2"
    DESCRIPTION = """
Generate PBR textured mesh from shape.

Takes shape_result from "Image to Shape" node and generates PBR materials:
- base_color (RGB)
- metallic
- roughness
- alpha

Parameters:
- model_config: The loaded model config
- conditioning: DinoV3 conditioning (same as used for shape)
- shape_result: Shape data from "Image to Shape" node
- seed: Random seed for texture variation
- tex_*: Texture sampling parameters

Returns:
- mesh_glb_path: Path to mesh geometry (.glb file)
- voxelgrid_npz_path: Path to voxelgrid data (.npz file with coords, attrs, vertices, faces, etc.)
"""

    def generate(
        self,
        model_config,
        conditioning,
        shape_result,
        seed=0,
        tex_guidance_strength=7.5,
        tex_sampling_steps=12,
    ):
        # All heavy imports happen inside subprocess
        import json
        import os
        import uuid
        import numpy as np
        from .trellis_utils import run_texture_generation

        texture_result = run_texture_generation(
            model_config=model_config,
            conditioning=conditioning,
            shape_result=shape_result,
            seed=seed,
            tex_guidance_strength=tex_guidance_strength,
            tex_sampling_steps=tex_sampling_steps,
        )

        # Create output directory
        cache_dir = '/tmp/trellis2_cache'
        os.makedirs(cache_dir, exist_ok=True)
        file_id = uuid.uuid4().hex[:8]

        # Convert layout slices to tuples for JSON serialization
        pbr_layout = texture_result['pbr_layout']
        layout_serializable = {k: (v.start, v.stop) for k, v in pbr_layout.items()}

        # Save voxelgrid data to .npz (contains mesh + PBR voxel data)
        voxelgrid_npz_path = os.path.join(cache_dir, f'voxelgrid_{file_id}.npz')
        np.savez(
            voxelgrid_npz_path,
            coords=texture_result['voxel_coords'],
            attrs=texture_result['voxel_attrs'],
            voxel_size=np.array([texture_result['voxel_size']]),
            vertices=texture_result['original_vertices'],
            faces=texture_result['original_faces'],
            layout=json.dumps(layout_serializable),
        )
        log.info(f"Voxelgrid saved to: {voxelgrid_npz_path}")

        return ("", voxelgrid_npz_path)


class Trellis2RemoveBackground:
    """Remove background from image using BiRefNet (TRELLIS rembg).

    Note: This is NOT isolated because BiRefNet runs fine in main process
    and doesn't conflict with other packages.
    """

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
        import gc
        import torch
        import numpy as np
        from PIL import Image

        import comfy.model_management as mm

        # Lazy import rembg from trellis2
        from .trellis2.pipelines import rembg

        device = mm.get_torch_device()

        # Load or reuse cached model
        if Trellis2RemoveBackground._model is None:
            log.info("Loading BiRefNet model for background removal...")
            Trellis2RemoveBackground._model = rembg.BiRefNet(model_name="briaai/RMBG-2.0")
            if not low_vram:
                Trellis2RemoveBackground._model.to(device)

        model = Trellis2RemoveBackground._model

        # Convert ComfyUI tensor to PIL
        if image.dim() == 4:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        log.info("Removing background...")

        if low_vram:
            model.to(device)

        # Run BiRefNet - returns RGBA image
        output = model(pil_image)

        if low_vram:
            model.cpu()
            gc.collect()
            mm.soft_empty_cache()

        # Extract mask from alpha channel
        output_np = np.array(output)
        mask_np = output_np[:, :, 3].astype(np.float32) / 255.0

        # Convert mask to ComfyUI format (B, H, W)
        mask_tensor = torch.tensor(mask_np).unsqueeze(0)

        log.info("Background removed successfully")

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

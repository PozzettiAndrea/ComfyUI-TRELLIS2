"""Inference nodes for TRELLIS.2 Image-to-3D generation."""
import logging
import comfy.model_management
from comfy_api.latest import io

log = logging.getLogger("trellis2")


class Trellis2GetConditioning(io.ComfyNode):
    """Extract image conditioning using DinoV3 for TRELLIS.2."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2GetConditioning",
            display_name="TRELLIS.2 Get Conditioning",
            category="TRELLIS2",
            description="""Preprocess image and extract visual features using DinoV3.

This node handles:
1. Applying mask as alpha channel
2. Cropping to object bounding box
3. Alpha premultiplication
4. DinoV3 feature extraction

Parameters:
- model_config: The loaded TRELLIS.2 config
- image: Input image (RGB)
- mask: Foreground mask (white=object, black=background)
Use any background removal node (BiRefNet, rembg, etc.) to generate the mask.""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config"),
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Combo.Input("background_color", options=["black", "gray", "white"],
                               default="black", optional=True),
            ],
            outputs=[
                io.Custom("TRELLIS2_CONDITIONING").Output(display_name="conditioning"),
                io.Image.Output(display_name="preprocessed_image"),
            ],
        )

    @classmethod
    def execute(cls, model_config, image, mask, background_color="black"):
        # All heavy imports happen inside subprocess
        from .stages import run_conditioning

        comfy.model_management.throw_exception_if_processing_interrupted()

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

        return io.NodeOutput(conditioning, preprocessed_image)


class Trellis2ImageToShape(io.ComfyNode):
    """Generate 3D shape from conditioning using TRELLIS.2."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ImageToShape",
            display_name="TRELLIS.2 Image to Shape",
            category="TRELLIS2",
            description="""Generate 3D shape from image conditioning.

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
- mesh: Untextured mesh for preview/export""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config",
                    tooltip="Model config from Load TRELLIS.2 Models node"),
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning",
                    tooltip="Image conditioning from Get Conditioning node"),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                             tooltip="Random seed for reproducible generation"),
                # Sparse Structure Sampler
                io.Float.Input("ss_guidance_strength", default=7.5, min=1.0, max=20.0, step=0.1, optional=True,
                               tooltip="Sparse structure CFG scale. Higher = stronger adherence to input image"),
                io.Int.Input("ss_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Sparse structure sampling steps. More steps = better quality but slower"),
                # Shape SLat Sampler
                io.Float.Input("shape_guidance_strength", default=7.5, min=1.0, max=20.0, step=0.1, optional=True,
                               tooltip="Shape CFG scale. Higher = stronger adherence to input image"),
                io.Int.Input("shape_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Shape sampling steps. More steps = better quality but slower"),
                # VRAM Control
                io.Int.Input("max_tokens", default=49152, min=16384, max=65536, step=4096, optional=True,
                             tooltip="Max tokens for 1024 cascade. Lower = less VRAM but potentially lower quality. Default 49152 (~9GB), try 32768 (~7GB) or 24576 (~6GB) for lower VRAM."),
                io.Boolean.Input("use_vb", default=True, optional=True,
                                 tooltip="Use o_voxel_vb (tiled decoder) vs o_voxel (upstream). Toggle to A/B test mesh extraction."),
            ],
            outputs=[
                io.Custom("TRELLIS2_SHAPE_RESULT").Output(display_name="shape_result"),
                io.Custom("TRIMESH").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config,
        conditioning,
        seed=0,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        shape_guidance_strength=7.5,
        shape_sampling_steps=12,
        max_tokens=49152,
        use_vb=True,
    ):
        # All heavy imports happen inside subprocess
        import trimesh as Trimesh
        from .stages import run_shape_generation

        comfy.model_management.throw_exception_if_processing_interrupted()

        # run_shape_generation returns (result_dict, vertices, faces)
        # result_dict is passed to downstream nodes, vertices/faces used for Trimesh
        import torch
        with torch.inference_mode():
            shape_result, vertices, faces = run_shape_generation(
                model_config=model_config,
                conditioning=conditioning,
                seed=seed,
                ss_guidance_strength=ss_guidance_strength,
                ss_sampling_steps=ss_sampling_steps,
                shape_guidance_strength=shape_guidance_strength,
                shape_sampling_steps=shape_sampling_steps,
                max_num_tokens=max_tokens,
                use_vb=use_vb,
            )

        # Create trimesh from vertices/faces
        tri_mesh = Trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )

        return io.NodeOutput(shape_result, tri_mesh)


class Trellis2ShapeToTexturedMesh(io.ComfyNode):
    """Generate PBR textured mesh from shape using TRELLIS.2."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ShapeToTexturedMesh",
            display_name="TRELLIS.2 Shape to Textured Mesh",
            category="TRELLIS2",
            description="""Generate PBR textured mesh from shape.

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
- voxelgrid_npz_path: Path to voxelgrid data (.npz file with coords, attrs, vertices, faces, etc.)""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config",
                    tooltip="Model config from Load TRELLIS.2 Models node"),
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning",
                    tooltip="Image conditioning from Get Conditioning node (same as used for shape)"),
                io.Custom("TRELLIS2_SHAPE_RESULT").Input("shape_result",
                    tooltip="Shape result from Image to Shape node"),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                             tooltip="Random seed for texture variation"),
                io.Float.Input("tex_guidance_strength", default=7.5, min=1.0, max=20.0, step=0.1, optional=True,
                               tooltip="Texture CFG scale. Higher = stronger adherence to input image"),
                io.Int.Input("tex_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Texture sampling steps. More steps = better quality but slower"),
            ],
            outputs=[
                io.String.Output(display_name="mesh_glb_path"),
                io.String.Output(display_name="voxelgrid_npz_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
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
        from .stages import run_texture_generation

        comfy.model_management.throw_exception_if_processing_interrupted()

        import torch
        with torch.inference_mode():
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

        return io.NodeOutput("", voxelgrid_npz_path)


class Trellis2RemoveBackground(io.ComfyNode):
    """Remove background from image using BiRefNet (TRELLIS rembg).

    Note: This is NOT isolated because BiRefNet runs fine in main process
    and doesn't conflict with other packages.
    """

    _model = None  # Class-level cache

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2RemoveBackground",
            display_name="TRELLIS.2 Remove Background",
            category="TRELLIS2",
            description="""Remove background from image using BiRefNet (same as TRELLIS rembg).

This node extracts a foreground mask using the BiRefNet segmentation model.
The mask can be used with the "Get Conditioning" node.

Parameters:
- image: Input RGB image
- low_vram: Move model to CPU when not in use (slower but saves VRAM)

Returns:
- image: Original image (unchanged)
- mask: Foreground mask (white=object, black=background)""",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("low_vram", default=True, optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, image, low_vram=True):
        import gc
        import torch
        import numpy as np
        from PIL import Image

        import comfy.model_management as mm

        # Lazy import rembg from trellis2
        from . import rembg

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

        comfy.model_management.throw_exception_if_processing_interrupted()

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
        return io.NodeOutput(image, mask_tensor)


class Trellis2LoadMesh(io.ComfyNode):
    """Load a 3D mesh file and return as TRIMESH."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2LoadMesh",
            display_name="TRELLIS.2 Load Mesh",
            category="TRELLIS2",
            description="""Load a 3D mesh from file.

Supports GLB, GLTF, OBJ, PLY, STL, 3MF, DAE, OFF and other formats
supported by the trimesh library.

Parameters:
- mesh_path: Absolute path to the mesh file""",
            inputs=[
                io.String.Input("mesh_path", default="",
                                tooltip="Absolute path to mesh file (GLB, OBJ, PLY, STL, etc.)"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(cls, mesh_path):
        import os
        import trimesh as Trimesh

        if not mesh_path or not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file not found: {mesh_path}")

        log.info(f"Loading mesh from: {mesh_path}")
        mesh = Trimesh.load(mesh_path, process=False, force='mesh')

        # If Scene returned, concatenate all geometry
        if isinstance(mesh, Trimesh.Scene):
            meshes = []
            for name, geom in mesh.geometry.items():
                if isinstance(geom, Trimesh.Trimesh):
                    meshes.append(geom)
            if not meshes:
                raise ValueError(f"No mesh geometry found in: {mesh_path}")
            mesh = Trimesh.util.concatenate(meshes)

        log.info(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return io.NodeOutput(mesh)


class Trellis2EncodeMesh(io.ComfyNode):
    """Encode a mesh into a TRELLIS.2 shape latent for retexturing or refinement."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2EncodeMesh",
            display_name="TRELLIS.2 Encode Mesh",
            category="TRELLIS2",
            description="""Encode a mesh into a TRELLIS.2 shape structured latent.

Uses the FlexiDualGrid VAE Encoder to convert mesh geometry into TRELLIS.2's
latent space. The latent can then be used for:
- Standalone retexturing (Texture Mesh node)
- Geometry refinement (Refine Mesh node)

The mesh is automatically centered and scaled to [-0.5, 0.5]^3.
First run will download the encoder weights (~950MB) from HuggingFace.

Parameters:
- model_config: The loaded model config
- mesh: Input TRIMESH to encode
- resolution: Grid resolution for voxelization (default 1024)""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config",
                    tooltip="Model config from Load TRELLIS.2 Models node"),
                io.Custom("TRIMESH").Input("mesh",
                    tooltip="Input mesh to encode"),
                io.Int.Input("resolution", default=1024, min=256, max=2048, step=128, optional=True,
                             tooltip="Encoding grid resolution. Higher = more detail but slower. 1024 recommended."),
            ],
            outputs=[
                io.Custom("TRELLIS2_SHAPE_LATENT").Output(display_name="shape_latent"),
            ],
        )

    @classmethod
    def execute(cls, model_config, mesh, resolution=1024):
        import torch
        import numpy as np
        from .stages import run_encode_mesh

        comfy.model_management.throw_exception_if_processing_interrupted()

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        with torch.inference_mode():
            shape_latent = run_encode_mesh(
                model_config=model_config,
                vertices=vertices,
                faces=faces,
                resolution=resolution,
            )

        return io.NodeOutput(shape_latent)


class Trellis2TextureMesh(io.ComfyNode):
    """Generate PBR textures for an existing mesh from a reference image."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2TextureMesh",
            display_name="TRELLIS.2 Texture Mesh (Standalone)",
            category="TRELLIS2",
            description="""Generate PBR textures for an existing mesh using a reference image.

This is the "retexture" workflow: take any mesh, encode it with Encode Mesh,
then generate new PBR materials (base_color, metallic, roughness, alpha)
guided by a reference image.

Unlike the standard texture path, this decodes WITHOUT subdivision guidance
since the mesh was not generated by TRELLIS.2's shape pipeline.

Output is a voxelgrid NPZ file compatible with Export GLB, Simplify, etc.

Parameters:
- model_config: Loaded model config
- conditioning: DinoV3 conditioning from the new reference image
- shape_latent: Encoded shape from Encode Mesh
- seed: Random seed
- tex_*: Texture sampling parameters""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config",
                    tooltip="Model config from Load TRELLIS.2 Models node"),
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning",
                    tooltip="Image conditioning from Get Conditioning node (the new texture reference)"),
                io.Custom("TRELLIS2_SHAPE_LATENT").Input("shape_latent",
                    tooltip="Encoded shape latent from Encode Mesh node"),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                             tooltip="Random seed for texture variation"),
                io.Float.Input("tex_guidance_strength", default=3.5, min=1.0, max=20.0, step=0.1, optional=True,
                               tooltip="Texture CFG scale. Higher = stronger adherence to input image"),
                io.Int.Input("tex_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Texture sampling steps"),
            ],
            outputs=[
                io.String.Output(display_name="voxelgrid_npz_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config,
        conditioning,
        shape_latent,
        seed=0,
        tex_guidance_strength=3.5,
        tex_sampling_steps=12,
    ):
        import json
        import os
        import uuid
        import numpy as np
        import torch
        from .stages import run_texture_mesh

        comfy.model_management.throw_exception_if_processing_interrupted()

        with torch.inference_mode():
            texture_result = run_texture_mesh(
                model_config=model_config,
                conditioning=conditioning,
                shape_latent=shape_latent,
                seed=seed,
                tex_guidance_strength=tex_guidance_strength,
                tex_sampling_steps=tex_sampling_steps,
            )

        # Save voxelgrid NPZ (same format as existing ShapeToTexturedMesh)
        cache_dir = '/tmp/trellis2_cache'
        os.makedirs(cache_dir, exist_ok=True)
        file_id = uuid.uuid4().hex[:8]

        pbr_layout = texture_result['pbr_layout']
        layout_serializable = {k: (v.start, v.stop) for k, v in pbr_layout.items()}

        voxelgrid_npz_path = os.path.join(cache_dir, f'voxelgrid_retex_{file_id}.npz')
        np.savez(
            voxelgrid_npz_path,
            coords=texture_result['voxel_coords'],
            attrs=texture_result['voxel_attrs'],
            voxel_size=np.array([texture_result['voxel_size']]),
            vertices=texture_result['original_vertices'],
            faces=texture_result['original_faces'],
            layout=json.dumps(layout_serializable),
        )
        log.info(f"Retexture voxelgrid saved to: {voxelgrid_npz_path}")

        return io.NodeOutput(voxelgrid_npz_path)


class Trellis2RefineMesh(io.ComfyNode):
    """Refine mesh geometry by re-sampling shape at higher resolution."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2RefineMesh",
            display_name="TRELLIS.2 Refine Mesh",
            category="TRELLIS2",
            description="""Refine mesh geometry by re-sampling shape at higher resolution.

Takes an encoded mesh shape latent and:
1. Upsamples via the shape decoder to get high-resolution coordinates
2. Re-samples a new shape latent at those coordinates
3. Decodes to a refined mesh with improved geometric detail

The output shape_result is compatible with the existing "Shape to Textured Mesh"
node for full PBR texturing with subdivision guidance.

Parameters:
- model_config: Loaded model config
- conditioning: DinoV3 conditioning (guides the refinement)
- shape_latent: Encoded shape from Encode Mesh
- seed: Random seed
- shape_*: Shape sampling parameters
- max_tokens: VRAM limit control
- use_vb: Toggle tiled vs upstream decoder""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config",
                    tooltip="Model config from Load TRELLIS.2 Models node"),
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning",
                    tooltip="Image conditioning from Get Conditioning node"),
                io.Custom("TRELLIS2_SHAPE_LATENT").Input("shape_latent",
                    tooltip="Encoded shape latent from Encode Mesh node"),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                             tooltip="Random seed for refinement"),
                io.Float.Input("shape_guidance_strength", default=7.5, min=1.0, max=20.0, step=0.1, optional=True,
                               tooltip="Shape CFG scale"),
                io.Int.Input("shape_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Shape sampling steps"),
                io.Int.Input("max_tokens", default=49152, min=16384, max=65536, step=4096, optional=True,
                             tooltip="Max tokens for HR resolution. Lower = less VRAM."),
                io.Boolean.Input("use_vb", default=True, optional=True,
                                 tooltip="Use tiled decoder (o_voxel_vb) vs upstream (o_voxel)"),
            ],
            outputs=[
                io.Custom("TRELLIS2_SHAPE_RESULT").Output(display_name="shape_result"),
                io.Custom("TRIMESH").Output(display_name="mesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config,
        conditioning,
        shape_latent,
        seed=0,
        shape_guidance_strength=7.5,
        shape_sampling_steps=12,
        max_tokens=49152,
        use_vb=True,
    ):
        import trimesh as Trimesh
        import torch
        from .stages import run_refine_mesh

        comfy.model_management.throw_exception_if_processing_interrupted()

        with torch.inference_mode():
            shape_result, vertices, faces = run_refine_mesh(
                model_config=model_config,
                conditioning=conditioning,
                shape_latent=shape_latent,
                seed=seed,
                shape_guidance_strength=shape_guidance_strength,
                shape_sampling_steps=shape_sampling_steps,
                max_num_tokens=max_tokens,
                use_vb=use_vb,
            )

        tri_mesh = Trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        return io.NodeOutput(shape_result, tri_mesh)


NODE_CLASS_MAPPINGS = {
    "Trellis2RemoveBackground": Trellis2RemoveBackground,
    "Trellis2GetConditioning": Trellis2GetConditioning,
    "Trellis2ImageToShape": Trellis2ImageToShape,
    "Trellis2ShapeToTexturedMesh": Trellis2ShapeToTexturedMesh,
    "Trellis2LoadMesh": Trellis2LoadMesh,
    "Trellis2EncodeMesh": Trellis2EncodeMesh,
    "Trellis2TextureMesh": Trellis2TextureMesh,
    "Trellis2RefineMesh": Trellis2RefineMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2RemoveBackground": "TRELLIS.2 Remove Background",
    "Trellis2GetConditioning": "TRELLIS.2 Get Conditioning",
    "Trellis2ImageToShape": "TRELLIS.2 Image to Shape",
    "Trellis2ShapeToTexturedMesh": "TRELLIS.2 Shape to Textured Mesh",
    "Trellis2LoadMesh": "TRELLIS.2 Load Mesh",
    "Trellis2EncodeMesh": "TRELLIS.2 Encode Mesh",
    "Trellis2TextureMesh": "TRELLIS.2 Texture Mesh (Standalone)",
    "Trellis2RefineMesh": "TRELLIS.2 Refine Mesh",
}

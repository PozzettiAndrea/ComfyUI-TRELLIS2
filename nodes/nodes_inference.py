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
- background_color: Color of background to fill (black, gray, or white)
- crop_margin: Padding ratio around the segmented object before DinoV3 extraction (0.1 = 10%).

Use any background removal node (BiRefNet, rembg, etc.) to generate the mask.""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config"),
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Combo.Input("background_color", options=["black", "gray", "white"],
                               default="black", optional=True),
                io.Float.Input("crop_margin", default=0.1, min=0.0, max=0.5, step=0.01, optional=True,
                               tooltip="Padding ratio around the segmented object before DinoV3 extraction (0.1 = 10%)."),
            ],
            outputs=[
                io.Custom("TRELLIS2_CONDITIONING").Output(display_name="conditioning"),
                io.Image.Output(display_name="preprocessed_image"),
            ],
        )

    @classmethod
    def execute(cls, model_config, image, mask, background_color="black", crop_margin=0.1):
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
            crop_margin=crop_margin,
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

Returns mesh, shape_slat (for texture generation), and subs (subdivision guides).""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config",
                    tooltip="Model config from Load TRELLIS.2 Models node"),
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning",
                    tooltip="Image conditioning from Get Conditioning node"),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                             tooltip="Random seed for reproducible generation"),
                # Sparse Structure Sampler
                io.Float.Input("ss_guidance_strength", default=6.5, min=0.0, max=99.99, step=0.01, optional=True,
                               tooltip="Sparse structure CFG scale. Higher = stronger adherence to input image"),
                io.Float.Input("ss_guidance_rescale", default=0.05, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Sparse structure guidance rescale. Reduces artifacts from high CFG"),
                io.Int.Input("ss_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Sparse structure sampling steps. More steps = better quality but slower"),
                # Shape SLat Sampler
                io.Float.Input("shape_guidance_strength", default=6.5, min=0.0, max=99.99, step=0.01, optional=True,
                               tooltip="Shape CFG scale. Higher = stronger adherence to input image"),
                io.Float.Input("shape_guidance_rescale", default=0.05, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Shape guidance rescale. Reduces artifacts from high CFG"),
                io.Int.Input("shape_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Shape sampling steps. More steps = better quality but slower"),
                # VRAM Control
                io.Int.Input("max_tokens", default=49152, min=16384, max=262144, step=4096, optional=True,
                             tooltip="Max tokens for 1024 cascade. Lower = less VRAM but potentially lower quality. Default 49152 (~9GB), try 32768 (~7GB) or 24576 (~6GB) for lower VRAM."),
                io.Float.Input("shape_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True,
                               tooltip="Shape generation flow-matching time rescaling parameter (rescale_t). Higher values spend more sampler steps on fine details, creating sharper geometry."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.Custom("TRELLIS2_SHAPE_SLAT").Output(display_name="shape_slat"),
                io.Custom("TRELLIS2_SUBS").Output(display_name="subs"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config,
        conditioning,
        seed=0,
        ss_guidance_strength=6.5,
        ss_guidance_rescale=0.05,
        ss_sampling_steps=12,
        shape_guidance_strength=6.5,
        shape_guidance_rescale=0.05,
        shape_sampling_steps=12,
        max_tokens=49152,
        shape_rescale_t=3.0,
    ):
        import numpy as np
        import trimesh as Trimesh
        from .stages import run_shape_generation

        comfy.model_management.throw_exception_if_processing_interrupted()

        import torch
        with torch.inference_mode():
            mesh_verts, mesh_faces, shape_slat_data, subs_data = run_shape_generation(
                model_config=model_config,
                conditioning=conditioning,
                seed=seed,
                ss_guidance_strength=ss_guidance_strength,
                ss_guidance_rescale=ss_guidance_rescale,
                ss_sampling_steps=ss_sampling_steps,
                shape_guidance_strength=shape_guidance_strength,
                shape_guidance_rescale=shape_guidance_rescale,
                shape_sampling_steps=shape_sampling_steps,
                max_num_tokens=max_tokens,
                shape_rescale_t=shape_rescale_t,
            )

        # TRIMESH stays in internal Z-up (conversion to Y-up happens only at export)
        vertices = mesh_verts.numpy().astype(np.float32)
        faces = mesh_faces.numpy()
        mesh = Trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        return io.NodeOutput(mesh, shape_slat_data, subs_data)


class Trellis2ShapeToTexturedMesh(io.ComfyNode):
    """Generate PBR voxel texture from shape latent using TRELLIS.2."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ShapeToTexturedMesh",
            display_name="TRELLIS.2 Shape to Textured Mesh",
            category="TRELLIS2",
            description="""Generate PBR voxel texture from shape latent.

Takes shape_slat and subs from "Image to Shape" and generates PBR materials
(base_color, metallic, roughness, alpha) as a voxel grid.""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config",
                    tooltip="Model config from Load TRELLIS.2 Models node"),
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning",
                    tooltip="Image conditioning from Get Conditioning node (same as used for shape)"),
                io.Custom("TRELLIS2_SHAPE_SLAT").Input("shape_slat",
                    tooltip="Shape structured latent from Image to Shape node"),
                io.Custom("TRELLIS2_SUBS").Input("subs",
                    tooltip="Subdivision guides from Image to Shape node"),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                             tooltip="Random seed for texture variation"),
                io.Float.Input("tex_guidance_strength", default=3.0, min=0.0, max=99.99, step=0.01, optional=True,
                               tooltip="Texture CFG scale. Higher = stronger adherence to input image"),
                io.Float.Input("tex_guidance_rescale", default=0.20, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Texture guidance rescale. Reduces artifacts from high CFG"),
                io.Int.Input("tex_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Texture sampling steps. More steps = better quality but slower"),
                io.Float.Input("tex_rescale_t", default=3.0, min=1.0, max=6.0, step=0.1, optional=True,
                               tooltip="Time-rescaling parameter for the texture flow sampler (Microsoft default: 3.0)."),
            ],
            outputs=[
                io.Custom("TRELLIS2_VOXELGRID").Output(display_name="voxelgrid"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config,
        conditioning,
        shape_slat,
        subs,
        seed=0,
        tex_guidance_strength=3.0,
        tex_guidance_rescale=0.20,
        tex_sampling_steps=12,
        tex_rescale_t=3.0,
    ):
        from .stages import run_texture_generation

        comfy.model_management.throw_exception_if_processing_interrupted()

        import torch
        with torch.inference_mode():
            voxelgrid = run_texture_generation(
                model_config=model_config,
                conditioning=conditioning,
                shape_slat_data=shape_slat,
                subs_data=subs,
                seed=seed,
                tex_guidance_strength=tex_guidance_strength,
                tex_guidance_rescale=tex_guidance_rescale,
                tex_sampling_steps=tex_sampling_steps,
                tex_rescale_t=tex_rescale_t,
            )

        return io.NodeOutput(voxelgrid)


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
                io.Float.Input("tex_guidance_strength", default=3.0, min=0.0, max=99.99, step=0.01, optional=True,
                               tooltip="Texture CFG scale. Higher = stronger adherence to input image"),
                io.Float.Input("tex_guidance_rescale", default=0.20, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Texture guidance rescale. Reduces artifacts from high CFG"),
                io.Int.Input("tex_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Texture sampling steps"),
                io.Float.Input("tex_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True,
                               tooltip="Texture flow-matching time rescaling parameter (rescale_t). Higher values spend more sampler steps on fine details, creating sharper textures."),
            ],
            outputs=[
                io.Custom("TRELLIS2_VOXELGRID").Output(display_name="voxelgrid"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config,
        conditioning,
        shape_latent,
        seed=0,
        tex_guidance_strength=3.0,
        tex_guidance_rescale=0.20,
        tex_sampling_steps=12,
        tex_rescale_t=3.0,
    ):
        import torch
        from .stages import run_texture_mesh

        comfy.model_management.throw_exception_if_processing_interrupted()

        with torch.inference_mode():
            voxelgrid = run_texture_mesh(
                model_config=model_config,
                conditioning=conditioning,
                shape_latent=shape_latent,
                seed=seed,
                tex_guidance_strength=tex_guidance_strength,
                tex_guidance_rescale=tex_guidance_rescale,
                tex_sampling_steps=tex_sampling_steps,
                tex_rescale_t=tex_rescale_t,
            )

        return io.NodeOutput(voxelgrid)


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

Outputs mesh, shape_slat, and subs — compatible with "Shape to Textured Mesh".

Parameters:
- model_config: Loaded model config
- conditioning: DinoV3 conditioning (guides the refinement)
- shape_latent: Encoded shape from Encode Mesh
- seed: Random seed
- shape_*: Shape sampling parameters
- max_tokens: VRAM limit control
""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config",
                    tooltip="Model config from Load TRELLIS.2 Models node"),
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning",
                    tooltip="Image conditioning from Get Conditioning node"),
                io.Custom("TRELLIS2_SHAPE_LATENT").Input("shape_latent",
                    tooltip="Encoded shape latent from Encode Mesh node"),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                             tooltip="Random seed for refinement"),
                io.Float.Input("shape_guidance_strength", default=6.5, min=0.0, max=99.99, step=0.01, optional=True,
                               tooltip="Shape CFG scale"),
                io.Float.Input("shape_guidance_rescale", default=0.05, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Shape guidance rescale. Reduces artifacts from high CFG"),
                io.Int.Input("shape_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Shape sampling steps"),
                io.Int.Input("max_tokens", default=49152, min=16384, max=262144, step=4096, optional=True,
                             tooltip="Max tokens for HR resolution. Lower = less VRAM."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.Custom("TRELLIS2_SHAPE_SLAT").Output(display_name="shape_slat"),
                io.Custom("TRELLIS2_SUBS").Output(display_name="subs"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config,
        conditioning,
        shape_latent,
        seed=0,
        shape_guidance_strength=6.5,
        shape_guidance_rescale=0.05,
        shape_sampling_steps=12,
        max_tokens=49152,
    ):
        import numpy as np
        import torch
        import trimesh as Trimesh
        from .stages import run_refine_mesh

        comfy.model_management.throw_exception_if_processing_interrupted()

        with torch.inference_mode():
            mesh_verts, mesh_faces, shape_slat_data, subs_data = run_refine_mesh(
                model_config=model_config,
                conditioning=conditioning,
                shape_latent=shape_latent,
                seed=seed,
                shape_guidance_strength=shape_guidance_strength,
                shape_guidance_rescale=shape_guidance_rescale,
                shape_sampling_steps=shape_sampling_steps,
                max_num_tokens=max_tokens,
            )

        # TRIMESH stays in internal Z-up (conversion to Y-up happens only at export)
        vertices = mesh_verts.numpy().astype(np.float32)
        faces = mesh_faces.numpy()
        mesh = Trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        return io.NodeOutput(mesh, shape_slat_data, subs_data)


class Trellis2ShapeToMesh(io.ComfyNode):
    """Simplify and/or fill holes in a mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ShapeToMesh",
            display_name="TRELLIS.2 Simplify Mesh",
            category="TRELLIS2",
            description="""Simplify mesh and/or fill holes.

Parameters:
- target_face_count: Target number of faces (0 = no simplification)
- fill_holes: Fill small holes before simplifying
- fill_holes_perimeter: Max hole perimeter to fill""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh",
                    tooltip="Mesh from Image to Shape or other mesh source"),
                io.Int.Input("target_face_count", default=200000, min=0, max=5000000, step=1000,
                    tooltip="Target face count. 0 = no simplification."),
                io.Boolean.Input("fill_holes", default=False, optional=True),
                io.Float.Input("fill_holes_perimeter", default=0.03, min=0.001, max=0.5, step=0.001, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, target_face_count=0, fill_holes=True, fill_holes_perimeter=0.03):
        import numpy as np
        import trimesh as Trimesh

        vertices = np.array(trimesh.vertices, dtype=np.float32)
        faces = np.array(trimesh.faces)

        log.info(f"SimplifyMesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

        simplify = target_face_count > 0 and faces.shape[0] > target_face_count

        if simplify or fill_holes:
            import torch
            import cumesh_vb as CuMesh

            device = comfy.model_management.get_torch_device()

            verts_t = torch.tensor(vertices, dtype=torch.float32).to(device)
            faces_t = torch.tensor(faces, dtype=torch.int32).to(device)

            cumesh = CuMesh.CuMesh()
            cumesh.init(verts_t, faces_t)

            if fill_holes:
                cumesh.fill_holes(max_hole_perimeter=fill_holes_perimeter)
                log.info(f"After fill holes: {cumesh.num_vertices} verts, {cumesh.num_faces} faces")

            if simplify:
                cumesh.simplify(target_face_count, verbose=True)
                log.info(f"After simplify: {cumesh.num_vertices} verts, {cumesh.num_faces} faces")

            out_v, out_f = cumesh.read()
            vertices = out_v.cpu().numpy()
            faces = out_f.cpu().numpy()

            del verts_t, faces_t, cumesh
            comfy.model_management.soft_empty_cache()

        mesh = Trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        log.info(f"SimplifyMesh output: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        return io.NodeOutput(mesh)


class Trellis2MultiViewImageToShape(io.ComfyNode):
    """Generate 3D shape from multiple view images using spatial blending."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2MultiViewImageToShape",
            display_name="TRELLIS.2 Multi-View Image to Shape",
            category="TRELLIS2",
            description="""Generate 3D shape from multiple view images.

At each sampling step, runs the model once per view and blends predictions
based on which voxels face which direction. Supports up to 6 views.

Front image + mask are required. Other views are optional — only provided
views participate in blending.""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config",
                    tooltip="Model config from Load TRELLIS.2 Models node"),
                # Front view (required)
                io.Image.Input("front_image", tooltip="Front view image (required)"),
                io.Mask.Input("front_mask", tooltip="Front view mask (required)"),
                # Optional views
                io.Image.Input("back_image", optional=True, tooltip="Back view image"),
                io.Mask.Input("back_mask", optional=True, tooltip="Back view mask"),
                io.Image.Input("left_image", optional=True, tooltip="Left view image"),
                io.Mask.Input("left_mask", optional=True, tooltip="Left view mask"),
                io.Image.Input("right_image", optional=True, tooltip="Right view image"),
                io.Mask.Input("right_mask", optional=True, tooltip="Right view mask"),
                io.Image.Input("top_image", optional=True, tooltip="Top view image"),
                io.Mask.Input("top_mask", optional=True, tooltip="Top view mask"),
                io.Image.Input("bottom_image", optional=True, tooltip="Bottom view image"),
                io.Mask.Input("bottom_mask", optional=True, tooltip="Bottom view mask"),
                # Sampling params
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1, optional=True,
                             tooltip="Random seed for reproducible generation"),
                io.Float.Input("ss_guidance_strength", default=6.5, min=0.0, max=99.99, step=0.01, optional=True,
                               tooltip="Sparse structure CFG scale"),
                io.Float.Input("ss_guidance_rescale", default=0.05, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Sparse structure guidance rescale"),
                io.Int.Input("ss_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Sparse structure sampling steps"),
                io.Float.Input("shape_guidance_strength", default=6.5, min=0.0, max=99.99, step=0.01, optional=True,
                               tooltip="Shape CFG scale"),
                io.Float.Input("shape_guidance_rescale", default=0.05, min=0.0, max=1.0, step=0.01, optional=True,
                               tooltip="Shape guidance rescale"),
                io.Int.Input("shape_sampling_steps", default=12, min=1, max=50, step=1, optional=True,
                             tooltip="Shape sampling steps"),
                io.Int.Input("max_tokens", default=49152, min=16384, max=262144, step=4096, optional=True,
                             tooltip="Max tokens for cascade. Lower = less VRAM."),
                # Multi-view params
                io.Combo.Input("front_axis", options=["z", "x"], default="z", optional=True,
                               tooltip="Which 3D axis the front view faces. Z for most models."),
                io.Float.Input("blend_temperature", default=2.0, min=0.1, max=10.0, step=0.1, optional=True,
                               tooltip="Softmax temperature for view blending. Higher = sharper transitions between views."),
                io.Combo.Input("background_color", options=["black", "gray", "white"], default="black", optional=True,
                               tooltip="Background color for masked regions"),
                io.Float.Input("shape_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True,
                               tooltip="Shape generation flow-matching time rescaling parameter (rescale_t). Higher values spend more sampler steps on fine details, creating sharper geometry."),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.Custom("TRELLIS2_SHAPE_SLAT").Output(display_name="shape_slat"),
                io.Custom("TRELLIS2_SUBS").Output(display_name="subs"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config,
        front_image,
        front_mask,
        back_image=None, back_mask=None,
        left_image=None, left_mask=None,
        right_image=None, right_mask=None,
        top_image=None, top_mask=None,
        bottom_image=None, bottom_mask=None,
        seed=0,
        ss_guidance_strength=6.5,
        ss_guidance_rescale=0.05,
        ss_sampling_steps=12,
        shape_guidance_strength=6.5,
        shape_guidance_rescale=0.05,
        shape_sampling_steps=12,
        max_tokens=49152,
        front_axis='z',
        blend_temperature=2.0,
        background_color="black",
        shape_rescale_t=3.0,
    ):
        import numpy as np
        import torch
        import trimesh as Trimesh
        from .stages import run_conditioning, run_multiview_shape_generation

        comfy.model_management.throw_exception_if_processing_interrupted()

        resolution = model_config.get("resolution", "1024_cascade")
        include_1024 = resolution in ("1024_cascade", "1536_cascade", "1024")

        # Collect view images/masks
        view_inputs = {'front': (front_image, front_mask)}
        for name, img, msk in [
            ('back', back_image, back_mask),
            ('left', left_image, left_mask),
            ('right', right_image, right_mask),
            ('top', top_image, top_mask),
            ('bottom', bottom_image, bottom_mask),
        ]:
            if img is not None:
                if msk is None:
                    # Default mask: all ones (fully visible)
                    msk = torch.ones(img.shape[0], img.shape[1], img.shape[2])
                view_inputs[name] = (img, msk)

        log.info(f"Multi-view conditioning for views: {list(view_inputs.keys())}")

        # Run conditioning for each view
        view_conditionings = {}
        for view_name, (img, msk) in view_inputs.items():
            cond, _ = run_conditioning(
                model_config=model_config,
                image=img,
                mask=msk,
                include_1024=include_1024,
                background_color=background_color,
            )
            view_conditionings[view_name] = cond

        with torch.inference_mode():
            mesh_verts, mesh_faces, shape_slat_data, subs_data = run_multiview_shape_generation(
                model_config=model_config,
                view_conditionings=view_conditionings,
                seed=seed,
                ss_guidance_strength=ss_guidance_strength,
                ss_guidance_rescale=ss_guidance_rescale,
                ss_sampling_steps=ss_sampling_steps,
                shape_guidance_strength=shape_guidance_strength,
                shape_guidance_rescale=shape_guidance_rescale,
                shape_sampling_steps=shape_sampling_steps,
                max_num_tokens=max_tokens,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                shape_rescale_t=shape_rescale_t,
            )

        # TRIMESH stays in internal Z-up (conversion to Y-up happens only at export)
        vertices = mesh_verts.numpy().astype(np.float32)
        faces = mesh_faces.numpy()
        mesh = Trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        return io.NodeOutput(mesh, shape_slat_data, subs_data)


NODE_CLASS_MAPPINGS = {
    "Trellis2RemoveBackground": Trellis2RemoveBackground,
    "Trellis2GetConditioning": Trellis2GetConditioning,
    "Trellis2ImageToShape": Trellis2ImageToShape,
    "Trellis2MultiViewImageToShape": Trellis2MultiViewImageToShape,
    "Trellis2ShapeToTexturedMesh": Trellis2ShapeToTexturedMesh,
    "Trellis2LoadMesh": Trellis2LoadMesh,
    "Trellis2EncodeMesh": Trellis2EncodeMesh,
    "Trellis2TextureMesh": Trellis2TextureMesh,
    "Trellis2RefineMesh": Trellis2RefineMesh,
    "Trellis2ShapeToMesh": Trellis2ShapeToMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2RemoveBackground": "TRELLIS.2 Remove Background",
    "Trellis2GetConditioning": "TRELLIS.2 Get Conditioning",
    "Trellis2ImageToShape": "TRELLIS.2 Image to Shape",
    "Trellis2MultiViewImageToShape": "TRELLIS.2 Multi-View Image to Shape",
    "Trellis2ShapeToTexturedMesh": "TRELLIS.2 Shape to Textured Mesh",
    "Trellis2LoadMesh": "TRELLIS.2 Load Mesh",
    "Trellis2EncodeMesh": "TRELLIS.2 Encode Mesh",
    "Trellis2TextureMesh": "TRELLIS.2 Texture Mesh (Standalone)",
    "Trellis2RefineMesh": "TRELLIS.2 Refine Mesh",
    "Trellis2ShapeToMesh": "TRELLIS.2 Shape to Mesh",
}

"""
TRELLIS2 pipeline stages.

Each stage loads models, runs inference, and optionally unloads.
These run inside the isolated subprocess.
"""

import sys
import gc
from fractions import Fraction
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import torch
import numpy as np
from PIL import Image

from .lazy_manager import get_model_manager
from .helpers import smart_crop_square


def _sparse_tensor_to_dict(st) -> Dict[str, Any]:
    """
    Convert a SparseTensor to a serializable dict.
    This allows SparseTensor to cross IPC boundaries without requiring
    trellis2.modules in the receiving process.
    """
    return {
        '_type': 'SparseTensor',
        'feats': st.feats.cpu(),
        'coords': st.coords.cpu(),
        'shape': tuple(st.shape) if st.shape else None,
        'scale': tuple((s.numerator, s.denominator) for s in st._scale),
    }


def _dict_to_sparse_tensor(d: Dict[str, Any], device: torch.device):
    """
    Reconstruct a SparseTensor from a serialized dict.
    Must be called within the isolated environment where trellis2 is available.
    """
    from trellis2.modules.sparse import SparseTensor

    feats = d['feats'].to(device)
    coords = d['coords'].to(device)
    shape = torch.Size(d['shape']) if d['shape'] else None
    scale = tuple(Fraction(n, den) for n, den in d['scale'])

    return SparseTensor(feats=feats, coords=coords, shape=shape, scale=scale)


def _serialize_for_ipc(obj: Any) -> Any:
    """
    Recursively convert SparseTensor objects to serializable dicts.
    """
    # Check if it's a SparseTensor by checking for the characteristic attributes
    if hasattr(obj, 'feats') and hasattr(obj, 'coords') and hasattr(obj, '_scale'):
        return _sparse_tensor_to_dict(obj)
    elif isinstance(obj, list):
        return [_serialize_for_ipc(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_serialize_for_ipc(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: _serialize_for_ipc(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.cpu()
    else:
        return obj


def _deserialize_from_ipc(obj: Any, device: torch.device) -> Any:
    """
    Recursively reconstruct SparseTensor objects from serialized dicts.
    """
    if isinstance(obj, dict) and obj.get('_type') == 'SparseTensor':
        return _dict_to_sparse_tensor(obj, device)
    elif isinstance(obj, list):
        return [_deserialize_from_ipc(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_deserialize_from_ipc(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: _deserialize_from_ipc(v, device) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def run_conditioning(
    model_config: Any,
    image: torch.Tensor,
    mask: torch.Tensor,
    include_1024: bool = True,
    background_color: str = "black",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Run DinoV3 conditioning extraction.

    Args:
        model_config: Trellis2ModelConfig
        image: ComfyUI IMAGE tensor [B, H, W, C]
        mask: ComfyUI MASK tensor [B, H, W] or [H, W]
        include_1024: Also extract 1024px features
        background_color: Background color name

    Returns:
        Tuple of (conditioning_dict, preprocessed_image_tensor)
    """
    print(f"[TRELLIS2] Running conditioning...", file=sys.stderr)

    # Background color mapping
    bg_colors = {
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "white": (255, 255, 255),
    }
    bg_color = bg_colors.get(background_color, (128, 128, 128))

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model manager
    manager = get_model_manager(
        model_config.model_name,
        model_config.resolution,
        model_config.attn_backend,
        model_config.vram_mode,
    )

    # Convert image to PIL
    if image.dim() == 4:
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    else:
        img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)

    # Process mask
    if mask.dim() == 3:
        mask_np = mask[0].cpu().numpy()
    else:
        mask_np = mask.cpu().numpy()

    # Resize mask to match image if needed
    if mask_np.shape[:2] != (pil_image.height, pil_image.width):
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((pil_image.width, pil_image.height), Image.LANCZOS)
        mask_np = np.array(mask_pil) / 255.0

    # Apply mask as alpha channel
    pil_image = pil_image.convert('RGB')
    alpha_np = (mask_np * 255).astype(np.uint8)
    rgba = np.dstack([np.array(pil_image), alpha_np])
    pil_image = Image.fromarray(rgba, 'RGBA')

    # Smart crop
    pil_image = smart_crop_square(pil_image, alpha_np, margin_ratio=0.1, background_color=bg_color)

    # Load DinoV3 and extract features
    model = manager.get_dinov3(device)

    # Get 512px conditioning
    model.image_size = 512
    cond_512 = model([pil_image])

    # Get 1024px conditioning if requested
    cond_1024 = None
    if include_1024:
        model.image_size = 1024
        cond_1024 = model([pil_image])

    # Unload DinoV3 immediately
    manager.unload_dinov3()

    # Create negative conditioning
    neg_cond = torch.zeros_like(cond_512)

    conditioning = {
        'cond_512': cond_512.cpu(),
        'neg_cond': neg_cond.cpu(),
    }
    if cond_1024 is not None:
        conditioning['cond_1024'] = cond_1024.cpu()

    # Convert preprocessed image to tensor
    pil_rgb = pil_image.convert('RGB') if pil_image.mode != 'RGB' else pil_image
    preprocessed_np = np.array(pil_rgb).astype(np.float32) / 255.0
    preprocessed_tensor = torch.from_numpy(preprocessed_np).unsqueeze(0)

    print(f"[TRELLIS2] Conditioning extracted", file=sys.stderr)
    return conditioning, preprocessed_tensor


def run_shape_generation(
    model_config: Any,
    conditioning: Dict[str, torch.Tensor],
    seed: int = 0,
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 12,
    shape_guidance_strength: float = 7.5,
    shape_sampling_steps: int = 12,
    max_num_tokens: int = 49152,
) -> Dict[str, Any]:
    """
    Run shape generation.

    Args:
        model_config: Trellis2ModelConfig
        conditioning: Dict with cond_512, neg_cond, optionally cond_1024
        seed: Random seed
        ss_*: Sparse structure sampling params
        shape_*: Shape latent sampling params
        max_num_tokens: Max tokens for 1024 cascade (lower = less VRAM)

    Returns:
        Dict with shape_slat, subs, mesh_vertices, mesh_faces, resolution, pipeline_type
        Plus raw_mesh_vertices/faces for texture stage reconstruction
    """
    import cumesh as CuMesh

    print(f"[TRELLIS2] Running shape generation (seed={seed})...", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move conditioning to device
    cond_on_device = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in conditioning.items()
    }

    # Get model manager and shape pipeline
    manager = get_model_manager(
        model_config.model_name,
        model_config.resolution,
        model_config.attn_backend,
        model_config.vram_mode,
    )
    pipeline = manager.get_shape_pipeline(device)

    # Build sampler params
    sampler_params = {
        "sparse_structure_sampler_params": {
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
        },
        "shape_slat_sampler_params": {
            "steps": shape_sampling_steps,
            "guidance_strength": shape_guidance_strength,
        },
    }

    # Run shape generation
    torch.cuda.reset_peak_memory_stats()
    meshes, shape_slat, subs, res = pipeline.run_shape(
        cond_on_device,
        seed=seed,
        pipeline_type=model_config.resolution,
        max_num_tokens=max_num_tokens,
        **sampler_params
    )
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[TRELLIS2] Shape generation peak VRAM: {peak_mem:.0f} MB", file=sys.stderr)
    mesh = meshes[0]
    mesh.fill_holes()

    # Save RAW mesh data for texture stage (before coordinate conversion)
    # These will be used to reconstruct Mesh objects in texture stage
    raw_mesh_vertices = mesh.vertices.cpu()
    raw_mesh_faces = mesh.faces.cpu()

    # Convert mesh to CPU arrays for output (with coordinate conversion)
    # Unify face orientations using CuMesh
    cumesh = CuMesh.CuMesh()
    cumesh.init(mesh.vertices, mesh.faces.int())
    cumesh.unify_face_orientations()
    unified_verts, unified_faces = cumesh.read()

    vertices = unified_verts.cpu().numpy().astype(np.float32)
    faces = unified_faces.cpu().numpy()
    del cumesh, unified_verts, unified_faces

    # Coordinate system conversion (Y-up to Z-up) for output mesh
    vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

    # Pack results - serialize SparseTensor objects to dicts for IPC
    # This allows the result to cross the subprocess boundary without requiring
    # trellis2.modules in the main ComfyUI process
    result = {
        'shape_slat': _serialize_for_ipc(shape_slat),
        'subs': _serialize_for_ipc(subs),
        'mesh_vertices': vertices,  # numpy, coordinate-converted for output
        'mesh_faces': faces,        # numpy, for output
        'resolution': res,
        'pipeline_type': model_config.resolution,
        # Raw mesh data for texture stage reconstruction (CPU tensors, original coords)
        'raw_mesh_vertices': raw_mesh_vertices.cpu(),
        'raw_mesh_faces': raw_mesh_faces.cpu(),
    }

    # Unload shape pipeline
    manager.unload_shape_pipeline()

    print(f"[TRELLIS2] Shape generated: {len(vertices)} verts, {len(faces)} faces", file=sys.stderr)
    return result


def run_texture_generation(
    model_config: Any,
    conditioning: Dict[str, torch.Tensor],
    shape_result: Dict[str, Any],
    seed: int = 0,
    tex_guidance_strength: float = 7.5,
    tex_sampling_steps: int = 12,
) -> Dict[str, Any]:
    """
    Run texture generation.

    Args:
        model_config: Trellis2ModelConfig
        conditioning: Dict with cond_512, neg_cond, optionally cond_1024
        shape_result: Result from run_shape_generation
        seed: Random seed
        tex_*: Texture sampling params

    Returns:
        Dict with textured mesh data
    """
    import cumesh as CuMesh
    from trellis2.representations.mesh import Mesh

    print(f"[TRELLIS2] Running texture generation (seed={seed})...", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move conditioning to device
    cond_on_device = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in conditioning.items()
    }

    # Deserialize and move shape data to device
    # SparseTensor objects were serialized as dicts for IPC, reconstruct them here
    shape_slat = _deserialize_from_ipc(shape_result['shape_slat'], device)
    subs = _deserialize_from_ipc(shape_result['subs'], device)
    resolution = shape_result['resolution']
    pipeline_type = shape_result['pipeline_type']

    # Reconstruct Mesh objects from saved data
    raw_vertices = shape_result['raw_mesh_vertices'].to(device)
    raw_faces = shape_result['raw_mesh_faces'].to(device)
    mesh = Mesh(vertices=raw_vertices, faces=raw_faces)
    mesh.fill_holes()  # Re-apply fill_holes since we reconstructed
    meshes = [mesh]

    # Get model manager and texture pipeline
    manager = get_model_manager(
        model_config.model_name,
        model_config.resolution,
        model_config.attn_backend,
        model_config.vram_mode,
    )
    pipeline = manager.get_texture_pipeline(device)

    # Build sampler params
    sampler_params = {
        "tex_slat_sampler_params": {
            "steps": tex_sampling_steps,
            "guidance_strength": tex_guidance_strength,
        },
    }

    # Run texture generation
    torch.cuda.reset_peak_memory_stats()
    textured_meshes = pipeline.run_texture_with_subs(
        cond_on_device,
        shape_slat,
        subs,
        meshes,
        resolution,
        seed=seed,
        pipeline_type=pipeline_type,
        **sampler_params
    )
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[TRELLIS2] Texture generation peak VRAM: {peak_mem:.0f} MB", file=sys.stderr)
    mesh = textured_meshes[0]
    mesh.simplify(16777216)

    # Get PBR layout from pipeline
    pbr_layout = pipeline.pbr_attr_layout

    # Convert mesh to outputs
    # Unify face orientations
    cumesh = CuMesh.CuMesh()
    cumesh.init(mesh.vertices, mesh.faces.int())
    cumesh.unify_face_orientations()
    unified_verts, unified_faces = cumesh.read()

    vertices = unified_verts.cpu().numpy().astype(np.float32)
    faces = unified_faces.cpu().numpy()
    del cumesh, unified_verts, unified_faces

    # Coordinate conversion
    vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

    # Get voxel grid data
    coords = mesh.coords.cpu().numpy().astype(np.float32)
    attrs = mesh.attrs.cpu().numpy()  # (L, 6) in [-1, 1]
    voxel_size = mesh.voxel_size

    result = {
        'mesh_vertices': vertices,
        'mesh_faces': faces,
        'voxel_coords': coords,
        'voxel_attrs': attrs,
        'voxel_size': voxel_size,
        'pbr_layout': pbr_layout,
        # Keep original mesh data for rasterization
        'original_vertices': mesh.vertices.cpu(),
        'original_faces': mesh.faces.cpu(),
    }

    # Cleanup
    manager.unload_texture_pipeline()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[TRELLIS2] Texture generated: {len(vertices)} verts, {len(coords)} voxels", file=sys.stderr)
    return result

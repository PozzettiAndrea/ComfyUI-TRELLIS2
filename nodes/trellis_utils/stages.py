"""
TRELLIS2 pipeline stages.

Each stage loads models, runs inference, and cleans up.
"""

import gc
import logging
import os
import tempfile
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import torch
import numpy as np
import comfy.model_management
from PIL import Image

log = logging.getLogger("trellis2")

from .helpers import smart_crop_square

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# Noise/conditioning stay float32 for sampling loop stability (error accumulation over 12 steps).
# Model weights are bf16 for memory savings. torch.autocast handles per-op precision:
# matmuls/convs/attention run on bf16 tensor cores, norms stay float32.
_DEFAULT_DTYPE = torch.float32
_MODEL_DTYPE = torch.bfloat16


# Shape models needed for each resolution mode
SHAPE_MODELS_BY_RESOLUTION = {
    '512': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
    ],
    '1024_cascade': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
    ],
    '1536_cascade': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
    ],
}

# Texture models needed for each resolution mode
TEXTURE_MODELS_BY_RESOLUTION = {
    '512': [
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
    ],
    '1024_cascade': [
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
    ],
    '1536_cascade': [
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
    ],
}

# Texture resolution mapping (texture maxes at 1024)
TEXTURE_RESOLUTION_MAP = {
    '512': '512',
    '1024_cascade': '1024_cascade',
    '1536_cascade': '1024_cascade',
}


# Disk-based tensor serialization for IPC
_TRELLIS2_TEMP_DIR = None


def _get_temp_dir():
    """Get or create temp directory for tensor files."""
    global _TRELLIS2_TEMP_DIR
    if _TRELLIS2_TEMP_DIR is None:
        _TRELLIS2_TEMP_DIR = tempfile.mkdtemp(prefix='trellis2_')
        log.info(f"Created temp dir: {_TRELLIS2_TEMP_DIR}")
    return _TRELLIS2_TEMP_DIR


def _save_to_disk(data: dict, prefix: str) -> dict:
    """Save tensor dict to disk, return reference dict."""
    path = os.path.join(_get_temp_dir(), f'{prefix}_{uuid.uuid4().hex[:8]}.pt')
    torch.save(data, path)
    log.info(f"Saved {prefix} to {path}")
    return {'_tensor_file': path}


def _load_from_disk(ref: Any) -> Any:
    """Load tensor dict from disk reference, or return as-is if not a reference."""
    if isinstance(ref, dict) and '_tensor_file' in ref:
        path = ref['_tensor_file']
        log.info(f"Loading from {path}")
        # Safe loading: these files contain only tensors and basic Python types (dicts, lists, tuples)
        # saved by _save_to_disk() for IPC between pipeline stages
        import comfy.utils
        return comfy.utils.load_torch_file(path)
    return ref


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
    from ..trellis2.modules.sparse import SparseTensor

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
    log.info("Running conditioning...")

    # Background color mapping
    bg_colors = {
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "white": (255, 255, 255),
    }
    bg_color = bg_colors.get(background_color, (128, 128, 128))

    # Get device
    device = comfy.model_management.get_torch_device()

    # Convert image to PIL
    if image.dim() == 4:
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    else:
        img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)

    # Process mask - handle various input formats and ensure 2D grayscale
    mask_np = mask.cpu().numpy()

    # Handle 4D [B, H, W, C] format (e.g., IMAGE passed as MASK)
    if mask_np.ndim == 4:
        mask_np = mask_np[0]  # Remove batch -> [H, W, C]

    # Handle 3D format - either [B, H, W] or [H, W, C]
    if mask_np.ndim == 3:
        if mask_np.shape[-1] in (1, 2, 3, 4):  # Likely [H, W, C]
            mask_np = mask_np[..., 0]  # Take first channel -> [H, W]
        else:  # Likely [B, H, W]
            mask_np = mask_np[0]  # Remove batch -> [H, W]

    # Handle 2D with channel dim after squeeze (e.g., [W, C] from squeezed [1, 1, W, C])
    # This catches cases like (1042, 3) where 3 is channels, not width
    if mask_np.ndim == 2 and mask_np.shape[-1] in (1, 2, 3, 4) and mask_np.shape[0] > 10:
        # Last dim looks like channels (small) and first dim looks like spatial (large)
        mask_np = mask_np[..., 0]  # Take first channel

    # Ensure we have at least 2D
    if mask_np.ndim == 1:
        mask_np = mask_np[np.newaxis, :]  # Add height dimension

    if mask_np.ndim != 2:
        raise ValueError(f"Mask must be 2D after processing, got shape {mask_np.shape}")

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

    # Load DinoV3 directly and extract features
    from ..trellis2.modules import image_feature_extractor
    log.info("Loading DinoV3 feature extractor...")
    dinov3_model = image_feature_extractor.DinoV3FeatureExtractor(
        model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    dinov3_model.to(device)
    log.info("DinoV3 loaded")

    # Get 512px conditioning
    dinov3_model.image_size = 512
    cond_512 = dinov3_model([pil_image])

    # Get 1024px conditioning if requested
    cond_1024 = None
    if include_1024:
        dinov3_model.image_size = 1024
        cond_1024 = dinov3_model([pil_image])

    # Unload DinoV3 immediately
    del dinov3_model
    gc.collect()
    comfy.model_management.soft_empty_cache()
    log.info("DinoV3 offloaded")

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

    # Save conditioning to disk for IPC
    conditioning_ref = _save_to_disk(conditioning, 'conditioning')

    log.info("Conditioning extracted")
    return conditioning_ref, preprocessed_tensor


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

    log.info(f"Running shape generation (seed={seed})...")

    device = comfy.model_management.get_torch_device()
    compute_dtype = _DEFAULT_DTYPE        # float32 — noise, conditioning, sampling loop
    model_dtype = _MODEL_DTYPE            # model weights
    resolution = model_config["resolution"]

    # Load conditioning from disk if needed
    conditioning = _load_from_disk(conditioning)

    # Move conditioning to device — keep float32 for sampling loop stability
    cond_on_device = {
        k: v.to(device=device, dtype=compute_dtype) if isinstance(v, torch.Tensor) else v
        for k, v in conditioning.items()
    }

    # Load shape pipeline
    from ..trellis2.pipelines import Trellis2ImageTo3DPipeline

    shape_models = SHAPE_MODELS_BY_RESOLUTION.get(
        resolution, SHAPE_MODELS_BY_RESOLUTION['1024_cascade']
    )

    log.info(f"Loading shape pipeline (model_dtype={model_dtype}, compute_dtype={compute_dtype})...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        model_config["model_name"],
        models_to_load=shape_models,
        dtype=model_dtype,
    )
    pipeline.default_pipeline_type = resolution
    pipeline._device = device
    pipeline._dtype = compute_dtype  # noise and normalization tensors
    log.info("Shape pipeline ready")

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

    # autocast: bf16 tensor cores for matmuls/convs/attention, float32 for norms
    # manual_cast() becomes no-op under autocast (by design)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=model_dtype, enabled=model_dtype is not None):
        meshes, shape_slat, subs, res = pipeline.run_shape(
            cond_on_device,
            seed=seed,
            pipeline_type=resolution,
            max_num_tokens=max_num_tokens,
            **sampler_params
        )
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    log.info(f"Shape generation peak VRAM: {peak_mem:.0f} MB")
    mesh = meshes[0]
    mesh.fill_holes()

    # Save RAW mesh data for texture stage (before coordinate conversion)
    raw_mesh_vertices = mesh.vertices.cpu()
    raw_mesh_faces = mesh.faces.cpu()

    # Convert mesh to CPU arrays for output (with coordinate conversion)
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
    log.debug(f"shape_slat before serialize:")
    log.debug(f"  feats.shape: {shape_slat.feats.shape}")
    log.debug(f"  coords.shape: {shape_slat.coords.shape}")
    log.debug(f"  shape: {shape_slat.shape}")
    log.debug(f"  scale: {shape_slat._scale}")
    result = {
        'shape_slat': _serialize_for_ipc(shape_slat),
        'subs': _serialize_for_ipc(subs),
        'mesh_vertices': vertices,
        'mesh_faces': faces,
        'resolution': res,
        'pipeline_type': resolution,
        'raw_mesh_vertices': raw_mesh_vertices.cpu(),
        'raw_mesh_faces': raw_mesh_faces.cpu(),
    }

    # Unload shape pipeline
    del pipeline
    gc.collect()
    comfy.model_management.soft_empty_cache()
    log.info("Shape pipeline offloaded")

    # Save result to disk for IPC
    result_ref = _save_to_disk(result, 'shape_result')

    log.info(f"Shape generated: {len(vertices)} verts, {len(faces)} faces")
    return result_ref, vertices, faces


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
    from ..trellis2.representations.mesh import Mesh

    log.info(f"Running texture generation (seed={seed})...")

    device = comfy.model_management.get_torch_device()
    compute_dtype = _DEFAULT_DTYPE        # float32 — noise, conditioning, sampling loop
    model_dtype = _MODEL_DTYPE            # model weights
    resolution = model_config["resolution"]

    # Load conditioning and shape_result from disk if needed
    conditioning = _load_from_disk(conditioning)
    shape_result = _load_from_disk(shape_result)

    # Move conditioning to device — keep float32 for sampling loop stability
    cond_on_device = {
        k: v.to(device=device, dtype=compute_dtype) if isinstance(v, torch.Tensor) else v
        for k, v in conditioning.items()
    }

    # Deserialize and move shape data to device
    shape_slat = _deserialize_from_ipc(shape_result['shape_slat'], device)
    subs = _deserialize_from_ipc(shape_result['subs'], device)
    pipeline_type = shape_result['pipeline_type']

    # Reconstruct Mesh objects from saved data
    raw_vertices = shape_result['raw_mesh_vertices'].to(device)
    raw_faces = shape_result['raw_mesh_faces'].to(device)
    mesh = Mesh(vertices=raw_vertices, faces=raw_faces)
    mesh.fill_holes()
    meshes = [mesh]

    # Load texture pipeline
    from ..trellis2.pipelines import Trellis2ImageTo3DPipeline

    texture_resolution = TEXTURE_RESOLUTION_MAP.get(resolution, '1024_cascade')
    texture_models = TEXTURE_MODELS_BY_RESOLUTION.get(
        resolution, TEXTURE_MODELS_BY_RESOLUTION['1024_cascade']
    )

    log.info(f"Loading texture pipeline (model_dtype={model_dtype}, compute_dtype={compute_dtype})...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        model_config["model_name"],
        models_to_load=texture_models,
        dtype=model_dtype,
    )
    pipeline.default_pipeline_type = texture_resolution
    pipeline._device = device
    pipeline._dtype = compute_dtype  # noise and normalization tensors
    log.info("Texture pipeline ready")

    # Build sampler params
    sampler_params = {
        "tex_slat_sampler_params": {
            "steps": tex_sampling_steps,
            "guidance_strength": tex_guidance_strength,
        },
    }

    # autocast: bf16 tensor cores for matmuls/convs/attention, float32 for norms
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=model_dtype, enabled=model_dtype is not None):
        textured_meshes = pipeline.run_texture_with_subs(
            cond_on_device,
            shape_slat,
            subs,
            meshes,
            shape_result['resolution'],
            seed=seed,
            pipeline_type=pipeline_type,
            **sampler_params
        )
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    log.info(f"Texture generation peak VRAM: {peak_mem:.0f} MB")
    mesh = textured_meshes[0]
    mesh.simplify(16777216)  # Light cleanup of degenerate geometry

    pbr_layout = pipeline.pbr_attr_layout

    result = {
        'voxel_coords': mesh.coords.cpu().numpy().astype(np.float32),
        'voxel_attrs': mesh.attrs.cpu().numpy(),
        'voxel_size': mesh.voxel_size,
        'pbr_layout': pbr_layout,
        'original_vertices': mesh.vertices.cpu(),
        'original_faces': mesh.faces.cpu(),
    }

    # Cleanup
    del pipeline
    gc.collect()
    comfy.model_management.soft_empty_cache()
    log.info("Texture pipeline offloaded")

    coords = result['voxel_coords']
    log.info(f"Texture generated: {mesh.vertices.shape[0]} verts, {len(coords)} voxels")
    return result

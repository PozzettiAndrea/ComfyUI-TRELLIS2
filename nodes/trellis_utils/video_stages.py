"""
TRELLIS2 Video-to-Animation pipeline stages.

Each stage handles temporal conditioning and animated output generation.
These run inside the isolated subprocess.
"""

import sys
import gc
from fractions import Fraction
from typing import Dict, Any, Tuple, Optional, List

import torch
import numpy as np
from PIL import Image

from .lazy_manager import get_model_manager
from .helpers import smart_crop_square
from .stages import _sparse_tensor_to_dict, _deserialize_from_ipc, _serialize_for_ipc


def run_video_conditioning(
    model_config: Any,
    images: torch.Tensor,
    masks: torch.Tensor,
    include_1024: bool = True,
    background_color: str = "black",
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Run DinoV3 conditioning extraction for multiple video frames.

    Args:
        model_config: Trellis2ModelConfig
        images: ComfyUI IMAGE tensor [B, H, W, C] where B = num_frames (e.g., 81)
        masks: ComfyUI MASK tensor [B, H, W] or [H, W] (single mask for all frames)
        include_1024: Also extract 1024px features
        background_color: Background color name
        batch_size: Number of frames to process at once (for VRAM efficiency)

    Returns:
        Dict with:
            - frame_conds_512: List[Tensor] of 512px features per frame
            - frame_conds_1024: List[Tensor] of 1024px features per frame (if include_1024)
            - neg_cond: Tensor of zeros for CFG
            - num_frames: int
    """
    print(f"[TRELLIS2] Running video conditioning for {images.shape[0]} frames...", file=sys.stderr)

    # Background color mapping
    bg_colors = {
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "white": (255, 255, 255),
    }
    bg_color = bg_colors.get(background_color, (128, 128, 128))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model manager
    manager = get_model_manager(
        model_config["model_name"],
        model_config["resolution"],
        model_config["attn_backend"],
        model_config["vram_mode"],
    )

    num_frames = images.shape[0]

    # Handle mask - either single mask or per-frame masks
    if masks.dim() == 2:
        # Single mask [H, W] - replicate for all frames
        masks = masks.unsqueeze(0).expand(num_frames, -1, -1)
    elif masks.dim() == 3:
        if masks.shape[0] == 1:
            # Single mask with batch dim [1, H, W] - expand to all frames
            masks = masks.expand(num_frames, -1, -1)
        elif masks.shape[0] != num_frames:
            raise ValueError(f"Mask batch size ({masks.shape[0]}) must match image batch size ({num_frames}) or be 1")
        # else: masks.shape[0] == num_frames, already per-frame masks [B, H, W]
    else:
        raise ValueError(f"Unexpected mask dimensions: {masks.dim()}, expected 2 or 3")

    # Preprocess all frames to PIL images
    pil_images = []
    for i in range(num_frames):
        img_np = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        mask_np = masks[i].cpu().numpy()

        # Resize mask if needed
        if mask_np.shape[:2] != (pil_image.height, pil_image.width):
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((pil_image.width, pil_image.height), Image.LANCZOS)
            mask_np = np.array(mask_pil) / 255.0

        # Apply mask as alpha
        pil_image = pil_image.convert('RGB')
        alpha_np = (mask_np * 255).astype(np.uint8)
        rgba = np.dstack([np.array(pil_image), alpha_np])
        pil_image = Image.fromarray(rgba, 'RGBA')

        # Smart crop
        pil_image = smart_crop_square(pil_image, alpha_np, margin_ratio=0.1, background_color=bg_color)
        pil_images.append(pil_image)

    # Load DinoV3
    model = manager.get_dinov3(device)

    # Extract features in batches
    frame_conds_512 = []
    frame_conds_1024 = [] if include_1024 else None

    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        batch_images = pil_images[batch_start:batch_end]

        print(f"[TRELLIS2] Processing frames {batch_start}-{batch_end-1}...", file=sys.stderr, flush=True)

        # 512px features
        model.image_size = 512
        cond_512_batch = model(batch_images)
        for i in range(cond_512_batch.shape[0]):
            frame_conds_512.append(cond_512_batch[i:i+1].cpu())

        # 1024px features
        if include_1024:
            model.image_size = 1024
            cond_1024_batch = model(batch_images)
            for i in range(cond_1024_batch.shape[0]):
                frame_conds_1024.append(cond_1024_batch[i:i+1].cpu())

        # Clear GPU memory between batches
        del cond_512_batch
        if include_1024:
            del cond_1024_batch
        torch.cuda.empty_cache()

    # Unload DinoV3
    manager.unload_dinov3()

    # Create negative conditioning (zeros like first frame)
    neg_cond = torch.zeros_like(frame_conds_512[0])

    result = {
        'frame_conds_512': frame_conds_512,
        'neg_cond': neg_cond,
        'num_frames': num_frames,
    }
    if include_1024:
        result['frame_conds_1024'] = frame_conds_1024

    print(f"[TRELLIS2] Video conditioning extracted for {num_frames} frames", file=sys.stderr)
    return result


def run_temporal_shape_generation(
    model_config: Any,
    video_conditioning: Dict[str, Any],
    seed: int = 0,
    aggregation_mode: str = "concat",
    ss_guidance_strength: float = 7.5,
    ss_sampling_steps: int = 12,
    shape_guidance_strength: float = 7.5,
    shape_sampling_steps: int = 12,
    max_num_tokens: int = 49152,
) -> Dict[str, Any]:
    """
    Run shape generation with temporal conditioning.

    Args:
        model_config: Trellis2ModelConfig
        video_conditioning: Dict from run_video_conditioning
        seed: Random seed
        aggregation_mode: How to combine frame features:
            - 'concat': Concatenate all frames (full temporal attention)
            - 'mean': Average all frame features
            - 'keyframe_middle': Use middle frame only
        ss_*: Sparse structure sampling params
        shape_*: Shape latent sampling params
        max_num_tokens: Max tokens for cascade

    Returns:
        Dict with shape_slat, subs, mesh data, resolution
    """
    import cumesh as CuMesh

    print(f"[TRELLIS2] Running temporal shape generation (mode={aggregation_mode})...", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get frame conditioning
    frame_conds_512 = video_conditioning['frame_conds_512']
    frame_conds_1024 = video_conditioning.get('frame_conds_1024', None)
    neg_cond = video_conditioning['neg_cond'].to(device)
    num_frames = video_conditioning['num_frames']

    # Aggregate conditioning based on mode
    if aggregation_mode == "concat":
        # Concatenate all frames along sequence dimension
        # Each frame: (1, N_patches, 1024) -> combined: (1, num_frames*N_patches, 1024)
        cond_512 = torch.cat([c.to(device) for c in frame_conds_512], dim=1)
        if frame_conds_1024:
            cond_1024 = torch.cat([c.to(device) for c in frame_conds_1024], dim=1)
        else:
            cond_1024 = None
        # Expand neg_cond to match
        neg_cond = neg_cond.repeat(1, num_frames, 1)

    elif aggregation_mode == "mean":
        # Average all frame features
        cond_512 = torch.stack([c.to(device) for c in frame_conds_512], dim=0).mean(dim=0)
        if frame_conds_1024:
            cond_1024 = torch.stack([c.to(device) for c in frame_conds_1024], dim=0).mean(dim=0)
        else:
            cond_1024 = None

    elif aggregation_mode == "keyframe_middle":
        # Use middle frame only
        mid_idx = num_frames // 2
        cond_512 = frame_conds_512[mid_idx].to(device)
        if frame_conds_1024:
            cond_1024 = frame_conds_1024[mid_idx].to(device)
        else:
            cond_1024 = None
    else:
        raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")

    # Build conditioning dict
    cond_on_device = {
        'cond_512': cond_512,
        'neg_cond': neg_cond,
    }
    if cond_1024 is not None:
        cond_on_device['cond_1024'] = cond_1024

    # Get pipeline
    manager = get_model_manager(
        model_config["model_name"],
        model_config["resolution"],
        model_config["attn_backend"],
        model_config["vram_mode"],
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
        pipeline_type=model_config["resolution"],
        max_num_tokens=max_num_tokens,
        **sampler_params
    )
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[TRELLIS2] Temporal shape generation peak VRAM: {peak_mem:.0f} MB", file=sys.stderr)

    mesh = meshes[0]
    mesh.fill_holes()

    # Save raw mesh data
    raw_mesh_vertices = mesh.vertices.cpu()
    raw_mesh_faces = mesh.faces.cpu()

    # Process mesh for output
    cumesh = CuMesh.CuMesh()
    cumesh.init(mesh.vertices, mesh.faces.int())
    cumesh.unify_face_orientations()
    unified_verts, unified_faces = cumesh.read()

    vertices = unified_verts.cpu().numpy().astype(np.float32)
    faces = unified_faces.cpu().numpy()
    del cumesh, unified_verts, unified_faces

    # Coordinate conversion
    vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()

    result = {
        'shape_slat': _serialize_for_ipc(shape_slat),
        'subs': _serialize_for_ipc(subs),
        'mesh_vertices': vertices,
        'mesh_faces': faces,
        'resolution': res,
        'pipeline_type': model_config["resolution"],
        'raw_mesh_vertices': raw_mesh_vertices.cpu(),
        'raw_mesh_faces': raw_mesh_faces.cpu(),
    }

    manager.unload_shape_pipeline()

    print(f"[TRELLIS2] Temporal shape generated: {len(vertices)} verts", file=sys.stderr)
    return result


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.

    Args:
        v0: Start tensor
        v1: End tensor
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated tensor
    """
    # Flatten for computation
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()

    # Normalize
    v0_norm = v0_flat / (v0_flat.norm() + 1e-8)
    v1_norm = v1_flat / (v1_flat.norm() + 1e-8)

    # Compute angle
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta = torch.acos(dot)

    # Handle near-parallel vectors
    if theta.abs() < 1e-6:
        return v0 * (1 - t) + v1 * t

    sin_theta = torch.sin(theta)
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta

    result_flat = s0 * v0_flat + s1 * v1_flat
    return result_flat.reshape(v0.shape)


def run_animated_texture_generation(
    model_config: Any,
    video_conditioning: Dict[str, Any],
    shape_result: Dict[str, Any],
    seed: int = 0,
    tex_guidance_strength: float = 7.5,
    tex_sampling_steps: int = 12,
    keyframe_interval: int = 10,
    interpolation_mode: str = "slerp",
) -> Dict[str, Any]:
    """
    Run animated texture generation with keyframe interpolation.

    Args:
        model_config: Trellis2ModelConfig
        video_conditioning: Dict from run_video_conditioning
        shape_result: Dict from run_temporal_shape_generation
        seed: Random seed
        tex_*: Texture sampling params
        keyframe_interval: Generate texture every N frames
        interpolation_mode: 'slerp' or 'linear'

    Returns:
        Dict with animation data:
            - coords: Shared voxel coordinates
            - attrs_sequence: List of per-frame attributes
            - voxel_size: float
            - layout: PBR channel mapping
            - num_frames: int
    """
    import cumesh as CuMesh
    from ..trellis2.representations.mesh import Mesh

    num_frames = video_conditioning['num_frames']
    print(f"[TRELLIS2] Running animated texture generation ({num_frames} frames, keyframe_interval={keyframe_interval})...", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get frame conditioning
    frame_conds_512 = video_conditioning['frame_conds_512']
    frame_conds_1024 = video_conditioning.get('frame_conds_1024', None)
    neg_cond = video_conditioning['neg_cond'].to(device)

    # Deserialize shape data
    shape_slat = _deserialize_from_ipc(shape_result['shape_slat'], device)
    subs = _deserialize_from_ipc(shape_result['subs'], device)
    resolution = shape_result['resolution']
    pipeline_type = shape_result['pipeline_type']

    # Reconstruct mesh
    raw_vertices = shape_result['raw_mesh_vertices'].to(device)
    raw_faces = shape_result['raw_mesh_faces'].to(device)
    mesh = Mesh(vertices=raw_vertices, faces=raw_faces)
    mesh.fill_holes()
    meshes = [mesh]

    # Get pipeline
    manager = get_model_manager(
        model_config["model_name"],
        model_config["resolution"],
        model_config["attn_backend"],
        model_config["vram_mode"],
    )
    pipeline = manager.get_texture_pipeline(device)

    # Determine keyframes
    keyframe_indices = list(range(0, num_frames, keyframe_interval))
    if keyframe_indices[-1] != num_frames - 1:
        keyframe_indices.append(num_frames - 1)

    print(f"[TRELLIS2] Generating {len(keyframe_indices)} keyframes: {keyframe_indices}", file=sys.stderr)

    # Generate texture for keyframes
    keyframe_tex_slats = {}
    keyframe_attrs = {}
    coords = None
    voxel_size = None
    pbr_layout = None

    sampler_params = {
        "tex_slat_sampler_params": {
            "steps": tex_sampling_steps,
            "guidance_strength": tex_guidance_strength,
        },
    }

    # Determine texture model based on pipeline resolution (not conditioning availability)
    # This ensures we use a model that's actually loaded
    if model_config["resolution"] in ('1024_cascade', '1536_cascade') and frame_conds_1024:
        tex_model_key = 'tex_slat_flow_model_1024'
        use_1024_cond = True
    else:
        tex_model_key = 'tex_slat_flow_model_512'
        use_1024_cond = False
    print(f"[TRELLIS2] Using texture model: {tex_model_key} (resolution={model_config['resolution']})", file=sys.stderr)

    # ========== Sample all tex_slats and save to disk ==========
    print(f"[TRELLIS2] Sampling texture latents...", file=sys.stderr)
    import os
    import tempfile

    # Create output folder for slats in temp directory
    base_temp = tempfile.gettempdir()
    slat_folder = os.path.join(base_temp, f"trellis2_tex_slats_{seed}")
    os.makedirs(slat_folder, exist_ok=True)
    print(f"[TRELLIS2] Saving texture latents to {slat_folder}", file=sys.stderr)

    for kf_idx in keyframe_indices:
        print(f"[TRELLIS2] Sampling keyframe {kf_idx}...", file=sys.stderr, flush=True)

        # Build conditioning for this frame based on model resolution
        cond_512 = frame_conds_512[kf_idx].to(device)
        if use_1024_cond and frame_conds_1024:
            cond_1024 = frame_conds_1024[kf_idx].to(device)
            tex_cond = {'cond': cond_1024, 'neg_cond': neg_cond}
        else:
            tex_cond = {'cond': cond_512, 'neg_cond': neg_cond}

        torch.manual_seed(seed + kf_idx)
        tex_slat = pipeline.sample_tex_slat(
            tex_cond, tex_model_key,
            shape_slat, sampler_params["tex_slat_sampler_params"]
        )

        # Save to disk immediately
        slat_path = os.path.join(slat_folder, f"slat_{kf_idx}.pt")
        torch.save({
            'feats': tex_slat.feats.cpu(),
            'coords': tex_slat.coords.cpu(),
        }, slat_path)

        # Also store feats for interpolation
        keyframe_tex_slats[kf_idx] = tex_slat.feats.cpu()

        del tex_slat
        torch.cuda.empty_cache()

    # Save metadata for decoding later
    metadata = {
        'keyframe_indices': keyframe_indices,
        'num_frames': num_frames,
        'resolution': resolution,
        'interpolation_mode': interpolation_mode,
        'model_name': model_config["model_name"],
        'mesh_vertices': shape_result['mesh_vertices'],
        'mesh_faces': shape_result['mesh_faces'],
        # Save subs for decoding
        'subs': [{
            'feats': s.feats.cpu(),
            'coords': s.coords.cpu(),
        } for s in subs],
        # Save keyframe feats for interpolation
        'keyframe_tex_slats': keyframe_tex_slats,
    }
    torch.save(metadata, os.path.join(slat_folder, 'metadata.pt'))

    # Cleanup
    manager.unload_texture_pipeline()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[TRELLIS2] Saved {len(keyframe_indices)} texture latents to {slat_folder}", file=sys.stderr)
    return {'slat_folder': slat_folder, 'num_keyframes': len(keyframe_indices), 'num_frames': num_frames}


def run_decode_and_export(
    model_config: Any,
    slat_folder: str,
    output_folder: str,
    decimation_target: int = 100000,
    texture_size: int = 2048,
) -> Dict[str, Any]:
    """
    Decode texture latents and export to GLB files.

    This runs in a separate node to avoid OOM - completely separate VRAM allocation.
    """
    import os
    print(f"[TRELLIS2] Loading metadata from {slat_folder}...", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata (weights_only=False needed for numpy arrays in metadata)
    metadata = torch.load(os.path.join(slat_folder, 'metadata.pt'), weights_only=False)
    keyframe_indices = metadata['keyframe_indices']
    num_frames = metadata['num_frames']
    resolution = metadata['resolution']
    interpolation_mode = metadata['interpolation_mode']
    model_name = metadata['model_name']
    mesh_vertices = metadata['mesh_vertices']
    mesh_faces = metadata['mesh_faces']
    subs_cpu = metadata['subs']
    keyframe_tex_slats = metadata['keyframe_tex_slats']

    os.makedirs(output_folder, exist_ok=True)

    # Load decoder only
    from ..trellis2.modules.sparse import SparseTensor
    from ..trellis2.pipelines import Trellis2ImageTo3DPipeline

    print(f"[TRELLIS2] Loading texture decoder...", file=sys.stderr)
    decoder_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        model_name,
        models_to_load=['tex_slat_decoder'],
    )
    decoder_pipeline._device = device

    # Restore subs to GPU
    subs = [SparseTensor(
        feats=s['feats'].to(device),
        coords=s['coords'].to(device),
    ) for s in subs_cpu]
    del subs_cpu

    # Decode keyframes and export
    keyframe_attrs = {}
    coords = None
    voxel_size = None
    pbr_layout = None

    print(f"[TRELLIS2] Decoding {len(keyframe_indices)} keyframes...", file=sys.stderr)

    for kf_idx in keyframe_indices:
        print(f"[TRELLIS2] Decoding keyframe {kf_idx}...", file=sys.stderr, flush=True)

        # Load slat from disk
        slat_path = os.path.join(slat_folder, f"slat_{kf_idx}.pt")
        slat_data = torch.load(slat_path, weights_only=False)
        tex_slat = SparseTensor(
            feats=slat_data['feats'].to(device),
            coords=slat_data['coords'].to(device),
        )
        del slat_data

        # Decode
        tex_voxels = decoder_pipeline.decode_tex_slat(tex_slat, subs)

        if coords is None:
            coords = tex_voxels[0].coords[:, 1:].cpu().numpy().astype(np.float32)
            voxel_size = 1.0 / resolution
            pbr_layout = decoder_pipeline.pbr_attr_layout

        keyframe_attrs[kf_idx] = tex_voxels[0].feats.cpu()

        del tex_slat, tex_voxels
        gc.collect()
        torch.cuda.empty_cache()

    # Unload decoder before export
    del decoder_pipeline, subs
    gc.collect()
    torch.cuda.empty_cache()

    # Interpolate and export
    print(f"[TRELLIS2] Interpolating and exporting {num_frames} frames...", file=sys.stderr)

    try:
        import o_voxel
        has_ovoxel = True
    except ImportError:
        has_ovoxel = False
        print(f"[TRELLIS2] o_voxel not available, skipping GLB export", file=sys.stderr)

    for frame_idx in range(num_frames):
        if frame_idx in keyframe_attrs:
            attrs = keyframe_attrs[frame_idx].numpy()
        else:
            # Interpolate
            prev_kf = max([k for k in keyframe_indices if k < frame_idx])
            next_kf = min([k for k in keyframe_indices if k > frame_idx])
            t = (frame_idx - prev_kf) / (next_kf - prev_kf)

            if interpolation_mode == "slerp":
                interp_slat = slerp(keyframe_tex_slats[prev_kf], keyframe_tex_slats[next_kf], t)
            else:
                interp_slat = (1 - t) * keyframe_tex_slats[prev_kf] + t * keyframe_tex_slats[next_kf]

            # Linear interpolate decoded attrs
            prev_attrs = keyframe_attrs[prev_kf]
            next_attrs = keyframe_attrs[next_kf]
            attrs = ((1 - t) * prev_attrs + t * next_attrs).numpy()

        # Export GLB
        if has_ovoxel:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh_vertices,
                faces=mesh_faces,
                attr_volume=attrs,
                coords=coords,
                attr_layout=pbr_layout,
                voxel_size=voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False
            )
            glb_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.glb")
            glb.export(glb_path, extension_webp=True)

        if frame_idx % 10 == 0:
            print(f"[TRELLIS2] Exported frame {frame_idx}", file=sys.stderr)

    print(f"[TRELLIS2] Exported {num_frames} frames to {output_folder}", file=sys.stderr)
    return {'output_folder': output_folder, 'num_frames': num_frames}



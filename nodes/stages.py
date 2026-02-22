"""
TRELLIS2 pipeline stages.

Each stage loads models on-demand, runs inference, and lets ComfyUI manage GPU offloading.
Models are cached at module level — first run loads from disk, subsequent runs reuse from RAM.
"""

import gc
import json
import logging
import os
from fractions import Fraction
from typing import Dict, Any, Tuple, Optional, List

import torch
import numpy as np
import comfy.model_management
from PIL import Image

log = logging.getLogger("trellis2")

from .helpers import smart_crop_square


# Noise/conditioning stay float32 for sampling loop stability (error accumulation over 12 steps).
# Model weights stay in safetensors dtype; manual_cast handles per-layer casting.
_DEFAULT_DTYPE = torch.float32

# PBR attribute layout for texture voxels
_PBR_ATTR_LAYOUT = {
    'base_color': slice(0, 3),
    'metallic': slice(3, 4),
    'roughness': slice(4, 5),
    'alpha': slice(5, 6),
}

# Texture resolution mapping (texture maxes at 1024)
TEXTURE_RESOLUTION_MAP = {
    '512': '512',
    '1024_cascade': '1024_cascade',
    '1536_cascade': '1024_cascade',
}


# ============================================================
# Module-level model management (persists across subprocess calls)
# ============================================================

_pipeline_config = None      # Parsed pipeline.json['args']
_model_paths = {}            # {model_key: local_safetensors_path}
_model_patchers = {}         # {model_key: ModelPatcher}
_dinov3_model = None         # DinoV3FeatureExtractor wrapper (cached across calls)


def _get_trellis2_models_dir():
    """Get the ComfyUI/models/trellis2 directory."""
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    except ImportError:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "trellis2")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _init_config():
    """Parse pipeline.json once and resolve all model paths to local files."""
    global _pipeline_config, _model_paths

    if _pipeline_config is not None:
        return

    models_dir = _get_trellis2_models_dir()
    config_file = os.path.join(models_dir, "pipeline.json")

    if not os.path.exists(config_file):
        from huggingface_hub import hf_hub_download
        log.info("Downloading pipeline config from HuggingFace...")
        hf_hub_download("microsoft/TRELLIS.2-4B", "pipeline.json", local_dir=models_dir)

    with open(config_file, 'r') as f:
        _pipeline_config = json.load(f)['args']

    # Resolve all model paths to local safetensors files
    for key, model_path in _pipeline_config['models'].items():
        path_parts = model_path.split('/')

        if len(path_parts) >= 3 and not model_path.startswith('ckpts/'):
            # Full HuggingFace path (e.g., "microsoft/TRELLIS-image-large/ckpts/...")
            repo_id = f'{path_parts[0]}/{path_parts[1]}'
            model_name = '/'.join(path_parts[2:])
        else:
            # Relative path, prepend base repo
            repo_id = "microsoft/TRELLIS.2-4B"
            model_name = model_path

        local_config = os.path.join(models_dir, f"{model_name}.json")
        local_weights = os.path.join(models_dir, f"{model_name}.safetensors")

        if os.path.exists(local_config) and os.path.exists(local_weights):
            _model_paths[key] = local_weights
        else:
            # Download if not cached
            from huggingface_hub import hf_hub_download
            os.makedirs(os.path.dirname(local_config), exist_ok=True)
            log.info(f"Downloading {model_name}...")
            hf_hub_download(repo_id, f"{model_name}.json", local_dir=models_dir)
            hf_hub_download(repo_id, f"{model_name}.safetensors", local_dir=models_dir)
            _model_paths[key] = local_weights

    log.info(f"Config loaded: {len(_model_paths)} models registered")


def _load_model(model_key, device=None):
    """
    Load a model using ComfyUI-native pattern.

    First call: build model on CPU, load weights, wrap in ModelPatcher.
    Subsequent calls: just call load_models_gpu (model already in RAM).
    """
    import time
    import comfy.model_patcher
    import comfy.utils

    if device is None:
        device = comfy.model_management.get_torch_device()

    offload_device = comfy.model_management.unet_offload_device()

    if model_key not in _model_patchers:
        t0 = time.perf_counter()
        pbar = comfy.utils.ProgressBar(3)

        safetensors_path = _model_paths[model_key]
        config_path = safetensors_path.replace('.safetensors', '.json')

        with open(config_path, 'r') as f:
            config = json.load(f)

        from .trellis2 import _get_model_class
        model_class = _get_model_class(config['name'])

        # Load state dict to CPU (ComfyUI-native: no device arg = CPU)
        sd = comfy.utils.load_torch_file(safetensors_path)
        pbar.update(1)

        # Determine target dtype: bf16 if GPU supports it, else keep disk dtype
        compute_dtype = comfy.model_management.vae_dtype(device, allowed_dtypes=[torch.bfloat16])
        weight_dtype = compute_dtype if compute_dtype == torch.bfloat16 else next(iter(sd.values())).dtype

        # Pre-cast state dict to target dtype (single pass, before entering model)
        if weight_dtype != next(iter(sd.values())).dtype:
            for k in sd:
                if sd[k].is_floating_point():
                    sd[k] = sd[k].to(weight_dtype)

        # Build model, assign=True: state dict tensors become parameters directly (no copy)
        model = model_class(**config['args'])
        model.load_state_dict(sd, strict=False, assign=True)
        del sd

        model.eval()
        pbar.update(1)

        # Wrap in ModelPatcher — model starts on CPU (offload_device)
        patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=device, offload_device=offload_device,
        )
        _model_patchers[model_key] = patcher

        # First load: move to GPU, then compute derived tensors (e.g. RoPE phases)
        comfy.model_management.load_models_gpu([patcher])
        if hasattr(model, '_post_load'):
            model._post_load(torch.device(device))
        pbar.update(1)

        elapsed = time.perf_counter() - t0
        import sys
        print(f"[TRELLIS2] Loaded {config['name']}: dtype={weight_dtype}, device={device}, {elapsed:.1f}s", file=sys.stderr)
    else:
        t0 = time.perf_counter()
        comfy.model_management.load_models_gpu([_model_patchers[model_key]])
        elapsed = time.perf_counter() - t0
        import sys
        print(f"[TRELLIS2] Reloaded {model_key} to GPU: {elapsed:.1f}s", file=sys.stderr)

    return _model_patchers[model_key].model


def _unload_model(model_key):
    """
    Move model weights back to CPU (offload_device) so subsequent load_models_gpu
    calls can properly manage VRAM — load_models_gpu fast-exits when a model is
    already in current_loaded_models, so without an explicit offload all models
    accumulate on GPU.
    """
    if model_key in _model_patchers:
        patcher = _model_patchers[model_key]
        patcher.unpatch_model(device_to=patcher.offload_device)
    comfy.model_management.soft_empty_cache()


def _load_dinov3(device=None):
    """
    Load (or reuse cached) DinoV3 feature extractor via ComfyUI ModelPatcher.

    First call: builds model, wraps in ModelPatcher, loads to GPU.
    Subsequent calls: ModelPatcher already in _model_patchers dict; load_models_gpu
    fast-exits (returns 0) if weights are already on GPU.
    """
    global _dinov3_model
    import comfy.model_patcher
    from .trellis2 import dinov3 as _dv3_mod

    if device is None:
        device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    if 'dinov3' not in _model_patchers:
        _dinov3_model = _dv3_mod.DinoV3FeatureExtractor(
            model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"
        )
        patcher = comfy.model_patcher.ModelPatcher(
            _dinov3_model.model,
            load_device=device,
            offload_device=offload_device,
        )
        _model_patchers['dinov3'] = patcher

    comfy.model_management.load_models_gpu([_model_patchers['dinov3']])
    return _dinov3_model


def _has_cascade_model():
    """True if pipeline has a separate HR SLat flow model (cascade mode)."""
    _init_config()
    return 'shape_slat_flow_model_1024' in _model_paths


# ============================================================
# IPC serialization helpers
# ============================================================

def _sparse_tensor_to_dict(st) -> Dict[str, Any]:
    """Convert a SparseTensor to a serializable dict for IPC."""
    return {
        '_type': 'SparseTensor',
        'feats': st.feats.cpu(),
        'coords': st.coords.cpu(),
        'shape': tuple(st.shape) if st.shape else None,
        'scale': tuple((s.numerator, s.denominator) for s in st._scale),
    }


def _dict_to_sparse_tensor(d: Dict[str, Any], device: torch.device):
    """Reconstruct a SparseTensor from a serialized dict."""
    from .trellis2.sparse import SparseTensor

    feats = d['feats'].to(device)
    coords = d['coords'].to(device)
    shape = torch.Size(d['shape']) if d['shape'] else None
    scale = tuple(Fraction(n, den) for n, den in d['scale'])

    return SparseTensor(feats=feats, coords=coords, shape=shape, scale=scale)


def _serialize_for_ipc(obj: Any) -> Any:
    """Recursively convert SparseTensor objects to serializable dicts."""
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
    """Recursively reconstruct SparseTensor objects from serialized dicts."""
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


# ============================================================
# Sampling functions (inlined from trellis2_image_to_3d.py)
# ============================================================

def _sample_sparse_structure(cond, ss_res, sampler_params, device, dtype):
    """
    Sample sparse structure: SS flow → SS decoder → voxel coords.

    Args:
        cond: dict with 'cond' and 'neg_cond' tensors
        ss_res: sparse structure resolution (32 or 64)
        sampler_params: dict with 'steps', 'guidance_strength', etc.
        device: torch device
        dtype: compute dtype for noise
    """
    from .trellis2.samplers import FlowEulerGuidanceIntervalSampler

    # Sample sparse structure latent
    flow_model = _load_model('sparse_structure_flow_model')
    reso = flow_model.resolution
    in_channels = flow_model.in_channels
    noise = torch.randn(1, in_channels, reso, reso, reso, device=device, dtype=dtype)

    default_params = _pipeline_config['sparse_structure_sampler']['params']
    params = {**default_params, **sampler_params}
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    z_s = sampler.sample(
        flow_model, noise, **cond, **params,
        verbose=True, tqdm_desc="Sampling sparse structure",
    ).samples

    del noise
    _unload_model('sparse_structure_flow_model')

    # Decode sparse structure latent
    decoder = _load_model('sparse_structure_decoder')
    model_dtype = next(decoder.parameters()).dtype
    z_s = z_s.to(dtype=model_dtype)
    decoded = decoder(z_s) > 0

    del z_s
    _unload_model('sparse_structure_decoder')

    if ss_res != decoded.shape[2]:
        ratio = decoded.shape[2] // ss_res
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    log.info(f"Active voxels: {coords.shape[0]}")

    del decoded
    gc.collect()
    comfy.model_management.soft_empty_cache()

    return coords


def _sample_shape_slat(cond, model_key, coords, sampler_params, device, dtype):
    """
    Sample structured latent (single resolution, no cascade).

    Args:
        cond: dict with 'cond' and 'neg_cond'
        model_key: e.g., 'shape_slat_flow_model_512'
        coords: voxel coordinates from sparse structure
        sampler_params: dict with 'steps', 'guidance_strength', etc.
        device: torch device
        dtype: compute dtype for noise
    """
    from .trellis2.sparse import SparseTensor
    from .trellis2.samplers import FlowEulerGuidanceIntervalSampler

    flow_model = _load_model(model_key)
    noise = SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model.in_channels, device=device, dtype=dtype),
        coords=coords,
    )

    default_params = _pipeline_config['shape_slat_sampler']['params']
    params = {**default_params, **sampler_params}
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    slat = sampler.sample(
        flow_model, noise, **cond, **params,
        verbose=True, tqdm_desc="Sampling shape SLat",
    ).samples

    del noise
    _unload_model(model_key)

    # Apply normalization
    std = torch.tensor(_pipeline_config['shape_slat_normalization']['std'])[None].to(device=slat.device, dtype=dtype)
    mean = torch.tensor(_pipeline_config['shape_slat_normalization']['mean'])[None].to(device=slat.device, dtype=dtype)
    slat = slat * std + mean

    return slat


def _sample_shape_slat_cascade(
    lr_cond, cond, lr_key, hr_key,
    lr_resolution, hr_resolution_target,
    coords, sampler_params, max_num_tokens,
    device, dtype,
):
    """
    Sample structured latent using cascade (LR flow → decoder upsample → HR flow).

    Returns:
        (slat, actual_hr_resolution)
    """
    from .trellis2.sparse import SparseTensor
    from .trellis2.samplers import FlowEulerGuidanceIntervalSampler

    default_params = _pipeline_config['shape_slat_sampler']['params']
    params = {**default_params, **sampler_params}
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)

    # ---- LR pass ----
    flow_model_lr = _load_model(lr_key)
    noise = SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model_lr.in_channels, device=device, dtype=dtype),
        coords=coords,
    )
    slat = sampler.sample(
        flow_model_lr, noise, **lr_cond, **params,
        verbose=True, tqdm_desc="Sampling shape SLat (LR)",
    ).samples

    del noise
    _unload_model(lr_key)

    # Free LR conditioning
    for k, v in lr_cond.items():
        if torch.is_tensor(v):
            lr_cond[k] = None
    del lr_cond, coords
    gc.collect()
    comfy.model_management.soft_empty_cache()

    # Apply normalization
    std = torch.tensor(_pipeline_config['shape_slat_normalization']['std'])[None].to(device=slat.device, dtype=dtype)
    mean = torch.tensor(_pipeline_config['shape_slat_normalization']['mean'])[None].to(device=slat.device, dtype=dtype)
    slat = slat * std + mean

    # ---- Upsample via decoder ----
    decoder = _load_model('shape_slat_decoder')
    model_dtype = next(decoder.parameters()).dtype
    slat = slat.replace(feats=slat.feats.to(dtype=model_dtype))
    hr_coords = decoder.upsample(slat, upsample_times=4)

    del slat, std, mean
    _unload_model('shape_slat_decoder')

    # Determine actual HR resolution (may reduce if too many tokens)
    hr_resolution = hr_resolution_target
    while True:
        quant_coords = torch.cat([
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
        ], dim=1)
        final_coords = quant_coords.unique(dim=0)
        num_tokens = final_coords.shape[0]
        if num_tokens < max_num_tokens or hr_resolution == 1024:
            if hr_resolution != hr_resolution_target:
                log.info(f"Resolution reduced to {hr_resolution} due to token limit")
            break
        hr_resolution -= 128

    # Move data to CPU to free GPU memory before loading HR model
    mem_before = torch.cuda.memory_allocated() / 1024**2
    cond_on_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in cond.items()}
    hr_coords_cpu = hr_coords.cpu()
    coords_cpu = final_coords.cpu()
    del hr_coords, final_coords, quant_coords, cond
    gc.collect()
    comfy.model_management.soft_empty_cache()
    mem_after = torch.cuda.memory_allocated() / 1024**2
    log.info(f"HR cleanup: {mem_before:.0f} → {mem_after:.0f} MB")

    # ---- HR pass ----
    flow_model = _load_model(hr_key)

    # Move conditioning and coords back to GPU
    hr_cond = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond_on_cpu.items()}
    del cond_on_cpu
    hr_final_coords = coords_cpu.to(device)
    del coords_cpu, hr_coords_cpu

    noise = SparseTensor(
        feats=torch.randn(hr_final_coords.shape[0], flow_model.in_channels, device=device, dtype=dtype),
        coords=hr_final_coords,
    )
    slat = sampler.sample(
        flow_model, noise, **hr_cond, **params,
        verbose=True, tqdm_desc="Sampling shape SLat (HR)",
    ).samples

    del noise, hr_final_coords, hr_cond
    _unload_model(hr_key)

    # Apply normalization
    std = torch.tensor(_pipeline_config['shape_slat_normalization']['std'])[None].to(device=slat.device, dtype=dtype)
    mean = torch.tensor(_pipeline_config['shape_slat_normalization']['mean'])[None].to(device=slat.device, dtype=dtype)
    slat = slat * std + mean

    return slat, hr_resolution


def _decode_shape_slat(slat, resolution, dtype):
    """Decode structured latent → meshes + subs."""
    import time as _time, sys as _sys
    decoder = _load_model('shape_slat_decoder')
    decoder.set_resolution(resolution)
    model_dtype = next(decoder.parameters()).dtype
    slat = slat.replace(feats=slat.feats.to(dtype=model_dtype))

    t0 = _time.perf_counter()
    meshes, subs = decoder(slat, return_subs=True)
    print(f"[TRELLIS2] Shape decode (FlexiDualGridVaeDecoder): {_time.perf_counter()-t0:.1f}s", file=_sys.stderr)

    _unload_model('shape_slat_decoder')
    return meshes, subs


def _sample_tex_slat(cond, model_key, shape_slat, sampler_params, device, dtype):
    """
    Sample texture structured latent.

    Args:
        cond: dict with 'cond' and 'neg_cond'
        model_key: e.g., 'tex_slat_flow_model_1024'
        shape_slat: shape structured latent (used as concat conditioning)
        sampler_params: dict with 'steps', 'guidance_strength', etc.
        device: torch device
        dtype: compute dtype for noise
    """
    from .trellis2.sparse import SparseTensor
    from .trellis2.samplers import FlowEulerGuidanceIntervalSampler
    import torch.nn as nn

    # Normalize shape_slat for conditioning
    std = torch.tensor(_pipeline_config['shape_slat_normalization']['std'])[None].to(device=shape_slat.device, dtype=dtype)
    mean = torch.tensor(_pipeline_config['shape_slat_normalization']['mean'])[None].to(device=shape_slat.device, dtype=dtype)
    shape_slat_normed = (shape_slat - mean) / std

    flow_model = _load_model(model_key)
    in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
    noise = shape_slat_normed.replace(
        feats=torch.randn(
            shape_slat_normed.coords.shape[0],
            in_channels - shape_slat_normed.feats.shape[1],
            device=device, dtype=dtype,
        )
    )

    default_params = _pipeline_config['tex_slat_sampler']['params']
    params = {**default_params, **sampler_params}
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    slat = sampler.sample(
        flow_model, noise, concat_cond=shape_slat_normed, **cond, **params,
        verbose=True, tqdm_desc="Sampling texture SLat",
    ).samples

    del noise
    _unload_model(model_key)

    # Apply texture normalization
    tex_std = torch.tensor(_pipeline_config['tex_slat_normalization']['std'])[None].to(device=slat.device, dtype=dtype)
    tex_mean = torch.tensor(_pipeline_config['tex_slat_normalization']['mean'])[None].to(device=slat.device, dtype=dtype)
    slat = slat * tex_std + tex_mean

    return slat


def _decode_tex_slat(slat, subs):
    """Decode texture structured latent → texture voxels."""
    import time as _time, sys as _sys
    decoder = _load_model('tex_slat_decoder')
    model_dtype = next(decoder.parameters()).dtype
    slat = slat.replace(feats=slat.feats.to(dtype=model_dtype))
    for i, sub in enumerate(subs):
        subs[i] = sub.replace(feats=sub.feats.to(dtype=model_dtype))

    t0 = _time.perf_counter()
    ret = decoder(slat, guide_subs=subs) * 0.5 + 0.5
    ret = ret.replace(feats=ret.feats.float())
    print(f"[TRELLIS2] Texture decode (SparseUnetVaeDecoder): {_time.perf_counter()-t0:.1f}s", file=_sys.stderr)

    _unload_model('tex_slat_decoder')
    return ret


# ============================================================
# Public stage functions
# ============================================================

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
    import comfy.utils
    log.info("Running conditioning...")
    pbar = comfy.utils.ProgressBar(3)

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

    # Load (or reuse cached) DinoV3 via ComfyUI ModelPatcher
    import time as _time
    import sys as _sys

    t0 = _time.perf_counter()
    dinov3_model = _load_dinov3(device)
    print(f"[TRELLIS2] DinoV3 load: {_time.perf_counter()-t0:.1f}s", file=_sys.stderr)

    # Get 512px conditioning
    dinov3_model.image_size = 512
    t0 = _time.perf_counter()
    cond_512 = dinov3_model([pil_image])
    print(f"[TRELLIS2] DinoV3 512px extract: {_time.perf_counter()-t0:.1f}s", file=_sys.stderr)
    pbar.update(1)

    # Get 1024px conditioning only if caller requested it AND pipeline has cascade model
    cond_1024 = None
    if include_1024 and _has_cascade_model():
        dinov3_model.image_size = 1024
        t0 = _time.perf_counter()
        cond_1024 = dinov3_model([pil_image])
        print(f"[TRELLIS2] DinoV3 1024px extract: {_time.perf_counter()-t0:.1f}s", file=_sys.stderr)
    pbar.update(1)

    t0 = _time.perf_counter()
    comfy.model_management.soft_empty_cache()
    print(f"[TRELLIS2] soft_empty_cache: {_time.perf_counter()-t0:.1f}s", file=_sys.stderr)
    pbar.update(1)

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

    log.info("Conditioning extracted")
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
    import comfy.utils
    import cumesh as CuMesh

    _init_config()

    log.info(f"Running shape generation (seed={seed})...")
    pbar = comfy.utils.ProgressBar(3)

    device = comfy.model_management.get_torch_device()
    compute_dtype = _DEFAULT_DTYPE
    resolution = model_config["resolution"]

    # Move conditioning to device — keep float32 for sampling loop stability
    cond_on_device = {
        k: v.to(device=device, dtype=compute_dtype) if isinstance(v, torch.Tensor) else v
        for k, v in conditioning.items()
    }

    # Sampler params (user overrides merged with pipeline.json defaults)
    ss_params = {
        "steps": ss_sampling_steps,
        "guidance_strength": ss_guidance_strength,
    }
    shape_params = {
        "steps": shape_sampling_steps,
        "guidance_strength": shape_guidance_strength,
    }

    torch.manual_seed(seed)

    # 1. Sample sparse structure
    ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[resolution]
    cond_512 = {'cond': cond_on_device['cond_512'], 'neg_cond': cond_on_device['neg_cond']}
    coords = _sample_sparse_structure(cond_512, ss_res, ss_params, device, compute_dtype)
    pbar.update(1)

    # 2. Sample shape structured latent
    torch.cuda.reset_peak_memory_stats()

    if resolution == '512':
        shape_slat = _sample_shape_slat(
            cond_512, 'shape_slat_flow_model_512',
            coords, shape_params, device, compute_dtype,
        )
        res = 512
    elif resolution == '1024':
        cond_1024 = {'cond': cond_on_device['cond_1024'], 'neg_cond': cond_on_device['neg_cond']}
        shape_slat = _sample_shape_slat(
            cond_1024, 'shape_slat_flow_model_1024',
            coords, shape_params, device, compute_dtype,
        )
        res = 1024
    elif resolution in ('1024_cascade', '1536_cascade'):
        cond_1024 = {'cond': cond_on_device['cond_1024'], 'neg_cond': cond_on_device['neg_cond']}
        target_res = 1024 if resolution == '1024_cascade' else 1536
        shape_slat, res = _sample_shape_slat_cascade(
            cond_512, cond_1024,
            'shape_slat_flow_model_512', 'shape_slat_flow_model_1024',
            512, target_res,
            coords, shape_params, max_num_tokens,
            device, compute_dtype,
        )
    else:
        raise ValueError(f"Invalid resolution: {resolution}")

    # Free conditioning refs
    for k in list(cond_on_device.keys()):
        cond_on_device[k] = None
    gc.collect()
    comfy.model_management.soft_empty_cache()

    # 3. Decode shape
    meshes, subs = _decode_shape_slat(shape_slat, res, compute_dtype)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    log.info(f"Shape generation peak VRAM: {peak_mem:.0f} MB")
    pbar.update(1)

    comfy.model_management.soft_empty_cache()

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
    result = {
        'shape_slat': _serialize_for_ipc(shape_slat),
        'subs': _serialize_for_ipc(subs),
        'mesh_vertices': vertices,
        'mesh_faces': faces,
        'resolution': res,
        'pipeline_type': resolution,
        'raw_mesh_vertices': raw_mesh_vertices,
        'raw_mesh_faces': raw_mesh_faces,
    }

    pbar.update(1)
    log.info(f"Shape generated: {len(vertices)} verts, {len(faces)} faces")
    return result, vertices, faces


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
    import comfy.utils
    from .trellis2.vae import Mesh, MeshWithVoxel

    _init_config()

    log.info(f"Running texture generation (seed={seed})...")
    pbar = comfy.utils.ProgressBar(3)

    device = comfy.model_management.get_torch_device()
    compute_dtype = _DEFAULT_DTYPE
    resolution = model_config["resolution"]

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

    # Determine texture model key
    texture_resolution = TEXTURE_RESOLUTION_MAP.get(resolution, '1024_cascade')
    if pipeline_type == '512':
        tex_model_key = 'tex_slat_flow_model_512'
        tex_cond = {'cond': cond_on_device['cond_512'], 'neg_cond': cond_on_device['neg_cond']}
    else:
        tex_model_key = 'tex_slat_flow_model_1024'
        tex_cond = {'cond': cond_on_device['cond_1024'], 'neg_cond': cond_on_device['neg_cond']}

    pbar.update(1)

    # Sample texture latent
    tex_params = {
        "steps": tex_sampling_steps,
        "guidance_strength": tex_guidance_strength,
    }

    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    tex_slat = _sample_tex_slat(
        tex_cond, tex_model_key, shape_slat,
        tex_params, device, compute_dtype,
    )

    # Free shape_slat and cond — no longer needed
    del shape_slat, tex_cond, cond_on_device
    gc.collect()
    comfy.model_management.soft_empty_cache()

    # Decode texture using pre-computed subs
    tex_voxels = _decode_tex_slat(tex_slat, subs)

    del tex_slat
    comfy.model_management.soft_empty_cache()

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    log.info(f"Texture generation peak VRAM: {peak_mem:.0f} MB")
    pbar.update(1)

    # Combine mesh with texture voxels (batch=0)
    m = meshes[0]
    v = tex_voxels[0]
    m.fill_holes()
    textured_mesh = MeshWithVoxel(
        m.vertices, m.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / shape_result['resolution'],
        coords=v.coords[:, 1:],
        attrs=v.feats,
        voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
        layout=_PBR_ATTR_LAYOUT,
    )
    textured_mesh.simplify(16777216)

    result = {
        'voxel_coords': textured_mesh.coords.detach().cpu().numpy().astype(np.float32),
        'voxel_attrs': textured_mesh.attrs.detach().cpu().numpy(),
        'voxel_size': textured_mesh.voxel_size,
        'pbr_layout': _PBR_ATTR_LAYOUT,
        'original_vertices': textured_mesh.vertices.detach().cpu(),
        'original_faces': textured_mesh.faces.detach().cpu(),
    }

    pbar.update(1)
    coords = result['voxel_coords']
    log.info(f"Texture generated: {textured_mesh.vertices.shape[0]} verts, {len(coords)} voxels")
    return result

import math
import logging
import os
from functools import lru_cache
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

import comfy.ops
import comfy.model_management
import comfy.utils
from .attention_sparse import scaled_dot_product_attention

ops = comfy.ops.manual_cast
log = logging.getLogger("trellis2")


# ---------------------------------------------------------------------------
# Config (hardcoded for ViT-L, matching the safetensors checkpoint)
# ---------------------------------------------------------------------------

VITL_CONFIG = dict(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    attention_dropout=0.0,
    layer_norm_eps=1e-6,
    patch_size=16,
    num_channels=3,
    query_bias=True,
    key_bias=False,
    value_bias=True,
    proj_bias=True,
    mlp_bias=True,
    layerscale_value=1e-5,
    drop_path_rate=0.4,
    num_register_tokens=4,
    rope_theta=100.0,
)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _get_patch_coords(num_h: int, num_w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Patch center coordinates in [-1, +1]."""
    ch = torch.arange(0.5, num_h, dtype=dtype, device=device) / num_h
    cw = torch.arange(0.5, num_w, dtype=dtype, device=device) / num_w
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).flatten(0, 1)
    return 2.0 * coords - 1.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    """Apply RoPE to q/k, skipping prefix tokens (CLS + register)."""
    n_prefix = q.shape[-2] - cos.shape[-2]
    q_pre, q_patch = q.split((n_prefix, cos.shape[-2]), dim=-2)
    k_pre, k_patch = k.split((n_prefix, cos.shape[-2]), dim=-2)
    q_patch = q_patch * cos + _rotate_half(q_patch) * sin
    k_patch = k_patch * cos + _rotate_half(k_patch) * sin
    return torch.cat((q_pre, q_patch), dim=-2), torch.cat((k_pre, k_patch), dim=-2)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class RoPEEmbedding(nn.Module):
    """Compute cos/sin RoPE embeddings from pixel_values shape."""

    def __init__(self, head_dim: int, patch_size: int, rope_theta: float = 100.0):
        super().__init__()
        self.patch_size = patch_size
        inv_freq = 1.0 / (rope_theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor):
        _, _, h, w = pixel_values.shape
        nh, nw = h // self.patch_size, w // self.patch_size
        device = pixel_values.device
        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=False):
            coords = _get_patch_coords(nh, nw, torch.float32, device)
            angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2).tile(2)
            cos, sin = torch.cos(angles), torch.sin(angles)
        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class Embeddings(nn.Module):
    def __init__(self, hidden_size, patch_size, num_channels, num_register_tokens, dtype=None, device=None, operations=ops):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size, dtype=dtype, device=device))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size, dtype=dtype, device=device))
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, hidden_size, dtype=dtype, device=device))
        self.patch_embeddings = operations.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, dtype=dtype, device=device)

    def forward(self, pixel_values, bool_masked_pos=None):
        B = pixel_values.shape[0]
        x = self.patch_embeddings(pixel_values)  # comfy.ops handles weight dtype casting
        x = x.flatten(2).transpose(1, 2)
        if bool_masked_pos is not None:
            x = torch.where(bool_masked_pos.unsqueeze(-1), self.mask_token.to(device=x.device, dtype=x.dtype), x)
        cls = self.cls_token.to(device=x.device, dtype=x.dtype).expand(B, -1, -1)
        reg = self.register_tokens.to(device=x.device, dtype=x.dtype).expand(B, -1, -1)
        return torch.cat([cls, reg, x], dim=1)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, query_bias, key_bias, value_bias, proj_bias, dtype=None, device=None, operations=ops):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = operations.Linear(hidden_size, hidden_size, bias=query_bias, dtype=dtype, device=device)
        self.k_proj = operations.Linear(hidden_size, hidden_size, bias=key_bias, dtype=dtype, device=device)
        self.v_proj = operations.Linear(hidden_size, hidden_size, bias=value_bias, dtype=dtype, device=device)
        self.o_proj = operations.Linear(hidden_size, hidden_size, bias=proj_bias, dtype=dtype, device=device)

    def forward(self, x, position_embeddings=None):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rope(q, k, cos, sin)
        # Use ComfyUI-native attention dispatch
        # scaled_dot_product_attention expects (B, L, H, D) format
        q = q.transpose(1, 2)  # (B, H, N, D) -> (B, N, H, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.contiguous().reshape(B, N, -1))


class LayerScale(nn.Module):
    def __init__(self, hidden_size, init_value, dtype=None, device=None):
        super().__init__()
        self.lambda1 = nn.Parameter(init_value * torch.ones(hidden_size, dtype=dtype, device=device))

    def forward(self, x):
        return x * self.lambda1.to(device=x.device, dtype=x.dtype)


def _drop_path(x, drop_prob, training):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep) * mask


class MLP(nn.Module):
    """Matches DINOv3ViTMLP key names: mlp.up_proj, mlp.down_proj."""
    def __init__(self, hidden_size, intermediate_size, bias, dtype=None, device=None, operations=ops):
        super().__init__()
        self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, layer_norm_eps,
                 layerscale_value, drop_path_rate, query_bias, key_bias, value_bias,
                 proj_bias, mlp_bias, dtype=None, device=None, operations=ops):
        super().__init__()
        self.norm1 = operations.LayerNorm(hidden_size, eps=layer_norm_eps, dtype=dtype, device=device)
        self.attention = Attention(hidden_size, num_heads, query_bias, key_bias, value_bias, proj_bias, dtype=dtype, device=device, operations=operations)
        self.layer_scale1 = LayerScale(hidden_size, layerscale_value, dtype=dtype, device=device)
        self.drop_path_rate = drop_path_rate

        self.norm2 = operations.LayerNorm(hidden_size, eps=layer_norm_eps, dtype=dtype, device=device)
        self.mlp = MLP(hidden_size, intermediate_size, mlp_bias, dtype=dtype, device=device, operations=operations)
        self.layer_scale2 = LayerScale(hidden_size, layerscale_value, dtype=dtype, device=device)

    def forward(self, x, position_embeddings=None):
        r = x
        x = self.attention(self.norm1(x), position_embeddings=position_embeddings)
        x = self.layer_scale1(x)
        x = _drop_path(x, self.drop_path_rate, self.training) + r

        r = x
        x = self.mlp(self.norm2(x))
        x = self.layer_scale2(x)
        x = _drop_path(x, self.drop_path_rate, self.training) + r
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DINOv3ViT(nn.Module):
    """
    DINOv3 ViT-L as a plain nn.Module. No HuggingFace PreTrainedModel.

    State dict keys match the transformers DINOv3ViTModel checkpoint exactly
    so existing safetensors files load with strict=True.
    """

    def __init__(self, cfg=None, dtype=None, device=None, operations=ops):
        super().__init__()
        c = {**VITL_CONFIG, **(cfg or {})}
        head_dim = c["hidden_size"] // c["num_attention_heads"]

        self.embeddings = Embeddings(c["hidden_size"], c["patch_size"], c["num_channels"], c["num_register_tokens"], dtype=dtype, device=device, operations=operations)
        self.rope_embeddings = RoPEEmbedding(head_dim, c["patch_size"], c.get("rope_theta", 100.0))
        self.layer = nn.ModuleList([
            Block(
                c["hidden_size"], c["num_attention_heads"], c["intermediate_size"],
                c["layer_norm_eps"], c["layerscale_value"], c["drop_path_rate"],
                c["query_bias"], c["key_bias"], c["value_bias"], c["proj_bias"], c["mlp_bias"],
                dtype=dtype, device=device, operations=operations,
            )
            for _ in range(c["num_hidden_layers"])
        ])
        self.norm = operations.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"], dtype=dtype, device=device)

    def forward(self, pixel_values, bool_masked_pos=None):
        x = self.embeddings(pixel_values, bool_masked_pos)
        pos = self.rope_embeddings(pixel_values)
        for block in self.layer:
            x = block(x, position_embeddings=pos)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

# Remap gated Facebook models to public reuploads
DINOV3_MODEL_REMAP = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m",
}

# Clean local safetensors filenames to check (in order of preference)
LOCAL_SAFETENSORS_NAMES = [
    "dinov3-vitl-pretrain.safetensors",
    "dinov3-vitl.safetensors",
    "model.safetensors",
]


def _find_local_safetensors(cache_dir: str) -> Optional[str]:
    """
    Check for clean local safetensors file in cache_dir.
    Returns the path if found, None otherwise.
    """
    for name in LOCAL_SAFETENSORS_NAMES:
        path = os.path.join(cache_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _load_dinov3_from_safetensors(safetensors_path: str) -> DINOv3ViT:
    """
    Load DINOv3 ViT-L model from a single safetensors file.
    Uses vendored DINOv3ViT (plain nn.Module with comfy-attn baked in).
    """
    model = DINOv3ViT()
    state_dict = comfy.utils.load_torch_file(safetensors_path)
    model.load_state_dict(state_dict, strict=True)
    return model


class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        device = comfy.model_management.get_torch_device()
        self.model.to(device)

    def cpu(self):
        self.model.cpu()

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.

        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.

        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        device = comfy.model_management.get_torch_device()
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.transform(image).to(device)
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens


class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.

    Supports loading from:
    1. Clean local safetensors file (preferred): models/dinov3/dinov3-vitl-pretrain.safetensors
    2. HuggingFace cache (fallback): downloads to models/dinov3/models--PIA-SPACE-LAB--...
    """
    def __init__(self, model_name: str, image_size=512):
        # Remap gated models to public reuploads
        actual_model_name = DINOV3_MODEL_REMAP.get(model_name, model_name)
        if actual_model_name != model_name:
            log.info(f"Remapping {model_name} -> {actual_model_name}")
        self.model_name = model_name

        # Use ComfyUI models directory for cache
        import folder_paths
        cache_dir = os.path.join(folder_paths.models_dir, "dinov3")
        os.makedirs(cache_dir, exist_ok=True)

        # Priority 1: Check for clean local safetensors file
        log.info(f"Checking for local DINOv3 safetensors in: {cache_dir}")
        local_safetensors = _find_local_safetensors(cache_dir)
        if local_safetensors:
            log.info(f"Loading DINOv3 from local safetensors: {local_safetensors}")
            self.model = _load_dinov3_from_safetensors(local_safetensors)
            log.info("DINOv3 model loaded successfully")
        else:
            # Priority 2: Download safetensors directly to models/dinov3/
            # (avoids HF cache structure that _find_local_safetensors can't find)
            from huggingface_hub import hf_hub_download
            log.info(f"Downloading DINOv3 model: {actual_model_name}...")
            hf_hub_download(actual_model_name, "model.safetensors", local_dir=cache_dir)
            local_safetensors = os.path.join(cache_dir, "model.safetensors")
            self.model = _load_dinov3_from_safetensors(local_safetensors)
            log.info("DINOv3 model loaded successfully")

        self.model.eval()

        device = comfy.model_management.get_torch_device()
        log.info(f"DINOv3 ViT-L loaded ({len(list(self.model.layer))} layers, comfy-attn baked in)")

        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        device = comfy.model_management.get_torch_device()
        self.model.to(device)

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        # Use ComfyUI-native dtype detection: bf16 if GPU supports it, else fp32.
        # fp16 is NOT allowed — DINOv3 ViT-L overflows at layer 1 in fp16.
        import sys
        from tqdm import tqdm
        device = comfy.model_management.get_torch_device()
        compute_dtype = comfy.model_management.vae_dtype(device, allowed_dtypes=[torch.bfloat16])
        print(f"[TRELLIS2] DinoV3 conditioning: dtype={compute_dtype}", file=sys.stderr)
        image = image.to(dtype=compute_dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for layer_module in tqdm(self.model.layer, desc="DinoV3 conditioning", ncols=80, leave=False, mininterval=0.5):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        out = F.layer_norm(hidden_states, hidden_states.shape[-1:])
        return out

    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.

        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.

        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        device = comfy.model_management.get_torch_device()
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.transform(image).to(device).float()
        features = self.extract_features(image)
        return features

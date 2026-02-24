"""
Standalone DINOv3 ViT-L implementation as plain nn.Module.

Vendored from HuggingFace transformers (Apache 2.0) to avoid PreTrainedModel
baggage (read-only .device property, config classes, etc.) and to bake in
comfy-attn attention dispatch directly.

Architecture: ViT-L/16 with RoPE, CLS token, register tokens, LayerScale, DropPath.
Weights are loaded from safetensors with identical parameter names.
"""

import math
import logging
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, hidden_size, patch_size, num_channels, num_register_tokens):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, hidden_size))
        self.patch_embeddings = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

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
    def __init__(self, hidden_size, num_heads, query_bias, key_bias, value_bias, proj_bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=query_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=key_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=value_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=proj_bias)

    def forward(self, x, position_embeddings=None):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rope(q, k, cos, sin)
        # comfy-attn dispatch (sage/flash/sdpa) with fallback to F.sdpa
        try:
            from comfy_attn import dispatch_attention
            out = dispatch_attention(q, k, v)
        except Exception:
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2)
        else:
            out = out.transpose(1, 2)
        return self.o_proj(out.contiguous().reshape(B, N, -1))


class LayerScale(nn.Module):
    def __init__(self, hidden_size, init_value):
        super().__init__()
        self.lambda1 = nn.Parameter(init_value * torch.ones(hidden_size))

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
    def __init__(self, hidden_size, intermediate_size, bias):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, layer_norm_eps,
                 layerscale_value, drop_path_rate, query_bias, key_bias, value_bias,
                 proj_bias, mlp_bias):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = Attention(hidden_size, num_heads, query_bias, key_bias, value_bias, proj_bias)
        self.layer_scale1 = LayerScale(hidden_size, layerscale_value)
        self.drop_path_rate = drop_path_rate

        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, mlp_bias)
        self.layer_scale2 = LayerScale(hidden_size, layerscale_value)

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

    def __init__(self, cfg=None):
        super().__init__()
        c = {**VITL_CONFIG, **(cfg or {})}
        head_dim = c["hidden_size"] // c["num_attention_heads"]

        self.embeddings = Embeddings(c["hidden_size"], c["patch_size"], c["num_channels"], c["num_register_tokens"])
        self.rope_embeddings = RoPEEmbedding(head_dim, c["patch_size"], c.get("rope_theta", 100.0))
        self.layer = nn.ModuleList([
            Block(
                c["hidden_size"], c["num_attention_heads"], c["intermediate_size"],
                c["layer_norm_eps"], c["layerscale_value"], c["drop_path_rate"],
                c["query_bias"], c["key_bias"], c["value_bias"], c["proj_bias"], c["mlp_bias"],
            )
            for _ in range(c["num_hidden_layers"])
        ])
        self.norm = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])

    def forward(self, pixel_values, bool_masked_pos=None):
        x = self.embeddings(pixel_values, bool_masked_pos)
        pos = self.rope_embeddings(pixel_values)
        for block in self.layer:
            x = block(x, position_embeddings=pos)
        return self.norm(x)

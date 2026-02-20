"""
Consolidated TRELLIS2 model file.

Combines utilities, norms, attention, transformer blocks (dense and sparse),
and model classes into a single module for simplified imports.
"""
from typing import *
import logging
import math
from abc import abstractmethod
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import comfy.ops
import comfy.patcher_extension

from .attention_sparse import scaled_dot_product_attention, sparse_scaled_dot_product_attention, dispatch_varlen_attention
from .sparse import VarLenTensor, SparseTensor, sparse_cat, get_debug
from .ops_sparse import manual_cast as sparse_ops

ops = comfy.ops.manual_cast

log = logging.getLogger("trellis2")


# =============================================================================
# 1. Utilities (from modules/utils.py and modules/spatial.py)
# =============================================================================

def str_to_dtype(dtype_str):
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    return {
        'f16': torch.float16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'f32': torch.float32,
        'fp32': torch.float32,
        'float32': torch.float32,
    }[dtype_str]


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D pixel shuffle.
    """
    B, C, H, W, D = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, C_, H*scale_factor, W*scale_factor, D*scale_factor)
    return x


# =============================================================================
# 2. Dense attention config (from modules/attention/config.py)
# =============================================================================

_DEBUG: bool = False

# Map TRELLIS2 backend names to internal names
_BACKEND_MAP = {
    'sageattn': 'sage',
    'flash_attn': 'flash_attn',
    'xformers': 'sdpa',
    'sdpa': 'sdpa',
    'naive': 'sdpa',
    'auto': 'auto',
}


def get_backend() -> str:
    """Get current backend name."""
    # Dense attention uses ComfyUI's optimized_attention_for_device,
    # which auto-selects the best backend.
    return 'auto'


def set_backend(backend: str) -> None:
    """
    Set backend explicitly.

    Args:
        backend: One of 'sageattn', 'flash_attn', 'xformers', 'sdpa', 'naive', 'auto'
    """
    comfy_name = _BACKEND_MAP.get(backend, 'auto')
    log.info(f"Dense attention backend request: {comfy_name} (ComfyUI auto-selects on device)")


def set_debug(debug: bool) -> None:
    """Enable or disable debug mode."""
    global _DEBUG
    _DEBUG = debug


# =============================================================================
# 3. Norms (from modules/norm.py)
# =============================================================================

class LayerNorm32(ops.LayerNorm):
    """
    A LayerNorm layer that computes in float32 for numerical stability.
    Inherits from comfy.ops.disable_weight_init.LayerNorm for ComfyUI
    weight management (CastWeightBiasOp, reset_parameters).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        o = F.layer_norm(
            x.float(), self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return o.to(x_dtype)


class GroupNorm32(ops.GroupNorm):
    """
    A GroupNorm layer that computes in float32 for numerical stability.
    Inherits from comfy.ops.disable_weight_init.GroupNorm for ComfyUI
    weight management (CastWeightBiasOp, reset_parameters).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        o = F.group_norm(
            x.float(), self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return o.to(x_dtype)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x


# =============================================================================
# 4. Dense RoPE (from modules/attention/rope.py)
# =============================================================================

class RotaryPositionEmbedder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0)
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** (self.freqs))

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        if self.freqs.device.type == 'meta':
            # Recompute freqs — meta tensors can't be copied
            freqs = torch.arange(self.freq_dim, dtype=torch.float32, device=indices.device) / self.freq_dim
            self.freqs = self.rope_freq[0] / (self.rope_freq[1] ** freqs)
        else:
            self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        assert indices.shape[-1] == self.dim, f"Last dim of indices must be {self.dim}"
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[-1] < self.head_dim // 2:
                padn = self.head_dim // 2 - phases.shape[-1]
                phases = torch.cat([phases, torch.polar(
                    torch.ones(*phases.shape[:-1], padn, device=phases.device),
                    torch.zeros(*phases.shape[:-1], padn, device=phases.device)
                )], dim=-1)
        return phases


# =============================================================================
# 5. Dense attention (from modules/attention/modules.py)
# =============================================================================

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int, dtype=None, device=None):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"

        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = operations.Linear(channels, channels * 3, bias=qkv_bias, dtype=dtype, device=device)
        else:
            self.to_q = operations.Linear(channels, channels, bias=qkv_bias, dtype=dtype, device=device)
            self.to_kv = operations.Linear(self.ctx_channels, channels * 2, bias=qkv_bias, dtype=dtype, device=device)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, dtype=dtype, device=device)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, dtype=dtype, device=device)

        self.to_out = operations.Linear(channels, channels, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, phases: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)

            if self.attn_mode == "full":
                if self.qk_rms_norm or self.use_rope:
                    q, k, v = qkv.unbind(dim=2)
                    if self.qk_rms_norm:
                        q = self.q_rms_norm(q)
                        k = self.k_rms_norm(k)
                    if self.use_rope:
                        assert phases is not None, "Phases must be provided for RoPE"
                        q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                        k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
                    h = scaled_dot_product_attention(q, k, v, transformer_options=transformer_options)
                else:
                    h = scaled_dot_product_attention(qkv, transformer_options=transformer_options)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v, transformer_options=transformer_options)
            else:
                h = scaled_dot_product_attention(q, kv, transformer_options=transformer_options)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h


# =============================================================================
# 6. Dense blocks (from modules/transformer/blocks.py)
# =============================================================================

class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, dtype=None, device=None, operations=ops):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(channels, int(channels * mlp_ratio), dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(int(channels * mlp_ratio), channels, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[int] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = True,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def _forward(self, x: torch.Tensor, phases: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, phases=phases, transformer_options=transformer_options)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, phases: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, phases, transformer_options, use_reentrant=False)
        else:
            return self._forward(x, phases, transformer_options=transformer_options)


class TransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def _forward(self, x: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(h, phases=phases, transformer_options=transformer_options)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context, transformer_options=transformer_options)
        x = x + h
        h = self.norm3(x)
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, phases, transformer_options, use_reentrant=False)
        else:
            return self._forward(x, context, phases, transformer_options=transformer_options)


# =============================================================================
# 7. Dense modulated blocks (from modules/transformer/modulated.py)
# =============================================================================

class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device),
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels, dtype=dtype, device=device) / channels ** 0.5)

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, phases: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        transformer_patches = transformer_options.get("patches", {})
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, phases=phases, transformer_options=transformer_options)
        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                h = p(h, transformer_options)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, phases: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, phases, transformer_options, use_reentrant=False)
        else:
            return self._forward(x, mod, phases, transformer_options=transformer_options)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device),
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels, dtype=dtype, device=device) / channels ** 0.5)

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        transformer_patches = transformer_options.get("patches", {})
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h, phases=phases, transformer_options=transformer_options)
        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                h = p(h, transformer_options)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context, transformer_options=transformer_options)
        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                h = p(h, transformer_options)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, phases, transformer_options, use_reentrant=False)
        else:
            return self._forward(x, mod, context, phases, transformer_options=transformer_options)


# =============================================================================
# 8. Sparse RoPE (from modules/sparse/attention/rope.py)
# =============================================================================

class SparseRotaryPositionEmbedder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: Tuple[float, float] = (1.0, 10000.0)
    ):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / (rope_freq[1] ** (self.freqs))

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        if self.freqs.device.type == 'meta':
            # Recompute freqs — meta tensors can't be copied
            freqs = torch.arange(self.freq_dim, dtype=torch.float32, device=indices.device) / self.freq_dim
            self.freqs = self.rope_freq[0] / (self.rope_freq[1] ** freqs)
        else:
            self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed

    def forward(self, q: SparseTensor, k: Optional[SparseTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (SparseTensor): [..., N, H, D] tensor of queries
            k (SparseTensor): [..., N, H, D] tensor of keys
        """
        assert q.coords.shape[-1] == self.dim + 1, "Last dimension of coords must be equal to dim+1"
        phases_cache_name = f'rope_phase_{self.dim}d_freq{self.rope_freq[0]}-{self.rope_freq[1]}_hd{self.head_dim}'
        phases = q.get_spatial_cache(phases_cache_name)
        if phases is None:
            coords = q.coords[..., 1:]
            phases = self._get_phases(coords.reshape(-1)).reshape(*coords.shape[:-1], -1)
            if phases.shape[-1] < self.head_dim // 2:
                padn = self.head_dim // 2 - phases.shape[-1]
                phases = torch.cat([phases, torch.polar(
                    torch.ones(*phases.shape[:-1], padn, device=phases.device),
                    torch.zeros(*phases.shape[:-1], padn, device=phases.device)
                )], dim=-1)
            q.register_spatial_cache(phases_cache_name, phases)
        q_embed = q.replace(self._rotary_embedding(q.feats, phases))
        if k is None:
            return q_embed
        k_embed = k.replace(self._rotary_embedding(k.feats, phases))
        return q_embed, k_embed


# =============================================================================
# 9. Sparse windowed attention (from modules/sparse/attention/windowed_attn.py)
# =============================================================================

def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], Dict]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (torch.Tensor): Sequence lengths.
        (dict): Attn func args (cu_seqlens, max_seqlen).
    """
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = [i + j for i, j in zip(tensor.spatial_shape, shift_window)]
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    mask = seq_lens != 0
    seq_lens = seq_lens[mask]

    # Build cumulative sequence lengths for comfy-attn varlen dispatch
    cu_seqlens = torch.cat([
        torch.tensor([0], device=tensor.device),
        torch.cumsum(seq_lens, dim=0)
    ], dim=0).int()
    max_seqlen = torch.max(seq_lens).item()

    attn_func_args = {
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max_seqlen,
    }

    return fwd_indices, bwd_indices, seq_lens, attn_func_args


def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.

    Returns:
        (SparseTensor): [N, *, H, C] sparse tensor containing the output features.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f'windowed_attention_{window_size}_{shift_window}'
    serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, attn_func_args))
    else:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = serialization_spatial_cache

    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]

    if get_debug():
        start = 0
        qkv_coords = qkv.coords[fwd_indices]
        for i in range(len(seq_lens)):
            seq_coords = qkv_coords[start:start+seq_lens[i]]
            assert (seq_coords[:, 1:].max(dim=0).values - seq_coords[:, 1:].min(dim=0).values < window_size).all(), \
                    f"SparseWindowedScaledDotProductSelfAttention: window size exceeded"
            start += seq_lens[i]

    q, k, v = qkv_feats.unbind(dim=1)  # each [M, H, C]

    cu_seqlens = attn_func_args['cu_seqlens']
    max_seqlen = attn_func_args['max_seqlen']

    out = dispatch_varlen_attention(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
    )  # [M, H, C]

    out = out[bwd_indices]      # [T, H, C]

    if get_debug():
        qkv_coords = qkv_coords[bwd_indices]
        assert torch.equal(qkv_coords, qkv.coords), "SparseWindowedScaledDotProductSelfAttention: coordinate mismatch"

    return qkv.replace(out)


def sparse_windowed_scaled_dot_product_cross_attention(
    q: SparseTensor,
    kv: SparseTensor,
    q_window_size: int,
    kv_window_size: int,
    q_shift_window: Tuple[int, int, int] = (0, 0, 0),
    kv_shift_window: Tuple[int, int, int] = (0, 0, 0),
) -> SparseTensor:
    """
    Apply windowed scaled dot product cross attention to two sparse tensors.

    Args:
        q (SparseTensor): [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor): [N, *, 2, H, C] sparse tensor containing Ks and Vs.
        q_window_size (int): The window size to use for Qs.
        kv_window_size (int): The window size to use for Ks and Vs.
        q_shift_window (Tuple[int, int, int]): The shift of serialized coordinates for Qs.
        kv_shift_window (Tuple[int, int, int]): The shift of serialized coordinates for Ks and Vs.

    Returns:
        (SparseTensor): [N, *, H, C] sparse tensor containing the output features.
    """
    assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
    assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"

    q_serialization_spatial_cache_name = f'windowed_attention_{q_window_size}_{q_shift_window}'
    q_serialization_spatial_cache = q.get_spatial_cache(q_serialization_spatial_cache_name)
    if q_serialization_spatial_cache is None:
        q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args = calc_window_partition(q, q_window_size, q_shift_window)
        q.register_spatial_cache(q_serialization_spatial_cache_name, (q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args))
    else:
        q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args = q_serialization_spatial_cache
    kv_serialization_spatial_cache_name = f'windowed_attention_{kv_window_size}_{kv_shift_window}'
    kv_serialization_spatial_cache = kv.get_spatial_cache(kv_serialization_spatial_cache_name)
    if kv_serialization_spatial_cache is None:
        kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args = calc_window_partition(kv, kv_window_size, kv_shift_window)
        kv.register_spatial_cache(kv_serialization_spatial_cache_name, (kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args))
    else:
        kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args = kv_serialization_spatial_cache

    assert len(q_seq_lens) == len(kv_seq_lens), "Number of sequences in q and kv must match"

    q_feats = q.feats[q_fwd_indices]      # [M, H, C]
    kv_feats = kv.feats[kv_fwd_indices]    # [M, 2, H, C]
    k, v = kv_feats.unbind(dim=1)          # each [M, H, C]

    out = dispatch_varlen_attention(
        q_feats, k, v,
        q_attn_func_args['cu_seqlens'], kv_attn_func_args['cu_seqlens'],
        q_attn_func_args['max_seqlen'], kv_attn_func_args['max_seqlen'],
    )  # [M, H, C]

    out = out[q_bwd_indices]      # [T, H, C]

    return q.replace(out)


# =============================================================================
# 10. Sparse attention (from modules/sparse/attention/modules.py)
# =============================================================================

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int, dtype=None, device=None):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim, dtype=dtype, device=device))

    def forward(self, x: Union[VarLenTensor, torch.Tensor]) -> Union[VarLenTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, VarLenTensor):
            x = x.replace(F.normalize(x.feats, dim=-1) * self.gamma * self.scale)
        else:
            x = F.normalize(x, dim=-1) * self.gamma * self.scale
        return x.to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed", "double_windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        dtype=None,
        device=None,
        operations=ops,
        sparse_operations=sparse_ops,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed", "double_windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        if attn_mode == 'double_windowed':
            assert window_size % 2 == 0, "Window size must be even for double windowed attention"
            assert num_heads % 2 == 0, "Number of heads must be even for double windowed attention"
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = sparse_operations.SparseLinear(channels, channels * 3, bias=qkv_bias, dtype=dtype, device=device)
        else:
            self.to_q = sparse_operations.SparseLinear(channels, channels, bias=qkv_bias, dtype=dtype, device=device)
            self.to_kv = sparse_operations.SparseLinear(self.ctx_channels, channels * 2, bias=qkv_bias, dtype=dtype, device=device)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads, dtype=dtype, device=device)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads, dtype=dtype, device=device)

        self.to_out = sparse_operations.SparseLinear(channels, channels, dtype=dtype, device=device)

        if use_rope:
            self.rope = SparseRotaryPositionEmbedder(self.head_dim, rope_freq=rope_freq)

    @staticmethod
    def _linear(module, x):
        return module(x)

    @staticmethod
    def _reshape_chs(x: Union[VarLenTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[VarLenTensor, torch.Tensor], num_fused: int) -> Union[VarLenTensor, torch.Tensor]:
        if isinstance(x, VarLenTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, VarLenTensor) else x_feats

    def forward(self, x: SparseTensor, context: Optional[Union[VarLenTensor, torch.Tensor]] = None, transformer_options={}) -> SparseTensor:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.qk_rms_norm or self.use_rope:
                q, k, v = qkv.unbind(dim=-3)
                if self.qk_rms_norm:
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                if self.use_rope:
                    q, k = self.rope(q, k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
            elif self.attn_mode == "double_windowed":
                qkv0 = qkv.replace(qkv.feats[:, :, self.num_heads//2:])
                qkv1 = qkv.replace(qkv.feats[:, :, :self.num_heads//2])
                h0 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv0, self.window_size, shift_window=(0, 0, 0)
                )
                h1 = sparse_windowed_scaled_dot_product_self_attention(
                    qkv1, self.window_size, shift_window=tuple([self.window_size//2] * 3)
                )
                h = qkv.replace(torch.cat([h0.feats, h1.feats], dim=1))
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=-3)
                k = self.k_rms_norm(k)
                h = sparse_scaled_dot_product_attention(q, k, v)
            else:
                h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h


# =============================================================================
# 11. Sparse blocks (from modules/sparse/transformer/blocks.py)
# =============================================================================

class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, dtype=None, device=None, operations=ops, sparse_operations=sparse_ops):
        super().__init__()
        self.mlp = nn.Sequential(
            sparse_operations.SparseLinear(channels, int(channels * mlp_ratio), dtype=dtype, device=device),
            sparse_operations.SparseGELU(approximate="tanh"),
            sparse_operations.SparseLinear(int(channels * mlp_ratio), channels, dtype=dtype, device=device),
        )

    def forward(self, x: VarLenTensor) -> VarLenTensor:
        return self.mlp(x)


class SparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
        dtype=None,
        device=None,
        operations=ops,
        sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )

    def _forward(self, x: SparseTensor, transformer_options={}) -> SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = self.attn(h, transformer_options=transformer_options)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor, transformer_options={}) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, transformer_options, use_reentrant=False)
        else:
            return self._forward(x, transformer_options=transformer_options)


class SparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN).
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
        dtype=None,
        device=None,
        operations=ops,
        sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.norm3 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6, dtype=dtype, device=device)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )

    def _forward(self, x: SparseTensor, context: Union[torch.Tensor, VarLenTensor], transformer_options={}) -> SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = self.self_attn(h, transformer_options=transformer_options)
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context, transformer_options=transformer_options)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = self.mlp(h)
        x = x + h
        return x

    def forward(self, x: SparseTensor, context: Union[torch.Tensor, VarLenTensor], transformer_options={}) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, context, transformer_options, use_reentrant=False)
        else:
            return self._forward(x, context, transformer_options=transformer_options)


# =============================================================================
# 12. Sparse modulated blocks (from modules/sparse/transformer/modulated.py)
# =============================================================================

class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        dtype=None,
        device=None,
        operations=ops,
        sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels, dtype=dtype, device=device) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor, transformer_options={}) -> SparseTensor:
        transformer_patches = transformer_options.get("patches", {})
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h, transformer_options=transformer_options)
        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                h = p(h, transformer_options)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, transformer_options={}) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, transformer_options, use_reentrant=False)
        else:
            return self._forward(x, mod, transformer_options=transformer_options)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        dtype=None,
        device=None,
        operations=ops,
        sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels, dtype=dtype, device=device) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor], transformer_options={}) -> SparseTensor:
        transformer_patches = transformer_options.get("patches", {})
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h, transformer_options=transformer_options)
        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                h = p(h, transformer_options)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context, transformer_options=transformer_options)
        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                h = p(h, transformer_options)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor], transformer_options={}) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, transformer_options, use_reentrant=False)
        else:
            return self._forward(x, mod, context, transformer_options=transformer_options)


# =============================================================================
# 13. TimestepEmbedder and SparseStructureFlowModel (from models/sparse_structure_flow.py)
# =============================================================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=ops):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'float32',
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        device=None,
        operations=ops,
        **kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels, dtype=self.dtype, device=device, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(model_channels, 6 * model_channels, bias=True, dtype=self.dtype, device=device)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)
        elif pe_mode == "rope":
            pos_embedder = RotaryPositionEmbedder(self.model_channels // self.num_heads, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            rope_phases = pos_embedder(coords)
            self.register_buffer("rope_phases", rope_phases)

        if pe_mode != "rope":
            self.rope_phases = None

        self.input_layer = operations.Linear(in_channels, model_channels, dtype=self.dtype, device=device)

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                dtype=self.dtype,
                device=device,
                operations=operations,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = operations.Linear(model_channels, out_channels, dtype=self.dtype, device=device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        pass

    def _post_load(self, device) -> None:
        """
        Recompute derived buffers after weight loading.

        rope_phases (RoPE positional encoding) is computed from model config,
        not stored in checkpoints. Must be recomputed after meta-device init.
        """
        if self.pe_mode == "rope":
            pos_embedder = RotaryPositionEmbedder(self.model_channels // self.num_heads, 3)
            coords = torch.meshgrid(*[torch.arange(self.resolution, device=device) for _ in range(3)], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            self.rope_phases = pos_embedder(coords)
        elif self.pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(self.model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(self.resolution, device=device) for _ in range(3)], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            self.pos_emb = pos_embedder(coords)

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

        elif self.initialization == 'scaled':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Scaled init for to_out and ffn2
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)

            # Initialize input layer to make the initial representation have variance 1
            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, transformer_options={}, **kwargs) -> torch.Tensor:
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, t, cond, transformer_options, **kwargs)

    def _forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, transformer_options={}, **kwargs) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()

        transformer_options = transformer_options.copy()
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        h = self.input_layer(h)
        if self.pe_mode == "ape":
            h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        transformer_options["block_type"] = "cross"
        transformer_options["total_blocks"] = len(self.blocks)

        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i

            if ("block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["x"] = block(args["x"], args["mod"], args["context"], args.get("phases"), transformer_options=args.get("transformer_options", {}))
                    return out

                out = blocks_replace[("block", i)](
                    {"x": h, "mod": t_emb, "context": cond, "phases": self.rope_phases, "transformer_options": transformer_options},
                    {"original_block": block_wrap}
                )
                h = out["x"]
            else:
                h = block(h, t_emb, cond, self.rope_phases, transformer_options=transformer_options)

        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()

        return h


# =============================================================================
# 14. SparseTransformerElasticMixin (from models/sparse_elastic_mixin.py)
# =============================================================================

class ElasticModuleMixin:
    """Mixin for training with elastic memory management."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_controller = None

    @abstractmethod
    def _get_input_size(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0):
        pass

    def register_memory_controller(self, memory_controller):
        self._memory_controller = memory_controller

    def forward(self, *args, **kwargs):
        if self._memory_controller is None or not torch.is_grad_enabled() or not self.training:
            ret = super().forward(*args, **kwargs)
        else:
            input_size = self._get_input_size(*args, **kwargs)
            mem_ratio = self._memory_controller.get_mem_ratio(input_size)
            with self.with_mem_ratio(mem_ratio) as exact_mem_ratio:
                ret = super().forward(*args, **kwargs)
            self._memory_controller.update_run_states(input_size, exact_mem_ratio)
        return ret


class SparseTransformerElasticMixin(ElasticModuleMixin):
    def _get_input_size(self, x: SparseTensor, *args, **kwargs):
        return x.feats.shape[0]

    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0):
        if mem_ratio == 1.0:
            yield 1.0
            return
        num_blocks = len(self.blocks)
        num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
        exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = i < num_checkpoint_blocks
        yield exact_mem_ratio
        for i in range(num_blocks):
            self.blocks[i].use_checkpoint = False


# =============================================================================
# 15. SLatFlowModel and ElasticSLatFlowModel (from models/structured_latent_flow.py)
# =============================================================================

class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'float32',
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        device=None,
        operations=ops,
        sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels, dtype=self.dtype, device=device, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(model_channels, 6 * model_channels, bias=True, dtype=self.dtype, device=device)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sparse_operations.SparseLinear(in_channels, model_channels, dtype=self.dtype, device=device)

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                dtype=self.dtype,
                device=device,
                operations=operations,
                sparse_operations=sparse_operations,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = sparse_operations.SparseLinear(model_channels, out_channels, dtype=self.dtype, device=device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        pass

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

        elif self.initialization == 'scaled':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Scaled init for to_out and ffn2
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)

            # Initialize input layer to make the initial representation have variance 1
            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[SparseTensor] = None,
        transformer_options={},
        **kwargs
    ) -> SparseTensor:
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, t, cond, concat_cond, transformer_options, **kwargs)

    def _forward(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[SparseTensor] = None,
        transformer_options={},
        **kwargs
    ) -> SparseTensor:
        if concat_cond is not None:
            x = sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = VarLenTensor.from_tensor_list(cond)

        h = self.input_layer(x)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        transformer_options = transformer_options.copy()
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + pe

        transformer_options["block_type"] = "cross"
        transformer_options["total_blocks"] = len(self.blocks)

        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i

            if ("block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["x"] = block(args["x"], args["mod"], args["context"], transformer_options=args.get("transformer_options", {}))
                    return out

                out = blocks_replace[("block", i)](
                    {"x": h, "mod": t_emb, "context": cond, "transformer_options": transformer_options},
                    {"original_block": block_wrap}
                )
                h = out["x"]
            else:
                h = block(h, t_emb, cond, transformer_options=transformer_options)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h


class ElasticSLatFlowModel(SparseTransformerElasticMixin, SLatFlowModel):
    """
    SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """
    pass

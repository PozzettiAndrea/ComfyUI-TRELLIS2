from typing import *
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import comfy.ops
import comfy.model_management
from .model import LayerNorm32, GroupNorm32, ChannelLayerNorm32, zero_module, pixel_shuffle_3d, str_to_dtype
from .sparse import SparseTensor, VarLenTensor, SparseDownsample, SparseUpsample, SparseSpatial2Channel, SparseChannel2Spatial, sparse_cat, SparseActivation
from .ops_sparse import manual_cast as sparse_ops
ops = comfy.ops.manual_cast


# ============================================================================
# Section 1: Dense 3D VAE (from sparse_structure_vae.py)
# ============================================================================

def norm_layer(norm_type: str, *args, dtype=None, device=None, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    if norm_type == "group":
        return GroupNorm32(32, *args, dtype=dtype, device=device, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, dtype=dtype, device=device, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels, dtype=dtype, device=device)
        self.norm2 = norm_layer(norm_type, self.out_channels, dtype=dtype, device=device)
        self.conv1 = operations.Conv3d(channels, self.out_channels, 3, padding=1, dtype=dtype, device=device)
        self.conv2 = zero_module(operations.Conv3d(self.out_channels, self.out_channels, 3, padding=1, dtype=dtype, device=device))
        self.skip_connection = operations.Conv3d(channels, self.out_channels, 1, dtype=dtype, device=device) if channels != self.out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
        dtype=None, device=None, operations=ops,
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = operations.Conv3d(in_channels, out_channels, 2, stride=2, dtype=dtype, device=device)
        elif mode == "avgpool":
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
        dtype=None, device=None, operations=ops,
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = operations.Conv3d(in_channels, out_channels*8, 3, padding=1, dtype=dtype, device=device)
        elif mode == "nearest":
            assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class SparseStructureEncoder(nn.Module):
    """
    Encoder for Sparse Structure (E_S in the paper Sec. 3.3).

    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16

        self.input_layer = operations.Conv3d(in_channels, channels[0], 3, padding=1, dtype=dtype, device=device)

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch, dtype=dtype, device=device, operations=operations)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i+1], dtype=dtype, device=device, operations=operations)
                )

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1], dtype=dtype, device=device, operations=operations)
            for _ in range(num_res_blocks_middle)
        ])

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1], dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv3d(channels[-1], latent_channels*2, 3, padding=1, dtype=dtype, device=device)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        pass

    def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
        h = self.input_layer(x)

        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)

        h = self.out_layer(h)

        mean, logvar = h.chunk(2, dim=1)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        if return_raw:
            return z, mean, logvar
        return z


class SparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure (D_S in the paper Sec. 3.3).

    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16

        self.input_layer = operations.Conv3d(latent_channels, channels[0], 3, padding=1, dtype=dtype, device=device)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0], dtype=dtype, device=device, operations=operations)
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch, dtype=dtype, device=device, operations=operations)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i+1], dtype=dtype, device=device, operations=operations)
                )

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1], dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv3d(channels[-1], out_channels, 3, padding=1, dtype=dtype, device=device)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        h = self.out_layer(h)
        return h


# ============================================================================
# Section 2: Sparse UNet VAE (from sparse_unet_vae.py)
# ============================================================================

# Use the sparse module as 'sp' namespace for SparseTensor operations
import importlib as _importlib
sp = _importlib.import_module('.sparse', __name__.rsplit('.', 1)[0] if '.' in __name__ else __name__)


def _apply_in_chunks(module: nn.Module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0 or x.shape[0] <= chunk_size:
        return module(x)

    outputs = []
    for start in range(0, x.shape[0], chunk_size):
        outputs.append(module(x[start : start + chunk_size]))
    return torch.cat(outputs, dim=0)


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        resample_mode: Literal['nearest', 'spatial2channel'] = 'nearest',
        use_checkpoint: bool = False,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.resample_mode = resample_mode
        self.use_checkpoint = use_checkpoint

        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        if resample_mode == 'nearest':
            self.conv1 = sparse_operations.SparseConv3d(channels, self.out_channels, 3, dtype=dtype, device=device)
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.conv1 = sparse_operations.SparseConv3d(channels, self.out_channels * 8, 3, dtype=dtype, device=device)
        elif resample_mode =='spatial2channel' and self.downsample:
            self.conv1 = sparse_operations.SparseConv3d(channels, self.out_channels // 8, 3, dtype=dtype, device=device)
        self.conv2 = zero_module(sparse_operations.SparseConv3d(self.out_channels, self.out_channels, 3, dtype=dtype, device=device))
        if resample_mode == 'nearest':
            self.skip_connection = sparse_operations.SparseLinear(channels, self.out_channels, dtype=dtype, device=device) if channels != self.out_channels else nn.Identity()
        elif resample_mode =='spatial2channel' and self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        self.updown = None
        if self.downsample:
            if resample_mode == 'nearest':
                self.updown = sp.SparseDownsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseSpatial2Channel(2)
        elif self.upsample:
            self.to_subdiv = sparse_operations.SparseLinear(channels, 8, dtype=dtype, device=device)
            if resample_mode == 'nearest':
                self.updown = sp.SparseUpsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseChannel2Spatial(2)

    def _updown(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.downsample:
            x = self.updown(x)
        elif self.upsample:
            x = self.updown(x, subdiv.replace(subdiv.feats > 0))
        return x

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        subdiv = None
        if self.upsample:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        if self.resample_mode == 'spatial2channel':
            h = self.conv1(h)
        h = self._updown(h, subdiv)
        x = self._updown(x, subdiv)
        if self.resample_mode == 'nearest':
            h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.upsample:
            return h, subdiv
        return h

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockDownsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.conv1 = sparse_operations.SparseConv3d(channels, self.out_channels, 3, dtype=dtype, device=device)
        self.conv2 = zero_module(sparse_operations.SparseConv3d(self.out_channels, self.out_channels, 3, dtype=dtype, device=device))
        self.skip_connection = sparse_operations.SparseLinear(channels, self.out_channels, dtype=dtype, device=device) if channels != self.out_channels else nn.Identity()
        self.updown = sp.SparseDownsample(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.updown(h)
        x = self.updown(x)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.conv1 = sparse_operations.SparseConv3d(channels, self.out_channels, 3, dtype=dtype, device=device)
        self.conv2 = zero_module(sparse_operations.SparseConv3d(self.out_channels, self.out_channels, 3, dtype=dtype, device=device))
        self.skip_connection = sparse_operations.SparseLinear(channels, self.out_channels, dtype=dtype, device=device) if channels != self.out_channels else nn.Identity()
        if self.pred_subdiv:
            self.to_subdiv = sparse_operations.SparseLinear(channels, 8, dtype=dtype, device=device)
        self.updown = sp.SparseUpsample(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockS2C3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.conv1 = sparse_operations.SparseConv3d(channels, self.out_channels // 8, 3, dtype=dtype, device=device)
        self.conv2 = zero_module(sparse_operations.SparseConv3d(self.out_channels, self.out_channels, 3, dtype=dtype, device=device))
        self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        self.updown = sp.SparseSpatial2Channel(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = self.updown(h)
        x = self.updown(x)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockC2S3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.conv1 = sparse_operations.SparseConv3d(channels, self.out_channels * 8, 3, dtype=dtype, device=device)
        self.conv2 = zero_module(sparse_operations.SparseConv3d(self.out_channels, self.out_channels, 3, dtype=dtype, device=device))
        self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        if pred_subdiv:
            self.to_subdiv = sparse_operations.SparseLinear(channels, 8, dtype=dtype, device=device)
        self.updown = sp.SparseChannel2Spatial(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h

    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, subdiv, use_reentrant=False)
        else:
            return self._forward(x, subdiv)


class SparseConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.low_vram = False
        self.mlp_chunk_size = 8192

        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.conv = sparse_operations.SparseConv3d(channels, channels, 3, dtype=dtype, device=device)
        self.mlp = nn.Sequential(
            operations.Linear(channels, int(channels * mlp_ratio), dtype=dtype, device=device),
            nn.SiLU(),
            zero_module(operations.Linear(int(channels * mlp_ratio), channels, dtype=dtype, device=device)),
        )

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        norm_feats = self.norm(h.feats)
        del h  # free conv output feats before MLP expansion
        steps = max(1, (norm_feats.shape[0] + self.mlp_chunk_size - 1) // self.mlp_chunk_size) if self.low_vram else 1
        while True:
            try:
                chunk_size = (norm_feats.shape[0] + steps - 1) // steps if steps > 1 else 0
                feats = _apply_in_chunks(self.mlp, norm_feats, chunk_size)
                break
            except comfy.model_management.OOM_EXCEPTION as e:
                comfy.model_management.soft_empty_cache(True)
                steps *= 2
                if steps > 64:
                    raise e
        del norm_feats  # free norm output before residual add
        feats.add_(x.feats)  # in-place residual
        return x.replace(feats)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseUnetVaeEncoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks

        self.input_layer = sparse_operations.SparseLinear(in_channels, model_channels[0], dtype=dtype, device=device)
        self.to_latent = sparse_operations.SparseLinear(model_channels[-1], 2 * latent_channels, dtype=dtype, device=device)

        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        dtype=dtype,
                        device=device,
                        operations=operations,
                        sparse_operations=sparse_operations,
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[down_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        dtype=dtype,
                        device=device,
                        operations=operations,
                        sparse_operations=sparse_operations,
                        **block_args[i],
                    )
                )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        pass

    def convert_to_fp16(self) -> None:
        pass

    def convert_to_fp32(self) -> None:
        pass

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, sample_posterior=False, return_raw=False):
        h = self.input_layer(x)
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                h = block(h)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.to_latent(h)

        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)

        if return_raw:
            return z, mean, logvar
        else:
            return z


class SparseUnetVaeDecoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
        dtype=None, device=None, operations=ops, sparse_operations=sparse_ops,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.pred_subdiv = pred_subdiv
        self._low_vram = False

        self.output_layer = sparse_operations.SparseLinear(model_channels[-1], out_channels, dtype=dtype, device=device)
        self.from_latent = sparse_operations.SparseLinear(latent_channels, model_channels[0], dtype=dtype, device=device)

        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        dtype=dtype,
                        device=device,
                        operations=operations,
                        sparse_operations=sparse_operations,
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[up_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        pred_subdiv=pred_subdiv,
                        dtype=dtype,
                        device=device,
                        operations=operations,
                        sparse_operations=sparse_operations,
                        **block_args[i],
                    )
                )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        pass

    @property
    def low_vram(self) -> bool:
        return self._low_vram

    @low_vram.setter
    def low_vram(self, value: bool) -> None:
        self._low_vram = bool(value)
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "low_vram"):
                try:
                    setattr(module, "low_vram", self._low_vram)
                except Exception:
                    pass

    def convert_to_fp16(self) -> None:
        pass

    def convert_to_fp32(self) -> None:
        pass

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, guide_subs: Optional[List[sp.SparseTensor]] = None, return_subs: bool = False) -> sp.SparseTensor:
        assert guide_subs is None or self.pred_subdiv == False, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert return_subs == False or self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with return_subs"

        import comfy.utils
        total_blocks = sum(len(res) for res in self.blocks)
        pbar = comfy.utils.ProgressBar(total_blocks + 1)  # +1 for output layer

        h = self.from_latent(x)
        subs_gt = []
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache('subdivision'))
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)
                pbar.update(1)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.output_layer(h)
        pbar.update(1)
        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        else:
            if return_subs:
                return h, subs
            else:
                return h

    def upsample(self, x: sp.SparseTensor, upsample_times: int) -> torch.Tensor:
        assert self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with upsampling"

        h = self.from_latent(x)
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    h, sub = block(h)
                else:
                    h = block(h)


# ============================================================================
# Section 3: FlexiDualGrid VAE (from fdg_vae.py)
# ============================================================================

from o_voxel.convert import flexible_dual_grid_to_mesh
from o_voxel_vb.convert import tiled_flexible_dual_grid_to_mesh
import comfy.model_management as _cmm


class Mesh:
    """Mesh representation (vertices + faces + optional vertex attributes)."""
    def __init__(self, vertices, faces, vertex_attrs=None):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs

    @property
    def device(self):
        return self.vertices.device

    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )

    def cuda(self, non_blocking=False):
        return self.to(_cmm.get_torch_device(), non_blocking=non_blocking)

    def cpu(self):
        return self.to('cpu')

    def fill_holes(self, max_hole_perimeter=3e-2):
        import cumesh
        device = _cmm.get_torch_device()
        vertices = self.vertices.to(device)
        faces = self.faces.to(device)
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.get_edges()
        mesh.get_boundary_info()
        if mesh.num_boundaries == 0:
            return
        mesh.get_vertex_edge_adjacency()
        mesh.get_vertex_boundary_adjacency()
        mesh.get_manifold_boundary_adjacency()
        mesh.read_manifold_boundary_adjacency()
        mesh.get_boundary_connected_components()
        mesh.get_boundary_loops()
        if mesh.num_boundary_loops == 0:
            return
        mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
        new_vertices, new_faces = mesh.read()
        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)

    def remove_faces(self, face_mask: torch.Tensor):
        import cumesh
        device = _cmm.get_torch_device()
        vertices = self.vertices.to(device)
        faces = self.faces.to(device)
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.remove_faces(face_mask)
        new_vertices, new_faces = mesh.read()
        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)

    def simplify(self, target=1000000, verbose: bool = False, options: dict = {}):
        import cumesh
        device = _cmm.get_torch_device()
        vertices = self.vertices.to(device)
        faces = self.faces.to(device)
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.simplify(target, verbose=verbose, options=options)
        new_vertices, new_faces = mesh.read()
        self.vertices = new_vertices.to(self.device)
        self.faces = new_faces.to(self.device)


class Voxel:
    """Voxel representation for 3D data."""
    def __init__(self, origin: list, voxel_size: float, coords: torch.Tensor = None,
                 attrs: torch.Tensor = None, layout: Dict = {}, device: torch.device = None):
        if device is None:
            device = _cmm.get_torch_device()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.layout = layout
        self.device = device

    @property
    def position(self):
        return (self.coords + 0.5) * self.voxel_size + self.origin[None, :]

    def split_attrs(self):
        return {k: self.attrs[:, self.layout[k]] for k in self.layout}


class MeshWithVoxel(Mesh, Voxel):
    """Combined mesh + voxel representation with PBR attribute querying."""
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, origin: list,
                 voxel_size: float, coords: torch.Tensor, attrs: torch.Tensor,
                 voxel_shape: torch.Size, layout: Dict = {}):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(), self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape, self.layout,
        )

    def query_attrs(self, xyz):
        from flex_gemm.ops.grid_sample import grid_sample_3d
        grid = ((xyz - self.origin) / self.voxel_size).reshape(1, -1, 3)
        vertex_attrs = grid_sample_3d(
            self.attrs,
            torch.cat([torch.zeros_like(self.coords[..., :1]), self.coords], dim=-1),
            self.voxel_shape, grid, mode='trilinear'
        )[0]
        return vertex_attrs

    def query_vertex_attrs(self):
        return self.query_attrs(self.vertices)


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        dtype=None,
        device=None,
        operations=ops,
        sparse_operations=sparse_ops,
    ):
        super().__init__(
            6,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )

    def forward(self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, sample_posterior=False, return_raw=False):
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        return super().forward(x, sample_posterior, return_raw)


class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
        dtype=None,
        device=None,
        operations=ops,
        sparse_operations=sparse_ops,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin

        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16,
            dtype=dtype,
            device=device,
            operations=operations,
            sparse_operations=sparse_operations,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

    def forward(self, x: sp.SparseTensor, gt_intersected: sp.SparseTensor = None, **kwargs):
        decoded = super().forward(x, **kwargs)
        if self.training:
            h, subs_gt, subs = decoded
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected_logits = h.replace(h.feats[..., 3:6])
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(flexible_dual_grid_to_mesh(
                h.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=True
            )) for v, i, q in zip(vertices, gt_intersected, quad_lerp)]
            return mesh, vertices, intersected_logits, subs_gt, subs
        else:
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected = h.replace(h.feats[..., 3:6] > 0)
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(*tiled_flexible_dual_grid_to_mesh(
                coords=h.coords[:, 1:],
                dual_vertices=v.feats,
                intersected_flag=i.feats,
                split_weight=q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                tile_size=128,
                train=False
            )) for v, i, q in zip(vertices, intersected, quad_lerp)]
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)

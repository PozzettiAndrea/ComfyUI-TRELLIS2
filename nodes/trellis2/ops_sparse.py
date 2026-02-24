"""
ComfyUI-native sparse operations for TRELLIS2.

Mirrors comfy/ops.py: provides `disable_weight_init` and `manual_cast` tiers
for sparse layers operating on VarLenTensor / SparseTensor.

Usage:
    # In model constructors:
    def __init__(self, ..., dtype=None, device=None, operations=None, sparse_operations=None):
        self.linear = sparse_operations.SparseLinear(dim, dim, dtype=dtype, device=device)
        self.norm = sparse_operations.SparseGroupNorm(groups, dim, dtype=dtype, device=device)
        self.conv = sparse_operations.SparseConv3d(in_ch, out_ch, 3, dtype=dtype, device=device)
"""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.model_management
from comfy.ops import cast_bias_weight, uncast_bias_weight, CastWeightBiasOp, run_every_op


# ---------------------------------------------------------------------------
# Lazy backend loader for sparse convolutions
# ---------------------------------------------------------------------------

_conv_backends = {}


def _get_conv_backend():
    from .modules.sparse.config import get_conv_backend
    backend = get_conv_backend()
    if backend not in _conv_backends:
        _conv_backends[backend] = importlib.import_module(
            f'.modules.sparse.conv.conv_{backend}', __package__,
        )
    return backend, _conv_backends[backend]


# ---------------------------------------------------------------------------
# disable_weight_init tier — skip random init, no auto-casting
# ---------------------------------------------------------------------------

class disable_weight_init:

    # -- SparseLinear -------------------------------------------------------

    class SparseLinear(comfy.ops.disable_weight_init.Linear):
        """Linear that accepts VarLenTensor: extract .feats, run linear, replace."""

        def forward_comfy_cast_weights(self, input):
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                weight, bias, offload = cast_bias_weight(self, input.feats, offloadable=True)
                out = F.linear(input.feats, weight, bias)
                uncast_bias_weight(self, weight, bias, offload)
                return input.replace(out)
            return super().forward_comfy_cast_weights(input)

        def forward(self, input, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(input)
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                return input.replace(F.linear(input.feats, self.weight, self.bias))
            return super().forward(input)

    # -- SparseGroupNorm ----------------------------------------------------

    class SparseGroupNorm(comfy.ops.disable_weight_init.GroupNorm):
        """GroupNorm that handles VarLenTensor with per-batch normalization."""

        @staticmethod
        def _sparse_group_norm(feats, layout, batch_size, num_channels, num_groups, weight, bias, eps):
            nfeats = torch.zeros_like(feats)
            for k in range(batch_size):
                bf = feats[layout[k]]
                bf = bf.permute(1, 0).reshape(1, num_channels, -1)
                bf = F.group_norm(bf, num_groups, weight, bias, eps)
                bf = bf.reshape(num_channels, -1).permute(1, 0)
                nfeats[layout[k]] = bf
            return nfeats

        def forward_comfy_cast_weights(self, input):
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                weight, bias, offload = cast_bias_weight(self, input.feats, offloadable=True)
                nfeats = self._sparse_group_norm(
                    input.feats, input.layout, input.shape[0], input.shape[1],
                    self.num_groups, weight, bias, self.eps,
                )
                uncast_bias_weight(self, weight, bias, offload)
                return input.replace(nfeats)
            return super().forward_comfy_cast_weights(input)

        def forward(self, input, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(input)
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                nfeats = self._sparse_group_norm(
                    input.feats, input.layout, input.shape[0], input.shape[1],
                    self.num_groups, self.weight, self.bias, self.eps,
                )
                return input.replace(nfeats)
            return super().forward(input)

    # -- SparseLayerNorm ----------------------------------------------------

    class SparseLayerNorm(comfy.ops.disable_weight_init.LayerNorm):
        """LayerNorm that handles VarLenTensor with per-batch normalization."""

        @staticmethod
        def _sparse_layer_norm(feats, layout, batch_size, num_channels, normalized_shape, weight, bias, eps):
            nfeats = torch.zeros_like(feats)
            for k in range(batch_size):
                bf = feats[layout[k]]
                bf = bf.permute(1, 0).reshape(1, num_channels, -1)
                bf = F.layer_norm(bf, normalized_shape, weight, bias, eps)
                bf = bf.reshape(num_channels, -1).permute(1, 0)
                nfeats[layout[k]] = bf
            return nfeats

        def forward_comfy_cast_weights(self, input):
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                if self.weight is not None:
                    weight, bias, offload = cast_bias_weight(self, input.feats, offloadable=True)
                else:
                    weight, bias, offload = None, None, None
                nfeats = self._sparse_layer_norm(
                    input.feats, input.layout, input.shape[0], input.shape[1],
                    self.normalized_shape, weight, bias, self.eps,
                )
                uncast_bias_weight(self, weight, bias, offload)
                return input.replace(nfeats)
            return super().forward_comfy_cast_weights(input)

        def forward(self, input, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(input)
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                nfeats = self._sparse_layer_norm(
                    input.feats, input.layout, input.shape[0], input.shape[1],
                    self.normalized_shape, self.weight, self.bias, self.eps,
                )
                return input.replace(nfeats)
            return super().forward(input)

    # -- SparseGroupNorm32 / SparseLayerNorm32 ------------------------------
    # Float32 computation wrappers.

    class SparseGroupNorm32(SparseGroupNorm):
        """SparseGroupNorm that computes in float32."""

        def forward_comfy_cast_weights(self, input):
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                orig_dtype = input.feats.dtype
                input = input.replace(input.feats.float())
                weight, bias, offload = cast_bias_weight(self, input.feats, offloadable=True)
                if weight is not None:
                    weight = weight.float()
                if bias is not None:
                    bias = bias.float()
                nfeats = self._sparse_group_norm(
                    input.feats, input.layout, input.shape[0], input.shape[1],
                    self.num_groups, weight, bias, self.eps,
                )
                uncast_bias_weight(self, weight, bias, offload)
                return input.replace(nfeats.to(orig_dtype))
            return super().forward_comfy_cast_weights(input)

        def forward(self, input, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(input)
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                orig_dtype = input.feats.dtype
                feats32 = input.feats.float()
                w = self.weight.float() if self.weight is not None else None
                b = self.bias.float() if self.bias is not None else None
                nfeats = self._sparse_group_norm(
                    feats32, input.layout, input.shape[0], input.shape[1],
                    self.num_groups, w, b, self.eps,
                )
                return input.replace(nfeats.to(orig_dtype))
            return super().forward(input)

    class SparseLayerNorm32(SparseLayerNorm):
        """SparseLayerNorm that computes in float32."""

        def forward_comfy_cast_weights(self, input):
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                orig_dtype = input.feats.dtype
                input = input.replace(input.feats.float())
                if self.weight is not None:
                    weight, bias, offload = cast_bias_weight(self, input.feats, offloadable=True)
                    weight = weight.float() if weight is not None else None
                    bias = bias.float() if bias is not None else None
                else:
                    weight, bias, offload = None, None, None
                nfeats = self._sparse_layer_norm(
                    input.feats, input.layout, input.shape[0], input.shape[1],
                    self.normalized_shape, weight, bias, self.eps,
                )
                uncast_bias_weight(self, weight, bias, offload)
                return input.replace(nfeats.to(orig_dtype))
            return super().forward_comfy_cast_weights(input)

        def forward(self, input, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(input)
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                orig_dtype = input.feats.dtype
                feats32 = input.feats.float()
                w = self.weight.float() if self.weight is not None else None
                b = self.bias.float() if self.bias is not None else None
                nfeats = self._sparse_layer_norm(
                    feats32, input.layout, input.shape[0], input.shape[1],
                    self.normalized_shape, w, b, self.eps,
                )
                return input.replace(nfeats.to(orig_dtype))
            return super().forward(input)

    # -- SparseConv3d -------------------------------------------------------

    class SparseConv3d(nn.Module):
        """
        Sparse 3D convolution with backend dispatch and ComfyUI auto-casting.

        Weight/bias live wherever the backend places them (self.conv.weight for
        spconv/torchsparse, self.weight for flex_gemm). Forward temporarily injects
        cast weights into the backend before running.
        """
        comfy_cast_weights = False
        weight_function = []
        bias_function = []

        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     dilation=1, padding=None, bias=True, indice_key=None,
                     dtype=None, device=None):
            super().__init__()
            _, backend_mod = _get_conv_backend()
            backend_mod.sparse_conv3d_init(
                self, in_channels, out_channels, kernel_size,
                stride, dilation, padding, bias, indice_key,
            )

        def reset_parameters(self):
            return None

        def _get_weight_bias(self):
            """Find weight/bias regardless of backend storage location."""
            if hasattr(self, 'conv'):
                return self.conv.weight, getattr(self.conv, 'bias', None)
            return self.weight, getattr(self, 'bias', None)

        def _forward(self, x):
            _, backend_mod = _get_conv_backend()
            return backend_mod.sparse_conv3d_forward(self, x)

        def forward_comfy_cast_weights(self, x):
            weight_param, bias_param = self._get_weight_bias()
            dtype = x.feats.dtype
            device = x.feats.device

            # Save original data, swap in cast versions
            orig_w = weight_param.data
            weight_param.data = comfy.model_management.cast_to(orig_w, dtype, device)

            orig_b = None
            if bias_param is not None:
                orig_b = bias_param.data
                bias_param.data = comfy.model_management.cast_to(orig_b, dtype, device)

            out = self._forward(x)

            # Restore originals
            weight_param.data = orig_w
            if bias_param is not None:
                bias_param.data = orig_b

            return out

        def forward(self, x):
            run_every_op()
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(x)
            return self._forward(x)

    # -- SparseInverseConv3d ------------------------------------------------

    class SparseInverseConv3d(nn.Module):
        """Sparse inverse (transposed) 3D convolution with auto-casting."""
        comfy_cast_weights = False
        weight_function = []
        bias_function = []

        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     dilation=1, bias=True, indice_key=None,
                     dtype=None, device=None):
            super().__init__()
            _, backend_mod = _get_conv_backend()
            backend_mod.sparse_inverse_conv3d_init(
                self, in_channels, out_channels, kernel_size,
                stride, dilation, bias, indice_key,
            )

        def reset_parameters(self):
            return None

        def _get_weight_bias(self):
            if hasattr(self, 'conv'):
                return self.conv.weight, getattr(self.conv, 'bias', None)
            return self.weight, getattr(self, 'bias', None)

        def _forward(self, x):
            _, backend_mod = _get_conv_backend()
            return backend_mod.sparse_inverse_conv3d_forward(self, x)

        def forward_comfy_cast_weights(self, x):
            weight_param, bias_param = self._get_weight_bias()
            dtype = x.feats.dtype
            device = x.feats.device

            orig_w = weight_param.data
            weight_param.data = comfy.model_management.cast_to(orig_w, dtype, device)

            orig_b = None
            if bias_param is not None:
                orig_b = bias_param.data
                bias_param.data = comfy.model_management.cast_to(orig_b, dtype, device)

            out = self._forward(x)

            weight_param.data = orig_w
            if bias_param is not None:
                bias_param.data = orig_b

            return out

        def forward(self, x):
            run_every_op()
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(x)
            return self._forward(x)

    # -- Sparse Activations -------------------------------------------------
    # These have no learnable parameters, so no casting needed.
    # Just wrap VarLenTensor → feats → activation → replace.

    class SparseReLU(nn.ReLU):
        def forward(self, input):
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                return input.replace(super().forward(input.feats))
            return super().forward(input)

    class SparseSiLU(nn.SiLU):
        def forward(self, input):
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                return input.replace(super().forward(input.feats))
            return super().forward(input)

    class SparseGELU(nn.GELU):
        def forward(self, input):
            from .modules.sparse.basic import VarLenTensor
            if isinstance(input, VarLenTensor):
                return input.replace(super().forward(input.feats))
            return super().forward(input)


# ---------------------------------------------------------------------------
# manual_cast tier — auto-cast weights to input dtype during forward
# ---------------------------------------------------------------------------

class manual_cast(disable_weight_init):

    class SparseLinear(disable_weight_init.SparseLinear):
        comfy_cast_weights = True

    class SparseGroupNorm(disable_weight_init.SparseGroupNorm):
        comfy_cast_weights = True

    class SparseLayerNorm(disable_weight_init.SparseLayerNorm):
        comfy_cast_weights = True

    class SparseGroupNorm32(disable_weight_init.SparseGroupNorm32):
        comfy_cast_weights = True

    class SparseLayerNorm32(disable_weight_init.SparseLayerNorm32):
        comfy_cast_weights = True

    class SparseConv3d(disable_weight_init.SparseConv3d):
        comfy_cast_weights = True

    class SparseInverseConv3d(disable_weight_init.SparseInverseConv3d):
        comfy_cast_weights = True

    class SparseReLU(disable_weight_init.SparseReLU):
        pass  # No weights to cast

    class SparseSiLU(disable_weight_init.SparseSiLU):
        pass

    class SparseGELU(disable_weight_init.SparseGELU):
        pass

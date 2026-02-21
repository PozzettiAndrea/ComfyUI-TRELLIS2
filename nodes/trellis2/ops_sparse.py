"""
ComfyUI-native sparse operations for TRELLIS2.

Mirrors comfy/ops.py: provides `disable_weight_init` and `manual_cast` tiers
for sparse layers operating on VarLenTensor / SparseTensor.

Also contains sparse conv backend detection and dispatch (spconv, torchsparse,
flex_gemm) — the authoritative source for all sparse convolution operations.

Usage:
    # In model constructors:
    def __init__(self, ..., dtype=None, device=None, operations=None, sparse_operations=None):
        self.linear = sparse_operations.SparseLinear(dim, dim, dtype=dtype, device=device)
        self.norm = sparse_operations.SparseGroupNorm(groups, dim, dtype=dtype, device=device)
        self.conv = sparse_operations.SparseConv3d(in_ch, out_ch, 3, dtype=dtype, device=device)
"""

import logging
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.model_management
from comfy.ops import cast_bias_weight, uncast_bias_weight, CastWeightBiasOp, run_every_op

log = logging.getLogger("trellis2")


# ==========================================================================
# Conv backend config and detection
# ==========================================================================

SPCONV_ALGO = 'auto'
FLEX_GEMM_ALGO = 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 2.0

_CONV = None  # Lazy initialization — detected on first use


def _detect_available_conv_backend() -> str:
    """Try to import conv backends in priority order, return first available."""
    env_backend = os.environ.get('SPARSE_CONV_BACKEND')
    if env_backend:
        valid_backends = ['none', 'spconv', 'torchsparse', 'flex_gemm']
        if env_backend in valid_backends:
            log.info(f"Using conv backend from SPARSE_CONV_BACKEND env var: {env_backend}")
            return env_backend
        else:
            log.warning(f"Invalid SPARSE_CONV_BACKEND '{env_backend}', must be one of {valid_backends}")

    backends = ['flex_gemm', 'spconv', 'torchsparse']
    for backend in backends:
        try:
            if backend == 'flex_gemm':
                import flex_gemm
                log.info("Auto-detected conv backend: flex_gemm")
                return backend
            elif backend == 'spconv':
                import spconv
                log.info("Auto-detected conv backend: spconv")
                return backend
            elif backend == 'torchsparse':
                import torchsparse
                log.info("Auto-detected conv backend: torchsparse")
                return backend
        except ImportError:
            continue
        except Exception as e:
            log.warning(f"{backend} import failed: {e}")
            continue

    log.info("No sparse conv backend available, using none")
    return 'none'


def get_conv_backend() -> str:
    """Get current conv backend, detecting on first call."""
    global _CONV
    if _CONV is None:
        _CONV = _detect_available_conv_backend()
    return _CONV


def set_conv_backend(backend: str) -> None:
    """Set conv backend explicitly."""
    global _CONV
    valid_backends = ['none', 'spconv', 'torchsparse', 'flex_gemm']
    if backend not in valid_backends:
        raise ValueError(f"Invalid conv backend '{backend}', must be one of {valid_backends}")
    if _CONV is not None and _CONV != backend:
        log.info(f"Changing conv backend from {_CONV} to {backend}")
    _CONV = backend
    log.info(f"Conv backend set to: {backend}")


# ==========================================================================
# Conv backend implementations (lazy-loaded)
# ==========================================================================

# --- spconv ---

_spconv_mod = None

def _load_spconv():
    global _spconv_mod
    if _spconv_mod is None:
        import spconv.pytorch as _spconv
        _spconv_mod = _spconv
    return _spconv_mod


def _spconv_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    spconv = _load_spconv()
    algo = None
    if SPCONV_ALGO == 'native':
        algo = spconv.ConvAlgo.Native
    elif SPCONV_ALGO == 'implicit_gemm':
        algo = spconv.ConvAlgo.MaskImplicitGemm
    if stride == 1 and (padding is None):
        self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
    else:
        self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
    self.padding = padding


def _spconv_conv3d_forward(self, x):
    from .sparse import SparseTensor
    spconv = _load_spconv()
    spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
    new_data = self.conv(x.data)
    new_shape = [x.shape[0], self.conv.out_channels]
    new_layout = None if spatial_changed else x.layout

    if spatial_changed and (x.shape[0] != 1):
        fwd = new_data.indices[:, 0].argsort()
        bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
        sorted_feats = new_data.features[fwd]
        sorted_coords = new_data.indices[fwd]
        unsorted_data = new_data
        new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)

    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )

    if spatial_changed and (x.shape[0] != 1):
        out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
        out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)

    return out


def _spconv_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
    spconv = _load_spconv()
    self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)


def _spconv_inverse_conv3d_forward(self, x):
    from .sparse import SparseTensor
    spatial_changed = any(s != 1 for s in self.stride)
    if spatial_changed:
        data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
        bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
        data = data.replace_feature(x.feats[bwd])
    else:
        data = x.data

    new_data = self.conv(data)
    new_shape = [x.shape[0], self.conv.out_channels]
    new_layout = None if spatial_changed else x.layout
    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )
    return out


# --- torchsparse ---

_torchsparse_mod = None

def _load_torchsparse():
    global _torchsparse_mod
    if _torchsparse_mod is None:
        import torchsparse as _ts
        _torchsparse_mod = _ts
    return _torchsparse_mod


def _torchsparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    torchsparse = _load_torchsparse()
    self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias)


def _torchsparse_conv3d_forward(self, x):
    from .sparse import SparseTensor
    out = self.conv(x.data)
    new_shape = [x.shape[0], self.conv.out_channels]
    out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
    out._spatial_cache = x._spatial_cache
    out._scale = tuple([s * stride for s, stride in zip(x._scale, self.conv.stride)])
    return out


def _torchsparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
    torchsparse = _load_torchsparse()
    self.conv = torchsparse.nn.Conv3d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias, transposed=True)


def _torchsparse_inverse_conv3d_forward(self, x):
    from .sparse import SparseTensor
    out = self.conv(x.data)
    new_shape = [x.shape[0], self.conv.out_channels]
    out = SparseTensor(out, shape=torch.Size(new_shape), layout=x.layout if all(s == 1 for s in self.conv.stride) else None)
    out._spatial_cache = x._spatial_cache
    out._scale = tuple([s / stride for s, stride in zip(x._scale, self.conv.stride)])
    return out


# --- flex_gemm ---

_flex_gemm_mod = None
_flex_gemm_spconv_ops = None

def _load_flex_gemm():
    global _flex_gemm_mod, _flex_gemm_spconv_ops
    if _flex_gemm_mod is None:
        import flex_gemm as _fg
        from flex_gemm.ops.spconv import sparse_submanifold_conv3d as _ssc
        _flex_gemm_mod = _fg
        _flex_gemm_spconv_ops = _ssc
    return _flex_gemm_mod, _flex_gemm_spconv_ops


def _flex_gemm_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    assert stride == 1 and (padding is None), 'Currently flex_gemm implementation only support submanifold sparse convolution (stride=1, padding=None)'

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, ) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, ) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, ) * 3

    self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    # Permute weight (Co, Ci, Kd, Kh, Kw) -> (Co, Kd, Kh, Kw, Ci)
    self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())


def _flex_gemm_conv3d_forward(self, x):
    flex_gemm_mod, sparse_submanifold_conv3d = _load_flex_gemm()
    flex_gemm_mod.ops.spconv.set_algorithm(FLEX_GEMM_ALGO)
    flex_gemm_mod.ops.spconv.set_hashmap_ratio(FLEX_GEMM_HASHMAP_RATIO)

    Co, Kd, Kh, Kw, Ci = self.weight.shape
    neighbor_cache_key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
    neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

    feats = x.feats
    if feats.dtype != self.weight.dtype:
        feats = feats.to(self.weight.dtype)
    out, neighbor_cache_ = sparse_submanifold_conv3d(
        feats,
        x.coords,
        torch.Size([*x.shape, *x.spatial_shape]),
        self.weight,
        self.bias,
        neighbor_cache,
        self.dilation
    )

    if neighbor_cache is None:
        x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)

    out = x.replace(out)
    return out


def _flex_gemm_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


def _flex_gemm_inverse_conv3d_forward(self, x):
    raise NotImplementedError('SparseInverseConv3d with flex_gemm is not implemented yet')


# --- Dispatch table ---

_conv_backend_dispatch = {
    'spconv': {
        'conv3d_init': _spconv_conv3d_init,
        'conv3d_forward': _spconv_conv3d_forward,
        'inverse_conv3d_init': _spconv_inverse_conv3d_init,
        'inverse_conv3d_forward': _spconv_inverse_conv3d_forward,
    },
    'torchsparse': {
        'conv3d_init': _torchsparse_conv3d_init,
        'conv3d_forward': _torchsparse_conv3d_forward,
        'inverse_conv3d_init': _torchsparse_inverse_conv3d_init,
        'inverse_conv3d_forward': _torchsparse_inverse_conv3d_forward,
    },
    'flex_gemm': {
        'conv3d_init': _flex_gemm_conv3d_init,
        'conv3d_forward': _flex_gemm_conv3d_forward,
        'inverse_conv3d_init': _flex_gemm_inverse_conv3d_init,
        'inverse_conv3d_forward': _flex_gemm_inverse_conv3d_forward,
    },
}


def _get_conv_backend():
    backend = get_conv_backend()
    return backend, _conv_backend_dispatch[backend]


# ==========================================================================
# disable_weight_init tier — skip random init, no auto-casting
# ==========================================================================

class disable_weight_init:

    # -- SparseLinear -------------------------------------------------------

    class SparseLinear(comfy.ops.disable_weight_init.Linear):
        """Linear that accepts VarLenTensor: extract .feats, run linear, replace."""

        def forward_comfy_cast_weights(self, input):
            from .sparse import VarLenTensor
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
            from .sparse import VarLenTensor
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
            from .sparse import VarLenTensor
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
            from .sparse import VarLenTensor
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
            from .sparse import VarLenTensor
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
            from .sparse import VarLenTensor
            if isinstance(input, VarLenTensor):
                nfeats = self._sparse_layer_norm(
                    input.feats, input.layout, input.shape[0], input.shape[1],
                    self.normalized_shape, self.weight, self.bias, self.eps,
                )
                return input.replace(nfeats)
            return super().forward(input)

    # -- SparseGroupNorm32 / SparseLayerNorm32 ------------------------------

    class SparseGroupNorm32(SparseGroupNorm):
        """SparseGroupNorm that computes in float32."""

        def forward_comfy_cast_weights(self, input):
            from .sparse import VarLenTensor
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
            from .sparse import VarLenTensor
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
            from .sparse import VarLenTensor
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
            from .sparse import VarLenTensor
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
            _, dispatch = _get_conv_backend()
            dispatch['conv3d_init'](
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
            _, dispatch = _get_conv_backend()
            return dispatch['conv3d_forward'](self, x)

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
            _, dispatch = _get_conv_backend()
            dispatch['inverse_conv3d_init'](
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
            _, dispatch = _get_conv_backend()
            return dispatch['inverse_conv3d_forward'](self, x)

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

    class SparseReLU(nn.ReLU):
        def forward(self, input):
            from .sparse import VarLenTensor
            if isinstance(input, VarLenTensor):
                return input.replace(super().forward(input.feats))
            return super().forward(input)

    class SparseSiLU(nn.SiLU):
        def forward(self, input):
            from .sparse import VarLenTensor
            if isinstance(input, VarLenTensor):
                return input.replace(super().forward(input.feats))
            return super().forward(input)

    class SparseGELU(nn.GELU):
        def forward(self, input):
            from .sparse import VarLenTensor
            if isinstance(input, VarLenTensor):
                return input.replace(super().forward(input.feats))
            return super().forward(input)


# ==========================================================================
# manual_cast tier — auto-cast weights to input dtype during forward
# ==========================================================================

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
        pass

    class SparseSiLU(disable_weight_init.SparseSiLU):
        pass

    class SparseGELU(disable_weight_init.SparseGELU):
        pass

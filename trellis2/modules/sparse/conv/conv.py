from .. import config
import importlib
import torch
import torch.nn as nn
from .. import SparseTensor


_backends = {}


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        backend = config.get_conv_backend()
        if backend not in _backends:
            _backends[backend] = importlib.import_module(f'..conv_{backend}', __name__)
        _backends[backend].sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, padding, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        backend = config.get_conv_backend()
        return _backends[backend].sparse_conv3d_forward(self, x)


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        backend = config.get_conv_backend()
        if backend not in _backends:
            _backends[backend] = importlib.import_module(f'..conv_{backend}', __name__)
        _backends[backend].sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        backend = config.get_conv_backend()
        return _backends[backend].sparse_inverse_conv3d_forward(self, x)

"""
TRELLIS2 latent format definitions.

Defines the latent spaces for each stage of the TRELLIS2 pipeline,
following the pattern established in comfy/latent_formats.py.

These formats describe the latent spaces so ComfyUI can understand
their properties (channel count, dimensionality, spatial scale).
"""
import torch
from comfy.latent_formats import LatentFormat


class TRELLIS2SparseStructure(LatentFormat):
    """
    Latent format for the sparse structure stage.

    Dense 3D grid (16x16x16) with 8 latent channels.
    Decoded to 128x128x128 binary voxel occupancy (8x spatial upscale).
    No normalization applied in this stage.
    """
    latent_channels = 8
    latent_dimensions = 3
    spacial_downscale_ratio = 8  # 128 / 16


class TRELLIS2ShapeSLat(LatentFormat):
    """
    Latent format for the shape structured latent (SLat) stage.

    Sparse 3D tokens with 32 latent channels per active voxel.
    Decoded to mesh geometry (7 output channels: vertices + intersected + quad_lerp).
    Per-channel mean/std normalization applied between sampling and decoding.
    """
    latent_channels = 32
    latent_dimensions = 3

    def __init__(self):
        self.latents_mean = torch.tensor([
            0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252,
            1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
            0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
            1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
            -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944,
            -3.186688, 1.578665,
        ]).view(1, 32)
        self.latents_std = torch.tensor([
            5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237,
            5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469,
            5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
            5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
            5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207,
            5.347008, 5.405691,
        ]).view(1, 32)

    def process_in(self, latent):
        """Normalize latent (encode direction)."""
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - mean) / std

    def process_out(self, latent):
        """Denormalize latent (decode direction)."""
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return latent * std + mean


class TRELLIS2TextureSLat(LatentFormat):
    """
    Latent format for the texture structured latent (SLat) stage.

    Sparse 3D tokens with 32 latent channels per active voxel.
    Decoded to PBR voxel attributes (6 output channels):
      - base_color: channels 0-2
      - metallic:   channel 3
      - roughness:  channel 4
      - alpha:      channel 5
    Per-channel mean/std normalization applied between sampling and decoding.

    Note: The texture flow model takes 64 input channels (32 shape_slat
    conditioning + 32 noise), but the latent space itself is 32 channels.
    """
    latent_channels = 32
    latent_dimensions = 3

    def __init__(self):
        self.latents_mean = torch.tensor([
            3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364,
            0.439898, -0.928075, 0.029398, -0.339596, -0.869527, 1.038479,
            -0.972385, 0.126042, -1.129303, 0.455149, -1.209521, 2.069067,
            0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717,
            1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524,
            -0.662387, -0.768862,
        ]).view(1, 32)
        self.latents_std = torch.tensor([
            2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316,
            2.144656, 2.911822, 2.969419, 2.501689, 2.154811, 3.163343,
            2.621215, 2.381943, 3.186697, 3.021588, 2.295916, 3.234985,
            3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999,
            2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849,
            2.786195, 2.775190,
        ]).view(1, 32)

    def process_in(self, latent):
        """Normalize latent (encode direction)."""
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - mean) / std

    def process_out(self, latent):
        """Denormalize latent (decode direction)."""
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return latent * std + mean

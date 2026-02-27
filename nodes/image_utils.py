"""Canonical image format conversions for ComfyUI custom nodes.

ComfyUI formats:
    IMAGE: [B, H, W, C] float32 in [0, 1], RGB
    MASK:  [B, H, W]    float32 in [0, 1]

Model formats:
    CHW:   [B, C, H, W] float32 (PyTorch conv/ViT input)
    numpy: [H, W, C]    uint8   [0, 255]
    PIL:   RGB Image
"""

import numpy as np
import torch
from PIL import Image


# ── ComfyUI ↔ numpy ──────────────────────────────────────────────


def comfy_to_numpy(image: torch.Tensor, index: int = 0) -> np.ndarray:
    """ComfyUI IMAGE [B,H,W,C] float32 → numpy [H,W,C] uint8."""
    return (image[index].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def numpy_to_comfy(image: np.ndarray) -> torch.Tensor:
    """numpy [H,W,C] uint8 → ComfyUI IMAGE [1,H,W,C] float32."""
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


# ── ComfyUI ↔ PIL ────────────────────────────────────────────────


def comfy_to_pil(image: torch.Tensor, index: int = 0) -> Image.Image:
    """ComfyUI IMAGE [B,H,W,C] float32 → PIL Image (RGB)."""
    return Image.fromarray(comfy_to_numpy(image, index))


def pil_to_comfy(image: Image.Image) -> torch.Tensor:
    """PIL Image → ComfyUI IMAGE [1,H,W,C] float32."""
    return numpy_to_comfy(np.array(image.convert("RGB")))


# ── ComfyUI ↔ CHW (model input) ─────────────────────────────────


def comfy_to_chw(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE [B,H,W,C] → model input [B,C,H,W]."""
    return image.movedim(-1, 1)


def chw_to_comfy(image: torch.Tensor) -> torch.Tensor:
    """Model output [B,C,H,W] → ComfyUI IMAGE [B,H,W,C]."""
    return image.movedim(1, -1)


# ── Mask conversions ─────────────────────────────────────────────


def mask_to_image(mask: torch.Tensor) -> torch.Tensor:
    """ComfyUI MASK [B,H,W] → ComfyUI IMAGE [B,H,W,3] (grayscale RGB)."""
    return mask.unsqueeze(-1).expand(-1, -1, -1, 3)


def image_to_mask(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE [B,H,W,C] → ComfyUI MASK [B,H,W] (luminance)."""
    if image.shape[-1] == 1:
        return image.squeeze(-1)
    # ITU-R BT.601 luminance
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])

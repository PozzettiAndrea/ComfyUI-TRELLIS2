"""Utility functions for ComfyUI-TRELLIS2."""
import logging
import torch
import numpy as np
from PIL import Image

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[TRELLIS2]")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI image tensor to PIL Image.

    ComfyUI tensors are [B, H, W, C] with values in [0, 1].
    Takes the first image if batch size > 1.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image

    # Convert to numpy
    np_image = tensor.cpu().numpy()

    # Scale to 0-255
    np_image = (np_image * 255).astype(np.uint8)

    # Create PIL image
    if np_image.shape[-1] == 4:
        return Image.fromarray(np_image, mode='RGBA')
    elif np_image.shape[-1] == 3:
        return Image.fromarray(np_image, mode='RGB')
    else:
        return Image.fromarray(np_image.squeeze(), mode='L')


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI image tensor.

    Returns tensor of shape [1, H, W, C] with values in [0, 1].
    """
    # Convert to numpy
    np_image = np.array(image).astype(np.float32) / 255.0

    # Ensure 3 dimensions [H, W, C]
    if np_image.ndim == 2:
        np_image = np_image[:, :, np.newaxis]

    # Add batch dimension [1, H, W, C]
    tensor = torch.from_numpy(np_image).unsqueeze(0)

    return tensor


def get_trellis_models_dir():
    """Get the directory for TRELLIS.2 models."""
    import folder_paths
    import os

    models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

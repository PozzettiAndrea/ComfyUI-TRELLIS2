"""
Helper utilities for TRELLIS2 nodes.

These run inside the isolated subprocess.
"""

import numpy as np
from PIL import Image
import torch


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI IMAGE tensor to PIL Image.

    Args:
        tensor: ComfyUI IMAGE format [B, H, W, C] or [H, W, C], float32 0-1

    Returns:
        PIL Image (RGB)
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch

    # Convert to numpy and scale to 0-255
    np_img = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI IMAGE tensor.

    Args:
        pil_image: PIL Image (RGB or RGBA)

    Returns:
        ComfyUI IMAGE format [1, H, W, C], float32 0-1
    """
    # Ensure RGB
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Convert to numpy
    np_img = np.array(pil_image).astype(np.float32) / 255.0

    # Add batch dimension [1, H, W, C]
    tensor = torch.from_numpy(np_img).unsqueeze(0)
    return tensor


def smart_crop_square(
    pil_image: Image.Image,
    mask_np: np.ndarray,
    margin_ratio: float = 0.1,
    background_color: tuple = (128, 128, 128),
) -> Image.Image:
    """
    Extract object with margin, pad to square.

    Args:
        pil_image: Input RGBA image (after mask applied)
        mask_np: Numpy mask array (H, W), values 0-255
        margin_ratio: Padding around object (default 10%)
        background_color: RGB tuple for background (default gray)

    Returns:
        RGB PIL Image - square, with specified background color
    """
    # 1. Find object bounding box from mask
    alpha_threshold = 0.8 * 255
    bbox_coords = np.argwhere(mask_np > alpha_threshold)

    if len(bbox_coords) == 0:
        # No object found, return as-is (fallback)
        print("[TRELLIS2] Warning: No object found in mask, returning original image")
        w, h = pil_image.size
        size = max(w, h)
        canvas = Image.new('RGB', (size, size), background_color)
        canvas.paste(pil_image.convert('RGB'), ((size - w) // 2, (size - h) // 2))
        return canvas

    y_min, x_min = bbox_coords.min(axis=0)
    y_max, x_max = bbox_coords.max(axis=0)

    # 2. Calculate object size and add margin
    obj_w = x_max - x_min
    obj_h = y_max - y_min
    obj_size = max(obj_w, obj_h)
    margin = int(obj_size * margin_ratio)

    # 3. Expand to square with margin (centered on object)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_size = (obj_size / 2) + margin

    crop_x1 = int(center_x - half_size)
    crop_y1 = int(center_y - half_size)
    crop_x2 = int(center_x + half_size)
    crop_y2 = int(center_y + half_size)
    crop_size = crop_x2 - crop_x1

    # Ensure crop_size is at least 1
    if crop_size < 1:
        crop_size = 1
        crop_x2 = crop_x1 + 1
        crop_y2 = crop_y1 + 1

    # 4. Create canvas with background color and paste cropped region
    img_w, img_h = pil_image.size
    canvas = Image.new('RGB', (crop_size, crop_size), background_color)

    # Calculate source region (clamp to image bounds)
    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(img_w, crop_x2)
    src_y2 = min(img_h, crop_y2)

    # Calculate destination offset (if crop extends beyond image)
    dst_x = src_x1 - crop_x1
    dst_y = src_y1 - crop_y1

    # Crop source region
    cropped = pil_image.crop((src_x1, src_y1, src_x2, src_y2))

    # 5. Alpha blend with background color
    cropped_np = np.array(cropped.convert('RGBA')).astype(np.float32) / 255
    alpha = cropped_np[:, :, 3:4]
    bg = np.array(background_color, dtype=np.float32) / 255
    # Blend: foreground * alpha + background * (1 - alpha)
    blended = cropped_np[:, :, :3] * alpha + bg * (1 - alpha)
    cropped_rgb = Image.fromarray((blended * 255).astype(np.uint8))

    canvas.paste(cropped_rgb, (dst_x, dst_y))

    return canvas

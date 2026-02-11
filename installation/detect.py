"""
Version detection utilities for ComfyUI-TRELLIS2.
Detects Python, PyTorch, and CUDA versions for wheel selection.
"""
import sys

from .config import WHEEL_CUDA_MAP, WHEEL_DIRS


def get_python_version():
    """Get Python version as major.minor string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_torch_info():
    """
    Get PyTorch and CUDA version info.
    Returns (torch_version, cuda_version) or (None, None) if not available.
    """
    try:
        import torch
        torch_ver = torch.__version__.split('+')[0]  # Remove +cu128 suffix if present
        cuda_ver = torch.version.cuda
        return torch_ver, cuda_ver
    except ImportError:
        return None, None


def get_wheel_cuda_suffix():
    """
    Determine which CUDA wheel suffix to use (cu124, cu128, or cu130).
    Returns suffix string or None if no matching wheel available.
    """
    torch_ver, cuda_ver = get_torch_info()
    if not torch_ver or not cuda_ver:
        return None

    # Extract major.minor versions
    cuda_mm = '.'.join(cuda_ver.split('.')[:2])
    torch_mm = '.'.join(torch_ver.split('.')[:2])

    suffix = WHEEL_CUDA_MAP.get((cuda_mm, torch_mm))
    if suffix:
        print(f"[ComfyUI-TRELLIS2] Detected CUDA {cuda_mm} + PyTorch {torch_mm} -> using {suffix} wheels")
        return suffix

    # Try to find closest match by CUDA version alone
    if cuda_mm.startswith("12.4"):
        print(f"[ComfyUI-TRELLIS2] CUDA {cuda_mm} detected, trying cu124 wheels")
        return "cu124"
    elif cuda_mm.startswith("12.") and float(cuda_mm.split('.')[1]) >= 6:
        print(f"[ComfyUI-TRELLIS2] CUDA {cuda_mm} detected, trying cu128 wheels")
        return "cu128"
    elif cuda_mm.startswith("13."):
        print(f"[ComfyUI-TRELLIS2] CUDA {cuda_mm} detected, trying cu130 wheels")
        return "cu130"

    print(f"[ComfyUI-TRELLIS2] No matching wheel for CUDA {cuda_mm} + PyTorch {torch_mm}")
    return None


def get_wheel_dir():
    """
    Get the wheel subdirectory for current CUDA + PyTorch version.
    Uses static mapping so any patch version (2.9.0, 2.9.1) maps to the same wheels.
    Returns directory string (e.g., 'cu128-torch291') or None if no match.
    """
    torch_ver, cuda_ver = get_torch_info()
    if not torch_ver or not cuda_ver:
        return None

    cuda_mm = '.'.join(cuda_ver.split('.')[:2])
    torch_mm = '.'.join(torch_ver.split('.')[:2])

    wheel_dir = WHEEL_DIRS.get((cuda_mm, torch_mm))
    if wheel_dir:
        print(f"[ComfyUI-TRELLIS2] Using wheel directory: {wheel_dir}")
        return wheel_dir

    return None

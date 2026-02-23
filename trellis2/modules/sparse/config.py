"""
Sparse operations configuration with lazy detection.

Attention supports: 'flash_attn', 'xformers', 'sdpa'
Convolution supports: 'none', 'spconv', 'torchsparse', 'flex_gemm'

Configure via:
- Environment variables: SPARSE_ATTN_BACKEND, SPARSE_CONV_BACKEND, ATTN_BACKEND (fallback)
- Runtime: set_attn_backend('sdpa') or set_conv_backend('flex_gemm') before first use
- ComfyUI: Trellis2Settings node
"""
from typing import *
import os

# Lazy initialization - backends detected on first use
_CONV: Optional[str] = None
_ATTN: Optional[str] = None
_DEBUG: bool = False


def _detect_available_conv_backend() -> str:
    """Try to import conv backends in priority order, return first available."""
    # Check env var first
    env_backend = os.environ.get('SPARSE_CONV_BACKEND')
    if env_backend:
        valid_backends = ['none', 'spconv', 'torchsparse', 'flex_gemm']
        if env_backend in valid_backends:
            print(f"[SPARSE] Using conv backend from SPARSE_CONV_BACKEND env var: {env_backend}")
            return env_backend
        else:
            print(f"[SPARSE] Warning: Invalid SPARSE_CONV_BACKEND '{env_backend}', must be one of {valid_backends}")

    # Auto-detect: try backends in priority order
    backends = ['flex_gemm', 'spconv', 'torchsparse']
    for backend in backends:
        try:
            if backend == 'flex_gemm':
                import flex_gemm
                print(f"[SPARSE] Auto-detected conv backend: flex_gemm")
                return backend
            elif backend == 'spconv':
                import spconv
                print(f"[SPARSE] Auto-detected conv backend: spconv")
                return backend
            elif backend == 'torchsparse':
                import torchsparse
                print(f"[SPARSE] Auto-detected conv backend: torchsparse")
                return backend
        except ImportError:
            continue
        except Exception as e:
            print(f"[SPARSE] Warning: {backend} import failed: {e}")
            continue

    print("[SPARSE] No sparse conv backend available, using none")
    return 'none'


def _detect_available_attn_backend() -> str:
    """Try to import attention backends in priority order, return first available."""
    # Check sparse-specific env var first, then fall back to general ATTN_BACKEND
    env_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_backend is None:
        env_backend = os.environ.get('ATTN_BACKEND')

    if env_backend:
        valid_backends = ['flash_attn', 'xformers', 'sdpa']
        if env_backend in valid_backends:
            # Verify the requested backend actually works
            if env_backend == 'flash_attn':
                try:
                    import flash_attn
                    if not callable(getattr(flash_attn, 'flash_attn_varlen_func', None)):
                        print(f"[SPARSE] Warning: flash_attn requested but not functional (common on Windows)")
                        print(f"[SPARSE] Falling back to auto-detection...")
                        env_backend = None
                except ImportError:
                    print(f"[SPARSE] Warning: flash_attn requested but not installed")
                    print(f"[SPARSE] Falling back to auto-detection...")
                    env_backend = None
            if env_backend:
                print(f"[SPARSE] Using attention backend from env var: {env_backend}")
                return env_backend
        elif env_backend == 'naive':
            # naive is valid for dense attention but not for sparse
            print(f"[SPARSE] Warning: 'naive' backend not supported for sparse attention, auto-detecting...")
        else:
            print(f"[SPARSE] Warning: Invalid sparse attention backend '{env_backend}', must be one of {valid_backends}")

    # Auto-detect: try backends in priority order
    backends = ['flash_attn', 'xformers', 'sdpa']
    for backend in backends:
        try:
            if backend == 'flash_attn':
                import flash_attn
                # Verify varlen functions work - on Windows flash_attn may import but functions are None
                if callable(getattr(flash_attn, 'flash_attn_varlen_func', None)):
                    print(f"[SPARSE] Auto-detected attention backend: flash_attn")
                    return backend
                else:
                    print(f"[SPARSE] flash_attn installed but not functional (common on Windows)")
                    continue
            elif backend == 'xformers':
                import xformers.ops as xops
                # Verify BlockDiagonalMask exists (needed for sparse)
                if hasattr(xops.fmha, 'BlockDiagonalMask'):
                    print(f"[SPARSE] Auto-detected attention backend: xformers")
                    return backend
            elif backend == 'sdpa':
                # sdpa is built into PyTorch >= 2.0
                from torch.nn.functional import scaled_dot_product_attention
                print(f"[SPARSE] Auto-detected attention backend: sdpa")
                return backend
        except ImportError:
            continue
        except Exception as e:
            print(f"[SPARSE] Warning: {backend} import failed: {e}")
            continue

    raise RuntimeError(
        "[SPARSE] No attention backend available for sparse operations! "
        "Please install one of: flash_attn, xformers, or use PyTorch >= 2.0 for sdpa"
    )


def get_conv_backend() -> str:
    """Get current conv backend, detecting on first call."""
    global _CONV
    if _CONV is None:
        _CONV = _detect_available_conv_backend()
    return _CONV


def get_attn_backend() -> str:
    """Get current attention backend, detecting on first call."""
    global _ATTN
    if _ATTN is None:
        _ATTN = _detect_available_attn_backend()
    return _ATTN


def set_conv_backend(backend: str) -> None:
    """
    Set conv backend explicitly. Can be called before or after first use.

    Args:
        backend: One of 'none', 'spconv', 'torchsparse', 'flex_gemm'
    """
    global _CONV
    valid_backends = ['none', 'spconv', 'torchsparse', 'flex_gemm']
    if backend not in valid_backends:
        raise ValueError(f"Invalid conv backend '{backend}', must be one of {valid_backends}")

    if _CONV is not None and _CONV != backend:
        print(f"[SPARSE] Changing conv backend from {_CONV} to {backend}")
    _CONV = backend
    print(f"[SPARSE] Conv backend set to: {backend}")


def set_attn_backend(backend: str) -> None:
    """
    Set attention backend explicitly. Can be called before or after first use.

    Args:
        backend: One of 'flash_attn', 'xformers', 'sdpa'
    """
    global _ATTN
    valid_backends = ['flash_attn', 'xformers', 'sdpa']
    if backend not in valid_backends:
        raise ValueError(f"Invalid attention backend '{backend}', must be one of {valid_backends}")

    if _ATTN is not None and _ATTN != backend:
        print(f"[SPARSE] Changing attention backend from {_ATTN} to {backend}")
    _ATTN = backend
    print(f"[SPARSE] Attention backend set to: {backend}")


def get_debug() -> bool:
    """Get debug mode status."""
    return _DEBUG


def set_debug(debug: bool) -> None:
    """Enable or disable debug mode."""
    global _DEBUG
    _DEBUG = debug



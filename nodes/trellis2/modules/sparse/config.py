"""
Sparse operations configuration.

Attention: delegates to comfy-attn for backend detection and dispatch.
Convolution: kept as-is (spconv, torchsparse, flex_gemm).

Configure via:
- Environment variables: SPARSE_CONV_BACKEND
- Runtime: set_conv_backend('flex_gemm') before first use
- ComfyUI: Trellis2Settings node
"""
from typing import *
import logging
import os
import comfy_attn

log = logging.getLogger("trellis2")

# Lazy initialization - conv backend detected on first use
_CONV: Optional[str] = None
_DEBUG: bool = False


def _detect_available_conv_backend() -> str:
    """Try to import conv backends in priority order, return first available."""
    # Check env var first
    env_backend = os.environ.get('SPARSE_CONV_BACKEND')
    if env_backend:
        valid_backends = ['none', 'spconv', 'torchsparse', 'flex_gemm']
        if env_backend in valid_backends:
            log.info(f"Using conv backend from SPARSE_CONV_BACKEND env var: {env_backend}")
            return env_backend
        else:
            log.warning(f"Invalid SPARSE_CONV_BACKEND '{env_backend}', must be one of {valid_backends}")

    # Auto-detect: try backends in priority order
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


def get_attn_backend() -> str:
    """Get current attention backend (delegates to comfy-attn)."""
    return comfy_attn.get_varlen_backend()


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
        log.info(f"Changing conv backend from {_CONV} to {backend}")
    _CONV = backend
    log.info(f"Conv backend set to: {backend}")


def set_attn_backend(backend: str) -> None:
    """
    Set attention backend (delegates to comfy-attn).

    Args:
        backend: One of 'flash_attn', 'xformers', 'sdpa'
    """
    # comfy-attn handles varlen backend detection automatically;
    # this is kept for backward compat but is a no-op since varlen
    # backend is auto-detected on first dispatch call.
    log.info(f"Sparse attn backend request: {backend} (comfy-attn handles automatically)")


def get_debug() -> bool:
    """Get debug mode status."""
    return _DEBUG


def set_debug(debug: bool) -> None:
    """Enable or disable debug mode."""
    global _DEBUG
    _DEBUG = debug

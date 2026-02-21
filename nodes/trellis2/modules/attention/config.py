"""
Attention backend configuration with lazy detection.

Supports: 'sageattn', 'flash_attn', 'xformers', 'sdpa', 'naive'
Priority: sageattn -> flash_attn -> xformers -> sdpa -> naive

sageattn is 2-5x faster than flash_attn with quantized kernels.

Configure via:
- Environment variable: ATTN_BACKEND
- Runtime: set_backend('sageattn') before first use
- ComfyUI: Trellis2Settings node
"""
from typing import *
import logging
import os

log = logging.getLogger("trellis2")

# Lazy initialization - backend detected on first use
_BACKEND: Optional[str] = None
_DEBUG: bool = False


def _detect_available_backend() -> str:
    """Try to import backends in priority order, return first available."""
    # Check env var first
    env_backend = os.environ.get('ATTN_BACKEND')
    if env_backend:
        valid_backends = ['sageattn', 'flash_attn', 'xformers', 'sdpa', 'naive']
        if env_backend in valid_backends:
            # Verify the requested backend actually works
            if env_backend == 'sageattn':
                try:
                    from sageattention import sageattn
                    log.info("Using backend from ATTN_BACKEND env var: sageattn")
                    return env_backend
                except ImportError:
                    log.warning("ATTN_BACKEND=sageattn but sageattention not installed")
                    log.info("Falling back to auto-detection...")
                    env_backend = None
            elif env_backend == 'flash_attn':
                try:
                    import flash_attn
                    if not callable(getattr(flash_attn, 'flash_attn_func', None)):
                        log.warning("ATTN_BACKEND=flash_attn but flash_attn not functional (common on Windows)")
                        log.info("Falling back to auto-detection...")
                        env_backend = None
                except ImportError:
                    log.warning("ATTN_BACKEND=flash_attn but flash_attn not installed")
                    log.info("Falling back to auto-detection...")
                    env_backend = None
            if env_backend:
                log.info(f"Using backend from ATTN_BACKEND env var: {env_backend}")
                return env_backend
        else:
            log.warning(f"Invalid ATTN_BACKEND '{env_backend}', must be one of {valid_backends}")

    # Auto-detect: try backends in priority order
    # sageattn is fastest (2-5x faster than flash_attn), then flash_attn, xformers, sdpa
    backends = ['sageattn', 'flash_attn', 'xformers', 'sdpa']
    for backend in backends:
        try:
            if backend == 'sageattn':
                from sageattention import sageattn
                log.info("Auto-detected backend: sageattn")
                return backend
            elif backend == 'flash_attn':
                import flash_attn
                # Verify it actually works - on Windows flash_attn may import but functions are None
                if callable(getattr(flash_attn, 'flash_attn_func', None)):
                    log.info("Auto-detected backend: flash_attn")
                    return backend
                else:
                    log.info("flash_attn installed but not functional (common on Windows)")
                    continue
            elif backend == 'xformers':
                import xformers.ops as xops
                # Verify memory_efficient_attention exists
                if hasattr(xops, 'memory_efficient_attention'):
                    log.info("Auto-detected backend: xformers")
                    return backend
            elif backend == 'sdpa':
                # sdpa is built into PyTorch >= 2.0
                from torch.nn.functional import scaled_dot_product_attention
                log.info("Auto-detected backend: sdpa")
                return backend
        except ImportError:
            continue
        except Exception as e:
            log.warning(f"{backend} import failed: {e}")
            continue

    log.info("No optimized backend available, using naive implementation")
    return 'naive'


def get_backend() -> str:
    """Get current backend, detecting on first call."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _detect_available_backend()
    return _BACKEND


def set_backend(backend: str) -> None:
    """
    Set backend explicitly. Can be called before or after first use.

    Args:
        backend: One of 'sageattn', 'flash_attn', 'xformers', 'sdpa', 'naive'
    """
    global _BACKEND
    valid_backends = ['sageattn', 'flash_attn', 'xformers', 'sdpa', 'naive']
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend '{backend}', must be one of {valid_backends}")

    if _BACKEND is not None and _BACKEND != backend:
        log.info(f"Changing backend from {_BACKEND} to {backend}")
    _BACKEND = backend
    log.info(f"Backend set to: {backend}")


def get_debug() -> bool:
    """Get debug mode status."""
    return _DEBUG


def set_debug(debug: bool) -> None:
    """Enable or disable debug mode."""
    global _DEBUG
    _DEBUG = debug


# Legacy compatibility - these are now functions, not module-level constants
# Code should use get_backend() instead of config.BACKEND
@property
def BACKEND():
    """Deprecated: Use get_backend() instead."""
    return get_backend()


@property
def DEBUG():
    """Deprecated: Use get_debug() instead."""
    return get_debug()

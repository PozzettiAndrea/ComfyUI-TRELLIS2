"""
Attention backend configuration.

Delegates to comfy-attn for backend detection and dispatch.
Kept for backward compatibility with TRELLIS2 code that calls config.set_backend().

Configure via:
- Runtime: set_backend('auto') before first use
- ComfyUI: Trellis2Settings node
"""
from typing import *
import logging
import comfy_attn

log = logging.getLogger("trellis2")

_DEBUG: bool = False

# Map TRELLIS2 backend names to comfy-attn backend names
_BACKEND_MAP = {
    'sageattn': 'sage',
    'flash_attn': 'flash_attn',
    'xformers': 'sdpa',   # comfy-attn handles xformers via sdpa path
    'sdpa': 'sdpa',
    'naive': 'sdpa',
    'auto': 'auto',
}


def get_backend() -> str:
    """Get current backend name."""
    return comfy_attn.get_backend()


def set_backend(backend: str) -> None:
    """
    Set backend explicitly via comfy-attn.

    Args:
        backend: One of 'sageattn', 'flash_attn', 'xformers', 'sdpa', 'naive', 'auto'
    """
    comfy_name = _BACKEND_MAP.get(backend, 'auto')
    label = comfy_attn.set_backend(comfy_name)
    log.info(f"Dense attention backend set to: {label} (requested: {backend})")


def get_debug() -> bool:
    """Get debug mode status."""
    return _DEBUG


def set_debug(debug: bool) -> None:
    """Enable or disable debug mode."""
    global _DEBUG
    _DEBUG = debug

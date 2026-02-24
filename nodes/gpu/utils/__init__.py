"""
TRELLIS2 utility modules.

Contains both:
- Host-process utilities (logger, model directories)
- Subprocess utilities imported inside @isolated methods
"""

import logging
import os

# Setup logger for host process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[TRELLIS2]")


def get_trellis_models_dir():
    """Get the directory for TRELLIS.2 models."""
    import folder_paths
    models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def get_dinov3_models_dir():
    """Get the directory for DINOv3 models."""
    import folder_paths
    models_dir = os.path.join(folder_paths.models_dir, "dinov3")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def get_birefnet_models_dir():
    """Get the directory for BiRefNet models."""
    import folder_paths
    models_dir = os.path.join(folder_paths.models_dir, "birefnet")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


# Subprocess utilities - only import when needed
# (These import trellis2 modules which require CUDA extensions)
def _import_subprocess_utils():
    """Lazy import of subprocess utilities."""
    from .lazy_manager import get_model_manager, LazyModelManager
    from .stages import (
        run_conditioning,
        run_shape_generation,
        run_texture_generation,
    )
    from .helpers import (
        tensor_to_pil,
        pil_to_tensor,
        smart_crop_square,
    )
    return {
        'get_model_manager': get_model_manager,
        'LazyModelManager': LazyModelManager,
        'run_conditioning': run_conditioning,
        'run_shape_generation': run_shape_generation,
        'run_texture_generation': run_texture_generation,
        'tensor_to_pil': tensor_to_pil,
        'pil_to_tensor': pil_to_tensor,
        'smart_crop_square': smart_crop_square,
    }


__all__ = [
    "logger",
    "get_trellis_models_dir",
    "get_dinov3_models_dir",
    "get_birefnet_models_dir",
]

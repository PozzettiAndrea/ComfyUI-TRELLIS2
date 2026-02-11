"""
ComfyUI-TRELLIS2: TRELLIS.2 Image-to-3D nodes for ComfyUI
"""

import os

# Only do imports when NOT running under pytest
# Use PYTEST_CURRENT_TEST env var which is only set when pytest is actually running tests
if 'PYTEST_CURRENT_TEST' not in os.environ:
    from .nodes_loader import (
        NODE_CLASS_MAPPINGS as LOADER_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as LOADER_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .nodes_inference import (
        NODE_CLASS_MAPPINGS as INFERENCE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as INFERENCE_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .nodes_export import (
        NODE_CLASS_MAPPINGS as EXPORT_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as EXPORT_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .nodes_unwrap import (
        NODE_CLASS_MAPPINGS as UNWRAP_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as UNWRAP_NODE_DISPLAY_NAME_MAPPINGS,
    )

    # Merge all node mappings
    NODE_CLASS_MAPPINGS = {
        **LOADER_NODE_CLASS_MAPPINGS,
        **INFERENCE_NODE_CLASS_MAPPINGS,
        **EXPORT_NODE_CLASS_MAPPINGS,
        **UNWRAP_NODE_CLASS_MAPPINGS,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        **LOADER_NODE_DISPLAY_NAME_MAPPINGS,
        **INFERENCE_NODE_DISPLAY_NAME_MAPPINGS,
        **EXPORT_NODE_DISPLAY_NAME_MAPPINGS,
        **UNWRAP_NODE_DISPLAY_NAME_MAPPINGS,
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
]

"""
ComfyUI-TRELLIS2: TRELLIS.2 Image-to-3D nodes for ComfyUI

Main nodes (run in main ComfyUI environment):
- nodes_loader: Model config loading
- nodes_export: GLB/mesh export
- nodes_unwrap: UV unwrapping

GPU nodes (run in isolated environment):
- gpu/nodes_inference: DinoV3 conditioning, shape/texture generation
- gpu/nodes_video: Video generation
"""

import os

# Only do imports when NOT running under pytest
if 'PYTEST_CURRENT_TEST' not in os.environ:
    from .nodes_loader import (
        NODE_CLASS_MAPPINGS as LOADER_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as LOADER_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .nodes_export import (
        NODE_CLASS_MAPPINGS as EXPORT_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as EXPORT_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .nodes_unwrap import (
        NODE_CLASS_MAPPINGS as UNWRAP_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as UNWRAP_NODE_DISPLAY_NAME_MAPPINGS,
    )

    # Main nodes only (GPU nodes are wrapped via comfy_env.wrap_isolated_nodes)
    NODE_CLASS_MAPPINGS = {
        **LOADER_NODE_CLASS_MAPPINGS,
        **EXPORT_NODE_CLASS_MAPPINGS,
        **UNWRAP_NODE_CLASS_MAPPINGS,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        **LOADER_NODE_DISPLAY_NAME_MAPPINGS,
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

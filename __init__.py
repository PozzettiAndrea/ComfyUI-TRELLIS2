"""ComfyUI-TRELLIS2: Microsoft TRELLIS.2 Image-to-3D nodes for ComfyUI."""

from comfy_env import wrap_nodes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

wrap_nodes()

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

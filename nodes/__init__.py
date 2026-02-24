"""ComfyUI-TRELLIS2 Nodes."""

from .nodes_loader import NODE_CLASS_MAPPINGS as loader_mappings
from .nodes_loader import NODE_DISPLAY_NAME_MAPPINGS as loader_display
from .nodes_export import NODE_CLASS_MAPPINGS as export_mappings
from .nodes_export import NODE_DISPLAY_NAME_MAPPINGS as export_display
from .nodes_unwrap import NODE_CLASS_MAPPINGS as unwrap_mappings
from .nodes_unwrap import NODE_DISPLAY_NAME_MAPPINGS as unwrap_display
from .nodes_inference import NODE_CLASS_MAPPINGS as inference_mappings
from .nodes_inference import NODE_DISPLAY_NAME_MAPPINGS as inference_display
from .nodes_native_sampling import NODE_CLASS_MAPPINGS as native_mappings
from .nodes_native_sampling import NODE_DISPLAY_NAME_MAPPINGS as native_display

NODE_CLASS_MAPPINGS = {
    **loader_mappings,
    **export_mappings,
    **unwrap_mappings,
    **inference_mappings,
    **native_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **loader_display,
    **export_display,
    **unwrap_display,
    **inference_display,
    **native_display,
}

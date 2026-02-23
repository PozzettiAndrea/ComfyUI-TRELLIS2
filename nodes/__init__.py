"""ComfyUI-TRELLIS2 Nodes."""

# Stage sparse primitives under the comfy namespace so that model code can
# import from comfy.sparse / comfy.ops_sparse / comfy.attention_sparse as if
# the upstream PR were already merged.  setup_link is a no-op when ComfyUI
# already ships those files as real (non-symlink) files.
import pathlib
import comfy_sparse_attn
from comfy_sparse_attn import setup_link
_PKG = pathlib.Path(comfy_sparse_attn.__file__).parent
setup_link(_PKG / "sparse.py",           "sparse.py")
setup_link(_PKG / "ops_sparse.py",       "ops_sparse.py")
setup_link(_PKG / "attention_sparse.py", "attention_sparse.py")
del pathlib, comfy_sparse_attn, setup_link, _PKG

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

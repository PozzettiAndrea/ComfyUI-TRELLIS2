import logging
log = logging.getLogger("trellis2")
log.info("loading...")
from comfy_env import register_nodes
log.info("calling register_nodes")

# Register TRELLIS2 model configs with ComfyUI's detection system
# so checkpoints can be auto-detected from state dict keys.
try:
    import comfy.supported_models
    from .nodes.trellis2.supported_models import TRELLIS2SparseStructure, TRELLIS2SLat
    comfy.supported_models.models.insert(0, TRELLIS2SparseStructure)
    comfy.supported_models.models.insert(1, TRELLIS2SLat)
    log.info("registered TRELLIS2 model configs with ComfyUI")
except Exception as e:
    log.warning(f"failed to register TRELLIS2 model configs: {e}")

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()


WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

import logging
log = logging.getLogger("trellis2")
log.info("loading...")
from comfy_env import register_nodes
log.info("calling register_nodes")

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()


WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

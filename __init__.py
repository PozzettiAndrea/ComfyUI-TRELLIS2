import sys
print("[trellis2] loading...", file=sys.stderr, flush=True)
from comfy_env import register_nodes
print("[trellis2] calling register_nodes", file=sys.stderr, flush=True)

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()


WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

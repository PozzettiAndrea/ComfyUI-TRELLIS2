from pathlib import Path
from comfy_env import setup_env, copy_files

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy assets
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")
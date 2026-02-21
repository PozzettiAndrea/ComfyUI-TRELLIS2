"""
Fallback for ComfyUI's folder_paths when running in an isolated environment
(e.g. a subprocess that does not have the ComfyUI root on sys.path).

Derives paths from this file's location, assuming the standard layout:

    ComfyUI/
      custom_nodes/
        ComfyUI-TRELLIS2/
          nodes/
            folder_paths_fallback.py   ← this file
      models/
      output/
"""
import os
import pathlib

# Walk up: nodes/ → ComfyUI-TRELLIS2/ → custom_nodes/ → ComfyUI/
_comfyui_root = pathlib.Path(__file__).resolve().parents[3]

models_dir: str = str(_comfyui_root / "models")
output_directory: str = str(_comfyui_root / "output")


def get_output_directory() -> str:
    return output_directory

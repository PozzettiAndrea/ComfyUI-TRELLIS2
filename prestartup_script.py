"""Pre-startup script for ComfyUI-TRELLIS2.

This script runs before ComfyUI initializes and copies example assets
to ComfyUI's input directory.
"""
import os
import shutil


def copy_assets_to_input():
    """Copy example assets to ComfyUI input directory."""
    script_dir = os.path.dirname(__file__)
    comfyui_root = os.path.dirname(os.path.dirname(script_dir))

    assets_src = os.path.join(script_dir, "assets")
    input_dst = os.path.join(comfyui_root, "input")

    if os.path.exists(assets_src):
        os.makedirs(input_dst, exist_ok=True)
        for asset in os.listdir(assets_src):
            src = os.path.join(assets_src, asset)
            dst = os.path.join(input_dst, asset)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"[TRELLIS2] Copied asset: {asset}")


copy_assets_to_input()

"""Pre-startup script for ComfyUI-TRELLIS2.

This script runs before ComfyUI initializes and copies workflows
to the appropriate ComfyUI directories.
"""
import os
import shutil


def copy_workflows_to_user():
    """Copy workflow JSON files to ComfyUI user workflows directory."""
    script_dir = os.path.dirname(__file__)
    comfyui_root = os.path.dirname(os.path.dirname(script_dir))

    workflows_src = os.path.join(script_dir, "workflows")
    workflows_dst = os.path.join(comfyui_root, "user", "default", "workflows")

    if os.path.exists(workflows_src):
        os.makedirs(workflows_dst, exist_ok=True)
        for workflow in os.listdir(workflows_src):
            if workflow.endswith('.json'):
                src = os.path.join(workflows_src, workflow)
                dst = os.path.join(workflows_dst, f"trellis2_{workflow}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"[TRELLIS2] Copied workflow: {workflow}")


copy_workflows_to_user()

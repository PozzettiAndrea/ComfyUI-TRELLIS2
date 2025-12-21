"""
ComfyUI-TRELLIS2: Microsoft TRELLIS.2 Image-to-3D nodes for ComfyUI
"""

import sys
import os
import traceback

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

# Web directory for JavaScript extensions (if needed in future)
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# Only run initialization and imports when loaded by ComfyUI, not during pytest
# Use PYTEST_CURRENT_TEST env var which is only set when pytest is actually running tests
if 'PYTEST_CURRENT_TEST' not in os.environ:
    print("[ComfyUI-TRELLIS2] Initializing custom node...")

    try:
        from .nodes import (
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS,
        )
        print("[ComfyUI-TRELLIS2] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[ComfyUI-TRELLIS2] [WARNING] {error_msg}")
        print(f"[ComfyUI-TRELLIS2] Traceback:\n{traceback.format_exc()}")

        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    if INIT_SUCCESS:
        print("[ComfyUI-TRELLIS2] [OK] Loaded successfully!")
    else:
        print(f"[ComfyUI-TRELLIS2] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[ComfyUI-TRELLIS2] Please check the errors above and your installation.")

else:
    print("[ComfyUI-TRELLIS2] Running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

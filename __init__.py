"""
ComfyUI-TRELLIS2: Microsoft TRELLIS.2 Image-to-3D nodes for ComfyUI
"""

import sys
import os
import traceback

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []


def _check_xatlas_conflict():
    """Check for potential xatlas pybind11 conflicts before loading cumesh."""
    conflicts = []

    # Check for standalone xatlas package
    try:
        import importlib.util
        if importlib.util.find_spec("xatlas") is not None:
            # Check if it's the cumesh-bundled version or standalone
            import xatlas
            xatlas_path = getattr(xatlas, '__file__', '')
            if 'cumesh' not in xatlas_path.lower():
                conflicts.append(f"Standalone 'xatlas' package detected at: {xatlas_path}")
    except ImportError:
        pass
    except Exception:
        pass

    # Check for already-loaded xatlas-related modules
    xatlas_modules = [m for m in sys.modules if 'xatlas' in m.lower() and 'cumesh' not in m.lower()]
    if xatlas_modules:
        conflicts.append(f"xatlas modules already loaded by another node: {xatlas_modules}")

    return conflicts

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

        # Check for xatlas/pybind11 conflicts
        if "already registered" in str(e).lower():
            print("[ComfyUI-TRELLIS2] [HINT] This error is caused by a pybind11 type conflict.")
            print("[ComfyUI-TRELLIS2] [HINT] Another custom node has loaded xatlas before this one.")
            print("[ComfyUI-TRELLIS2] [HINT] Common fixes:")
            print("[ComfyUI-TRELLIS2]   1. Remove ComfyUI_TRELLIS (v1) if installed")
            print("[ComfyUI-TRELLIS2]   2. Run: pip uninstall xatlas")
            print("[ComfyUI-TRELLIS2]   3. Disable other 3D mesh processing nodes")
            conflicts = _check_xatlas_conflict()
            if conflicts:
                print("[ComfyUI-TRELLIS2] [DETECTED CONFLICTS]:")
                for c in conflicts:
                    print(f"[ComfyUI-TRELLIS2]   - {c}")

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

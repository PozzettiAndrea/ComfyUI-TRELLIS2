"""
ComfyUI-TRELLIS2: Microsoft TRELLIS.2 Image-to-3D nodes for ComfyUI

Architecture:
- Main nodes (loader, export, unwrap) run in main ComfyUI environment
- GPU nodes (inference, video) run in isolated pixi environment
"""

import sys
import os
import traceback
from pathlib import Path
from unittest.mock import MagicMock

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []


def _inject_cuda_mocks():
    """Inject mocks for CUDA packages when running in test mode."""
    mock_packages_str = os.environ.get("COMFY_TEST_MOCK_PACKAGES", "")
    if not mock_packages_str:
        return []

    mock_packages = [p.strip() for p in mock_packages_str.split(",") if p.strip()]
    mocked = []

    for pkg in mock_packages:
        if pkg not in sys.modules:
            mock = MagicMock()
            mock.__name__ = pkg
            mock.__package__ = pkg
            sys.modules[pkg] = mock
            mocked.append(pkg)

            for submodule in ["torch", "cuda", "ops", "core"]:
                full_name = f"{pkg}.{submodule}"
                if full_name not in sys.modules:
                    sys.modules[full_name] = MagicMock()

    return mocked


def _check_xatlas_conflict():
    """Check for potential xatlas pybind11 conflicts."""
    conflicts = []
    try:
        import importlib.util
        if importlib.util.find_spec("xatlas") is not None:
            import xatlas
            xatlas_path = getattr(xatlas, '__file__', '')
            if 'cumesh' not in xatlas_path.lower():
                conflicts.append(f"Standalone 'xatlas' package detected at: {xatlas_path}")
    except ImportError:
        pass
    except Exception:
        pass

    xatlas_modules = [m for m in sys.modules if 'xatlas' in m.lower() and 'cumesh' not in m.lower()]
    if xatlas_modules:
        conflicts.append(f"xatlas modules already loaded by another node: {xatlas_modules}")

    return conflicts


# Web directory for JavaScript extensions
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# Only run initialization when loaded by ComfyUI, not during pytest
if 'PYTEST_CURRENT_TEST' not in os.environ:
    print("[ComfyUI-TRELLIS2] Initializing custom node...")

    # Inject CUDA mocks if running in test mode (CPU-only CI)
    mocked_packages = _inject_cuda_mocks()
    if mocked_packages:
        print(f"[ComfyUI-TRELLIS2] [TEST MODE] Mocked CUDA packages: {', '.join(mocked_packages)}")

    try:
        # Import main nodes (run in main ComfyUI environment)
        from .nodes import (
            NODE_CLASS_MAPPINGS as MAIN_NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS as MAIN_NODE_DISPLAY_NAME_MAPPINGS,
        )
        print(f"[ComfyUI-TRELLIS2] Main nodes loaded ({len(MAIN_NODE_CLASS_MAPPINGS)} nodes)")

        # Import GPU nodes (will be wrapped for isolation)
        from .nodes.gpu import (
            NODE_CLASS_MAPPINGS as GPU_NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS as GPU_NODE_DISPLAY_NAME_MAPPINGS,
        )
        print(f"[ComfyUI-TRELLIS2] GPU nodes loaded ({len(GPU_NODE_CLASS_MAPPINGS)} nodes)")

        # Wrap GPU nodes for process isolation
        try:
            from comfy_env import wrap_isolated_nodes
            gpu_dir = Path(__file__).parent / "nodes" / "gpu"
            GPU_NODE_CLASS_MAPPINGS = wrap_isolated_nodes(GPU_NODE_CLASS_MAPPINGS, gpu_dir)
            print(f"[ComfyUI-TRELLIS2] [OK] GPU nodes wrapped for isolation")
        except ImportError:
            print("[ComfyUI-TRELLIS2] comfy-env not installed, GPU isolation disabled")
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] Failed to wrap GPU nodes: {e}")

        # Merge all node mappings
        NODE_CLASS_MAPPINGS = {
            **MAIN_NODE_CLASS_MAPPINGS,
            **GPU_NODE_CLASS_MAPPINGS,
        }
        NODE_DISPLAY_NAME_MAPPINGS = {
            **MAIN_NODE_DISPLAY_NAME_MAPPINGS,
            **GPU_NODE_DISPLAY_NAME_MAPPINGS,
        }

        print(f"[ComfyUI-TRELLIS2] Total nodes: {len(NODE_CLASS_MAPPINGS)}")
        INIT_SUCCESS = True

    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[ComfyUI-TRELLIS2] [WARNING] {error_msg}")
        print(f"[ComfyUI-TRELLIS2] Traceback:\n{traceback.format_exc()}")

        if "already registered" in str(e).lower():
            print("[ComfyUI-TRELLIS2] [HINT] This error is caused by a pybind11 type conflict.")
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
        print(f"[ComfyUI-TRELLIS2] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s))")

else:
    print("[ComfyUI-TRELLIS2] Running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

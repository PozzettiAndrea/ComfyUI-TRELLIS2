#!/usr/bin/env python3
"""
Installation script for ComfyUI-TRELLIS2.
Called by ComfyUI Manager during installation/update.

Automatically detects PyTorch/CUDA versions and installs pre-built wheels
for CUDA extensions (nvdiffrast, flex_gemm, cumesh, o_voxel, nvdiffrec_render).
Falls back to compilation from source if no wheel is available.
"""
import subprocess
import sys
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Ensure script directory is in path (for users with outdated installations)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from modular installation package
from installation import (
    PACKAGES,
    get_python_version,
    get_torch_info,
    get_wheel_cuda_suffix,
    is_package_installed,
    try_install_from_direct_url,
    try_install_from_wheel,
    try_compile_from_source,
    try_install_flash_attn,
    try_install_vcredist,
)


def install_cuda_package(package_config):
    """
    Install a CUDA package - tries wheel first, falls back to compilation.
    Returns True if successful, False otherwise.
    """
    name = package_config["name"]
    import_name = package_config["import_name"]
    wheel_index = package_config.get("wheel_index")
    git_url = package_config.get("git_url")
    wheel_type = package_config.get("wheel_type")

    print(f"\n[ComfyUI-TRELLIS2] Installing {name}...")

    # Always reinstall to ensure users get the latest fixed wheels
    # (Previous wheel builds had naming bugs that caused installation failures)

    # Check if we have CUDA
    torch_ver, cuda_ver = get_torch_info()
    if not cuda_ver:
        print(f"[ComfyUI-TRELLIS2] [SKIP] PyTorch CUDA not available, skipping {name}")
        return False

    # Special handling for flash_attn
    if wheel_type == "flash_attn":
        if try_install_flash_attn():
            return True
        # Fall through to compilation
    else:
        # Try 1: direct GitHub release URL (most reliable - bypasses pip index parsing)
        if try_install_from_direct_url(package_config):
            return True

        # Try 2: wheel index as fallback (pip --find-links)
        if wheel_index and try_install_from_wheel(name, wheel_index, import_name):
            return True

    # Try 3: compile from source
    print(f"[ComfyUI-TRELLIS2] No pre-built wheel found, attempting compilation...")
    if try_compile_from_source(name, git_url):
        return True

    print(f"[ComfyUI-TRELLIS2] [FAILED] Could not install {name}")
    return False


def install_requirements():
    """Install dependencies from requirements.txt."""
    print("[ComfyUI-TRELLIS2] Installing requirements.txt dependencies...")

    script_dir = Path(__file__).parent.absolute()
    requirements_path = script_dir / "requirements.txt"

    if not requirements_path.exists():
        print("[ComfyUI-TRELLIS2] [WARNING] requirements.txt not found")
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode == 0:
            print("[ComfyUI-TRELLIS2] [OK] Requirements installed successfully")
            return True
        else:
            print("[ComfyUI-TRELLIS2] [WARNING] Some requirements failed to install")
            if result.stderr:
                print(f"[ComfyUI-TRELLIS2] Error: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print("[ComfyUI-TRELLIS2] [WARNING] Requirements installation timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] [WARNING] Requirements error: {e}")
        return False


def try_install_python_headers():
    """
    Install Python development headers for embedded Python on Windows.
    Required for Triton to compile CUDA utilities at runtime.

    Downloads include/ and libs/ folders from triton-windows releases.
    """
    # Only needed on Windows
    if sys.platform != "win32":
        return True

    # Check if we're using embedded Python (ComfyUI portable)
    python_path = Path(sys.executable)
    python_dir = python_path.parent

    # Look for python_embeded folder pattern
    if "python_embeded" not in str(python_dir).lower() and "python_embedded" not in str(python_dir).lower():
        # Not embedded Python, probably has headers already
        print("[ComfyUI-TRELLIS2] [SKIP] Not using embedded Python, headers should exist")
        return True

    include_dir = python_dir / "include"
    libs_dir = python_dir / "libs"

    # Check if Python.h already exists
    python_h = include_dir / "Python.h"
    if python_h.exists():
        print("[ComfyUI-TRELLIS2] [OK] Python headers already installed")
        return True

    print("[ComfyUI-TRELLIS2] Installing Python headers for Triton compatibility...")
    print(f"[ComfyUI-TRELLIS2] Target directory: {python_dir}")

    # Determine Python version for download URL
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Try version-specific URL first, fall back to 3.12.7 (most common for ComfyUI)
    urls_to_try = [
        f"https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_{py_version}_include_libs.zip",
        "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip",
    ]

    for url in urls_to_try:
        try:
            print(f"[ComfyUI-TRELLIS2] Downloading from {url}...")

            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / "python_headers.zip"

                # Download the zip file
                urllib.request.urlretrieve(url, zip_path)
                print("[ComfyUI-TRELLIS2] Download complete, extracting...")

                # Extract to python_embeded directory
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(python_dir)

                # Verify extraction
                if python_h.exists():
                    print("[ComfyUI-TRELLIS2] [OK] Python headers installed successfully")
                    return True
                else:
                    print("[ComfyUI-TRELLIS2] [WARNING] Extraction completed but Python.h not found")

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"[ComfyUI-TRELLIS2] [INFO] Headers for Python {py_version} not available, trying fallback...")
                continue
            print(f"[ComfyUI-TRELLIS2] [WARNING] Download failed: {e}")
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] [WARNING] Failed to install Python headers: {e}")

    print("[ComfyUI-TRELLIS2] [WARNING] Could not install Python headers")
    print("[ComfyUI-TRELLIS2] [WARNING] Triton may fail to compile CUDA utilities")
    print("[ComfyUI-TRELLIS2] [WARNING] Manual fix: download Python include/libs folders from")
    print("[ComfyUI-TRELLIS2] [WARNING] https://github.com/woct0rdho/triton-windows/releases")
    return False


def main():
    print("\n" + "=" * 60)
    print("ComfyUI-TRELLIS2 Installation")
    print("=" * 60)

    # Show detected environment
    py_ver = get_python_version()
    torch_ver, cuda_ver = get_torch_info()

    print(f"[ComfyUI-TRELLIS2] Python: {py_ver}")
    if torch_ver:
        print(f"[ComfyUI-TRELLIS2] PyTorch: {torch_ver}")
    if cuda_ver:
        print(f"[ComfyUI-TRELLIS2] CUDA: {cuda_ver}")

    # Warn about Python version compatibility
    if sys.version_info >= (3, 13):
        print(f"[ComfyUI-TRELLIS2] [WARNING] Python {py_ver} detected - pre-built wheels may not be available")
        print(f"[ComfyUI-TRELLIS2] [WARNING] Will attempt compilation from source where possible")

    wheel_suffix = get_wheel_cuda_suffix()
    if wheel_suffix:
        print(f"[ComfyUI-TRELLIS2] Wheel suffix: {wheel_suffix}")

    # Install Visual C++ Redistributable on Windows (required for opencv, etc.)
    if sys.platform == "win32":
        try_install_vcredist()

    # Install triton-windows on Windows (required by flex_gemm)
    if sys.platform == "win32":
        print("\n[ComfyUI-TRELLIS2] Installing triton-windows (required for flex_gemm)...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "triton-windows"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print("[ComfyUI-TRELLIS2] [OK] triton-windows installed")
            else:
                print(f"[ComfyUI-TRELLIS2] [WARNING] triton-windows install failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] [WARNING] triton-windows install error: {e}")

        # Install Python headers for embedded Python (required for Triton to compile CUDA utils)
        try_install_python_headers()

    # Install requirements.txt first
    print("\n" + "-" * 60)
    install_requirements()

    # Install CUDA packages
    results = {}
    for pkg in PACKAGES:
        print("-" * 60)
        results[pkg["name"]] = install_cuda_package(pkg)

    # Summary
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  {name}: {status}")
    print("=" * 60)

    # Overall status
    if all(results.values()):
        print("[ComfyUI-TRELLIS2] Installation completed successfully!")
    elif any(results.values()):
        print("[ComfyUI-TRELLIS2] Installation completed with some failures")
        print("[ComfyUI-TRELLIS2] TRELLIS2 may still work with reduced functionality")
    else:
        print("[ComfyUI-TRELLIS2] Installation failed")
        print("[ComfyUI-TRELLIS2] Check that you have PyTorch with CUDA support installed")


if __name__ == "__main__":
    main()

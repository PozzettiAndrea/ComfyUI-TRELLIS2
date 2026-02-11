"""
Source compilation utilities for ComfyUI-TRELLIS2.
Handles building CUDA extensions from source when pre-built wheels are unavailable.
"""
import os
import shutil
import subprocess
import sys

from .cuda_env import find_cuda_home, setup_cuda_environment, get_cuda_arch_list, try_install_cuda_toolkit


def clone_and_init_submodules(git_url, target_dir):
    """
    Clone a git repo and initialize submodules with HTTPS URLs.
    Returns the path to the source (or subdirectory if specified), or None on failure.
    """
    # Extract repo URL (remove git+ prefix and #subdirectory suffix)
    repo_url = git_url
    subdirectory = None
    if repo_url.startswith("git+"):
        repo_url = repo_url[4:]
    if "#subdirectory=" in repo_url:
        repo_url, subdirectory = repo_url.split("#subdirectory=")

    print(f"[ComfyUI-TRELLIS2] Cloning {repo_url}...")
    try:
        # Clone the repo
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"[ComfyUI-TRELLIS2] Git clone failed: {result.stderr[:200]}")
            return None

        # Initialize submodules
        print(f"[ComfyUI-TRELLIS2] Initializing submodules...")
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=target_dir, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            # Try to fix SSH URLs to HTTPS
            print(f"[ComfyUI-TRELLIS2] Submodule init failed, trying to fix SSH URLs...")
            gitmodules_path = os.path.join(target_dir, ".gitmodules")
            if os.path.exists(gitmodules_path):
                with open(gitmodules_path, 'r') as f:
                    content = f.read()
                # Replace git@github.com: with https://github.com/
                content = content.replace("git@github.com:", "https://github.com/")
                with open(gitmodules_path, 'w') as f:
                    f.write(content)
                # Sync and retry
                subprocess.run(["git", "submodule", "sync"], cwd=target_dir, capture_output=True)
                result = subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=target_dir, capture_output=True, text=True, timeout=300
                )

        if subdirectory:
            return os.path.join(target_dir, subdirectory)
        return target_dir

    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Clone error: {e}")
        return None


def try_compile_from_source(package_name, git_url):
    """
    Try compiling a package from source.
    Returns True if successful, False otherwise.
    """
    import tempfile

    if not git_url:
        print(f"[ComfyUI-TRELLIS2] No source URL available for {package_name}")
        return False

    # Check for CUDA compiler
    cuda_env = setup_cuda_environment()
    if not cuda_env:
        # Try to install CUDA toolkit via pip
        if try_install_cuda_toolkit():
            cuda_env = setup_cuda_environment()

    if not cuda_env:
        print(f"[ComfyUI-TRELLIS2] CUDA compiler not found, cannot compile {package_name}")
        return False

    # Set CUDA architecture
    cuda_arch = get_cuda_arch_list()
    if cuda_arch:
        cuda_env["TORCH_CUDA_ARCH_LIST"] = cuda_arch
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch

    # Platform-specific compiler setup
    if sys.platform == "win32":
        if not shutil.which("cl"):
            print(f"[ComfyUI-TRELLIS2] MSVC C++ compiler not found.")
            print(f"[ComfyUI-TRELLIS2] To install build tools, run: python install_compilers.py")
            print(f"[ComfyUI-TRELLIS2] Or run ComfyUI from 'Developer Command Prompt for VS'")
            return False
    else:
        gcc = shutil.which("gcc-11") or shutil.which("gcc")
        gxx = shutil.which("g++-11") or shutil.which("g++")
        if not gxx:
            print(f"[ComfyUI-TRELLIS2] g++ compiler not found.")
            print(f"[ComfyUI-TRELLIS2] To install build tools, run: python install_compilers.py")
            return False
        if gcc:
            cuda_env["CC"] = gcc
        if gxx:
            cuda_env["CXX"] = gxx

    print(f"[ComfyUI-TRELLIS2] Compiling {package_name} from source (this may take several minutes)...")

    # First try direct pip install
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", git_url],
            capture_output=True, text=True, timeout=900, env=cuda_env
        )
        if result.returncode == 0:
            print(f"[ComfyUI-TRELLIS2] [OK] Compiled {package_name} successfully")
            return True
    except:
        pass

    # If direct install fails, try manual clone with submodule fix
    print(f"[ComfyUI-TRELLIS2] Direct install failed, trying manual clone with submodule fix...")
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "repo")
        source_path = clone_and_init_submodules(git_url, clone_dir)

        if source_path and os.path.exists(source_path):
            # Install any missing pure-Python dependencies first
            print(f"[ComfyUI-TRELLIS2] Installing build dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "plyfile", "zstandard", "ninja"],
                capture_output=True, text=True, timeout=120
            )

            # Try to install with --no-deps to skip problematic dependencies
            print(f"[ComfyUI-TRELLIS2] Building from cloned source...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", source_path],
                capture_output=True, text=True, timeout=900, env=cuda_env
            )
            if result.returncode == 0:
                print(f"[ComfyUI-TRELLIS2] [OK] Compiled {package_name} successfully")
                return True
            else:
                print(f"[ComfyUI-TRELLIS2] Compilation failed for {package_name}")
                if result.stderr:
                    print(f"[ComfyUI-TRELLIS2] Error: {result.stderr[:500]}")
                return False

    print(f"[ComfyUI-TRELLIS2] Compilation failed for {package_name}")
    return False

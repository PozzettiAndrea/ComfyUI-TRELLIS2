#!/usr/bin/env python3
"""
Installation script for ComfyUI-TRELLIS2.
Called by ComfyUI Manager during installation/update.

Automatically detects PyTorch/CUDA versions and installs pre-built wheels
for CUDA extensions (nvdiffrast, flex_gemm, cumesh).
Falls back to compilation from source if no wheel is available.
"""
import os
import subprocess
import sys
import site
import shutil
from pathlib import Path


# =============================================================================
# Package Configuration
# =============================================================================

PACKAGES = [
    {
        "name": "nvdiffrast",
        "import_name": "nvdiffrast",
        "wheel_index": "https://pozzettiandrea.github.io/nvdiffrast-full-wheels/",
        "git_url": "git+https://github.com/NVlabs/nvdiffrast",
    },
    {
        "name": "flex_gemm",
        "import_name": "flex_gemm",
        "wheel_index": "https://pozzettiandrea.github.io/flexgemm-wheels/",
        "git_url": "git+https://github.com/JeffreyXiang/FlexGEMM",
    },
    {
        "name": "cumesh",
        "import_name": "cumesh",
        "wheel_index": "https://pozzettiandrea.github.io/cumesh-wheels/",
        "git_url": "git+https://github.com/JeffreyXiang/CuMesh",
    },
]

# Map (CUDA major.minor, PyTorch major.minor) -> wheel CUDA suffix
# Wheels are built for cu124 and cu128
WHEEL_CUDA_MAP = {
    ("12.4", "2.5"): "cu124",
    ("12.6", "2.6"): "cu124",
    ("12.6", "2.8"): "cu128",
    ("12.8", "2.8"): "cu128",
    ("12.8", "2.9"): "cu128",
}


# =============================================================================
# Version Detection
# =============================================================================

def get_python_version():
    """Get Python version as major.minor string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_torch_info():
    """
    Get PyTorch and CUDA version info.
    Returns (torch_version, cuda_version) or (None, None) if not available.
    """
    try:
        import torch
        torch_ver = torch.__version__.split('+')[0]  # Remove +cu128 suffix if present
        cuda_ver = torch.version.cuda
        return torch_ver, cuda_ver
    except ImportError:
        return None, None


def get_wheel_cuda_suffix():
    """
    Determine which CUDA wheel suffix to use (cu124 or cu128).
    Returns suffix string or None if no matching wheel available.
    """
    torch_ver, cuda_ver = get_torch_info()
    if not torch_ver or not cuda_ver:
        return None

    # Extract major.minor versions
    cuda_mm = '.'.join(cuda_ver.split('.')[:2])
    torch_mm = '.'.join(torch_ver.split('.')[:2])

    suffix = WHEEL_CUDA_MAP.get((cuda_mm, torch_mm))
    if suffix:
        print(f"[ComfyUI-TRELLIS2] Detected CUDA {cuda_mm} + PyTorch {torch_mm} -> using {suffix} wheels")
        return suffix

    # Try to find closest match by CUDA version alone
    if cuda_mm.startswith("12.4"):
        print(f"[ComfyUI-TRELLIS2] CUDA {cuda_mm} detected, trying cu124 wheels")
        return "cu124"
    elif cuda_mm.startswith("12.") and float(cuda_mm.split('.')[1]) >= 6:
        print(f"[ComfyUI-TRELLIS2] CUDA {cuda_mm} detected, trying cu128 wheels")
        return "cu128"

    print(f"[ComfyUI-TRELLIS2] No matching wheel for CUDA {cuda_mm} + PyTorch {torch_mm}")
    return None


# =============================================================================
# CUDA Environment (for compilation fallback)
# =============================================================================

def find_cuda_home():
    """Find CUDA installation directory."""
    # Check CUDA_HOME environment variable
    if "CUDA_HOME" in os.environ and os.path.exists(os.environ["CUDA_HOME"]):
        cuda_home = os.environ["CUDA_HOME"]
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc_path) or os.path.exists(nvcc_path + ".exe"):
            return cuda_home

    # Check nvcc in PATH
    nvcc_in_path = shutil.which("nvcc")
    if nvcc_in_path:
        nvcc_dir = os.path.dirname(nvcc_in_path)
        if os.path.basename(nvcc_dir) == "bin":
            cuda_home = os.path.dirname(nvcc_dir)
            if cuda_home != "/usr":
                return cuda_home

    # Check system CUDA installations
    system_paths = [
        "/usr/local/cuda-12.8", "/usr/local/cuda-12.6", "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.2", "/usr/local/cuda-12.0", "/usr/local/cuda",
    ]
    for cuda_path in system_paths:
        if os.path.exists(cuda_path):
            nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
            if os.path.exists(nvcc_path):
                return cuda_path

    # Check Windows locations
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        cuda_base = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        if os.path.exists(cuda_base):
            versions = sorted(os.listdir(cuda_base), reverse=True)
            for v in versions:
                cuda_path = os.path.join(cuda_base, v)
                if os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
                    return cuda_path

    # Check pip-installed CUDA
    for sp in site.getsitepackages():
        for subdir in ["nvidia/cuda_nvcc", "nvidia_cuda_nvcc"]:
            cuda_path = os.path.join(sp, subdir)
            if os.path.exists(cuda_path):
                nvcc = os.path.join(cuda_path, "bin", "nvcc")
                if os.path.exists(nvcc) or os.path.exists(nvcc + ".exe"):
                    return cuda_path

    # Check conda
    if "CONDA_PREFIX" in os.environ:
        nvcc = os.path.join(os.environ["CONDA_PREFIX"], "bin", "nvcc")
        if os.path.exists(nvcc):
            return os.environ["CONDA_PREFIX"]

    return None


def setup_cuda_environment():
    """Setup CUDA environment variables for compilation."""
    cuda_home = find_cuda_home()
    if not cuda_home:
        return None

    env = os.environ.copy()
    env["CUDA_HOME"] = cuda_home
    os.environ["CUDA_HOME"] = cuda_home

    cuda_bin = os.path.join(cuda_home, "bin")
    if cuda_bin not in env.get("PATH", ""):
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
        os.environ["PATH"] = env["PATH"]

    return env


def get_cuda_arch_list():
    """Detect GPU compute capability."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}.{minor}"
    except Exception:
        return None


# =============================================================================
# Installation Functions
# =============================================================================

def is_package_installed(import_name):
    """Check if a package is already installed."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def try_install_from_wheel(package_name, wheel_index_url):
    """
    Try installing a package from pre-built wheels.
    Returns True if successful, False otherwise.
    """
    py_ver = get_python_version()
    wheel_suffix = get_wheel_cuda_suffix()
    print(f"[ComfyUI-TRELLIS2] Looking for {package_name} wheel (Python {py_ver}, {wheel_suffix or 'unknown CUDA'})")
    print(f"[ComfyUI-TRELLIS2] Wheel index: {wheel_index_url}")

    try:
        # Use --no-index to ONLY look at our wheel index, not PyPI
        # This avoids conflicts with similarly-named packages on PyPI
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            package_name, "--no-index", "--find-links", wheel_index_url
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"[ComfyUI-TRELLIS2] [OK] Installed {package_name} from pre-built wheel")
            return True
        else:
            if result.stderr:
                # Check if it's a "no matching distribution" error
                if "No matching distribution" in result.stderr or "no matching distribution" in result.stderr.lower():
                    print(f"[ComfyUI-TRELLIS2] No matching wheel for {package_name} (Python {py_ver}, {wheel_suffix})")
                    print(f"[ComfyUI-TRELLIS2] Available wheels are for Python 3.10-3.12, CUDA 12.4/12.8")
                else:
                    print(f"[ComfyUI-TRELLIS2] Wheel install failed: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[ComfyUI-TRELLIS2] Wheel install timed out for {package_name}")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Wheel install error: {e}")
        return False


def try_compile_from_source(package_name, git_url):
    """
    Try compiling a package from source.
    Returns True if successful, False otherwise.
    """
    if not git_url:
        print(f"[ComfyUI-TRELLIS2] No source URL available for {package_name}")
        return False

    # Check for CUDA compiler
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
            print(f"[ComfyUI-TRELLIS2] MSVC not found, cannot compile {package_name}")
            return False
    else:
        gcc = shutil.which("gcc-11") or shutil.which("gcc")
        gxx = shutil.which("g++-11") or shutil.which("g++")
        if gcc:
            cuda_env["CC"] = gcc
        if gxx:
            cuda_env["CXX"] = gxx

    print(f"[ComfyUI-TRELLIS2] Compiling {package_name} from source (this may take several minutes)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", git_url],
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
    except subprocess.TimeoutExpired:
        print(f"[ComfyUI-TRELLIS2] Compilation timed out for {package_name}")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Compilation error: {e}")
        return False


def install_cuda_package(package_config):
    """
    Install a CUDA package - tries wheel first, falls back to compilation.
    Returns True if successful, False otherwise.
    """
    name = package_config["name"]
    import_name = package_config["import_name"]
    wheel_index = package_config["wheel_index"]
    git_url = package_config.get("git_url")

    print(f"\n[ComfyUI-TRELLIS2] Installing {name}...")

    # Check if already installed
    if is_package_installed(import_name):
        print(f"[ComfyUI-TRELLIS2] [OK] {name} already installed")
        return True

    # Check if we have CUDA
    torch_ver, cuda_ver = get_torch_info()
    if not cuda_ver:
        print(f"[ComfyUI-TRELLIS2] [SKIP] PyTorch CUDA not available, skipping {name}")
        return False

    # Try wheel installation first
    if try_install_from_wheel(name, wheel_index):
        return True

    # Fall back to compilation
    print(f"[ComfyUI-TRELLIS2] No pre-built wheel found, attempting compilation...")
    if try_compile_from_source(name, git_url):
        return True

    print(f"[ComfyUI-TRELLIS2] [FAILED] Could not install {name}")
    return False


# =============================================================================
# Requirements Installation
# =============================================================================

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


# =============================================================================
# Main
# =============================================================================

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

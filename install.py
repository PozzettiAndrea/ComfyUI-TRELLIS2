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

# GitHub release URL template: {repo}/releases/download/{tag}/{wheel_name}
# wheel_name format: {package}-{version}+{cuda_suffix}-cp{py}-cp{py}-{platform}.whl
PACKAGES = [
    {
        "name": "nvdiffrast",
        "import_name": "nvdiffrast",
        "wheel_index": "https://pozzettiandrea.github.io/nvdiffrast-full-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/nvdiffrast-full-wheels/releases/download",
        "wheel_version": "0.4.0",
        "git_url": "git+https://github.com/NVlabs/nvdiffrast",
    },
    {
        "name": "flex_gemm",
        "import_name": "flex_gemm",
        "wheel_index": "https://pozzettiandrea.github.io/flexgemm-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/flexgemm-wheels/releases/download",
        "wheel_version": "0.0.1",
        "git_url": "git+https://github.com/JeffreyXiang/FlexGEMM",
    },
    {
        "name": "cumesh",
        "import_name": "cumesh",
        "wheel_index": "https://pozzettiandrea.github.io/cumesh-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/cumesh-wheels/releases/download",
        "wheel_version": "0.0.1",
        "git_url": "git+https://github.com/JeffreyXiang/CuMesh",
    },
    {
        "name": "o_voxel",
        "import_name": "o_voxel",
        "wheel_index": "https://pozzettiandrea.github.io/ovoxel-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/ovoxel-wheels/releases/download",
        "wheel_version": "0.0.1",
        "git_url": "git+https://github.com/microsoft/TRELLIS.2#subdirectory=o-voxel",
    },
    {
        "name": "flash_attn",
        "import_name": "flash_attn",
        "wheel_release_base": "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download",
        "wheel_type": "flash_attn",  # Special handling for torch version in wheel name
        "git_url": "git+https://github.com/Dao-AILab/flash-attention",
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


def try_install_cuda_toolkit():
    """
    Try to install minimal CUDA toolkit components needed for compilation.
    - Linux: Uses apt to install cuda-nvcc and cuda-cudart-dev
    - Windows: Downloads and runs CUDA network installer with minimal components
    Returns True if successful, False otherwise.
    """
    print("[ComfyUI-TRELLIS2] CUDA compiler not found, attempting to install...")

    # Get CUDA version from PyTorch
    try:
        import torch
        cuda_ver = torch.version.cuda
        if not cuda_ver:
            print("[ComfyUI-TRELLIS2] PyTorch CUDA version not available")
            return False
        cuda_parts = cuda_ver.split('.')
        cuda_major = cuda_parts[0]  # "12"
        cuda_minor = cuda_parts[1] if len(cuda_parts) > 1 else "0"  # "8"
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Could not detect CUDA version: {e}")
        return False

    if sys.platform == "linux":
        return _install_cuda_linux(cuda_major, cuda_minor)
    elif sys.platform == "win32":
        return _install_cuda_windows(cuda_major, cuda_minor)
    else:
        print(f"[ComfyUI-TRELLIS2] Unsupported platform: {sys.platform}")
        return False


def _install_cuda_linux(cuda_major, cuda_minor):
    """Install minimal CUDA components on Linux via apt."""
    print("[ComfyUI-TRELLIS2] Installing CUDA compiler via apt (minimal components)...")

    # Add NVIDIA repository if not present
    keyring_path = "/usr/share/keyrings/cuda-archive-keyring.gpg"
    if not os.path.exists(keyring_path):
        print("[ComfyUI-TRELLIS2] Adding NVIDIA CUDA repository...")
        try:
            # Download and install keyring
            subprocess.run([
                "wget", "-q", "-O", "/tmp/cuda-keyring.deb",
                "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
            ], check=True, timeout=60)
            subprocess.run(["sudo", "dpkg", "-i", "/tmp/cuda-keyring.deb"], check=True, timeout=30)
            subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True, timeout=120)
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-TRELLIS2] Failed to add NVIDIA repository: {e}")
            return False
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] Repository setup error: {e}")
            return False

    # Install minimal CUDA components
    packages = [
        f"cuda-nvcc-{cuda_major}-{cuda_minor}",
        f"cuda-cudart-dev-{cuda_major}-{cuda_minor}",
    ]

    print(f"[ComfyUI-TRELLIS2] Installing: {', '.join(packages)}")
    try:
        result = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "-qq"] + packages,
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"[ComfyUI-TRELLIS2] apt install failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        print("[ComfyUI-TRELLIS2] Installation timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Installation error: {e}")
        return False

    # Update PATH
    cuda_path = f"/usr/local/cuda-{cuda_major}.{cuda_minor}"
    if os.path.exists(cuda_path):
        os.environ["CUDA_HOME"] = cuda_path
        os.environ["PATH"] = f"{cuda_path}/bin:" + os.environ.get("PATH", "")

    print("[ComfyUI-TRELLIS2] [OK] CUDA compiler installed via apt")
    return True


def _install_cuda_windows(cuda_major, cuda_minor):
    """Install minimal CUDA components on Windows via network installer."""
    import tempfile
    import urllib.request

    print("[ComfyUI-TRELLIS2] Installing CUDA compiler via network installer (minimal components)...")

    # CUDA network installer URL (adjust version as needed)
    cuda_version = f"{cuda_major}.{cuda_minor}"
    # Map to installer version (e.g., 12.8 -> 12.8.1)
    installer_versions = {
        "12.8": "12.8.1",
        "12.6": "12.6.3",
        "12.4": "12.4.1",
    }
    installer_ver = installer_versions.get(cuda_version, f"{cuda_version}.0")

    installer_url = f"https://developer.download.nvidia.com/compute/cuda/{installer_ver}/network_installers/cuda_{installer_ver}_windows_network.exe"
    installer_path = os.path.join(tempfile.gettempdir(), "cuda_network_installer.exe")

    # Download installer
    print(f"[ComfyUI-TRELLIS2] Downloading CUDA {cuda_version} network installer...")
    try:
        urllib.request.urlretrieve(installer_url, installer_path)
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Failed to download installer: {e}")
        return False

    # Run installer with minimal components (silent mode)
    # Components: nvcc (compiler), cudart (runtime)
    components = [
        f"nvcc_{cuda_major}.{cuda_minor}",
        f"cudart_{cuda_major}.{cuda_minor}",
    ]

    print(f"[ComfyUI-TRELLIS2] Installing components: {', '.join(components)}")
    try:
        cmd = [installer_path, "-s"] + components
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[ComfyUI-TRELLIS2] Installer failed with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("[ComfyUI-TRELLIS2] Installation timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Installation error: {e}")
        return False

    # Update environment
    cuda_path = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_major}.{cuda_minor}"
    if os.path.exists(cuda_path):
        os.environ["CUDA_HOME"] = cuda_path
        os.environ["PATH"] = f"{cuda_path}\\bin;" + os.environ.get("PATH", "")

    # Cleanup
    try:
        os.remove(installer_path)
    except:
        pass

    print("[ComfyUI-TRELLIS2] [OK] CUDA compiler installed")
    return True


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


def get_direct_wheel_urls(package_config):
    """
    Build direct wheel URLs from GitHub releases.
    Returns list of URLs to try (new format first, then old format).
    """
    wheel_release_base = package_config.get("wheel_release_base")
    wheel_version = package_config.get("wheel_version")
    wheel_type = package_config.get("wheel_type")

    # flash_attn uses different URL building
    if wheel_type == "flash_attn":
        return []  # Handled by get_flash_attn_wheel_urls

    if not wheel_release_base or not wheel_version:
        return []

    cuda_suffix = get_wheel_cuda_suffix()
    if not cuda_suffix:
        return []

    torch_ver, _ = get_torch_info()
    if not torch_ver:
        return []

    py_major, py_minor = sys.version_info[:2]
    platform = "linux_x86_64" if sys.platform == "linux" else "win_amd64"
    package_name = package_config["name"]

    # Extract torch major.minor for release tag (e.g., "2.8.0" -> "280")
    torch_parts = torch_ver.split('.')[:2]
    torch_tag = f"{torch_parts[0]}{torch_parts[1]}0"  # "2.8" -> "280"

    urls = []

    # New format: tag=cu128-torch280, wheel=package-version-cpXX-cpXX-platform.whl (no cuda suffix)
    new_tag = f"{cuda_suffix}-torch{torch_tag}"
    new_wheel = f"{package_name}-{wheel_version}-cp{py_major}{py_minor}-cp{py_major}{py_minor}-{platform}.whl"
    urls.append(f"{wheel_release_base}/{new_tag}/{new_wheel}")

    # Old format: tag=cu128, wheel=package-version+cu128-cpXX-cpXX-platform.whl (with cuda suffix)
    old_wheel = f"{package_name}-{wheel_version}+{cuda_suffix}-cp{py_major}{py_minor}-cp{py_major}{py_minor}-{platform}.whl"
    urls.append(f"{wheel_release_base}/{cuda_suffix}/{old_wheel}")

    return urls


def get_flash_attn_wheel_urls():
    """
    Build flash_attn wheel URLs to try.
    Returns list of (url, version) tuples to try in order.
    flash_attn uses: flash_attn-{ver}+cu{cuda}torch{torch}-cp{py}-cp{py}-{platform}.whl
    """
    torch_ver, cuda_ver = get_torch_info()
    if not torch_ver or not cuda_ver:
        return []

    cuda_mm = cuda_ver.split('.')[0] + cuda_ver.split('.')[1]  # "12.8" -> "128"
    torch_mm = '.'.join(torch_ver.split('.')[:2])  # "2.8.0" -> "2.8"
    py_major, py_minor = sys.version_info[:2]
    platform = "linux_x86_64" if sys.platform == "linux" else "win_amd64"

    base_url = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download"

    # flash_attn versions to try (newest first) with their release tags
    versions_to_try = [
        ("2.7.4", "v0.5.4"),
        ("2.6.3", "v0.5.4"),
        ("2.7.4", "v0.4.19"),
    ]

    urls = []
    for flash_ver, release_tag in versions_to_try:
        wheel_name = f"flash_attn-{flash_ver}+cu{cuda_mm}torch{torch_mm}-cp{py_major}{py_minor}-cp{py_major}{py_minor}-{platform}.whl"
        url = f"{base_url}/{release_tag}/{wheel_name}"
        urls.append((url, flash_ver))

    return urls


def try_install_from_direct_url(package_config):
    """
    Try installing a package from direct GitHub release wheel URLs.
    Tries multiple URL formats (new torch-versioned first, then old).
    Returns True if successful, False otherwise.
    """
    urls = get_direct_wheel_urls(package_config)
    if not urls:
        return False

    package_name = package_config["name"]

    for wheel_url in urls:
        print(f"[ComfyUI-TRELLIS2] Trying: {wheel_url}")

        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", wheel_url
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"[ComfyUI-TRELLIS2] [OK] Installed {package_name} from wheel")
                return True
            else:
                if result.stderr and "404" in result.stderr:
                    continue  # Try next URL
                elif result.stderr and ("not found" in result.stderr.lower() or "no such file" in result.stderr.lower()):
                    continue  # Try next URL
        except subprocess.TimeoutExpired:
            print(f"[ComfyUI-TRELLIS2] Download timed out")
            continue
        except Exception as e:
            continue

    print(f"[ComfyUI-TRELLIS2] No wheel found at any URL for {package_name}")
    return False


def try_install_from_wheel(package_name, wheel_index_url):
    """
    Try installing a package from pre-built wheels via index.
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
                    print(f"[ComfyUI-TRELLIS2] No matching wheel in index for {package_name}")
                else:
                    print(f"[ComfyUI-TRELLIS2] Wheel index install failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[ComfyUI-TRELLIS2] Wheel install timed out for {package_name}")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Wheel install error: {e}")
        return False


def clone_and_init_submodules(git_url, target_dir):
    """
    Clone a git repo and initialize submodules with HTTPS URLs.
    Returns True if successful, False otherwise.
    """
    import tempfile

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

    # First try direct pip install
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", git_url],
            capture_output=True, text=True, timeout=900, env=cuda_env
        )
        if result.returncode == 0:
            print(f"[ComfyUI-TRELLIS2] [OK] Compiled {package_name} successfully")
            return True
    except:
        pass

    # If direct install fails, try manual clone with submodule fix
    print(f"[ComfyUI-TRELLIS2] Direct install failed, trying manual clone with submodule fix...")
    import tempfile
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


def try_install_flash_attn():
    """
    Try installing flash_attn from pre-built wheels.
    Returns True if successful, False otherwise.
    """
    urls = get_flash_attn_wheel_urls()
    if not urls:
        print("[ComfyUI-TRELLIS2] Could not determine flash_attn wheel URL for this environment")
        return False

    for url, version in urls:
        print(f"[ComfyUI-TRELLIS2] Trying flash_attn {version}: {url}")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", url
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"[ComfyUI-TRELLIS2] [OK] Installed flash_attn {version} from wheel")
                return True
            elif "404" in (result.stderr or ""):
                continue  # Try next URL
        except:
            continue

    return False


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

    # Check if already installed
    if is_package_installed(import_name):
        print(f"[ComfyUI-TRELLIS2] [OK] {name} already installed")
        return True

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
        # Try 1: wheel index (pip --find-links)
        if wheel_index and try_install_from_wheel(name, wheel_index):
            return True

        # Try 2: direct GitHub release URL
        print(f"[ComfyUI-TRELLIS2] Trying direct GitHub release URL...")
        if try_install_from_direct_url(package_config):
            return True

    # Try 3: compile from source
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

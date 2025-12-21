"""
CUDA environment setup for ComfyUI-TRELLIS2.
Handles finding and configuring CUDA for compilation fallback.
"""
import os
import shutil
import site
import subprocess
import sys


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
        "/usr/local/cuda-13.0", "/usr/local/cuda-12.8", "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.4", "/usr/local/cuda-12.2", "/usr/local/cuda-12.0",
        "/usr/local/cuda",
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
        "13.0": "13.0.0",
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

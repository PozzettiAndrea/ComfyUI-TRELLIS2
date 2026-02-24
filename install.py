#!/usr/bin/env python3
"""
Installation script for ComfyUI-TRELLIS2 with isolated environment.

This script sets up an isolated Python virtual environment with all dependencies
required for TRELLIS2. The environment is completely isolated from
ComfyUI's main environment, preventing any dependency conflicts.

Uses comfy-env package for environment management.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path


# =============================================================================
# VC++ Redistributable Check (Windows only)
# =============================================================================

VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def check_vcredist_installed():
    """Check if VC++ Redistributable DLLs are actually present on the system."""
    if platform.system() != "Windows":
        return True  # Not needed on non-Windows

    required_dlls = ['vcruntime140.dll', 'msvcp140.dll']

    # Search locations in order of preference
    search_paths = []

    # 1. System directory (most reliable)
    system_root = os.environ.get('SystemRoot', r'C:\Windows')
    search_paths.append(os.path.join(system_root, 'System32'))

    # 2. Python environment directories
    if hasattr(sys, 'base_prefix'):
        search_paths.append(os.path.join(sys.base_prefix, 'Library', 'bin'))
        search_paths.append(os.path.join(sys.base_prefix, 'DLLs'))
    if hasattr(sys, 'prefix') and sys.prefix != getattr(sys, 'base_prefix', sys.prefix):
        search_paths.append(os.path.join(sys.prefix, 'Library', 'bin'))
        search_paths.append(os.path.join(sys.prefix, 'DLLs'))

    # Check each required DLL
    for dll in required_dlls:
        found = False
        for search_path in search_paths:
            dll_path = os.path.join(search_path, dll)
            if os.path.exists(dll_path):
                found = True
                break
        if not found:
            return False

    return True


def install_vcredist():
    """Download and install VC++ Redistributable with UAC elevation."""
    import urllib.request
    import tempfile

    print("[TRELLIS2] Downloading VC++ Redistributable...")

    # Download to temp file
    temp_path = os.path.join(tempfile.gettempdir(), "vc_redist.x64.exe")
    try:
        urllib.request.urlretrieve(VCREDIST_URL, temp_path)
    except Exception as e:
        print(f"[TRELLIS2] Failed to download VC++ Redistributable: {e}")
        print(f"[TRELLIS2] Please download manually from: {VCREDIST_URL}")
        return False

    print("[TRELLIS2] Installing VC++ Redistributable (UAC prompt may appear)...")

    # Run with elevation - /passive shows progress, /quiet is fully silent
    try:
        result = subprocess.run(
            [temp_path, '/install', '/passive', '/norestart'],
            capture_output=True
        )
    except Exception as e:
        print(f"[TRELLIS2] Failed to run installer: {e}")
        print(f"[TRELLIS2] Please run manually: {temp_path}")
        return False

    # Clean up
    try:
        os.remove(temp_path)
    except:
        pass

    if result.returncode == 0:
        print("[TRELLIS2] VC++ Redistributable installer completed.")
    elif result.returncode == 1638:
        # 1638 = newer version already installed
        print("[TRELLIS2] VC++ Redistributable already installed (newer version)")
    else:
        print(f"[TRELLIS2] Installation returned code {result.returncode}")
        print(f"[TRELLIS2] Please install manually from: {VCREDIST_URL}")
        return False

    # Verify DLLs are actually present after installation
    if check_vcredist_installed():
        print("[TRELLIS2] VC++ Redistributable DLLs verified!")
        return True
    else:
        print("[TRELLIS2] Installation completed but DLLs not found in expected locations.")
        print("[TRELLIS2] You may need to restart your system or terminal.")
        return False


def ensure_vcredist():
    """Check and install VC++ Redistributable if needed (Windows only)."""
    if platform.system() != "Windows":
        return True

    if check_vcredist_installed():
        print("[TRELLIS2] VC++ Redistributable: OK (DLLs found)")
        return True

    print("[TRELLIS2] VC++ Redistributable DLLs not found - attempting automatic install...")

    if install_vcredist():
        return True

    # Fallback: provide clear manual instructions
    print("")
    print("=" * 70)
    print("[TRELLIS2] MANUAL INSTALLATION REQUIRED")
    print("=" * 70)
    print("")
    print("  The automatic installation of VC++ Redistributable failed.")
    print("  This is required for PyTorch CUDA and other native extensions.")
    print("")
    print("  Please download and install manually:")
    print(f"    {VCREDIST_URL}")
    print("")
    print("  After installation, restart your terminal and try again.")
    print("=" * 70)
    print("")
    return False


# =============================================================================
# Main Installation
# =============================================================================

def ensure_comfy_env():
    """Install/upgrade comfy-env package."""
    print("[TRELLIS2] Installing comfy-env package...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "comfy-env>=0.0.12"
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"[TRELLIS2] Failed to install comfy-env: {e}")
        return False


def main():
    """Main installation function."""
    print("\n" + "=" * 60)
    print("ComfyUI-TRELLIS2 Installation (Isolated Environment)")
    print("=" * 60)

    # Check VC++ Redistributable first (required for PyTorch CUDA and native extensions)
    if not ensure_vcredist():
        print("[TRELLIS2] WARNING: VC++ Redistributable installation failed.")
        print("[TRELLIS2] Some features may not work. Continuing anyway...")

    # Ensure comfy-env is installed
    if not ensure_comfy_env():
        print("[TRELLIS2] Cannot continue without comfy-env package.")
        return 1

    from comfyui_envmanager import IsolatedEnvManager, discover_config

    node_root = Path(__file__).parent.absolute()

    # Load environment config from comfyui_env.toml (v2 schema)
    config = discover_config(node_root)
    if config is None:
        print("[TRELLIS2] ERROR: Could not find comfyui_env.toml")
        return 1

    # Get the trellis2 isolated environment
    if "trellis2" not in config.envs:
        print("[TRELLIS2] ERROR: No 'trellis2' environment defined in config")
        return 1

    env_config = config.envs["trellis2"]

    print(f"[TRELLIS2] Loaded config: {env_config.name}")
    print(f"[TRELLIS2]   CUDA: {env_config.cuda}")
    print(f"[TRELLIS2]   PyTorch: {env_config.pytorch_version}")
    print(f"[TRELLIS2]   Requirements: {len(env_config.requirements)} packages")
    print(f"[TRELLIS2]   CUDA packages: {len(env_config.no_deps_requirements)} packages")

    # Create environment manager
    def log(msg):
        print(f"[TRELLIS2] {msg}")

    manager = IsolatedEnvManager(base_dir=node_root, log_callback=log)

    # Check if already ready
    if manager.is_ready(env_config, verify_packages=["torch", "nvdiffrast"]):
        env_dir = manager.get_env_dir(env_config)
        print("[TRELLIS2] Isolated environment already exists and is ready!")
        print(f"[TRELLIS2] Location: {env_dir}")
        print("[TRELLIS2] To reinstall, delete the environment directory.")
        return 0

    # Setup environment
    try:
        manager.setup(env_config, verify_packages=["torch", "nvdiffrast"])
        print("\n" + "=" * 60)
        print("[TRELLIS2] Installation completed successfully!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n[TRELLIS2] Installation FAILED: {e}")
        print("[TRELLIS2] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2/issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())

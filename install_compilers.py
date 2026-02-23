#!/usr/bin/env python3
"""
Install C++ compilers for building CUDA extensions from source.

Run this only if wheel installation failed and you need to compile.

Usage:
    python install_compilers.py

For ComfyUI Portable (Windows):
    path\\to\\python_embeded\\python.exe install_compilers.py
"""
import subprocess
import sys
import os


def install_linux():
    """Install build-essential (g++, make) on Linux via apt."""
    print("[TRELLIS2] Installing build-essential (g++, make)...")
    print("[TRELLIS2] This requires sudo access.")

    try:
        # Update package list
        result = subprocess.run(
            ["sudo", "apt-get", "update", "-qq"],
            timeout=120
        )
        if result.returncode != 0:
            print("[TRELLIS2] [WARNING] apt-get update failed, continuing anyway...")

        # Install build-essential
        result = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "build-essential"],
            timeout=300
        )

        if result.returncode == 0:
            print("[TRELLIS2] [OK] Build tools installed successfully!")
            print("[TRELLIS2] Now run: python install.py")
            return True
        else:
            print("[TRELLIS2] [FAILED] Could not install build-essential")
            return False

    except subprocess.TimeoutExpired:
        print("[TRELLIS2] [FAILED] Installation timed out")
        return False
    except FileNotFoundError:
        print("[TRELLIS2] [FAILED] apt-get not found. Are you on a Debian/Ubuntu system?")
        print("[TRELLIS2] For other distros, install gcc/g++ manually:")
        print("[TRELLIS2]   Fedora/RHEL: sudo dnf install gcc gcc-c++")
        print("[TRELLIS2]   Arch: sudo pacman -S base-devel")
        return False
    except Exception as e:
        print(f"[TRELLIS2] [FAILED] Error: {e}")
        return False


def install_windows():
    """Download and install Visual Studio Build Tools on Windows."""
    import urllib.request
    import tempfile

    print("[TRELLIS2] " + "=" * 50)
    print("[TRELLIS2] Visual Studio Build Tools Installer")
    print("[TRELLIS2] " + "=" * 50)
    print("[TRELLIS2]")
    print("[TRELLIS2] This will download and install the C++ build tools")
    print("[TRELLIS2] required to compile CUDA extensions.")
    print("[TRELLIS2]")
    print("[TRELLIS2] Download size: ~1.5 MB (installer)")
    print("[TRELLIS2] Install size: ~2-3 GB")
    print("[TRELLIS2] Install time: 10-15 minutes")
    print("[TRELLIS2]")

    # Ask for confirmation
    response = input("[TRELLIS2] Continue? [y/N]: ").strip().lower()
    if response not in ('y', 'yes'):
        print("[TRELLIS2] Cancelled.")
        return False

    installer_url = "https://aka.ms/vs/17/release/vs_buildtools.exe"
    installer_path = os.path.join(tempfile.gettempdir(), "vs_buildtools.exe")

    try:
        # Download the installer
        print(f"[TRELLIS2] Downloading from {installer_url}...")
        urllib.request.urlretrieve(installer_url, installer_path)
        print(f"[TRELLIS2] Downloaded to {installer_path}")

        # Run the installer
        print("[TRELLIS2] Starting installation...")
        print("[TRELLIS2] This will take 10-15 minutes. Please wait...")

        result = subprocess.run([
            installer_path,
            "--quiet",
            "--wait",
            "--norestart",
            "--add", "Microsoft.VisualStudio.Workload.VCTools",
            "--includeRecommended"
        ], timeout=1800)  # 30 minute timeout

        if result.returncode == 0:
            print("[TRELLIS2] [OK] Build tools installed successfully!")
            print("[TRELLIS2]")
            print("[TRELLIS2] IMPORTANT: You may need to restart your computer")
            print("[TRELLIS2] or run ComfyUI from 'Developer Command Prompt for VS'")
            print("[TRELLIS2]")
            print("[TRELLIS2] After restart, run: python install.py")
            return True
        elif result.returncode == 3010:
            # 3010 = success but restart required
            print("[TRELLIS2] [OK] Build tools installed!")
            print("[TRELLIS2] A system restart is required to complete installation.")
            print("[TRELLIS2] After restarting, run: python install.py")
            return True
        else:
            print(f"[TRELLIS2] [FAILED] Installer returned code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("[TRELLIS2] [FAILED] Installation timed out after 30 minutes")
        return False
    except urllib.error.URLError as e:
        print(f"[TRELLIS2] [FAILED] Could not download installer: {e}")
        return False
    except Exception as e:
        print(f"[TRELLIS2] [FAILED] Error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(installer_path):
            try:
                os.remove(installer_path)
            except:
                pass


def main():
    print("[TRELLIS2] C++ Compiler Installation Script")
    print("[TRELLIS2] " + "=" * 40)
    print()

    if sys.platform == "linux":
        success = install_linux()
    elif sys.platform == "win32":
        success = install_windows()
    elif sys.platform == "darwin":
        print("[TRELLIS2] macOS detected.")
        print("[TRELLIS2] Install Xcode Command Line Tools:")
        print("[TRELLIS2]   xcode-select --install")
        success = False
    else:
        print(f"[TRELLIS2] Unsupported platform: {sys.platform}")
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

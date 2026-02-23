"""
Visual C++ Redistributable and Media Feature Pack installation for Windows.
Required for opencv-python and other native extensions.
"""
import os
import sys
import subprocess
import tempfile
import urllib.request


# Direct download URL for VC++ 2015-2022 Redistributable x64
VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def is_windows_n_edition():
    """Check if running Windows N or KN edition (missing Media Feature Pack)."""
    if sys.platform != "win32":
        return False
    try:
        result = subprocess.run(
            ["powershell", "-Command", "(Get-CimInstance Win32_OperatingSystem).Caption"],
            capture_output=True, text=True, timeout=30
        )
        caption = result.stdout.strip()
        return " N" in caption or " KN" in caption
    except Exception:
        return False


def try_install_media_feature_pack():
    """
    Install Media Feature Pack for Windows.
    Required for OpenCV to work properly.
    Returns True if successful or not needed, False otherwise.
    """
    if sys.platform != "win32":
        return True

    print("[ComfyUI-TRELLIS2] Checking Media Feature Pack (required for OpenCV)...")

    try:
        # Try to install Media Feature Pack
        result = subprocess.run(
            ["DISM", "/Online", "/Add-Capability",
             "/CapabilityName:Media.MediaFeaturePack~~~~0.0.1.0"],
            capture_output=True, text=True, timeout=600  # 10 min timeout
        )

        if result.returncode == 0:
            print("[ComfyUI-TRELLIS2] [OK] Media Feature Pack installed (reboot required)")
            return True
        elif "already installed" in result.stdout.lower() or result.returncode == 87:
            print("[ComfyUI-TRELLIS2] [OK] Media Feature Pack already installed")
            return True
        elif result.returncode == 740:
            # Requires elevation
            print("[ComfyUI-TRELLIS2] [WARNING] Administrator privileges required")
            print("[ComfyUI-TRELLIS2] Please run the following command as Administrator:")
            print('    DISM /Online /Add-Capability /CapabilityName:Media.MediaFeaturePack~~~~0.0.1.0')
            print("[ComfyUI-TRELLIS2] Or install via Settings > Apps > Optional features > Add a feature")
            return False
        else:
            print(f"[ComfyUI-TRELLIS2] [WARNING] DISM returned code {result.returncode}")
            if result.stderr:
                print(f"[ComfyUI-TRELLIS2] Error: {result.stderr[:300]}")
            print("[ComfyUI-TRELLIS2] Please install Media Feature Pack manually via:")
            print("    Settings > Apps > Optional features > Add a feature > Media Feature Pack")
            return False

    except subprocess.TimeoutExpired:
        print("[ComfyUI-TRELLIS2] [WARNING] Media Feature Pack installation timed out")
        print("[ComfyUI-TRELLIS2] This is normal - it may still be installing in the background")
        return False
    except FileNotFoundError:
        print("[ComfyUI-TRELLIS2] [WARNING] DISM not found")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] [WARNING] Error: {e}")
        return False


def is_vcredist_installed():
    """
    Check if Visual C++ Redistributable is properly installed by testing cv2 import.
    Returns True if cv2 can be imported, False otherwise.
    """
    try:
        import cv2
        return True
    except ImportError as e:
        if "DLL load failed" in str(e):
            return False
        # Other import errors (module not installed) - assume vcredist is fine
        return True
    except Exception:
        return True


def try_install_vcredist():
    """
    Download and install Visual C++ Redistributable silently.
    Also installs Media Feature Pack for Windows N/KN editions.
    Returns True if successful, False otherwise.

    Note: This requires administrator privileges for silent install.
    If not elevated, will prompt the user.
    """
    if sys.platform != "win32":
        return True  # Not needed on non-Windows

    # First, try to install Media Feature Pack (required for OpenCV)
    try_install_media_feature_pack()

    print("[ComfyUI-TRELLIS2] Checking Visual C++ Redistributable...")

    if is_vcredist_installed():
        print("[ComfyUI-TRELLIS2] [OK] Visual C++ Redistributable is working")
        return True

    print("[ComfyUI-TRELLIS2] Visual C++ Redistributable may be missing or broken")
    print("[ComfyUI-TRELLIS2] Attempting to download and install...")

    try:
        # Download to temp directory
        temp_dir = tempfile.gettempdir()
        installer_path = os.path.join(temp_dir, "vc_redist.x64.exe")

        print(f"[ComfyUI-TRELLIS2] Downloading from {VCREDIST_URL}...")
        urllib.request.urlretrieve(VCREDIST_URL, installer_path)
        print("[ComfyUI-TRELLIS2] Download complete")

        # Try silent install first (requires admin)
        print("[ComfyUI-TRELLIS2] Installing Visual C++ Redistributable...")
        result = subprocess.run(
            [installer_path, "/install", "/quiet", "/norestart"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("[ComfyUI-TRELLIS2] [OK] Visual C++ Redistributable installed successfully")
            return True
        elif result.returncode == 1638:
            # Already installed (newer version)
            print("[ComfyUI-TRELLIS2] [OK] Visual C++ Redistributable already installed")
            return True
        elif result.returncode == 3010:
            # Success but reboot required
            print("[ComfyUI-TRELLIS2] [OK] Visual C++ Redistributable installed (reboot may be required)")
            return True
        elif result.returncode == 5:
            # Access denied - need admin
            print("[ComfyUI-TRELLIS2] [WARNING] Administrator privileges required")
            print("[ComfyUI-TRELLIS2] Please run the following command as Administrator:")
            print(f'    "{installer_path}" /install /quiet /norestart')
            print("[ComfyUI-TRELLIS2] Or download and install manually from:")
            print(f"    {VCREDIST_URL}")
            return False
        else:
            print(f"[ComfyUI-TRELLIS2] [WARNING] Installation returned code {result.returncode}")
            if result.stderr:
                print(f"[ComfyUI-TRELLIS2] Error: {result.stderr[:200]}")
            return False

    except urllib.error.URLError as e:
        print(f"[ComfyUI-TRELLIS2] [WARNING] Failed to download: {e}")
        print("[ComfyUI-TRELLIS2] Please install Visual C++ Redistributable manually from:")
        print(f"    {VCREDIST_URL}")
        return False
    except subprocess.TimeoutExpired:
        print("[ComfyUI-TRELLIS2] [WARNING] Installation timed out")
        return False
    except PermissionError:
        print("[ComfyUI-TRELLIS2] [WARNING] Permission denied - need administrator privileges")
        print("[ComfyUI-TRELLIS2] Please install Visual C++ Redistributable manually from:")
        print(f"    {VCREDIST_URL}")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] [WARNING] Installation error: {e}")
        return False

"""
Flash Attention installation for ComfyUI-TRELLIS2.
Handles finding and installing flash_attn from multiple wheel sources.
"""
import subprocess
import sys

from .config import FLASH_ATTN_SOURCES
from .detect import get_torch_info


def get_flash_attn_wheel_urls():
    """
    Build flash_attn wheel URLs from multiple sources.
    Returns list of (url, version, source_name) tuples to try in order.
    Different repos use different naming conventions.
    """
    torch_ver, cuda_ver = get_torch_info()
    if not torch_ver or not cuda_ver:
        return []

    cuda_mm = cuda_ver.split('.')[0] + cuda_ver.split('.')[1]  # "12.8" -> "128"
    torch_mm = '.'.join(torch_ver.split('.')[:2])  # "2.8.0" -> "2.8"
    py_major, py_minor = sys.version_info[:2]
    platform = "linux_x86_64" if sys.platform == "linux" else "win_amd64"

    # CUDA versions to try - flash_attn repos often don't have cu128, so try cu126 as fallback
    cuda_versions_to_try = [cuda_mm]
    if cuda_mm == "128":
        cuda_versions_to_try.append("126")  # cu126 wheels work on CUDA 12.8

    # Torch versions to try - not all python/cuda combos have wheels for latest torch
    # e.g., cu128+torch2.8+cp310 doesn't exist but cu128+torch2.7+cp310 does
    torch_versions_to_try = [torch_mm]
    if torch_mm == "2.8":
        torch_versions_to_try.append("2.7")  # torch2.7 wheels are often compatible
    elif torch_mm == "2.9":
        torch_versions_to_try.extend(["2.8", "2.7"])

    urls = []
    for source in FLASH_ATTN_SOURCES:
        if platform not in source["platforms"]:
            continue

        for flash_ver, release_tag in source["versions"]:
            for cuda_ver_try in cuda_versions_to_try:
                for torch_ver_try in torch_versions_to_try:
                    if source["format"] == "cxx11abi":
                        # bdashore3/oobabooga format: +cu128torch2.8.0cxx11abiFALSE
                        wheel_name = f"flash_attn-{flash_ver}+cu{cuda_ver_try}torch{torch_ver_try}.0cxx11abiFALSE-cp{py_major}{py_minor}-cp{py_major}{py_minor}-{platform}.whl"
                    else:
                        # mjun0812 format: +cu128torch2.8
                        wheel_name = f"flash_attn-{flash_ver}+cu{cuda_ver_try}torch{torch_ver_try}-cp{py_major}{py_minor}-cp{py_major}{py_minor}-{platform}.whl"

                    url = f"{source['base_url']}/{release_tag}/{wheel_name}"
                    urls.append((url, flash_ver, source["name"]))

    return urls


def try_install_flash_attn():
    """
    Try installing flash_attn from pre-built wheels.
    Tries multiple sources (bdashore3, mjun0812, oobabooga) until one works.
    Returns True if successful, False otherwise.
    """
    urls = get_flash_attn_wheel_urls()
    if not urls:
        print("[ComfyUI-TRELLIS2] Could not determine flash_attn wheel URL for this environment")
        return False

    for url, version, source in urls:
        print(f"[ComfyUI-TRELLIS2] Trying flash_attn {version} from {source}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "--no-deps", url
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"[ComfyUI-TRELLIS2] [OK] Installed flash_attn {version} from {source}")
                return True
            else:
                # Check for 404 or other errors
                error_msg = result.stderr or result.stdout or ""
                if "404" in error_msg or "not found" in error_msg.lower():
                    continue  # Try next URL
                else:
                    # Log non-404 errors but continue trying
                    print(f"[ComfyUI-TRELLIS2] {source} failed: {error_msg[:100]}")
                    continue
        except subprocess.TimeoutExpired:
            print(f"[ComfyUI-TRELLIS2] {source} timed out")
            continue
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] {source} error: {e}")
            continue

    print("[ComfyUI-TRELLIS2] Could not find flash_attn wheel from any source")
    return False

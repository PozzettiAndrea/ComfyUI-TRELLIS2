"""
Flash Attention installation for ComfyUI-TRELLIS2.
Dynamically queries GitHub API to find matching wheels across all releases.
"""
import json
import re
import subprocess
import sys
import urllib.request

from .detect import get_torch_info

# mjun0812's pre-built wheels repository
GITHUB_API_URL = "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases"


def find_flash_attn_wheels():
    """
    Query GitHub API to find flash_attn wheels matching current environment.
    Returns list of (url, flash_version) tuples, sorted by version (newest first).
    """
    torch_ver, cuda_ver = get_torch_info()
    if not torch_ver or not cuda_ver:
        return []

    # Build search pattern: cu128torch2.8-cp310-cp310-linux_x86_64
    cuda_short = cuda_ver.replace(".", "")  # "12.8" -> "128"
    torch_mm = ".".join(torch_ver.split(".")[:2])  # "2.8.0" -> "2.8"
    py_ver = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    platform = "linux_x86_64" if sys.platform == "linux" else "win_amd64"

    pattern = f"cu{cuda_short}torch{torch_mm}-{py_ver}-{py_ver}-{platform}"
    print(f"[ComfyUI-TRELLIS2] Searching for flash_attn wheel: *{pattern}.whl")

    try:
        req = urllib.request.Request(
            GITHUB_API_URL,
            headers={"User-Agent": "ComfyUI-TRELLIS2"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            releases = json.loads(response.read())
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] GitHub API error: {e}")
        return []

    matches = []
    for release in releases:
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if pattern in name and name.endswith(".whl"):
                # Extract flash_attn version from wheel name
                match = re.match(r"flash_attn-([^+]+)\+", name)
                if match:
                    version = match.group(1)
                    url = asset["browser_download_url"]
                    matches.append((url, version))

    # Sort by version descending (newest first)
    # Handle versions like "2.8.3" or "2.7.4.post1"
    def version_key(item):
        version = item[1]
        # Extract numeric parts for sorting
        parts = re.findall(r'\d+', version)
        return [int(p) for p in parts[:4]]  # Take up to 4 parts

    matches.sort(key=version_key, reverse=True)

    if matches:
        print(f"[ComfyUI-TRELLIS2] Found {len(matches)} matching wheel(s)")
    else:
        print(f"[ComfyUI-TRELLIS2] No matching wheels found for pattern: {pattern}")

    return matches


def try_install_flash_attn():
    """
    Try installing flash_attn from pre-built wheels.
    Queries GitHub API to find matching wheels for current environment.
    Returns True if successful, False otherwise.
    """
    wheels = find_flash_attn_wheels()

    if not wheels:
        print("[ComfyUI-TRELLIS2] No flash_attn wheels found for this environment")
        return False

    for url, version in wheels:
        print(f"[ComfyUI-TRELLIS2] Trying flash_attn {version}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "--no-deps", "--force-reinstall", "--no-cache-dir", url],
                capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                # Verify the package can be imported (check for ABI issues)
                if _verify_flash_attn_import():
                    print(f"[ComfyUI-TRELLIS2] [OK] Installed flash_attn {version}")
                    return True
                else:
                    # ABI mismatch - uninstall and try next
                    print(f"[ComfyUI-TRELLIS2] ABI mismatch, trying next wheel...")
                    _uninstall_flash_attn()
                    continue
            else:
                # Log error but continue trying
                error_msg = result.stderr or result.stdout or ""
                if "404" in error_msg or "not found" in error_msg.lower():
                    continue  # Wheel doesn't exist at this URL
                else:
                    print(f"[ComfyUI-TRELLIS2] pip install failed: {error_msg[:100]}")
                    continue

        except subprocess.TimeoutExpired:
            print(f"[ComfyUI-TRELLIS2] Download timed out")
            continue
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] Install error: {e}")
            continue

    print("[ComfyUI-TRELLIS2] Could not install flash_attn from any wheel")
    return False


def _verify_flash_attn_import():
    """
    Verify flash_attn can be imported without ABI errors.
    Returns True if import succeeds, False otherwise.
    """
    try:
        # Force reimport by removing from cache
        if "flash_attn" in sys.modules:
            del sys.modules["flash_attn"]

        import flash_attn
        return True
    except ImportError as e:
        error_str = str(e)
        if "undefined symbol" in error_str:
            print(f"[ComfyUI-TRELLIS2] [WARNING] flash_attn ABI incompatibility: {error_str[:100]}")
            return False
        # Other import error
        print(f"[ComfyUI-TRELLIS2] [WARNING] flash_attn import error: {error_str}")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] [WARNING] flash_attn verification error: {e}")
        return False


def _uninstall_flash_attn():
    """Uninstall flash_attn package."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "flash-attn"],
            capture_output=True, timeout=60
        )
    except Exception:
        pass

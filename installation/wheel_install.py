"""
Wheel installation functions for ComfyUI-TRELLIS2.
Handles downloading and installing pre-built wheels.
"""
import subprocess
import sys

from .detect import get_python_version, get_torch_info, get_wheel_cuda_suffix, get_wheel_dir


def is_package_installed(import_name):
    """Check if a package is already installed."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def verify_package_import(import_name, package_name):
    """
    Verify that a package can be imported without errors (e.g., ABI mismatch).
    If import fails, uninstall the broken package.
    Returns True if import succeeds, False otherwise.
    """
    try:
        __import__(import_name)
        return True
    except ImportError as e:
        # Check for ABI/symbol errors
        error_str = str(e)
        if "undefined symbol" in error_str or "cannot import" in error_str:
            print(f"[ComfyUI-TRELLIS2] [WARNING] {package_name} has ABI incompatibility: {error_str[:100]}")
            print(f"[ComfyUI-TRELLIS2] Uninstalling broken wheel and will try compilation...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "uninstall", "-y", package_name
                ], capture_output=True, timeout=60)
            except:
                pass
            return False
        elif "No module named" in error_str:
            print(f"[ComfyUI-TRELLIS2] [WARNING] {package_name} missing dependency: {error_str}")
            return False
        # Regular import error - package not installed
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] [WARNING] {package_name} import error: {e}")
        return False


def get_direct_wheel_urls(package_config):
    """
    Build direct wheel URL from GitHub releases.
    Uses static WHEEL_DIRS mapping for reliable URL construction.
    Wheel name format: {package}-{version}+cu{cuda_short}torch{torch_mm}-cpXX-cpXX-{platform}.whl
    Example: flex_gemm-0.0.1+cu128torch29-cp312-cp312-linux_x86_64.whl
    """
    wheel_release_base = package_config.get("wheel_release_base")
    wheel_version = package_config.get("wheel_version")
    wheel_type = package_config.get("wheel_type")

    # flash_attn uses different URL building
    if wheel_type == "flash_attn":
        return []  # Handled by get_flash_attn_wheel_urls

    if not wheel_release_base or not wheel_version:
        return []

    # Use static mapping for wheel directory
    wheel_dir = get_wheel_dir()
    if not wheel_dir:
        return []

    # Get CUDA suffix (e.g., "cu128") and torch version for wheel naming
    cuda_suffix = get_wheel_cuda_suffix()
    if not cuda_suffix:
        return []

    torch_ver, _ = get_torch_info()
    if not torch_ver:
        return []

    # Build torch_mm: "2.9.1" -> "29"
    torch_parts = torch_ver.split('.')[:2]
    torch_mm = ''.join(torch_parts)

    py_major, py_minor = sys.version_info[:2]
    platform = "linux_x86_64" if sys.platform == "linux" else "win_amd64"
    package_name = package_config["name"]

    # Build wheel URL with new naming: {package}-{version}+cu{cuda}torch{mm}-cpXX-cpXX-{platform}.whl
    # cuda_suffix is already "cu128", so we just add "torch{mm}"
    # Windows wheels have duplicated CUDA suffix due to PowerShell -replace bug in wheel build workflows
    if platform == "win_amd64":
        wheel_name = f"{package_name}-{wheel_version}+{cuda_suffix}torch{torch_mm}-cp{py_major}{py_minor}+{cuda_suffix}torch{torch_mm}-cp{py_major}{py_minor}-{platform}.whl"
    else:
        wheel_name = f"{package_name}-{wheel_version}+{cuda_suffix}torch{torch_mm}-cp{py_major}{py_minor}-cp{py_major}{py_minor}-{platform}.whl"
    wheel_url = f"{wheel_release_base}/{wheel_dir}/{wheel_name}"

    return [wheel_url]


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
                sys.executable, "-m", "pip", "install", "--no-deps", wheel_url
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Verify the package can be imported (check for ABI issues)
                import_name = package_config.get("import_name", package_name)
                if verify_package_import(import_name, package_name):
                    print(f"[ComfyUI-TRELLIS2] [OK] Installed {package_name} from wheel")
                    return True
                else:
                    # ABI mismatch - wheel installed but can't be imported
                    return False
            else:
                # Log the actual error so we can debug
                error_msg = result.stderr[:300] if result.stderr else result.stdout[:300] if result.stdout else "unknown error"
                if "404" in error_msg or "not found" in error_msg.lower() or "no such file" in error_msg.lower():
                    print(f"[ComfyUI-TRELLIS2] Wheel not found at URL")
                    continue
                else:
                    print(f"[ComfyUI-TRELLIS2] pip install failed: {error_msg}")
                    continue
        except subprocess.TimeoutExpired:
            print(f"[ComfyUI-TRELLIS2] Download timed out")
            continue
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] Exception during install: {e}")
            continue

    print(f"[ComfyUI-TRELLIS2] No wheel found at any URL for {package_name}")
    return False


def try_install_from_wheel(package_name, wheel_index_url, import_name=None):
    """
    Try installing a package from pre-built wheels via index.
    Returns True if successful, False otherwise.
    """
    if import_name is None:
        import_name = package_name

    py_ver = get_python_version()

    # Use static mapping for wheel directory
    wheel_dir = get_wheel_dir()
    if not wheel_dir:
        print(f"[ComfyUI-TRELLIS2] No matching wheel directory for {package_name}")
        return False

    # Build the subdirectory URL for this CUDA/torch version
    # e.g., https://pozzettiandrea.github.io/ovoxel-wheels/cu128-torch291/
    wheel_index_with_subdir = f"{wheel_index_url.rstrip('/')}/{wheel_dir}/"

    print(f"[ComfyUI-TRELLIS2] Looking for {package_name} wheel (Python {py_ver}, {wheel_dir})")
    print(f"[ComfyUI-TRELLIS2] Wheel index: {wheel_index_with_subdir}")

    try:
        # Use --no-index to ONLY look at our wheel index, not PyPI
        # Use --no-deps to avoid pulling in dependencies (we manage them ourselves)
        # This avoids conflicts with similarly-named packages on PyPI
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-deps",
            package_name, "--no-index", "--find-links", wheel_index_with_subdir
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Verify the package can be imported (check for ABI issues)
            if verify_package_import(import_name, package_name):
                print(f"[ComfyUI-TRELLIS2] [OK] Installed {package_name} from pre-built wheel")
                return True
            else:
                # ABI mismatch - wheel installed but can't be imported
                return False
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

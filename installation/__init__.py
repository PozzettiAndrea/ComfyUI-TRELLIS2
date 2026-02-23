"""
Installation modules for ComfyUI-TRELLIS2.
Provides modular installation logic for CUDA extensions.
"""

from .config import PACKAGES, WHEEL_CUDA_MAP, WHEEL_DIRS
from .detect import (
    get_python_version,
    get_torch_info,
    get_wheel_cuda_suffix,
    get_wheel_dir,
)
from .cuda_env import (
    find_cuda_home,
    setup_cuda_environment,
    get_cuda_arch_list,
    try_install_cuda_toolkit,
)
from .wheel_install import (
    is_package_installed,
    verify_package_import,
    get_direct_wheel_urls,
    try_install_from_direct_url,
    try_install_from_wheel,
)
from .compile import (
    clone_and_init_submodules,
    try_compile_from_source,
)
from .flash_attn import (
    find_flash_attn_wheels,
    try_install_flash_attn,
)
from .vcredist import (
    is_vcredist_installed,
    try_install_vcredist,
)

__all__ = [
    # Config
    "PACKAGES",
    "WHEEL_CUDA_MAP",
    "WHEEL_DIRS",
    # Detection
    "get_python_version",
    "get_torch_info",
    "get_wheel_cuda_suffix",
    "get_wheel_dir",
    # CUDA Environment
    "find_cuda_home",
    "setup_cuda_environment",
    "get_cuda_arch_list",
    "try_install_cuda_toolkit",
    # Wheel Installation
    "is_package_installed",
    "verify_package_import",
    "get_direct_wheel_urls",
    "try_install_from_direct_url",
    "try_install_from_wheel",
    # Compilation
    "clone_and_init_submodules",
    "try_compile_from_source",
    # Flash Attention
    "find_flash_attn_wheels",
    "try_install_flash_attn",
    # VC++ Redistributable (Windows)
    "is_vcredist_installed",
    "try_install_vcredist",
]

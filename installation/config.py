"""
Package configuration and version mappings for ComfyUI-TRELLIS2.
"""

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
        "name": "nvdiffrec_render",
        "import_name": "nvdiffrec_render",
        "wheel_index": "https://pozzettiandrea.github.io/nvdiffrec_render-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/nvdiffrec_render-wheels/releases/download",
        "wheel_version": "0.0.1",
        "git_url": None,  # Source requires custom build process, wheel-only
    },
    {
        "name": "flash_attn",
        "import_name": "flash_attn",
        "wheel_release_base": "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download",
        "wheel_type": "flash_attn",  # Special handling for torch version in wheel name
        "git_url": "git+https://github.com/Dao-AILab/flash-attention",
    },
]


# =============================================================================
# CUDA/PyTorch Version Mappings
# =============================================================================

# Map (CUDA major.minor, PyTorch major.minor) -> wheel CUDA suffix
# Wheels are built for cu124, cu128, cu130
WHEEL_CUDA_MAP = {
    ("12.4", "2.5"): "cu124",
    ("12.6", "2.6"): "cu126",
    ("12.6", "2.8"): "cu126",
    ("12.8", "2.8"): "cu128",
    ("12.8", "2.9"): "cu128",
    ("12.9", "2.8"): "cu128",  # CUDA 12.9 can use 12.8 wheels
    ("12.9", "2.9"): "cu128",
    ("13.0", "2.9"): "cu130",  # CUDA 13.0 for RTX 50 series
}

# Map (CUDA major.minor, PyTorch major.minor) -> wheel subdirectory
# Static mapping ensures any patch version (2.9.0, 2.9.1, etc.) finds the right wheels
WHEEL_DIRS = {
    ("12.4", "2.5"): "cu124-torch251",
    ("12.6", "2.6"): "cu126-torch260",
    ("12.6", "2.8"): "cu126-torch280",
    ("12.8", "2.8"): "cu128-torch280",
    ("12.8", "2.9"): "cu128-torch291",
    ("12.9", "2.8"): "cu128-torch280",  # CUDA 12.9 uses 12.8 wheels
    ("12.9", "2.9"): "cu128-torch291",
    ("13.0", "2.9"): "cu130-torch291",  # CUDA 13.0 for RTX 50 series
}


# =============================================================================
# Flash Attention Sources
# =============================================================================

# Flash attention wheel sources - tried in order until one works
# Different repos use different naming conventions
FLASH_ATTN_SOURCES = [
    {
        # mjun0812 v0.7.2 has Linux wheels with cu128
        "name": "mjun0812",
        "base_url": "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download",
        "versions": [("2.8.3", "v0.7.2"), ("2.7.4", "v0.7.2"), ("2.6.3", "v0.7.2")],
        # Format: flash_attn-{ver}+cu{cuda}torch{torch}-cp{py}-cp{py}-{platform}.whl
        "format": "simple",
        "platforms": ["win_amd64", "linux_x86_64"],
    },
    {
        "name": "bdashore3",
        "base_url": "https://github.com/bdashore3/flash-attention/releases/download",
        "versions": [("2.8.3", "v2.8.3")],
        # Format: flash_attn-{ver}+cu{cuda}torch{torch}.0cxx11abiFALSE-cp{py}-cp{py}-{platform}.whl
        "format": "cxx11abi",
        "platforms": ["win_amd64", "linux_x86_64"],
    },
    {
        "name": "oobabooga",
        "base_url": "https://github.com/oobabooga/flash-attention/releases/download",
        "versions": [("2.7.4.post1", "v2.7.4.post1"), ("2.7.3", "v2.7.3")],
        # Format: flash_attn-{ver}+cu{cuda}torch{torch}.0cxx11abiFALSE-cp{py}-cp{py}-{platform}.whl
        "format": "cxx11abi",
        "platforms": ["win_amd64", "linux_x86_64"],
    },
]

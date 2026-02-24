# ComfyUI-TRELLIS2 Issues & Fixes Tracker

## Overview

This document tracks all known issues reported via GitHub Issues, Reddit, and code analysis.
Last updated: 2025-12-21

---

## CATEGORY 1: Wheel Build & Distribution Issues

### 1.1 o_voxel METADATA contains git URL dependencies [CRITICAL]
**Status:** Open
**Impact:** Breaks installation even when dependencies are installed
**Source:** Code analysis

The o_voxel wheel's METADATA contains:
```
Requires-Dist: cumesh@ git+https://github.com/JeffreyXiang/CuMesh.git
Requires-Dist: flex_gemm@ git+https://github.com/JeffreyXiang/FlexGEMM.git
```

This forces pip to clone from git even when cumesh/flex_gemm are already installed, causing installation failures.

**Fix:** Modify o_voxel's pyproject.toml to remove git URL dependencies or use plain package names.

---

### 1.2 flex_gemm requires `triton` instead of `triton-windows` [CRITICAL]
**Status:** Open
**Impact:** Breaks all Windows installations
**Source:** Reddit, GitHub Issues, Code analysis

The flex_gemm wheel METADATA has:
```
Requires-Dist: triton >=3.2.0
```

On Windows, the package is named `triton-windows`, not `triton`.

**Fix:** Use conditional dependency in pyproject.toml:
```toml
dependencies = [
    "triton>=3.2.0; platform_system != 'Windows'",
    "triton-windows>=3.2.0; platform_system == 'Windows'",
]
```

---

### 1.3 Missing CUDA 13.0 wheels [CRITICAL]
**Status:** Open
**Impact:** RTX 50 series (5090, 5080, etc.) users cannot install
**Source:** GitHub Issue #7, Reddit

No wheels exist for CUDA 13.0. New NVIDIA drivers and RTX 50 series GPUs use CUDA 13.0.

**Fix:** Add GitHub Actions workflow for `cu130-torch2XX` builds.

---

### 1.4 Missing PyTorch 2.9.x wheels [HIGH]
**Status:** Partial (some repos have cu128-torch291, others don't)
**Impact:** ComfyUI portable ships PyTorch 2.9.1, many users affected
**Source:** GitHub Issues, Reddit

Current wheel availability:
- ovoxel-wheels: Has cu128-torch291 ✓
- flexgemm-wheels: Has cu128-torch291 ✓
- cumesh-wheels: Has cu128-torch280 only ✗
- nvdiffrast-full-wheels: Needs verification
- nvdiffrec_render-wheels: Needs verification

**Fix:** Ensure all repos have cu128-torch291 builds.

---

### 1.5 Missing Python 3.13 wheels [MEDIUM]
**Status:** Open
**Impact:** Users on Python 3.13 cannot install
**Source:** GitHub Issue #7, Reddit

Python 3.13 is becoming more common. No pre-built wheels available.

**Fix:** Add Python 3.13 to build matrix in all wheel repos.

---

### 1.6 o_voxel compiled incorrectly [HIGH]
**Status:** Open (per maintainer)
**Impact:** Unknown runtime issues
**Source:** Maintainer knowledge

Maintainer notes o_voxel wheels are compiled wrong. Needs investigation.

**Fix:** Review o_voxel build process and fix compilation issues.

---

### 1.7 Inconsistent wheel naming across repos [LOW]
**Status:** Open
**Impact:** Confusion, potential pip resolution issues
**Source:** Maintainer request

Wheel naming should be standardized closer to numpy conventions.

Current naming:
- `flex_gemm-0.0.1-cp312-cp312-linux_x86_64.whl` (underscore)
- `cumesh-0.0.1-cp312-cp312-linux_x86_64.whl` (no underscore)
- `o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl` (underscore)

**Fix:** Standardize naming convention across all packages.

---

## CATEGORY 2: install.py Issues

### 2.1 install.py doesn't use --no-deps [CRITICAL]
**Status:** Open
**Impact:** Dependency resolution failures during wheel install
**Source:** Code analysis

In `try_install_from_direct_url()` (line ~570):
```python
result = subprocess.run([
    sys.executable, "-m", "pip", "install", wheel_url
], ...)
```

Missing `--no-deps` flag causes pip to try resolving dependencies (which includes the broken git URLs in o_voxel).

**Fix:** Add `--no-deps` to pip install command for wheel installations.

---

### 2.2 No fallback for user-side compilation [MEDIUM]
**Status:** Open
**Impact:** Users without matching wheels cannot easily compile
**Source:** Maintainer request

Need automatic/semi-automatic way for users to compile wheels on their machine when pre-built wheels don't exist.

**Fix:** Create a compile helper script that:
1. Detects user environment (CUDA, PyTorch, Python)
2. Checks for matching pre-built wheel
3. If none, guides user through local compilation

---

## CATEGORY 3: Runtime Errors

### 3.1 HuggingFace 401/404 model download errors [HIGH]
**Status:** Open
**Impact:** Model loading fails completely
**Source:** GitHub Issue #12, Reddit

Error message:
```
Repository Not Found for url: https://huggingface.co/ckpts/shape_dec_next_dc_f16c32_fp16/resolve/main/.json
```

The code is constructing malformed HuggingFace URLs, treating config names as repo paths.

**Fix:** Review and fix HuggingFace model path construction in the codebase.

---

### 3.2 cumesh CUDA error - "invalid configuration argument" [HIGH]
**Status:** Open
**Impact:** Only works once, then fails on subsequent runs
**Source:** GitHub Issue #11, Reddit

Error:
```
[CuMesh] CUDA error:
    File: .../CuMesh/src/remesh/svox2vert.cu
    Line: 174
    Error code: 9
    Error text: invalid configuration argument
```

Works first run, fails on subsequent runs until ComfyUI restart.

**Fix:** Investigate CUDA kernel configuration in cumesh. May be related to incorrect o_voxel compilation.

---

### 3.3 "WARNING TOO BIG (stack overflow)" in texture workflow [HIGH]
**Status:** Open
**Impact:** Texture generation fails
**Source:** GitHub Issue #10

Reported on RTX 3090 with 48GB RAM. Mesh reconstruction works, texture workflow fails.

**Fix:** Investigate memory management in texture pipeline.

---

### 3.4 DINOv3 loading hangs [MEDIUM]
**Status:** Open
**Impact:** Workflow stalls during model loading
**Source:** Reddit

Gets stuck at:
```
Loading DINOv3 model: PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m...
```

**Fix:** Investigate DINOv3 model loading, check for network issues or model availability.

---

### 3.5 'NoneType' object is not callable [MEDIUM]
**Status:** Open
**Impact:** Trellis2ImageToShape node fails
**Source:** GitHub Issue (Basic-Specific2447), Reddit

Runtime error in shape generation node.

**Fix:** Debug Trellis2ImageToShape node, likely missing model or failed initialization.

---

## CATEGORY 4: Output Quality Issues

### 4.1 Weird spikes/artifacts in generated models [MEDIUM]
**Status:** Open
**Impact:** Poor quality 3D output, especially with textures
**Source:** Reddit (Educational_Smell292, Muddled-Neurons)

Users report:
- "weird spikes in the generated model"
- "geometry is off"
- Non-textured models are fine, textured models have spikes

**Fix:** May be related to o_voxel compilation issues or cumesh remeshing bugs.

---

### 4.2 Holes in meshes [LOW]
**Status:** Open
**Impact:** Incomplete 3D geometry
**Source:** Reddit (Rizzlord)

Generated meshes have holes.

**Fix:** Investigate mesh generation and remeshing pipeline.

---

## CATEGORY 5: Documentation & UX Issues

### 5.1 Unclear installation instructions [MEDIUM]
**Status:** Open
**Impact:** Users struggle to install manually
**Source:** Reddit, GitHub Issues

Need clearer documentation for:
- Manual installation steps
- Which models go where (GitHub Issue #8)
- Troubleshooting common errors

---

### 5.2 ComfyUI Manager security settings required [LOW]
**Status:** Open
**Impact:** Installation blocked by default security
**Source:** Reddit

Users need to set `security_level = weak` in ComfyUI Manager config.

**Fix:** Document this requirement or work with ComfyUI Manager team.

---

### 5.3 Nodes don't appear after Manager install [MEDIUM]
**Status:** Open
**Impact:** Users think installation failed
**Source:** GitHub Issue #5, #6, Reddit

After installing via ComfyUI Manager, nodes don't appear. Users need to manually run `python install.py`.

**Fix:** Ensure install.py is properly triggered by Manager, or document manual step.

---

## CATEGORY 6: Platform-Specific Issues

### 6.1 Windows MSVC compiler not found [MEDIUM]
**Status:** Open
**Impact:** Windows users cannot compile from source
**Source:** GitHub Issue #7, Reddit

When pre-built wheels don't exist, compilation fails because MSVC is not found.

**Fix:**
1. Provide more pre-built wheels
2. Better error messages with installation instructions
3. Consider providing install_compilers.py helper

---

### 6.2 Linux compilation requires specific gcc version [LOW]
**Status:** Open
**Impact:** Some Linux users have compilation issues
**Source:** Code analysis

Build may require gcc-11/g++-11 for ABI compatibility.

**Fix:** Document compiler requirements, consider static linking.

---

### 6.3 AMD/ROCm not supported [LOW]
**Status:** Known limitation
**Impact:** AMD GPU users cannot use TRELLIS2
**Source:** Reddit (DrBearJ3w)

TRELLIS2 requires NVIDIA CUDA.

**Fix:** Document as known limitation. ROCm support would require significant work.

---

## Priority Matrix

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| P0 | 1.1 o_voxel git URL deps | Critical | Low |
| P0 | 2.1 install.py --no-deps | Critical | Low |
| P0 | 1.2 flex_gemm triton-windows | Critical | Low |
| P1 | 1.3 CUDA 13.0 wheels | High | Medium |
| P1 | 3.1 HuggingFace paths | High | Medium |
| P1 | 1.4 PyTorch 2.9.x wheels | High | Medium |
| P1 | 3.2 cumesh CUDA error | High | High |
| P1 | 1.6 o_voxel compilation | High | High |
| P2 | 1.5 Python 3.13 wheels | Medium | Medium |
| P2 | 3.3 Stack overflow texture | Medium | High |
| P2 | 4.1 Geometry spikes | Medium | High |
| P2 | 2.2 User compile helper | Medium | Medium |
| P3 | 5.x Documentation | Low | Low |

---

## Quick Wins (Can fix immediately)

1. **Add `--no-deps` to install.py** - 1 line change
2. **Document the issues** - This file
3. **Fix install order** - Ensure dependencies installed before dependents

## Next Steps

1. Fix install.py (--no-deps)
2. Rebuild o_voxel wheels with fixed METADATA
3. Rebuild flex_gemm wheels with triton-windows conditional
4. Add CUDA 13.0 build workflows
5. Investigate HuggingFace path issue
6. Investigate cumesh CUDA error

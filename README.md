# ComfyUI-TRELLIS2

ComfyUI custom nodes for [TRELLIS.2](https://github.com/microsoft/TRELLIS) - Microsoft's state-of-the-art image-to-3D generation model.

Generate high-quality 3D meshes with PBR (Physically Based Rendering) materials from a single image.

## Features

- **Image to 3D Generation**: Convert any image to a detailed 3D mesh
- **PBR Materials**: Generated meshes include base color, metallic, and roughness maps
- **Multiple Resolutions**: Support for 512, 1024, and 1536 resolution outputs
- **GLB Export**: Export meshes to industry-standard GLB format
- **Preview Rendering**: Render preview images and videos of generated meshes
- **Background Removal**: Automatic background removal using BiRefNet

## Installation

### Prerequisites

1. **TRELLIS.2**: You must have TRELLIS.2 installed. Clone it alongside your ComfyUI installation:
   ```bash
   git clone https://github.com/microsoft/TRELLIS TRELLIS.2
   cd TRELLIS.2
   pip install -r requirements.txt
   ```

2. **o_voxel**: Required for GLB export (included with TRELLIS.2 setup)

### Install via ComfyUI Manager (Recommended)

Search for "ComfyUI-TRELLIS2" in ComfyUI Manager and click install.

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-TRELLIS2
cd ComfyUI-TRELLIS2
pip install -r requirements.txt
```

## Nodes

### (Down)Load TRELLIS.2 Model
Loads the TRELLIS.2 pipeline from HuggingFace. Models are automatically downloaded on first use.

**Inputs:**
- `model`: Model variant (currently `microsoft/TRELLIS.2-4B`)
- `low_vram`: Enable low-VRAM mode (recommended)

### TRELLIS.2 Sampler Parameters
Configure advanced sampling parameters for the three generation stages.

### TRELLIS.2 Preprocess Image
Preprocesses input image (background removal, cropping).

### TRELLIS.2 Image to 3D
Main generation node - converts image to 3D mesh.

**Inputs:**
- `pipeline`: Loaded TRELLIS.2 pipeline
- `image`: Input image (RGB or RGBA)
- `seed`: Random seed for reproducibility
- `resolution`: Output resolution (512, 1024, 1536)
- `preprocess_image`: Enable automatic preprocessing

**Outputs:**
- `mesh`: Generated 3D mesh with PBR materials
- `latent`: Latent codes for further manipulation

### TRELLIS.2 Export GLB
Export mesh to GLB format with baked PBR textures.

### TRELLIS.2 Render Preview
Render multi-view preview images.

### TRELLIS.2 Render Video
Render a rotating video of the 3D mesh.

## Example Workflow

![Simple Workflow](docs/simple.png)

1. Load Image
2. Load TRELLIS.2 Model
3. (Optional) Configure Sampler Parameters
4. Run Image to 3D
5. Export GLB or Render Preview

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU with 8GB+ VRAM (16GB+ recommended)
- TRELLIS.2 and its dependencies

## License

MIT License - See LICENSE file for details.

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS) by Microsoft Research
- Inspired by [ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3)

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

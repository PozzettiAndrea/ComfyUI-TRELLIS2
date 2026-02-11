# ComfyUI-TRELLIS2

ComfyUI custom nodes for [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) - Microsoft's state-of-the-art image-to-3D generation model.

Generate high-quality 3D meshes with PBR (Physically Based Rendering) materials from a single image.

### Install via ComfyUI Manager (Recommended)

Search for "ComfyUI-TRELLIS2" in ComfyUI Manager and click install.

## Example Workfloww

![tpose](docs/tpose.png)

![rmbg](docs/rmbg.png)


https://github.com/user-attachments/assets/e28e4a74-b119-4303-8e30-63361f26aa88


## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU with 8GB VRAM (16GB+ recommended)

All dependencies should theoretically install through wheel by running install.py!
This node was made for comfyui-manager install, but you can also pip install requirements and "python install.py" manually.

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

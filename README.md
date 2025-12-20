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

## Notes

The video render nodes will be implemented soon! Not yet working.

TRELLIS2 Checkpoints (downloaded from microsoft/TRELLIS-image-large)

```text
ComfyUI/models/trellis2/
├── ss_dec_conv3d_16l8_fp16/          # sparse structure decoder
├── ss_flow_img_dit_1_3B_64_bf16/     # sparse structure flow
├── shape_dec_next_dc_f16c32_fp16/    # shape decoder
├── slat_flow_img2shape_dit_1_3B_512_bf16/   # shape flow 512
├── slat_flow_img2shape_dit_1_3B_1024_bf16/  # shape flow 1024
├── tex_dec_next_dc_f16c32_fp16/      # texture decoder
└── slat_flow_imgshape2tex_dit_1_3B_1024_bf16/  # texture flow 1024

HuggingFace Models

ComfyUI/models/dinov3/
└── models--PIA-SPACE-LAB--dinov3-vitl-pretrain-lvd1689m/

ComfyUI/models/birefnet/
└── models--ZhengPeng7--BiRefNet/
```

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

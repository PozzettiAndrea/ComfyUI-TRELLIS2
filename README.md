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

## Manual Model Download

Models are downloaded automatically from HuggingFace on first run. If you need to download manually:

**From [microsoft/TRELLIS-image-large/ckpts](https://huggingface.co/microsoft/TRELLIS-image-large/tree/main/ckpts):**
- `ss_dec_conv3d_16l8_fp16.json` + `ss_dec_conv3d_16l8_fp16.safetensors`

**From [microsoft/TRELLIS.2-4B/ckpts](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/main/ckpts):**
- `ss_flow_img_dit_1_3B_64_bf16.json` + `.safetensors`
- `shape_dec_next_dc_f16c32_fp16.json` + `.safetensors`
- `slat_flow_img2shape_dit_1_3B_512_bf16.json` + `.safetensors`
- `slat_flow_img2shape_dit_1_3B_1024_bf16.json` + `.safetensors`
- `tex_dec_next_dc_f16c32_fp16.json` + `.safetensors`
- `slat_flow_imgshape2tex_dit_1_3B_512_bf16.json` + `.safetensors`
- `slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json` + `.safetensors`

**Place files in:** `ComfyUI/models/trellis2/ckpts/`

```text
ComfyUI/models/trellis2/
└── ckpts/
    ├── ss_dec_conv3d_16l8_fp16.json
    ├── ss_dec_conv3d_16l8_fp16.safetensors
    ├── ss_flow_img_dit_1_3B_64_bf16.json
    ├── ss_flow_img_dit_1_3B_64_bf16.safetensors
    ├── shape_dec_next_dc_f16c32_fp16.json
    ├── shape_dec_next_dc_f16c32_fp16.safetensors
    ├── ... (etc)
```

**Other HuggingFace models (auto-downloaded):**
- DinoV3: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- BiRefNet: `briaai/RMBG-2.0`

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

Thank you @fire for pointing out the segment_reduce bug!

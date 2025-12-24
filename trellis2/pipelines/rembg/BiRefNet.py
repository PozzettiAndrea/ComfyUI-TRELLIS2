from typing import *
import os
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image

# Remap gated models to public alternatives
RMBG_MODEL_REMAP = {
    "briaai/RMBG-2.0": "ZhengPeng7/BiRefNet",
}


def _is_offline_mode() -> bool:
    """Check if offline mode is enabled via HF_HUB_OFFLINE environment variable."""
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1"


def _is_model_cached(model_name: str, cache_dir: str) -> bool:
    """Check if a HuggingFace model is already cached locally."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import _CACHED_NO_EXIST
        cached = try_to_load_from_cache(model_name, "config.json", cache_dir=cache_dir)
        return cached is not None and cached != _CACHED_NO_EXIST
    except Exception:
        return False


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # Remap gated models to public reuploads
        actual_model_name = RMBG_MODEL_REMAP.get(model_name, model_name)
        if actual_model_name != model_name:
            print(f"[ComfyUI-TRELLIS2] Remapping {model_name} -> {actual_model_name}")

        # Use ComfyUI models directory for cache
        import folder_paths
        cache_dir = os.path.join(folder_paths.models_dir, "birefnet")
        os.makedirs(cache_dir, exist_ok=True)

        # Use local_files_only if model is cached or offline mode is enabled
        local_files_only = _is_offline_mode() or _is_model_cached(actual_model_name, cache_dir)
        if local_files_only:
            print(f"[ComfyUI-TRELLIS2] Loading BiRefNet model from cache: {actual_model_name}...")
        else:
            print(f"[ComfyUI-TRELLIS2] Downloading BiRefNet model: {actual_model_name}...")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            actual_model_name, trust_remote_code=True, cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        print(f"[ComfyUI-TRELLIS2] BiRefNet model loaded successfully")
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    
from typing import *
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
try:
    from transformers import DINOv3ViTModel
except ImportError:
    raise ImportError(
        "DINOv3ViTModel requires transformers>=4.56.0. "
        "Please upgrade: pip install --upgrade transformers"
    )
import numpy as np
from PIL import Image

# Remap gated Facebook models to public reuploads
DINOV3_MODEL_REMAP = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m",
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


class DinoV2FeatureExtractor:
    """
    Feature extractor for DINOv2 models.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
    
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    

class DinoV3FeatureExtractor:
    """
    Feature extractor for DINOv3 models.
    """
    def __init__(self, model_name: str, image_size=512):
        # Remap gated models to public reuploads
        actual_model_name = DINOV3_MODEL_REMAP.get(model_name, model_name)
        if actual_model_name != model_name:
            print(f"[ComfyUI-TRELLIS2] Remapping {model_name} -> {actual_model_name}")
        self.model_name = model_name

        # Use ComfyUI models directory for cache
        import folder_paths
        cache_dir = os.path.join(folder_paths.models_dir, "dinov3")
        os.makedirs(cache_dir, exist_ok=True)

        # Use local_files_only if model is cached or offline mode is enabled
        local_files_only = _is_offline_mode() or _is_model_cached(actual_model_name, cache_dir)
        if local_files_only:
            print(f"[ComfyUI-TRELLIS2] Loading DINOv3 model from cache: {actual_model_name}...")
        else:
            print(f"[ComfyUI-TRELLIS2] Downloading DINOv3 model: {actual_model_name}...")

        self.model = DINOv3ViTModel.from_pretrained(
            actual_model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        print(f"[ComfyUI-TRELLIS2] DINOv3 model loaded successfully")
        self.model.eval()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)

        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
        
    @torch.no_grad()
    def __call__(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features from the image.
        
        Args:
            image: A batch of images as a tensor of shape (B, C, H, W) or a list of PIL images.
        
        Returns:
            A tensor of shape (B, N, D) where N is the number of patches and D is the feature dimension.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.transform(image).cuda()
        features = self.extract_features(image)
        return features

from typing import *
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
try:
    from transformers import DINOv3ViTModel, DINOv3ViTConfig
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

# Embedded config for DINOv3 ViT-L (avoids needing config.json download)
DINOV3_VITL_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "mlp_ratio": 4,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-6,
    "image_size": 512,
    "patch_size": 16,
    "num_channels": 3,
    "qkv_bias": True,
    "layerscale_value": 1e-5,
    "drop_path_rate": 0.4,
    "use_swiglu_ffn": True,
    "num_register_tokens": 4,
    "interpolate_pos_encoding": True,
    "interpolate_offset": 0.0,
    "model_type": "dinov3_vit",
}

# Clean local safetensors filenames to check (in order of preference)
LOCAL_SAFETENSORS_NAMES = [
    "dinov3-vitl-pretrain.safetensors",
    "dinov3-vitl.safetensors",
    "model.safetensors",
]


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


def _find_local_safetensors(cache_dir: str) -> Optional[str]:
    """
    Check for clean local safetensors file in cache_dir.
    Returns the path if found, None otherwise.
    """
    for name in LOCAL_SAFETENSORS_NAMES:
        path = os.path.join(cache_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _load_dinov3_from_safetensors(safetensors_path: str) -> DINOv3ViTModel:
    """
    Load DINOv3 ViT-L model from a single safetensors file.
    Uses embedded config to avoid needing config.json.
    """
    from safetensors.torch import load_file

    # Create model from embedded config
    config = DINOv3ViTConfig(**DINOV3_VITL_CONFIG)
    model = DINOv3ViTModel(config)

    # Load weights from safetensors
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=True)

    return model


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

    Supports loading from:
    1. Clean local safetensors file (preferred): models/dinov3/dinov3-vitl-pretrain.safetensors
    2. HuggingFace cache (fallback): downloads to models/dinov3/models--PIA-SPACE-LAB--...
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

        # Priority 1: Check for clean local safetensors file
        print(f"[TRELLIS2] Checking for local DINOv3 safetensors in: {cache_dir}", file=sys.stderr, flush=True)
        local_safetensors = _find_local_safetensors(cache_dir)
        if local_safetensors:
            print(f"[TRELLIS2] Loading DINOv3 from local safetensors: {local_safetensors}", file=sys.stderr, flush=True)
            self.model = _load_dinov3_from_safetensors(local_safetensors)
            print(f"[ComfyUI-TRELLIS2] DINOv3 model loaded successfully")
        else:
            # Priority 2: Use HuggingFace (cache or download)
            local_files_only = _is_offline_mode() or _is_model_cached(actual_model_name, cache_dir)
            if local_files_only:
                print(f"[ComfyUI-TRELLIS2] Loading DINOv3 from HF cache: {actual_model_name}...")
            else:
                print(f"[ComfyUI-TRELLIS2] Downloading DINOv3 model: {actual_model_name}...")
                print(f"[ComfyUI-TRELLIS2] TIP: For cleaner storage, download model.safetensors directly to {cache_dir}/")

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

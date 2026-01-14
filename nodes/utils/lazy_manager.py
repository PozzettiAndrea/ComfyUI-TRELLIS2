"""
On-demand model loading manager for TRELLIS2.

Loads models only when needed and can unload them after use.
This runs inside the isolated subprocess.
"""

import sys
import gc
from pathlib import Path
from typing import Optional

import torch


# Global model manager instance
_LAZY_MANAGER: Optional["LazyModelManager"] = None


def get_model_manager(
    model_name: str = "microsoft/TRELLIS.2-4B",
    resolution: str = "1024_cascade",
    attn_backend: str = "flash_attn",
    vram_mode: str = "keep_loaded",
) -> "LazyModelManager":
    """
    Get or create the global model manager.

    Args:
        model_name: HuggingFace model name
        resolution: Output resolution mode
        attn_backend: Attention backend
        vram_mode: VRAM usage mode (normal, low, minimal)

    Returns:
        LazyModelManager instance
    """
    global _LAZY_MANAGER

    if _LAZY_MANAGER is None:
        _LAZY_MANAGER = LazyModelManager(model_name, resolution, attn_backend, vram_mode)
    elif (_LAZY_MANAGER.model_name != model_name or
          _LAZY_MANAGER.resolution != resolution or
          _LAZY_MANAGER.vram_mode != vram_mode):
        # Config changed, recreate manager
        _LAZY_MANAGER.cleanup()
        _LAZY_MANAGER = LazyModelManager(model_name, resolution, attn_backend, vram_mode)

    return _LAZY_MANAGER


# Shape models needed for each resolution mode
SHAPE_MODELS_BY_RESOLUTION = {
    '512': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
    ],
    '1024_cascade': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
    ],
    '1536_cascade': [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
    ],
}

# Texture models needed for each resolution mode
TEXTURE_MODELS_BY_RESOLUTION = {
    '512': [
        'tex_slat_decoder',
        'tex_slat_flow_model_512',
    ],
    '1024_cascade': [
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ],
    '1536_cascade': [
        'tex_slat_decoder',
        'tex_slat_flow_model_1024',
    ],
}

# Texture resolution mapping (texture maxes at 1024)
TEXTURE_RESOLUTION_MAP = {
    '512': '512',
    '1024_cascade': '1024_cascade',
    '1536_cascade': '1024_cascade',
}


class LazyModelManager:
    """
    Lazy loading manager for TRELLIS2 models.

    Loads models only when needed and unloads them after use.
    Each subprocess call creates a fresh manager, ensuring clean state.

    VRAM modes:
    - keep_loaded: Keep all models in VRAM (fastest, ~12GB VRAM)
    - cpu_offload: Offload unused models to CPU RAM (~3-4GB VRAM)
    - disk_offload: Delete unused models, reload from disk (~3GB VRAM)
    """

    def __init__(
        self,
        model_name: str = "microsoft/TRELLIS.2-4B",
        resolution: str = "1024_cascade",
        attn_backend: str = "flash_attn",
        vram_mode: str = "keep_loaded",
    ):
        self.model_name = model_name
        self.resolution = resolution
        self.attn_backend = attn_backend
        self.vram_mode = vram_mode

        # Track loaded models
        self.dinov3_model = None
        self.shape_pipeline = None
        self.texture_pipeline = None

        # Setup attention backend
        self._setup_attention_backend()

        vram_desc = {
            "keep_loaded": "keep all models in VRAM",
            "cpu_offload": "offload unused to CPU",
            "disk_offload": "offload unused to disk"
        }.get(vram_mode, vram_mode)
        print(f"[TRELLIS2] LazyModelManager initialized (resolution={resolution}, vram_mode={vram_mode}: {vram_desc})", file=sys.stderr)

    def _setup_attention_backend(self):
        """Setup attention backend before any model loading."""
        # Block sageattn - not yet implemented
        if self.attn_backend == "sageattn":
            raise NotImplementedError("sage_attn not yet implemented!")

        try:
            from trellis2.modules.attention import config as dense_config
            from trellis2.modules.sparse import config as sparse_config
            dense_config.set_backend(self.attn_backend)
            sparse_config.set_attn_backend(self.attn_backend)
            print(f"[TRELLIS2] Attention backend set to: {self.attn_backend}", file=sys.stderr)
        except Exception as e:
            print(f"[TRELLIS2] Warning: Could not set attention backend: {e}", file=sys.stderr)

    def get_dinov3(self, device: torch.device) -> "DinoV3FeatureExtractor":
        """Load DinoV3 model on demand."""
        if self.dinov3_model is None:
            from trellis2.modules import image_feature_extractor
            print(f"[TRELLIS2] Loading DinoV3 feature extractor...", file=sys.stderr)
            self.dinov3_model = image_feature_extractor.DinoV3FeatureExtractor(
                model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"
            )
            print(f"[TRELLIS2] DinoV3 loaded successfully", file=sys.stderr)

        self.dinov3_model.to(device)
        return self.dinov3_model

    def unload_dinov3(self):
        """Unload DinoV3 to free VRAM."""
        if self.dinov3_model is not None:
            self.dinov3_model.cpu()
            self.dinov3_model = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[TRELLIS2] DinoV3 offloaded", file=sys.stderr)

    def get_shape_pipeline(self, device: torch.device) -> "Trellis2ImageTo3DPipeline":
        """Load shape pipeline on demand."""
        if self.shape_pipeline is None:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            shape_models = SHAPE_MODELS_BY_RESOLUTION.get(
                self.resolution, SHAPE_MODELS_BY_RESOLUTION['1024_cascade']
            )

            # Enable disk offload for disk_offload mode (models deleted after use, reloaded from disk)
            enable_disk_offload = (self.vram_mode == "disk_offload")

            print(f"[TRELLIS2] Loading shape pipeline...", file=sys.stderr)
            self.shape_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                self.model_name,
                models_to_load=shape_models,
                enable_disk_offload=enable_disk_offload,
            )
            self.shape_pipeline.default_pipeline_type = self.resolution
            self.shape_pipeline._device = device

            # Set keep_model_loaded based on vram_mode:
            # - keep_loaded: True (keep all models in VRAM)
            # - cpu_offload/disk_offload: False (unload models after each operation)
            self.shape_pipeline.keep_model_loaded = (self.vram_mode == "keep_loaded")

            if self.vram_mode != "keep_loaded":
                print(f"[TRELLIS2] Shape pipeline: progressive loading enabled (vram_mode={self.vram_mode})", file=sys.stderr)
                print(f"[TRELLIS2] Models will be loaded on-demand and unloaded after use to minimize VRAM", file=sys.stderr)
                # Enable low_vram mode for chunked processing to reduce peak memory
                self.shape_pipeline.low_vram = True
            print(f"[TRELLIS2] Shape pipeline ready", file=sys.stderr)

        return self.shape_pipeline

    def unload_shape_pipeline(self):
        """Unload shape pipeline to free VRAM."""
        if self.shape_pipeline is not None:
            self.shape_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[TRELLIS2] Shape pipeline offloaded", file=sys.stderr)

    def get_texture_pipeline(self, device: torch.device) -> "Trellis2ImageTo3DPipeline":
        """Load texture pipeline on demand."""
        if self.texture_pipeline is None:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            texture_resolution = TEXTURE_RESOLUTION_MAP.get(self.resolution, '1024_cascade')
            texture_models = TEXTURE_MODELS_BY_RESOLUTION.get(
                self.resolution, TEXTURE_MODELS_BY_RESOLUTION['1024_cascade']
            )

            # Enable disk offload for disk_offload mode (models deleted after use, reloaded from disk)
            enable_disk_offload = (self.vram_mode == "disk_offload")

            print(f"[TRELLIS2] Loading texture pipeline...", file=sys.stderr)
            self.texture_pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                self.model_name,
                models_to_load=texture_models,
                enable_disk_offload=enable_disk_offload,
            )
            self.texture_pipeline.default_pipeline_type = texture_resolution
            self.texture_pipeline._device = device

            # Set keep_model_loaded based on vram_mode:
            # - keep_loaded: True (keep all models in VRAM)
            # - cpu_offload/disk_offload: False (unload models after each operation)
            self.texture_pipeline.keep_model_loaded = (self.vram_mode == "keep_loaded")

            if self.vram_mode != "keep_loaded":
                print(f"[TRELLIS2] Texture pipeline: progressive loading enabled (vram_mode={self.vram_mode})", file=sys.stderr)
                print(f"[TRELLIS2] Models will be loaded on-demand and unloaded after use to minimize VRAM", file=sys.stderr)
                # Enable low_vram mode for chunked processing to reduce peak memory
                self.texture_pipeline.low_vram = True
            print(f"[TRELLIS2] Texture pipeline ready", file=sys.stderr)

        return self.texture_pipeline

    def unload_texture_pipeline(self):
        """Unload texture pipeline to free VRAM."""
        if self.texture_pipeline is not None:
            self.texture_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[TRELLIS2] Texture pipeline offloaded", file=sys.stderr)

    def cleanup(self):
        """Unload all models and free VRAM."""
        self.unload_dinov3()
        self.unload_shape_pipeline()
        self.unload_texture_pipeline()
        print(f"[TRELLIS2] All models cleaned up", file=sys.stderr)

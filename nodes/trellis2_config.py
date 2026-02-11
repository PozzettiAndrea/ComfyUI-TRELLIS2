"""
TRELLIS2 model configuration holder.

This is a simple data class that holds model configuration.
Actual inference happens in @isolated decorated node methods.
"""


class Trellis2ModelConfig:
    """
    Configuration holder for TRELLIS2 models.

    This doesn't load any models - it just stores the configuration
    that @isolated node methods need to run inference.
    """

    def __init__(
        self,
        model_name: str = "microsoft/TRELLIS.2-4B",
        resolution: str = "1024_cascade",
        attn_backend: str = "auto",
        vram_mode: str = "keep_loaded",
    ):
        """
        Initialize model configuration.

        Args:
            model_name: HuggingFace model name
            resolution: Output resolution mode (512, 1024_cascade, 1536_cascade)
            attn_backend: Attention backend (auto, flash_attn, xformers, sdpa)
            vram_mode: VRAM usage mode
                - "keep_loaded": Keep all models in VRAM (fastest, ~12GB VRAM)
                - "cpu_offload": Offload unused models to CPU RAM (~3-4GB VRAM)
                - "disk_offload": Delete unused models, reload from disk (~3GB VRAM)
        """
        self.model_name = model_name
        self.resolution = resolution
        self.attn_backend = attn_backend
        self.vram_mode = vram_mode

    def __repr__(self) -> str:
        return (
            f"Trellis2ModelConfig(model={self.model_name}, "
            f"resolution={self.resolution}, attn_backend={self.attn_backend}, "
            f"vram_mode={self.vram_mode})"
        )

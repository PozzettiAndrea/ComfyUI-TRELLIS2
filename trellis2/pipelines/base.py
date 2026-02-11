from typing import *
import gc
import torch
import torch.nn as nn
from .. import models
from ..utils.disk_offload import DiskOffloadManager


def _get_trellis2_models_dir():
    """Get the ComfyUI/models/trellis2 directory."""
    import os
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    except ImportError:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "trellis2")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        disk_offload_manager: DiskOffloadManager = None,
    ):
        if models is None:
            return
        self.models = models
        self.disk_offload_manager = disk_offload_manager
        self.keep_model_loaded = True  # Default: keep models on GPU
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(
        path: str,
        models_to_load: list = None,
        enable_disk_offload: bool = False
    ) -> "Pipeline":
        """
        Load a pretrained model.

        Args:
            path: Path to the model (local or HuggingFace repo)
            models_to_load: Optional list of model keys to load. If None, loads all models.
            enable_disk_offload: If True, models can be deleted and reloaded from disk.
        """
        import os
        import json
        import shutil

        is_local = os.path.exists(f"{path}/pipeline.json")

        # Check for cached pipeline.json in ComfyUI/models/trellis2
        models_dir = _get_trellis2_models_dir()
        cached_config = os.path.join(models_dir, "pipeline.json")

        if is_local:
            print(f"[ComfyUI-TRELLIS2] Loading pipeline config from local path...")
            config_file = f"{path}/pipeline.json"
        elif os.path.exists(cached_config):
            print(f"[ComfyUI-TRELLIS2] Loading pipeline config from local cache...")
            config_file = cached_config
        else:
            from huggingface_hub import hf_hub_download
            print(f"[ComfyUI-TRELLIS2] Downloading pipeline config from HuggingFace...")
            hf_config = hf_hub_download(path, "pipeline.json")
            # Cache it
            shutil.copy2(hf_config, cached_config)
            config_file = cached_config

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        # Create disk offload manager if enabled
        disk_offload_manager = DiskOffloadManager() if enable_disk_offload else None

        _models = {}
        # Filter to only load requested models
        model_items = [(k, v) for k, v in args['models'].items()
                       if models_to_load is None or k in models_to_load]
        total_models = len(model_items)

        if models_to_load:
            skipped = len(args['models']) - total_models
            print(f"[ComfyUI-TRELLIS2] Loading {total_models} models (skipping {skipped} not needed for this resolution)")

        for i, (k, v) in enumerate(model_items, 1):
            print(f"[ComfyUI-TRELLIS2] Loading model [{i}/{total_models}]: {k}...")
            try:
                _models[k] = models.from_pretrained(
                    f"{path}/{v}",
                    disk_offload_manager=disk_offload_manager,
                    model_key=k
                )
            except Exception as e:
                _models[k] = models.from_pretrained(
                    v,
                    disk_offload_manager=disk_offload_manager,
                    model_key=k
                )
            print(f"[ComfyUI-TRELLIS2] Loaded {k} successfully")

        new_pipeline = Pipeline(_models, disk_offload_manager=disk_offload_manager)
        new_pipeline._pretrained_args = args
        print(f"[ComfyUI-TRELLIS2] All {total_models} models loaded!")
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                try:
                    return next(model.parameters()).device
                except StopIteration:
                    continue  # Model might be unloaded
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            if model is not None:
                model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))

    def _load_model(self, model_key: str, device: torch.device = None) -> nn.Module:
        """
        Load a model to GPU - either move existing or recreate from disk.
        """
        if device is None:
            device = self.device

        model = self.models.get(model_key)

        # If model was deleted, recreate it from disk
        if model is None and self.disk_offload_manager is not None:
            safetensors_path = self.disk_offload_manager.get_path(model_key)
            if safetensors_path:
                # Config is same path with .json extension
                config_path = safetensors_path.replace('.safetensors', '')
                print(f"[ComfyUI-TRELLIS2] Reloading {model_key} from disk directly to {device}...")
                model = models.from_pretrained(config_path, device=str(device))
                model.eval()
                self.models[model_key] = model
                print(f"[ComfyUI-TRELLIS2] {model_key} reloaded to {device}")
        elif model is not None:
            model.to(device)

        return model

    def _unload_model(self, model_key: str) -> None:
        """
        Unload a model to free memory - just delete it entirely.
        """
        if self.keep_model_loaded:
            return  # Keep model loaded, do nothing

        model = self.models.get(model_key)
        if model is not None:
            # Delete the model entirely
            self.models[model_key] = None
            del model
            gc.collect()
            torch.cuda.empty_cache()

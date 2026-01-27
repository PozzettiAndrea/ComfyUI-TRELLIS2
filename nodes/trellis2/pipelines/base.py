from typing import *
import gc
import sys
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
            if model is not None:  # Skip None placeholders (progressive loading)
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
            enable_disk_offload: If True, models are NOT loaded upfront - they're loaded
                                 on-demand when first needed, then unloaded after use.
                                 This enables running on GPUs with limited VRAM.
        """
        import os
        import json
        import shutil

        is_local = os.path.exists(f"{path}/pipeline.json")

        # Check for cached pipeline.json in ComfyUI/models/trellis2
        models_dir = _get_trellis2_models_dir()
        cached_config = os.path.join(models_dir, "pipeline.json")

        if is_local:
            print(f"[TRELLIS2] Loading pipeline config from local path...", file=sys.stderr, flush=True)
            config_file = f"{path}/pipeline.json"
        elif os.path.exists(cached_config):
            print(f"[TRELLIS2] Loading pipeline config from local cache...", file=sys.stderr, flush=True)
            config_file = cached_config
        else:
            from huggingface_hub import hf_hub_download
            print(f"[TRELLIS2] Downloading pipeline config from HuggingFace...", file=sys.stderr, flush=True)
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
            if enable_disk_offload:
                print(f"[TRELLIS2] Registering {total_models} models for progressive loading (skipping {skipped} not needed)", file=sys.stderr, flush=True)
            else:
                print(f"[TRELLIS2] Loading {total_models} models (skipping {skipped} not needed for this resolution)", file=sys.stderr, flush=True)

        for i, (k, v) in enumerate(model_items, 1):
            # Check if v is already a full HuggingFace path (org/repo/file pattern)
            # Full paths have 3+ parts; relative paths like "ckpts/model" have only 2
            v_parts = v.split('/')
            if len(v_parts) >= 3 and not v.startswith('ckpts/'):
                # Already a full path (e.g., "microsoft/TRELLIS-image-large/ckpts/...")
                model_path = v
            else:
                # Relative path, prepend the base repo
                model_path = f"{path}/{v}"

            if enable_disk_offload:
                # PROGRESSIVE LOADING: Don't load model now, just ensure files are cached
                # and register path for on-demand loading later
                print(f"[TRELLIS2] Registering model [{i}/{total_models}]: {k}...", file=sys.stderr, flush=True)
                safetensors_path = Pipeline._ensure_model_cached(model_path, models_dir)
                disk_offload_manager.register(k, safetensors_path)
                _models[k] = None  # Placeholder - will be loaded on-demand
                print(f"[TRELLIS2] Registered {k} (will load on-demand)", file=sys.stderr, flush=True)
            else:
                # IMMEDIATE LOADING: Load model to GPU now (original behavior)
                print(f"[TRELLIS2] Loading model [{i}/{total_models}]: {k}...", file=sys.stderr, flush=True)
                _models[k] = models.from_pretrained(
                    model_path,
                    disk_offload_manager=disk_offload_manager,
                    model_key=k
                )
                print(f"[TRELLIS2] Loaded {k} successfully", file=sys.stderr, flush=True)

        new_pipeline = Pipeline(_models, disk_offload_manager=disk_offload_manager)
        new_pipeline._pretrained_args = args
        if enable_disk_offload:
            print(f"[TRELLIS2] All {total_models} models registered for progressive loading!", file=sys.stderr, flush=True)
        else:
            print(f"[TRELLIS2] All {total_models} models loaded!", file=sys.stderr, flush=True)
        return new_pipeline

    @staticmethod
    def _ensure_model_cached(model_path: str, models_dir: str) -> str:
        """
        Ensure model config and weights are cached locally.
        Returns the path to the safetensors file.

        This downloads files if needed but does NOT load them into GPU memory.
        """
        import os
        import shutil

        # Parse the path to determine if local or HuggingFace
        path_parts = model_path.split('/')

        # Check if it's a direct local path
        if os.path.exists(f"{model_path}.json") and os.path.exists(f"{model_path}.safetensors"):
            return f"{model_path}.safetensors"

        # HuggingFace path
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])

        local_config = os.path.join(models_dir, f"{model_name}.json")
        local_weights = os.path.join(models_dir, f"{model_name}.safetensors")

        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_config), exist_ok=True)

        if os.path.exists(local_config) and os.path.exists(local_weights):
            # Already cached
            return local_weights

        # Download from HuggingFace
        from huggingface_hub import hf_hub_download
        print(f"[TRELLIS2]   Downloading {model_name} config...", file=sys.stderr, flush=True)
        hf_config = hf_hub_download(repo_id, f"{model_name}.json")
        print(f"[TRELLIS2]   Downloading {model_name} weights (this may take a while)...", file=sys.stderr, flush=True)
        hf_weights = hf_hub_download(repo_id, f"{model_name}.safetensors")

        # Copy to local models folder
        print(f"[TRELLIS2]   Caching to {models_dir}...", file=sys.stderr, flush=True)
        shutil.copy2(hf_config, local_config)
        shutil.copy2(hf_weights, local_weights)

        return local_weights

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

    @property
    def low_vram(self) -> bool:
        return getattr(self, '_low_vram', False)

    @low_vram.setter
    def low_vram(self, value: bool) -> None:
        self._low_vram = value
        # Propagate to all loaded models
        for model in self.models.values():
            if model is not None and hasattr(model, 'low_vram'):
                model.low_vram = value

    def _load_model(self, model_key: str, device: torch.device = None) -> nn.Module:
        """
        Load a model to GPU - either move existing or load from disk.

        With progressive loading (disk_offload mode), models are loaded on-demand
        the first time they're needed, then unloaded after use to free VRAM.
        """
        if device is None:
            device = self.device

        model = self.models.get(model_key)

        # If model is None, load it from disk (first-time or after unload)
        if model is None and self.disk_offload_manager is not None:
            safetensors_path = self.disk_offload_manager.get_path(model_key)
            if safetensors_path:
                # Config is same path with .json extension
                config_path = safetensors_path.replace('.safetensors', '')
                mem_before = torch.cuda.memory_allocated() / 1024**2
                print(f"[TRELLIS2] Loading {model_key} to {device}... (VRAM before: {mem_before:.0f} MB)", file=sys.stderr, flush=True)
                model = models.from_pretrained(config_path, device=str(device))
                model.eval()
                # Apply low_vram setting if enabled
                if self.low_vram and hasattr(model, 'low_vram'):
                    model.low_vram = True
                self.models[model_key] = model
                # Enable activation checkpointing for memory reduction (cpu_offload or disk_offload mode)
                if not self.keep_model_loaded and hasattr(model, 'blocks'):
                    for block in model.blocks:
                        if hasattr(block, 'use_checkpoint'):
                            block.use_checkpoint = True
                    print(f"[TRELLIS2] {model_key} checkpointing enabled", file=sys.stderr, flush=True)
                mem_after = torch.cuda.memory_allocated() / 1024**2
                print(f"[TRELLIS2] {model_key} loaded (VRAM after: {mem_after:.0f} MB)", file=sys.stderr, flush=True)
        elif model is not None:
            # Model exists, just move to device if needed
            model.to(device)

        return model

    def _unload_model(self, model_key: str) -> None:
        """
        Unload a model to free VRAM.

        With progressive loading, the model is deleted entirely and will be
        reloaded from disk the next time it's needed.
        """
        if self.keep_model_loaded:
            return  # Keep model loaded, do nothing

        model = self.models.get(model_key)
        if model is not None:
            mem_before = torch.cuda.memory_allocated() / 1024**2
            reserved_before = torch.cuda.memory_reserved() / 1024**2
            print(f"[TRELLIS2] Unloading {model_key}... (allocated: {mem_before:.0f} MB, reserved: {reserved_before:.0f} MB)", file=sys.stderr, flush=True)
            # Delete the model entirely
            self.models[model_key] = None
            del model
            gc.collect()
            torch.cuda.empty_cache()
            mem_after = torch.cuda.memory_allocated() / 1024**2
            reserved_after = torch.cuda.memory_reserved() / 1024**2
            print(f"[TRELLIS2] {model_key} unloaded (allocated: {mem_after:.0f} MB, reserved: {reserved_after:.0f} MB)", file=sys.stderr, flush=True)

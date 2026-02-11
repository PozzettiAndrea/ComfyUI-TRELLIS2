from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str, models_to_load: list = None) -> "Pipeline":
        """
        Load a pretrained model.

        Args:
            path: Path to the model (local or HuggingFace repo)
            models_to_load: Optional list of model keys to load. If None, loads all models.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        print(f"[ComfyUI-TRELLIS2] Loading pipeline config from {'local' if is_local else 'HuggingFace'}...")
        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

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
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                _models[k] = models.from_pretrained(v)
            print(f"[ComfyUI-TRELLIS2] Loaded {k} successfully")

        new_pipeline = Pipeline(_models)
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
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
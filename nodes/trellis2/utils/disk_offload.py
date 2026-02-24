"""
Simple disk-based model management - delete models when not in use,
reload from disk when needed.
"""

import gc
from typing import Dict, Optional

import torch


class DiskOffloadManager:
    """
    Tracks model paths so we can delete and recreate models on demand.

    When a model is unloaded, it's deleted entirely.
    When needed again, it's recreated from the safetensors file.
    """

    def __init__(self):
        self.model_paths: Dict[str, str] = {}  # model_key -> safetensors_path

    def register(self, key: str, safetensors_path: str) -> None:
        """Register a model's safetensors path for later reloading."""
        self.model_paths[key] = safetensors_path

    def get_path(self, key: str) -> Optional[str]:
        """Get the safetensors path for a model."""
        return self.model_paths.get(key)

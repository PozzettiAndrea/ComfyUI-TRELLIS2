import importlib

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def _get_trellis2_models_dir():
    """Get the ComfyUI/models/trellis2 directory."""
    import os
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    except ImportError:
        # Fallback if folder_paths not available
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "models", "trellis2")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def from_pretrained(path: str, disk_offload_manager=None, model_key: str = None, device=None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        disk_offload_manager: Optional DiskOffloadManager for RAM-efficient loading.
                              When provided, the model's safetensors path will be registered
                              for later disk-to-GPU direct loading.
        model_key: Optional key to identify this model in the disk_offload_manager.
                   Required if disk_offload_manager is provided.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    import shutil
    from safetensors.torch import load_file

    # Check if it's a direct local path
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        # Parse HuggingFace path
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])

        # Check if cached in ComfyUI/models/trellis2
        models_dir = _get_trellis2_models_dir()
        local_config = os.path.join(models_dir, f"{model_name}.json")
        local_weights = os.path.join(models_dir, f"{model_name}.safetensors")

        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_config), exist_ok=True)

        if os.path.exists(local_config) and os.path.exists(local_weights):
            print(f"[ComfyUI-TRELLIS2]   Loading {model_name} from local cache...")
            config_file = local_config
            model_file = local_weights
        else:
            from huggingface_hub import hf_hub_download
            print(f"[ComfyUI-TRELLIS2]   Downloading {model_name} config...")
            hf_config = hf_hub_download(repo_id, f"{model_name}.json")
            print(f"[ComfyUI-TRELLIS2]   Downloading {model_name} weights...")
            hf_weights = hf_hub_download(repo_id, f"{model_name}.safetensors")

            # Copy to local models folder
            print(f"[ComfyUI-TRELLIS2]   Saving to {models_dir}...")
            shutil.copy2(hf_config, local_config)
            shutil.copy2(hf_weights, local_weights)

            config_file = local_config
            model_file = local_weights

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Auto-detect device: prefer CUDA, fallback to CPU
    if device is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[ComfyUI-TRELLIS2]   Building model: {config['name']}")
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.to(device)  # Move empty model to GPU before loading weights
    print(f"[ComfyUI-TRELLIS2]   Loading weights directly to {device}...")
    model.load_state_dict(load_file(model_file, device=device), strict=False)

    # Register with disk offload manager if provided
    if disk_offload_manager is not None:
        if model_key is None:
            raise ValueError(
                "model_key is required when disk_offload_manager is provided"
            )
        disk_offload_manager.register(model_key, model_file)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder

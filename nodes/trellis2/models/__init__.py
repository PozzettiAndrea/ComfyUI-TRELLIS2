import importlib
import logging

log = logging.getLogger("trellis2")

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


def from_pretrained(path: str, disk_offload_manager=None, model_key: str = None, device=None, dtype=None, **kwargs):
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
    import comfy.utils

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
            log.info(f"Loading {model_name} from local cache...")
            config_file = local_config
            model_file = local_weights
        else:
            # Download directly to models folder (no intermediate HF cache)
            from huggingface_hub import hf_hub_download
            log.info(f"Downloading {model_name} config...")
            hf_hub_download(repo_id, f"{model_name}.json", local_dir=models_dir)
            log.info(f"Downloading {model_name} weights (this may take a while)...")
            hf_hub_download(repo_id, f"{model_name}.safetensors", local_dir=models_dir)

            config_file = local_config
            model_file = local_weights

    with open(config_file, 'r') as f:
        config = json.load(f)

    import torch

    # Auto-detect device
    if device is None:
        import comfy.model_management
        device = str(comfy.model_management.get_torch_device())

    # Build model on meta device (zero memory, no random init)
    model_class = __getattr__(config['name'])
    log.info(f"Building model: {config['name']} (meta device)...")
    with torch.device("meta"):
        model = model_class(**config['args'], **kwargs)
    log.info(f"Loading weights directly to {device}...")
    model.load_state_dict(comfy.utils.load_torch_file(model_file, device=torch.device(device)), strict=False, assign=True)

    # Reinitialize any buffers left on meta device after assign=True loading
    # (strict=False means buffers not in the checkpoint stay on meta)
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            parent._buffers[parts[-1]] = torch.zeros_like(buf, device=device)

    if dtype is not None:
        model = model.to(dtype=dtype)
        # Sync model.dtype with weight dtype so manual_cast() uses matching precision.
        # model.dtype is a plain attribute (not a parameter) — model.to() doesn't touch it.
        if hasattr(model, 'dtype'):
            model.dtype = dtype
        log.info(f"Model {config['name']}: weights={dtype}, model.dtype={getattr(model, 'dtype', 'N/A')}")

    # Recompute derived buffers (e.g., RoPE phases) AFTER dtype conversion.
    # These are computed from model config, not stored in checkpoints.
    # Must run after model.to(dtype) because that destroys complex buffers.
    if hasattr(model, '_post_load'):
        model._post_load(torch.device(device))
        log.info(f"Recomputed derived buffers for {config['name']}")

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

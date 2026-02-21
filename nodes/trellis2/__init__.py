# Disable HuggingFace progress bars (weight loading, downloads, etc.)
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Suppress verbose HTTP request logs from huggingface_hub/httpx
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

log = logging.getLogger("trellis2")

# Model class lookup table: name → (module, class_name)
_MODEL_CLASSES = {
    # Sparse Structure
    'SparseStructureEncoder': ('.vae', 'SparseStructureEncoder'),
    'SparseStructureDecoder': ('.vae', 'SparseStructureDecoder'),
    'SparseStructureFlowModel': ('.model', 'SparseStructureFlowModel'),
    # SLat Generation
    'SLatFlowModel': ('.model', 'SLatFlowModel'),
    'ElasticSLatFlowModel': ('.model', 'ElasticSLatFlowModel'),
    # SC-VAEs
    'SparseUnetVaeEncoder': ('.vae', 'SparseUnetVaeEncoder'),
    'SparseUnetVaeDecoder': ('.vae', 'SparseUnetVaeDecoder'),
    'FlexiDualGridVaeEncoder': ('.vae', 'FlexiDualGridVaeEncoder'),
    'FlexiDualGridVaeDecoder': ('.vae', 'FlexiDualGridVaeDecoder'),
}


def _get_trellis2_models_dir():
    """Get the ComfyUI/models/trellis2 directory."""
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "trellis2")
    except ImportError:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "trellis2")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _get_model_class(name: str):
    """Resolve a model class name to the actual class."""
    import importlib
    if name not in _MODEL_CLASSES:
        raise ValueError(f"Unknown model class: {name}")
    module_path, class_name = _MODEL_CLASSES[name]
    module = importlib.import_module(module_path, __name__)
    return getattr(module, class_name)


def from_pretrained(path: str, disk_offload_manager=None, model_key: str = None, device=None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        disk_offload_manager: Optional DiskOffloadManager for RAM-efficient loading.
        model_key: Optional key to identify this model in the disk_offload_manager.
        **kwargs: Additional arguments for the model constructor.
    """
    import json
    import torch
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

    # Auto-detect device
    if device is None:
        import comfy.model_management
        device = str(comfy.model_management.get_torch_device())

    # Build model on meta device (zero memory, no random init)
    model_class = _get_model_class(config['name'])
    log.info(f"Building model: {config['name']} (meta device)...")
    with torch.device("meta"):
        model = model_class(**config['args'], **kwargs)
    log.info(f"Loading weights directly to {device}...")
    model.load_state_dict(comfy.utils.load_torch_file(model_file, device=torch.device(device)), strict=False, assign=True)
    weight_dtype = next(model.parameters()).dtype
    import sys; print(f"[TRELLIS2] Model {config['name']} loaded: weights={weight_dtype}, device={device}", file=sys.stderr)

    # Reinitialize any buffers left on meta device after assign=True loading
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            parent._buffers[parts[-1]] = torch.zeros_like(buf, device=device)

    # Recompute derived buffers (e.g., RoPE phases).
    if hasattr(model, '_post_load'):
        model._post_load(torch.device(device))
        log.info(f"Recomputed derived buffers for {config['name']}")

    # Register with disk offload manager if provided
    if disk_offload_manager is not None:
        if model_key is None:
            raise ValueError("model_key is required when disk_offload_manager is provided")
        disk_offload_manager.register(model_key, model_file)

    return model

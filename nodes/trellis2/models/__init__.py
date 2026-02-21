import importlib
import sys

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
        from ... import folder_paths_fallback as folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "trellis2")
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
            print(f"[TRELLIS2]   Loading {model_name} from local cache...", file=sys.stderr, flush=True)
            config_file = local_config
            model_file = local_weights
        else:
            # Download directly to models folder (no intermediate HF cache)
            from huggingface_hub import hf_hub_download
            print(f"[TRELLIS2]   Downloading {model_name} config...", file=sys.stderr, flush=True)
            hf_hub_download(repo_id, f"{model_name}.json", local_dir=models_dir)
            print(f"[TRELLIS2]   Downloading {model_name} weights (this may take a while)...", file=sys.stderr, flush=True)
            hf_hub_download(repo_id, f"{model_name}.safetensors", local_dir=models_dir)

            config_file = local_config
            model_file = local_weights

    with open(config_file, 'r') as f:
        config = json.load(f)

    import torch
    import torch.nn.init as init

    # Auto-detect device: prefer CUDA, fallback to CPU
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Skip ALL random weight initialization during construction:
    # 1. Monkey-patch initialize_weights (the model-level sweep)
    # 2. Patch torch.nn.init functions (catches per-layer reset_parameters in
    #    nn.Linear, nn.LayerNorm, etc.)
    # All weights are immediately overwritten by safetensors below.
    model_class = __getattr__(config['name'])
    _orig_init_weights = getattr(model_class, 'initialize_weights', None)
    if _orig_init_weights:
        model_class.initialize_weights = lambda self: None

    _init_funcs = ['normal_', 'kaiming_uniform_', 'uniform_', 'zeros_', 'ones_',
                   'kaiming_normal_', 'xavier_uniform_', 'xavier_normal_', 'constant_']
    _orig_inits = {name: getattr(init, name) for name in _init_funcs if hasattr(init, name)}
    _noop = lambda tensor, *args, **kwargs: tensor
    for name in _orig_inits:
        setattr(init, name, _noop)

    try:
        print(f"[TRELLIS2]   Building model: {config['name']} (skip_init)...", file=sys.stderr, flush=True)
        model = model_class(**config['args'], **kwargs)
    finally:
        for name, fn in _orig_inits.items():
            setattr(init, name, fn)
        if _orig_init_weights:
            model_class.initialize_weights = _orig_init_weights
    model.to(device)
    print(f"[TRELLIS2]   Loading weights directly to {device}...", file=sys.stderr, flush=True)
    model.load_state_dict(load_file(model_file, device=str(device)), strict=False)

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

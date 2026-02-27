"""
Native ComfyUI sampling nodes for TRELLIS2.

These nodes bridge TRELLIS2's SparseStructureFlowModel to work with
ComfyUI's standard KSampler via the BaseModel/ModelPatcher system.

Workflow:
  LoadTrellis2Models -> Trellis2LoadSSFlowModel -> MODEL
  Trellis2GetConditioning -> Trellis2SSConditioning -> CONDITIONING (pos/neg)
  Trellis2Empty3DLatent -> LATENT
  (optionally) Trellis2ApplyGuidanceInterval -> MODEL
  KSampler(MODEL, pos, neg, LATENT) -> LATENT
  Trellis2DecodeSSLatent -> TRELLIS2_SS_COORDS
"""
import gc
import json
import logging

import torch
import comfy.model_management
import comfy.model_patcher
import comfy.utils
from comfy_api.latest import io

log = logging.getLogger("trellis2")


class Trellis2LoadSSFlowModel(io.ComfyNode):
    """Load TRELLIS2 sparse structure flow model as ComfyUI MODEL for KSampler."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2LoadSSFlowModel",
            display_name="TRELLIS.2 Load SS Flow Model (Native)",
            category="TRELLIS2/Native",
            description="""Load the sparse structure flow model as a standard ComfyUI MODEL.

This wraps the flow model in ComfyUI's BaseModel/ModelPatcher system
so it can be driven by KSampler with any sampler/scheduler.

Connect the output MODEL to KSampler's 'model' input.
Use Trellis2SSConditioning for positive/negative conditioning.
Use Trellis2Empty3DLatent for the initial latent.""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model_config):
        from .stages import _init_config, _model_paths
        from .trellis2.supported_models import TRELLIS2SparseStructure as SSConfig

        _init_config()

        # Load state dict
        safetensors_path = _model_paths['sparse_structure_flow_model']
        config_path = safetensors_path.replace('.safetensors', '.json')

        sd = comfy.utils.load_torch_file(safetensors_path)

        with open(config_path) as f:
            model_json = json.load(f)

        # Build unet_config from model JSON args
        args = model_json['args']
        unet_config = {
            'image_model': 'trellis2_sparse_structure',
            'resolution': args['resolution'],
            'in_channels': args['in_channels'],
            'model_channels': args['model_channels'],
            'cond_channels': args['cond_channels'],
            'out_channels': args['out_channels'],
            'num_blocks': args['num_blocks'],
        }
        # Optional params
        for key in ['num_heads', 'num_head_channels', 'pe_mode', 'mlp_ratio',
                     'share_mod', 'qk_rms_norm', 'qk_rms_norm_cross']:
            if key in args:
                unet_config[key] = args[key]

        # Create model config instance
        ss_config = SSConfig(unet_config)

        # Determine dtype
        device = comfy.model_management.get_torch_device()
        weight_dtype = comfy.utils.weight_dtype(sd)
        parameters = comfy.utils.calculate_parameters(sd)
        unet_dtype = comfy.model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=ss_config.supported_inference_dtypes,
            weight_dtype=weight_dtype,
        )
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            unet_dtype, device, ss_config.supported_inference_dtypes,
        )
        ss_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

        # Create BaseModel
        model = ss_config.get_model(sd, device=device)

        # Wrap in ModelPatcher
        offload_device = comfy.model_management.unet_offload_device()
        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=device,
            offload_device=offload_device,
        )

        # Move to offload device before loading weights (standard ComfyUI pattern)
        if not comfy.model_management.is_device_cpu(offload_device):
            model.to(offload_device)

        # Load weights
        model.load_model_weights(sd, "")

        log.info(f"SS flow model loaded for KSampler: dtype={unet_dtype}")
        return io.NodeOutput(patcher)


class Trellis2SSConditioning(io.ComfyNode):
    """Convert TRELLIS2 conditioning to ComfyUI CONDITIONING format for KSampler."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2SSConditioning",
            display_name="TRELLIS.2 SS Conditioning (Native)",
            category="TRELLIS2/Native",
            description="""Convert TRELLIS2 DinoV3 conditioning to standard ComfyUI CONDITIONING.

Takes the output of 'TRELLIS.2 Get Conditioning' and produces
positive/negative conditioning compatible with KSampler.

Connect 'positive' to KSampler's positive input.
Connect 'negative' to KSampler's negative input.""",
            inputs=[
                io.Custom("TRELLIS2_CONDITIONING").Input("conditioning"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, conditioning):
        # Get 512px features (always present)
        cond = conditioning['cond_512']
        neg_cond = conditioning['neg_cond']

        # ComfyUI conditioning format: [[tensor, dict]]
        # tensor becomes "cross_attn" via convert_cond() in sampler_helpers
        positive = [[cond, {}]]
        negative = [[neg_cond, {}]]

        return io.NodeOutput(positive, negative)


class Trellis2Empty3DLatent(io.ComfyNode):
    """Create empty 3D latent tensor for sparse structure sampling."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2Empty3DLatent",
            display_name="TRELLIS.2 Empty 3D Latent",
            category="TRELLIS2/Native",
            description="""Create an empty 3D latent tensor for sparse structure sampling.

Reads the model's resolution and channel count to create the
correct shape: [batch, channels, res, res, res].

For the default TRELLIS2 model: [1, 8, 16, 16, 16].""",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("batch_size", default=1, min=1, max=4, optional=True),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, batch_size=1):
        # Get model params from the diffusion model
        diff_model = model.model.diffusion_model
        resolution = diff_model.resolution
        in_channels = diff_model.in_channels

        # Create empty 5D latent: [B, C, R, R, R]
        latent = torch.zeros(batch_size, in_channels, resolution, resolution, resolution)

        return io.NodeOutput({"samples": latent})


class Trellis2ApplyGuidanceInterval(io.ComfyNode):
    """Apply guidance interval to model for TRELLIS2 flow matching sampling.

    When guidance_interval is (0.0, 1.0), CFG is applied at every step (standard).
    Narrower intervals only apply CFG during a portion of the sampling process.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ApplyGuidanceInterval",
            display_name="TRELLIS.2 Apply Guidance Interval",
            category="TRELLIS2/Native",
            description="""Apply a guidance interval to control when CFG is active during sampling.

For TRELLIS2 flow matching, sigma goes from 1.0 (pure noise) to 0.0 (clean).
- (0.0, 1.0) = always apply CFG (default, same as standard KSampler)
- (0.2, 0.8) = only apply CFG when sigma is between 0.2 and 0.8

Outside the interval, only the conditional prediction is used (no CFG).""",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("guidance_interval_start", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Start of guidance interval (sigma value). 0.0 = apply from clean end."),
                io.Float.Input("guidance_interval_end", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="End of guidance interval (sigma value). 1.0 = apply until noise end."),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, guidance_interval_start=0.0, guidance_interval_end=1.0):
        # Clone model patcher so we don't modify the original
        m = model.clone()

        interval_start = guidance_interval_start
        interval_end = guidance_interval_end

        def guidance_interval_cfg(args):
            """Custom CFG function that only applies guidance within a sigma interval."""
            cond_eps = args["cond"]       # noise prediction from conditional
            uncond_eps = args["uncond"]    # noise prediction from unconditional
            scale = args["cond_scale"]
            sigma = args["sigma"]

            # For flow matching, sigma goes from 1.0 (pure noise) to 0.0 (clean)
            t = float(sigma.max())

            if t >= interval_start and t <= interval_end:
                # Inside guidance interval: apply standard CFG
                return uncond_eps + (cond_eps - uncond_eps) * scale
            else:
                # Outside guidance interval: no guidance
                return cond_eps

        m.model_options["sampler_cfg_function"] = guidance_interval_cfg
        return io.NodeOutput(m)


class Trellis2DecodeSSLatent(io.ComfyNode):
    """Decode sparse structure latent to voxel coordinates.

    Takes the sampled latent from KSampler and runs it through the
    SparseStructureDecoder to produce binary voxel occupancy,
    then extracts the active voxel coordinates.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2DecodeSSLatent",
            display_name="TRELLIS.2 Decode SS Latent",
            category="TRELLIS2/Native",
            description="""Decode sparse structure latent into voxel coordinates.

Takes the sampled latent from KSampler and:
1. Runs the SparseStructureDecoder (logits -> binary occupancy)
2. Optionally downsamples to target resolution via max pooling
3. Extracts active voxel coordinates

The output coords can be passed to downstream shape generation nodes.""",
            inputs=[
                io.Custom("TRELLIS2_MODEL_CONFIG").Input("model_config"),
                io.Latent.Input("samples"),
                io.Int.Input("ss_resolution", default=32, min=16, max=128, step=16, optional=True,
                             tooltip="Target sparse structure resolution. If decoder outputs higher res, maxpool downsamples."),
            ],
            outputs=[
                io.Custom("TRELLIS2_SS_COORDS").Output(display_name="coords"),
            ],
        )

    @classmethod
    def execute(cls, model_config, samples, ss_resolution=32):
        from .stages import _init_config, _load_model, _unload_model

        _init_config()

        comfy.model_management.throw_exception_if_processing_interrupted()

        latent = samples["samples"]
        device = comfy.model_management.get_torch_device()
        latent = latent.to(device)

        # Load decoder
        decoder = _load_model('sparse_structure_decoder')
        model_dtype = next(decoder.parameters()).dtype
        latent = latent.to(dtype=model_dtype)

        # Decode: output is logits [B, 1, R, R, R], threshold at 0
        decoded = decoder(latent) > 0

        _unload_model('sparse_structure_decoder')

        # Optionally downsample to target resolution
        if ss_resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // ss_resolution
            if ratio > 1:
                decoded = torch.nn.functional.max_pool3d(
                    decoded.float(), ratio, ratio, 0
                ) > 0.5

        # Extract voxel coordinates: [N, 4] with (batch_idx, x, y, z)
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        log.info(f"Decoded sparse structure: {coords.shape[0]} active voxels")

        del decoded, latent
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(coords.cpu())


NODE_CLASS_MAPPINGS = {
    "Trellis2LoadSSFlowModel": Trellis2LoadSSFlowModel,
    "Trellis2SSConditioning": Trellis2SSConditioning,
    "Trellis2Empty3DLatent": Trellis2Empty3DLatent,
    "Trellis2ApplyGuidanceInterval": Trellis2ApplyGuidanceInterval,
    "Trellis2DecodeSSLatent": Trellis2DecodeSSLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2LoadSSFlowModel": "TRELLIS.2 Load SS Flow Model (Native)",
    "Trellis2SSConditioning": "TRELLIS.2 SS Conditioning (Native)",
    "Trellis2Empty3DLatent": "TRELLIS.2 Empty 3D Latent",
    "Trellis2ApplyGuidanceInterval": "TRELLIS.2 Apply Guidance Interval",
    "Trellis2DecodeSSLatent": "TRELLIS.2 Decode SS Latent",
}

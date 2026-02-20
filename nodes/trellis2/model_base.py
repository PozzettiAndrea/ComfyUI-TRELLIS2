"""
TRELLIS2 BaseModel subclasses.

Follows the pattern in comfy/model_base.py to integrate TRELLIS2
flow models with ComfyUI's apply_model() / model_sampling system.

Heavy model imports are deferred to __init__ to avoid loading sparse
ops at registration time.
"""
import torch
import comfy.model_base
import comfy.model_management
import comfy.model_sampling
import comfy.conds
import comfy.ops
import comfy.patcher_extension
from comfy.model_base import ModelType


class TRELLIS2SparseStructure(comfy.model_base.BaseModel):
    """
    BaseModel wrapper for SparseStructureFlowModel.

    This is a dense 3D flow model (operates on regular [B,C,R,R,R] tensors).
    Uses CONST predictor + ModelSamplingDiscreteFlow (linear flow matching
    from sigma=1 to sigma=0, timesteps scaled to 0-1000).

    The model predicts velocity v, and x_0 = x_t - t * v.
    This matches ComfyUI's CONST.calculate_denoised: model_input - model_output * sigma.
    """

    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        from .model import SparseStructureFlowModel
        super().__init__(
            model_config,
            model_type,
            device=device,
            unet_model=SparseStructureFlowModel,
        )

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        """
        Override to adapt calling convention.

        SparseStructureFlowModel.forward(x, t, cond, transformer_options, **kwargs)
        expects 'cond' as 3rd positional arg, while the default BaseModel._apply_model
        passes 'context=' as keyword arg.
        """
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat([xc, comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)], dim=1)

        dtype = self.get_dtype_inference()
        xc = xc.to(dtype)
        device = xc.device
        t = self.model_sampling.timestep(t).float()

        cond = c_crossattn
        if cond is not None:
            cond = comfy.model_management.cast_to_device(cond, device, dtype)

        model_output = self.diffusion_model(xc, t, cond, transformer_options=transformer_options)

        return self.model_sampling.calculate_denoised(sigma, model_output.float(), x)

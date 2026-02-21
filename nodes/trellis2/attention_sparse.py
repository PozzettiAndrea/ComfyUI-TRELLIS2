"""
ComfyUI-native attention dispatch for TRELLIS2.

Dense attention: wraps ComfyUI's optimized_attention_for_device with TRELLIS2
layout conversion (N,L,H,C <-> B,H,N,D).

Varlen attention: native backend dispatch for variable-length sparse sequences.
Priority: sage2 > flash > xformers > sdpa (matching comfy_attn ordering).

Replaces the comfy_attn dependency entirely.

Usage:
    # Dense attention (TRELLIS2 layout):
    from .attention_sparse import scaled_dot_product_attention
    out = scaled_dot_product_attention(q, k, v)  # [N, L, H, C] tensors

    # Sparse varlen attention (VarLenTensor):
    from .attention_sparse import sparse_scaled_dot_product_attention
    out = sparse_scaled_dot_product_attention(q, k, v)  # VarLenTensor / dense mixes
"""

import logging

import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention_for_device

log = logging.getLogger("trellis2")

__all__ = [
    'scaled_dot_product_attention',
    'sparse_scaled_dot_product_attention',
    'dispatch_varlen_attention',
]


# ---------------------------------------------------------------------------
# Dense attention dispatch (TRELLIS2 layout)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(*args, **kwargs):
    """
    Scaled dot product attention for TRELLIS2 dense tensors.

    Supports 1, 2, or 3 argument forms:
        scaled_dot_product_attention(qkv)       # qkv: [N, L, 3, H, C]
        scaled_dot_product_attention(q, kv)      # q: [N, L, H, C], kv: [N, L, 2, H, C]
        scaled_dot_product_attention(q, k, v)    # each: [N, L, H, C]

    Returns: [N, L, H, C] tensor.
    """
    transformer_options = kwargs.pop('transformer_options', {})

    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        q, k, v = qkv.unbind(dim=2)

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        k, v = kv.unbind(dim=2)

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"

    # TRELLIS2 [N, L, H, C] -> ComfyUI [N, H, L, C]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    heads = q.shape[1]
    attn_fn = optimized_attention_for_device(q.device)
    out = attn_fn(q, k, v, heads=heads, skip_reshape=True, skip_output_reshape=True, transformer_options=transformer_options)

    # ComfyUI [N, H, L, C] -> TRELLIS2 [N, L, H, C]
    return out.permute(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Varlen attention backend selection
# ---------------------------------------------------------------------------

_varlen_fn = None
_varlen_backend = None


def _get_gpu_arch():
    """Return (major, minor) compute capability, cached implicitly by CUDA."""
    try:
        import comfy.model_management
        if comfy.model_management.get_torch_device().type == "cuda":
            return torch.cuda.get_device_capability()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)


def _can_import(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _resolve_varlen_backend():
    """Pick the best varlen backend. Returns (fn, name).

    Priority: sage2 > flash > xformers > sdpa
    """
    major, _ = _get_gpu_arch()

    if major >= 8 and _can_import("sageattention"):
        try:
            from sageattention import sageattn_varlen
            return sageattn_varlen, "sage2"
        except ImportError:
            pass

    if _can_import("flash_attn"):
        try:
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func, "flash"
        except ImportError:
            pass

    if _can_import("xformers"):
        return _xformers_varlen, "xformers"

    return _sdpa_varlen, "sdpa"


# ---------------------------------------------------------------------------
# Varlen backend implementations (xformers / sdpa fallbacks)
# ---------------------------------------------------------------------------

def _xformers_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                     max_seqlen_q, max_seqlen_kv, **kwargs):
    """xFormers varlen via BlockDiagonalMask."""
    import xformers.ops as xops

    B = cu_seqlens_q.shape[0] - 1
    q_seqlen = [(cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item() for i in range(B)]
    kv_seqlen = [(cu_seqlens_kv[i + 1] - cu_seqlens_kv[i]).item() for i in range(B)]
    mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
    return xops.memory_efficient_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask,
    )[0]


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                 max_seqlen_q, max_seqlen_kv, **kwargs):
    """SDPA fallback with block-diagonal additive mask."""
    B = cu_seqlens_q.shape[0] - 1
    T_q, H, D = q.shape
    T_kv = k.shape[0]

    # Build block-diagonal mask: 0 where allowed, -inf where masked
    mask = torch.full((T_q, T_kv), float("-inf"), device=q.device, dtype=q.dtype)
    for i in range(B):
        qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        ks, ke = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        mask[qs:qe, ks:ke] = 0.0

    q = q.unsqueeze(0).permute(0, 2, 1, 3)   # [1, H, T_q, D]
    k = k.unsqueeze(0).permute(0, 2, 1, 3)
    v = v.unsqueeze(0).permute(0, 2, 1, 3)
    mask = mask.unsqueeze(0).unsqueeze(0)     # [1, 1, T_q, T_kv]

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.permute(0, 2, 1, 3).squeeze(0)  # [T_q, H, D]


# ---------------------------------------------------------------------------
# Varlen attention dispatch
# ---------------------------------------------------------------------------

def dispatch_varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                              max_seqlen_q, max_seqlen_kv):
    """Dispatch variable-length attention to the best available backend.

    Args:
        q: [T_q, H, D] packed query tensor.
        k: [T_kv, H, D] packed key tensor.
        v: [T_kv, H, D_v] packed value tensor.
        cu_seqlens_q: [B+1] cumulative query sequence lengths.
        cu_seqlens_kv: [B+1] cumulative kv sequence lengths.
        max_seqlen_q: Maximum query sequence length.
        max_seqlen_kv: Maximum kv sequence length.

    Returns:
        Output tensor of shape [T_q, H, D_v].
    """
    global _varlen_fn, _varlen_backend
    if _varlen_fn is None:
        _varlen_fn, _varlen_backend = _resolve_varlen_backend()
        import sys; print(f"[TRELLIS2] Attention backend: {_varlen_backend}", file=sys.stderr)

    return _varlen_fn(
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv,
    )


# ---------------------------------------------------------------------------
# Sparse attention dispatch (VarLenTensor)
# ---------------------------------------------------------------------------

def sparse_scaled_dot_product_attention(*args, **kwargs):
    """
    Scaled dot product attention for sparse/variable-length tensors.

    Supports combinations of VarLenTensor and dense torch.Tensor inputs:
        sparse_scaled_dot_product_attention(qkv)       # qkv: VarLenTensor [N, *, 3, H, C]
        sparse_scaled_dot_product_attention(q, kv)      # mixed VarLenTensor/Tensor
        sparse_scaled_dot_product_attention(q, k, v)    # mixed VarLenTensor/Tensor

    Returns VarLenTensor if q is VarLenTensor, else dense [N, L, H, C] tensor.
    """
    from .sparse import VarLenTensor

    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, VarLenTensor), f"qkv must be a VarLenTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv_feats = qkv.feats       # [T, 3, H, C]
        q, k, v = qkv_feats.unbind(dim=1)  # each [T, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, VarLenTensor) and isinstance(kv, (VarLenTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, VarLenTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)

        if isinstance(kv, VarLenTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv_feats = kv.feats
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv_feats = kv.reshape(N * L, 2, H, C)
        k, v = kv_feats.unbind(dim=1)

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, VarLenTensor) and isinstance(k, (VarLenTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, VarLenTensor) and isinstance(v, VarLenTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)

        if isinstance(k, VarLenTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats
            v = v.feats
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)
            v = v.reshape(N * L, H, CO)

    # Build cumulative sequence lengths
    cu_seqlens_q = torch.cat([
        torch.tensor([0], device=device),
        torch.cumsum(torch.tensor(q_seqlen, device=device), dim=0)
    ]).int()
    cu_seqlens_kv = torch.cat([
        torch.tensor([0], device=device),
        torch.cumsum(torch.tensor(kv_seqlen, device=device), dim=0)
    ]).int()

    out = dispatch_varlen_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen),
    )

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)

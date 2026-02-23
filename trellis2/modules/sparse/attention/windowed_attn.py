from typing import *
import torch
import math
from .. import SparseTensor
from .. import config


__all__ = [
    'sparse_windowed_scaled_dot_product_self_attention',
    'sparse_windowed_scaled_dot_product_cross_attention',
]


def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (torch.Tensor): Sequence lengths.
        (dict): Attn func args.
    """
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = [i + j for i, j in zip(tensor.spatial_shape, shift_window)]
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    mask = seq_lens != 0
    seq_lens = seq_lens[mask]
    
    backend = config.get_attn_backend()

    if backend == 'xformers':
        import xformers.ops as xops
        attn_func_args = {
            'attn_bias': xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
        }
    elif backend == 'flash_attn':
        attn_func_args = {
            'cu_seqlens': torch.cat([torch.tensor([0], device=tensor.device), torch.cumsum(seq_lens, dim=0)], dim=0).int(),
            'max_seqlen': torch.max(seq_lens)
        }
    elif backend == 'sdpa':
        # For sdpa, we need to store seq_lens to build the mask later
        attn_func_args = {
            'seq_lens': seq_lens.tolist() if isinstance(seq_lens, torch.Tensor) else seq_lens
        }
    else:
        raise ValueError(f"Unknown sparse attention backend: {backend}")

    return fwd_indices, bwd_indices, seq_lens, attn_func_args
    

def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
        
    Returns:
        (SparseTensor): [N, *, H, C] sparse tensor containing the output features.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f'windowed_attention_{window_size}_{shift_window}'
    serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, attn_func_args))
    else:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = serialization_spatial_cache
    
    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]

    if config.get_debug():
        start = 0
        qkv_coords = qkv.coords[fwd_indices]
        for i in range(len(seq_lens)):
            seq_coords = qkv_coords[start:start+seq_lens[i]]
            assert (seq_coords[:, 1:].max(dim=0).values - seq_coords[:, 1:].min(dim=0).values < window_size).all(), \
                    f"SparseWindowedScaledDotProductSelfAttention: window size exceeded"
            start += seq_lens[i]

    backend = config.get_attn_backend()

    if backend == 'xformers':
        import xformers.ops as xops
        q, k, v = qkv_feats.unbind(dim=1)                                               # [M, H, C]
        q = q.unsqueeze(0)                                                              # [1, M, H, C]
        k = k.unsqueeze(0)                                                              # [1, M, H, C]
        v = v.unsqueeze(0)                                                              # [1, M, H, C]
        out = xops.memory_efficient_attention(q, k, v, **attn_func_args)[0]             # [M, H, C]
    elif backend == 'flash_attn':
        import flash_attn
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)  # [M, H, C]
    elif backend == 'sdpa':
        # SDPA fallback: pad variable-length sequences and use attention mask
        from torch.nn.functional import scaled_dot_product_attention as sdpa

        q, k, v = qkv_feats.unbind(dim=1)  # Each: [M, H, C]
        seq_lens_list = attn_func_args['seq_lens']
        device = q.device

        N = len(seq_lens_list)
        max_len = max(seq_lens_list)
        H = q.shape[1]
        C = q.shape[2]

        # Create padded tensors [N, max_len, H, C]
        q_padded = torch.zeros(N, max_len, H, C, device=device, dtype=q.dtype)
        k_padded = torch.zeros(N, max_len, H, C, device=device, dtype=k.dtype)
        v_padded = torch.zeros(N, max_len, H, C, device=device, dtype=v.dtype)

        # Fill in the actual values
        offset = 0
        for i in range(N):
            seq_len = seq_lens_list[i]
            q_padded[i, :seq_len] = q[offset:offset + seq_len]
            k_padded[i, :seq_len] = k[offset:offset + seq_len]
            v_padded[i, :seq_len] = v[offset:offset + seq_len]
            offset += seq_len

        # Create attention mask: True for valid positions
        attn_mask = torch.zeros(N, 1, max_len, max_len, device=device, dtype=torch.bool)
        for i in range(N):
            seq_len = seq_lens_list[i]
            attn_mask[i, :, :seq_len, :seq_len] = True

        # Convert to float mask for sdpa (0 for valid, -inf for invalid)
        attn_mask_float = torch.where(attn_mask, 0.0, float('-inf')).to(dtype=q.dtype)

        # Permute to [N, H, L, C] for sdpa
        q_padded = q_padded.permute(0, 2, 1, 3)
        k_padded = k_padded.permute(0, 2, 1, 3)
        v_padded = v_padded.permute(0, 2, 1, 3)

        # Run sdpa
        out_padded = sdpa(q_padded, k_padded, v_padded, attn_mask=attn_mask_float)

        # Permute back to [N, L, H, C]
        out_padded = out_padded.permute(0, 2, 1, 3)

        # Extract valid outputs back to flat tensor
        out_list = []
        for i in range(N):
            out_list.append(out_padded[i, :seq_lens_list[i]])
        out = torch.cat(out_list, dim=0)
    else:
        raise ValueError(f"Unknown sparse attention backend: {backend}")

    out = out[bwd_indices]      # [T, H, C]

    if config.get_debug():
        qkv_coords = qkv_coords[bwd_indices]
        assert torch.equal(qkv_coords, qkv.coords), "SparseWindowedScaledDotProductSelfAttention: coordinate mismatch"

    return qkv.replace(out)


def sparse_windowed_scaled_dot_product_cross_attention(
    q: SparseTensor,
    kv: SparseTensor,
    q_window_size: int,
    kv_window_size: int,
    q_shift_window: Tuple[int, int, int] = (0, 0, 0),
    kv_shift_window: Tuple[int, int, int] = (0, 0, 0),
) -> SparseTensor:
    """
    Apply windowed scaled dot product cross attention to two sparse tensors.

    Args:
        q (SparseTensor): [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor): [N, *, 2, H, C] sparse tensor containing Ks and Vs.
        q_window_size (int): The window size to use for Qs.
        kv_window_size (int): The window size to use for Ks and Vs.
        q_shift_window (Tuple[int, int, int]): The shift of serialized coordinates for Qs.
        kv_shift_window (Tuple[int, int, int]): The shift of serialized coordinates for Ks and Vs.
        
    Returns:
        (SparseTensor): [N, *, H, C] sparse tensor containing the output features.
    """
    assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
    assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"

    q_serialization_spatial_cache_name = f'windowed_attention_{q_window_size}_{q_shift_window}'
    q_serialization_spatial_cache = q.get_spatial_cache(q_serialization_spatial_cache_name)
    if q_serialization_spatial_cache is None:
        q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args = calc_window_partition(q, q_window_size, q_shift_window)
        q.register_spatial_cache(q_serialization_spatial_cache_name, (q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args))
    else:
        q_fwd_indices, q_bwd_indices, q_seq_lens, q_attn_func_args = q_serialization_spatial_cache
    kv_serialization_spatial_cache_name = f'windowed_attention_{kv_window_size}_{kv_shift_window}'
    kv_serialization_spatial_cache = kv.get_spatial_cache(kv_serialization_spatial_cache_name)
    if kv_serialization_spatial_cache is None:
        kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args = calc_window_partition(kv, kv_window_size, kv_shift_window)
        kv.register_spatial_cache(kv_serialization_spatial_cache_name, (kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args))
    else:
        kv_fwd_indices, kv_bwd_indices, kv_seq_lens, kv_attn_func_args = kv_serialization_spatial_cache

    assert len(q_seq_lens) == len(kv_seq_lens), "Number of sequences in q and kv must match"

    q_feats = q.feats[q_fwd_indices]      # [M, H, C]
    kv_feats = kv.feats[kv_fwd_indices]    # [M, 2, H, C]

    backend = config.get_attn_backend()

    if backend == 'xformers':
        import xformers.ops as xops
        k, v = kv_feats.unbind(dim=1)                                                   # [M, H, C]
        q_tensor = q_feats.unsqueeze(0)                                                 # [1, M, H, C]
        k = k.unsqueeze(0)                                                              # [1, M, H, C]
        v = v.unsqueeze(0)                                                              # [1, M, H, C]
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seq_lens, kv_seq_lens)
        out = xops.memory_efficient_attention(q_tensor, k, v, attn_bias=mask)[0]        # [M, H, C]
    elif backend == 'flash_attn':
        import flash_attn
        out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats,
            cu_seqlens_q=q_attn_func_args['cu_seqlens'], cu_seqlens_k=kv_attn_func_args['cu_seqlens'],
            max_seqlen_q=q_attn_func_args['max_seqlen'], max_seqlen_k=kv_attn_func_args['max_seqlen'],
        )  # [M, H, C]
    elif backend == 'sdpa':
        # SDPA fallback: pad variable-length sequences and use attention mask
        from torch.nn.functional import scaled_dot_product_attention as sdpa

        k, v = kv_feats.unbind(dim=1)  # Each: [M, H, C]
        q_seq_lens_list = q_attn_func_args['seq_lens']
        kv_seq_lens_list = kv_attn_func_args['seq_lens']
        device = q_feats.device

        N = len(q_seq_lens_list)
        max_q_len = max(q_seq_lens_list)
        max_kv_len = max(kv_seq_lens_list)
        H = q_feats.shape[1]
        C = q_feats.shape[2]

        # Create padded tensors [N, max_len, H, C]
        q_padded = torch.zeros(N, max_q_len, H, C, device=device, dtype=q_feats.dtype)
        k_padded = torch.zeros(N, max_kv_len, H, C, device=device, dtype=k.dtype)
        v_padded = torch.zeros(N, max_kv_len, H, C, device=device, dtype=v.dtype)

        # Fill in the actual values
        q_offset = 0
        kv_offset = 0
        for i in range(N):
            q_len = q_seq_lens_list[i]
            kv_len = kv_seq_lens_list[i]
            q_padded[i, :q_len] = q_feats[q_offset:q_offset + q_len]
            k_padded[i, :kv_len] = k[kv_offset:kv_offset + kv_len]
            v_padded[i, :kv_len] = v[kv_offset:kv_offset + kv_len]
            q_offset += q_len
            kv_offset += kv_len

        # Create attention mask: True for valid positions
        attn_mask = torch.zeros(N, 1, max_q_len, max_kv_len, device=device, dtype=torch.bool)
        for i in range(N):
            attn_mask[i, :, :q_seq_lens_list[i], :kv_seq_lens_list[i]] = True

        # Convert to float mask for sdpa (0 for valid, -inf for invalid)
        attn_mask_float = torch.where(attn_mask, 0.0, float('-inf')).to(dtype=q.dtype)

        # Permute to [N, H, L, C] for sdpa
        q_padded = q_padded.permute(0, 2, 1, 3)
        k_padded = k_padded.permute(0, 2, 1, 3)
        v_padded = v_padded.permute(0, 2, 1, 3)

        # Run sdpa
        out_padded = sdpa(q_padded, k_padded, v_padded, attn_mask=attn_mask_float)

        # Permute back to [N, L, H, C]
        out_padded = out_padded.permute(0, 2, 1, 3)

        # Extract valid outputs back to flat tensor
        out_list = []
        for i in range(N):
            out_list.append(out_padded[i, :q_seq_lens_list[i]])
        out = torch.cat(out_list, dim=0)
    else:
        raise ValueError(f"Unknown sparse attention backend: {backend}")

    out = out[q_bwd_indices]      # [T, H, C]

    return q.replace(out)

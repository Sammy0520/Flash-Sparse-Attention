import math

import torch

from nsa_ref.ops.topk_sparse_attention import _topk_sparse_attention_fwd


def _topk_sparse_attention_decode(
    q: torch.Tensor,  # [total_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_k_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_k_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    sm_scale=None,
    attention_mask: torch.Tensor = None,
):
    # attention_mask: optional (total_q_len, total_k_len) or (num_heads, total_q_len, total_k_len), 1=attend 0=mask. Applied in kernel.
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype and k.dtype == v.dtype
    assert topk_idx.dtype == torch.int32
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    # softmax scale
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(q.shape[-1])

    mask_2d = None
    if attention_mask is not None:
        if attention_mask.dim() == 3:
            mask_2d = attention_mask.max(dim=0).values
        else:
            mask_2d = attention_mask
        if mask_2d.dtype != torch.float32:
            mask_2d = mask_2d.to(torch.float32)

    o, lse = _topk_sparse_attention_fwd(
        q,
        k,
        v,
        topk_idx,
        block_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
        attention_mask=mask_2d,
    )

    return o

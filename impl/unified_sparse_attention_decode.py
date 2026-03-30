import math
import torch
import triton
import triton.language as tl
from impl.utils import is_hopper_gpu, get_num_warps_stages, is_power_of_two


IS_HOPPER_GPU = is_hopper_gpu()
INT32_MAX = 2147483647


@triton.jit
def forward_kernel_unified(
    # tensors
    q_ptr, # [total_q_len, num_q_heads, head_dim]
    k_ptr, # [total_k_len, num_k_heads, head_dim]
    v_ptr, # [total_k_len, num_k_heads, head_dim]
    topk_idx_ptr, # [num_kv_heads, total_len, topk]
    o_ptr, # [total_q_len, num_q_heads, head_dim]
    cu_seqlens_q_ptr, # [batch_size + 1]
    cu_seqlens_k_ptr, # [batch_size + 1]
    gate_ptr, # [total_q_len, 3]
    attention_mask_ptr, # [total_q_len, total_k_len]

    # shape
    total_q_len,
    total_k_len,
    num_kv_heads,
    topk,

    # scale
    sm_scale,

    # stride
    stride_ql, stride_qh, stride_qd, # q_ptr, [total_q_len, num_q_heads, head_dim]
    stride_kl, stride_kh, stride_kd, # k_ptr, [total_k_len, num_k_heads, head_dim]
    stride_vl, stride_vh, stride_vd, # v_ptr, [total_k_len, num_k_heads, head_dim]
    stride_th, stride_tl, stride_tk, # top_idx_ptr, [num_kv_heads, total_q_len, topk]
    stride_ol, stride_oh, stride_od, # o_ptr, [total_q_len, num_q_heads, head_dim
    stride_gl, stride_g3, # gate_ptr, [total_q_len, 3]
    stride_aql, stride_akl, # attention_mask_ptr, [total_q_len, total_k_len]

    # meta parameters
    causal: tl.constexpr,
    block_size_k: tl.constexpr,
    head_dim: tl.constexpr,
    num_share_q_heads: tl.constexpr, # num_q_heads // num_k_heads
    BLOCK_SIZE_topk: tl.constexpr,
    window_size: tl.constexpr,
    max_seqlen_q: tl.constexpr,
    HAS_MASK: tl.constexpr,
    INT32_MAX: tl.constexpr,
):
    # Process pid 0
    pid_0 = tl.program_id(0)
    pid_batch = pid_0 // (num_kv_heads * max_seqlen_q) # [0, batch_size)
    pid_k_heads = (pid_0 % (num_kv_heads * max_seqlen_q)) // max_seqlen_q # [0, num_k_heads)
    pid_seq_q_start = pid_0 % max_seqlen_q
    pid_q_heads = pid_k_heads * num_share_q_heads
 
    # Compute start and len for each batch
    q_start = tl.load(cu_seqlens_q_ptr + pid_batch)
    q_len = tl.load(cu_seqlens_q_ptr + pid_batch + 1) - q_start
    k_start = tl.load(cu_seqlens_k_ptr + pid_batch)
    k_len = tl.load(cu_seqlens_k_ptr + pid_batch + 1) - k_start
 
    if pid_seq_q_start >= q_len:
        return
 
    BLOCK_H: tl.constexpr = num_share_q_heads if num_share_q_heads >= 16 else 16
    pid_q_start_abs = q_start + pid_seq_q_start # [0, total_q_len)
    abs_q_pos = k_len - q_len + pid_seq_q_start
 
    # common offsets
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, head_dim)
    offs_k_blk = tl.arange(0, block_size_k)
    row_valid = offs_h < num_share_q_heads
 
    # Load q: [BLOCK_H, head_dim]
    q_ptrs = q_ptr + pid_q_start_abs * stride_ql + (pid_q_heads + offs_h)[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0)
 
    # Load gate (per-token scalar, broadcast to all BLOCK_H head rows)
    gate_t_val = tl.load(gate_ptr + pid_q_start_abs * stride_gl + 1 * stride_g3)
    gate_w_val = tl.load(gate_ptr + pid_q_start_abs * stride_gl + 2 * stride_g3)
 
    # Load topk indices
    offs_block_topk = tl.arange(0, BLOCK_SIZE_topk)
    topk_idx_base = topk_idx_ptr + pid_k_heads * stride_th + pid_q_start_abs * stride_tl
    topk_vals = tl.load(
        topk_idx_base + offs_block_topk * stride_tk,
        mask=offs_block_topk < topk,
        other=INT32_MAX,
    )

    if causal:
        max_valid_block_idx = abs_q_pos // block_size_k
        real_topk = tl.sum(tl.where((topk_vals >= 0) & (topk_vals <= max_valid_block_idx), 1, 0), axis=0)
    else:
        real_topk = tl.sum(tl.where((topk_vals >= 0) & (topk_vals < INT32_MAX), 1, 0), axis=0)
 
    # TopK branch
    qk_scale = sm_scale * 1.44269504
    m_tk   = tl.full((BLOCK_H,), float('-inf'), dtype=tl.float32)
    lse_tk = tl.full((BLOCK_H,), float('-inf'), dtype=tl.float32)
    acc_tk = tl.zeros((BLOCK_H, head_dim), dtype=tl.float32)
 
    k_ptrs_tk = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kl + pid_k_heads * stride_kh,
        shape=(head_dim, k_len),
        strides=(stride_kd, stride_kl),
        offsets=(0, 0),
        block_shape=(head_dim, block_size_k),
        order=(0, 1),
    )
    v_ptrs_tk = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vl + pid_k_heads * stride_vh,
        shape=(k_len, head_dim),
        strides=(stride_vl, stride_vd),
        offsets=(0, 0),
        block_shape=(block_size_k, head_dim),
        order=(1, 0),
    )
 
    for i in range(real_topk):
        kv_idx = tl.load(topk_idx_base + i * stride_tk)
        c = kv_idx * block_size_k
 
        k = tl.load(tl.advance(k_ptrs_tk, (0, c)), boundary_check=(1, 0), padding_option="zero")
        qk = tl.dot(q, k) * qk_scale  # [BLOCK_H, block_size_k], log2 scale
 
        if causal:
            qk += tl.where(abs_q_pos >= (c + offs_k_blk)[None, :], 0.0, float('-inf'))

        if HAS_MASK:
            mask_ptrs = tl.make_block_ptr(
                base=attention_mask_ptr,
                shape=(total_q_len, total_k_len),
                strides=(stride_aql, stride_akl),
                offsets=(pid_q_start_abs, k_start + c),
                block_shape=(1, block_size_k),
                order=(1, 0),
            )
            mask_block = tl.load(mask_ptrs, boundary_check=(0, 1), padding_option="zero")
            qk = tl.where(mask_block > 0, qk, float("-inf"))
 
        m_ij = tl.maximum(m_tk, tl.max(qk, axis=1))
        qk_s = tl.where(m_ij[:, None] > float('-inf'), qk - m_ij[:, None], float('-inf'))
        p = tl.exp2(qk_s)
        l_ij = tl.sum(p, axis=1)
 
        v = tl.load(tl.advance(v_ptrs_tk, (c, 0)), boundary_check=(0, 1), padding_option="zero")
 
        scl = tl.where(m_ij > float('-inf'), tl.exp2(m_tk - m_ij), 1.0)
        acc_tk = acc_tk * scl[:, None] + tl.dot(p.to(v.dtype), v)
 
        lse_update = m_ij + tl.math.log2(tl.exp2(lse_tk - m_ij) + l_ij)
        lse_tk = tl.where(m_ij > float('-inf'), lse_update, lse_tk)
        m_tk = m_ij
 
    # normalize
    ea_t = tl.where(lse_tk > float('-inf'), m_tk - lse_tk, float('-inf'))
    o_tk = acc_tk * tl.exp2(ea_t)[:, None]
 
    # Sliding window branch
    m_win = tl.full((BLOCK_H,), float('-inf'), dtype=tl.float32)
    l_win = tl.zeros((BLOCK_H,), dtype=tl.float32)   # direct sum, NOT log-space
    acc_win = tl.zeros((BLOCK_H, head_dim), dtype=tl.float32)
 
    # Left boundary (local k index, 0-based within this batch's K)
    n_min = pid_seq_q_start + k_len - q_len - window_size
    n_min = tl.maximum(n_min, 0)
    first_w_block = n_min // block_size_k
    num_k_blocks  = tl.cdiv(k_len, block_size_k)
 
    k_ptrs_win = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kl + pid_k_heads * stride_kh,
        shape=(head_dim, k_len),
        strides=(stride_kd, stride_kl),
        offsets=(0, first_w_block * block_size_k),
        block_shape=(head_dim, block_size_k),
        order=(0, 1),
    )
    v_ptrs_win = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vl + pid_k_heads * stride_vh,
        shape=(k_len, head_dim),
        strides=(stride_vl, stride_vd),
        offsets=(first_w_block * block_size_k, 0),
        block_shape=(block_size_k, head_dim),
        order=(1, 0),
    )
 
    for blk in range(first_w_block, num_k_blocks):
        c = blk * block_size_k
 
        k = tl.load(k_ptrs_win, boundary_check=(1, 0), padding_option="zero")
        qk = tl.dot(q, k) * sm_scale  # [BLOCK_H, block_size_k], natural-log units
 
        # Left boundary mask: k positions < n_min are masked out (affects first block only)
        k_local = c + offs_k_blk
        qk = qk + tl.where(k_local[None, :] >= n_min, 0.0, float('-inf'))

        if causal:
            qk += tl.where(abs_q_pos >= k_local[None, :], 0.0, float('-inf'))
 
        if HAS_MASK:
            mask_ptrs = tl.make_block_ptr(
                base=attention_mask_ptr,
                shape=(total_q_len, total_k_len),
                strides=(stride_aql, stride_akl),
                offsets=(pid_q_start_abs, k_start + c),
                block_shape=(1, block_size_k),
                order=(1, 0),
            )
            mask_block = tl.load(mask_ptrs, boundary_check=(0, 1), padding_option="zero")
            qk = tl.where(mask_block > 0, qk, float("-inf"))
 
        # Online softmax update (base-e), l_i direct-sum formulation like flash_attn
        m_ij = tl.maximum(m_win, tl.max(qk, axis=1))
        # NaN guard: fully-masked block + m_win=-inf → exp(-inf - (-inf)) = nan
        p = tl.math.exp(tl.where(m_ij[:, None] > float('-inf'), qk - m_ij[:, None], float('-inf')))
        l_ij = tl.sum(p, axis=1)
 
        v = tl.load(v_ptrs_win, boundary_check=(0, 1), padding_option="zero")
 
        # alpha = exp(m_win - m_ij); NaN guard: if m_win = m_ij = -inf, use 1.0
        alpha = tl.where(m_ij > float('-inf'), tl.math.exp(m_win - m_ij), 1.0)
        l_win   = l_win * alpha + l_ij
        acc_win = acc_win * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_win = m_ij
 
        k_ptrs_win = tl.advance(k_ptrs_win, (0, block_size_k))
        v_ptrs_win = tl.advance(v_ptrs_win, (block_size_k, 0))
 
    # normalize window: acc / l  (same as flash_attn's l_recip = 1 / l_i)
    l_recip_win = tl.where(l_win > 0, 1.0 / l_win, 0.0)
    o_win = acc_win * l_recip_win[:, None]
 
    # final output
    out = gate_t_val * o_tk + gate_w_val * o_win
 
    # Write output
    o_ptrs = o_ptr + pid_q_start_abs * stride_ol + (pid_q_heads + offs_h)[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=row_valid[:, None])


def _unified_sparse_attention_decode(
    q: torch.Tensor, # [total_q_len, num_q_heads, head_dim]
    k: torch.Tensor, # [total_k_len, num_k_heads, head_dim]
    v: torch.Tensor, # [total_k_len, num_k_heads, head_dim]
    topk_idx: torch.Tensor, # [num_kv_heads, total_len, topk], for topk branch, <= 16
    block_size: int, # for topk branch, usually 64
    window_size: int, # for sliding window branch, usually 512 (8 * 64)
    cu_seqlens_q: torch.Tensor, # [batch_size + 1]
    cu_seqlens_k: torch.Tensor, # [batch_size + 1]
    max_seqlen_q: int,
    max_seqlen_k: int,
    gate: torch.Tensor, # [total_q_len, 3], gate[:, 1:2] is for sparse topk branch, gate[:, 2:3] is for sliding window branch
    sm_scale: float = None,
    attention_mask: torch.Tensor = None, # [total_q_len, total_k_len]
    causal: bool = True,
) -> torch.Tensor: # [total_q_len, num_q_heads, head_dim]
    """
    Unified sparse attention (topk + sliding window)
    """
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype and k.dtype == v.dtype
    assert topk_idx.dtype == torch.int32
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert block_size in {32, 64, 128, 256}

    # shape
    total_q_len, num_q_heads, head_dim = q.shape
    total_k_len, num_k_heads, head_dim = k.shape
    batch_size = len(cu_seqlens_q) - 1
    topk = topk_idx.shape[-1]
    assert is_power_of_two(block_size)
    assert is_power_of_two(head_dim)

    # block_size
    BLOCK_SIZE_topk = triton.next_power_of_2(topk)

    # attention_mask
    if attention_mask is not None:
        assert attention_mask.shape == (total_q_len, total_k_len), (
            f"attention_mask shape {attention_mask.shape} != (total_q_len, total_k_len) ({total_q_len}, {total_k_len})"
        )
        assert attention_mask.dtype == torch.float32, "attention_mask must be float32"

    has_mask = attention_mask is not None
    if not has_mask:
        attention_mask = q # dummy
    stride_mask_q = attention_mask.stride(0) if has_mask else 0
    stride_mask_k = attention_mask.stride(1) if has_mask else 0

    # gqa
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    assert is_power_of_two(num_share_q_heads)

    # softmax scale
    if sm_scale is None or sm_scale == 0.0:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # output tensor
    o = torch.zeros_like(q)

    # launch kernel
    num_warps, num_stages = get_num_warps_stages(head_dim, block_size, IS_HOPPER_GPU)
    grid = (batch_size * num_k_heads * max_seqlen_q,)
    forward_kernel_unified[grid](
        # tensors
        q,
        k,
        v,
        topk_idx,
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        gate,
        attention_mask,

        # shape
        total_q_len,
        total_k_len,
        num_k_heads,
        topk,

        # scale
        sm_scale,

        # stride
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        gate.stride(0), gate.stride(1),
        stride_mask_q, stride_mask_k,

        # meta parameters
        causal=causal,
        block_size_k=block_size,
        head_dim=head_dim,
        num_share_q_heads=num_share_q_heads,
        BLOCK_SIZE_topk=BLOCK_SIZE_topk,
        window_size=window_size,
        max_seqlen_q=max_seqlen_q,
        HAS_MASK=has_mask,
        INT32_MAX=INT32_MAX,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o

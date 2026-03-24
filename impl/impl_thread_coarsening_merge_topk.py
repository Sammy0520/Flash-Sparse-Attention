import torch
import triton
import triton.language as tl

from impl.utils import is_hopper_gpu, get_num_warps_stages, is_power_of_two

IS_HOPPER_GPU = is_hopper_gpu()
INT32_MAX = (1 << 31) - 1


# ---------------------------------------------------------------------------
# merge_topk_idx
#   Input : topk_idx [num_kv_heads, total_q_len, topk], sentinel = -1
#   Output: topk_idx_coarsened [num_kv_heads, total_q_len_coarsened, topk_coarsened]
#           cu_seqlens_topk_idx_coarsened [batch_size + 1]
#   Each coarsened row = union of CFACTOR consecutive rows, sorted ascending,
#   deduped, padded with INT32_MAX, truncated to topk_coarsened.
# ---------------------------------------------------------------------------

def merge_topk_idx(
    topk_idx: torch.Tensor,       # [num_kv_heads, total_q_len, topk]
    cu_seqlens_q: torch.Tensor,   # [batch_size + 1]
    topk_coarsened: int,
    CFACTOR: int,
):
    num_kv_heads, total_q_len, topk = topk_idx.shape
    device = topk_idx.device
    dtype  = topk_idx.dtype

    seq_lens           = cu_seqlens_q[1:] - cu_seqlens_q[:-1]           # [B]
    coarsened_seq_lens = (seq_lens + CFACTOR - 1) // CFACTOR             # [B]

    cu_seqlens_c = torch.zeros_like(cu_seqlens_q)
    cu_seqlens_c[1:] = torch.cumsum(coarsened_seq_lens, dim=0)
    total_q_len_c = cu_seqlens_c[-1].item()

    topk_idx_c = torch.full(
        (num_kv_heads, total_q_len_c, topk_coarsened),
        INT32_MAX, dtype=dtype, device=device,
    )

    batch_size = len(seq_lens)
    for b in range(batch_size):
        q_s  = cu_seqlens_q[b].item()
        q_e  = cu_seqlens_q[b + 1].item()
        seq_len = q_e - q_s
        if seq_len == 0:
            continue

        # Gather this sequence's topk rows: [H, seq_len, topk]
        seq_idx = topk_idx[:, q_s:q_e, :]

        # Pad to multiple of CFACTOR
        pad_len = (-seq_len) % CFACTOR
        if pad_len > 0:
            pad = torch.full((num_kv_heads, pad_len, topk), INT32_MAX, dtype=dtype, device=device)
            seq_idx = torch.cat([seq_idx, pad], dim=1)

        num_chunks = seq_idx.shape[1] // CFACTOR

        # [H, num_chunks, CFACTOR * topk] — flatten each group
        chunked = seq_idx.reshape(num_kv_heads, num_chunks, CFACTOR * topk)

        # Replace -1 sentinel with INT32_MAX so sort puts invalids at the end
        chunked = torch.where(chunked == -1, torch.tensor(INT32_MAX, dtype=dtype, device=device), chunked)

        # Sort
        sorted_chunk, _ = torch.sort(chunked, dim=-1)   # [H, C, CFACTOR*topk]

        # Deduplicate: compare each element with its left neighbour
        pad_left = torch.full((num_kv_heads, num_chunks, 1), -1, dtype=dtype, device=device)
        shifted  = torch.cat([pad_left, sorted_chunk[..., :-1]], dim=-1)
        is_dup   = sorted_chunk == shifted
        dedup    = torch.where(is_dup, torch.tensor(INT32_MAX, dtype=dtype, device=device), sorted_chunk)

        # Re-sort so valid entries (< INT32_MAX) come first
        final, _ = torch.sort(dedup, dim=-1)

        # Write out (truncate to topk_coarsened)
        take = min(topk_coarsened, final.shape[-1])
        c_s  = cu_seqlens_c[b].item()
        c_e  = cu_seqlens_c[b + 1].item()
        topk_idx_c[:, c_s:c_e, :take] = final[:, :c_e - c_s, :take]

    return topk_idx_c, cu_seqlens_c


@triton.jit
def forward_kernel_split_k(
    # tensors
    q_ptr, # [total_q_len, num_q_heads, head_dim]
    k_ptr, # [total_k_len, num_k_heads, head_dim]
    v_ptr, # [total_k_len, num_k_heads, head_dim]
    topk_idx_c_ptr, # [num_kv_heads, total_q_len_coarsened, topk_coarsened]
    partial_o_ptr, # [total_q_len, num_q_heads, SPLIT_K, head_dim]
    partial_lse_ptr, # [num_q_heads, total_q_len, SPLIT_K]
    cu_seqlens_q_ptr, # [batch_size + 1]
    cu_seqlens_k_ptr, # [batch_size + 1]
    cu_seqlens_c_ptr,  # [batch_size + 1]  coarsened seq offsets
    attention_mask_ptr, # [total_q_len, total_k_len]

    # shapes
    total_q_len,
    total_k_len,
    num_kv_heads,

    # scale
    sm_scale,

    # strides — q, k, v
    stride_ql, stride_qh, stride_qd,
    stride_kl, stride_kh, stride_kd,
    stride_vl, stride_vh, stride_vd,
    # strides — topk_idx_coarsened
    stride_th, stride_tl, stride_tk,
    # strides — partial_o, partial_lse
    stride_pol, stride_poh, stride_pospk, stride_pod,
    stride_plh, stride_pll, stride_plspk,
    # strides — attention_mask
    stride_aql, stride_akl,

    # meta parameters
    block_size_k: tl.constexpr,
    head_dim: tl.constexpr,
    num_share_q_heads: tl.constexpr,
    topk_coarsened: tl.constexpr,   # must be power-of-2
    max_seqlen_q: tl.constexpr,
    SPLIT_K: tl.constexpr,
    CFACTOR: tl.constexpr,
    HAS_MASK: tl.constexpr,
    INT32_MAX_VAL: tl.constexpr,
):
    max_q_groups: tl.constexpr = (max_seqlen_q + CFACTOR - 1) // CFACTOR

    pid_0 = tl.program_id(0)
    pid_split_k = tl.program_id(1)
    pid_batch    = pid_0 // (num_kv_heads * max_q_groups)
    pid_k_heads  = (pid_0 % (num_kv_heads * max_q_groups)) // max_q_groups
    pid_group    = pid_0 % max_q_groups
    pid_q_heads  = pid_k_heads * num_share_q_heads

    q_start = tl.load(cu_seqlens_q_ptr + pid_batch)
    q_len = tl.load(cu_seqlens_q_ptr + pid_batch + 1) - q_start
    k_start = tl.load(cu_seqlens_k_ptr + pid_batch)
    k_len = tl.load(cu_seqlens_k_ptr + pid_batch + 1) - k_start
    c_start = tl.load(cu_seqlens_c_ptr + pid_batch)
    c_len = tl.load(cu_seqlens_c_ptr + pid_batch + 1) - c_start

    pid_seq_q_start = pid_group * CFACTOR   # first absolute seq-pos in this group

    # Early-exit if this group is out of bounds for this batch
    if pid_group >= c_len:
        return
    if pid_seq_q_start >= q_len:
        return

    BLOCK_H: tl.constexpr = num_share_q_heads if num_share_q_heads >= 16 else 16
    BLOCK_CF_H: tl.constexpr = CFACTOR * BLOCK_H
    pid_q_len_start = q_start + pid_seq_q_start   # absolute token ptr of group start
    coarsened_row   = c_start + pid_group          # row in topk_idx_coarsened

    # Load topk
    topk_idx_base_ptr = topk_idx_c_ptr + pid_k_heads * stride_th + coarsened_row * stride_tl
    offs_topk = tl.arange(0, topk_coarsened)
    topk_all = tl.load(topk_idx_base_ptr + offs_topk * stride_tk)  # [topk_coarsened]

    # Calculate topk range
    real_topk = tl.sum(tl.where(topk_all < INT32_MAX_VAL, 1, 0), axis=0)
    chunk_size: tl.constexpr = topk_coarsened // SPLIT_K
    start_topk    = pid_split_k * chunk_size
    real_end_topk = tl.minimum(start_topk + chunk_size, real_topk)

    offs_cf_h = tl.arange(0, BLOCK_CF_H)
    cf_idx    = offs_cf_h // BLOCK_H    # which token in group [0..CFACTOR-1]
    h_idx     = offs_cf_h % BLOCK_H    # which head in tile   [0..BLOCK_H-1]
    offs_d    = tl.arange(0, head_dim)
    offs_bk   = tl.arange(0, block_size_k)

    row_valid = (
        ((pid_seq_q_start + cf_idx) < q_len) &
        (h_idx < num_share_q_heads)
    )

    # ── load Q: [BLOCK_CF_H, head_dim] ────────────────────────────────────
    q = tl.load(
        q_ptr
        + (pid_q_len_start + cf_idx[:, None]) * stride_ql
        + (pid_q_heads     + h_idx[:, None])  * stride_qh
        + offs_d[None, :]                      * stride_qd,
        mask=row_valid[:, None],
        other=0.0,
    )

    # Per-row absolute sequence position for causal mask
    seq_q_for_row = pid_seq_q_start + cf_idx   # [BLOCK_CF_H]

    # Init
    m_i = tl.full((BLOCK_CF_H,), float('-inf'), dtype=tl.float32)
    lse_i = tl.full((BLOCK_CF_H,), float('-inf'), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_CF_H, head_dim), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    # Ptrs for k, v
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kl + pid_k_heads * stride_kh,
        shape=(head_dim, k_len),
        strides=(stride_kd, stride_kl),
        offsets=(0, 0),
        block_shape=(head_dim, block_size_k),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vl + pid_k_heads * stride_vh,
        shape=(k_len, head_dim),
        strides=(stride_vl, stride_vd),
        offsets=(0, 0),
        block_shape=(block_size_k, head_dim),
        order=(1, 0),
    )

    for i in range(start_topk, real_end_topk):
        kv_idx = tl.load(topk_idx_base_ptr + i * stride_tk)

        if kv_idx >= 0 and kv_idx < INT32_MAX_VAL:
            c = kv_idx * block_size_k  # relative to k_start
            k = tl.load(tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero")
            qk = tl.dot(q, k) * qk_scale  # [BLOCK_CF_H, block_size_k]

            # Per-row causal mask — each token sees only its own past
            # qk += tl.where(
            #     seq_q_for_row[:, None] >= (kv_idx * block_size_k + offs_bk)[None, :],
            #     0, float("-inf"),
            # )

            if HAS_MASK:
                # Load mask for every token row in the group
                # Shape: [CFACTOR, block_size_k] broadcast to [BLOCK_CF_H, block_size_k]
                offs_q_abs = (pid_q_len_start + cf_idx)[:, None]  # [BLOCK_CF_H, 1]
                offs_k_abs = (k_start + kv_idx * block_size_k + offs_bk)[None, :]  # [1, bk]
                attn_mask = tl.load(
                    attention_mask_ptr
                    + offs_q_abs * stride_aql
                    + offs_k_abs * stride_akl,
                    mask=row_valid[:, None],
                    other=0.0,
                )
                qk = tl.where(attn_mask > 0, qk, float("-inf"))

            # Compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            qk_shift = tl.where(m_ij[:, None] > float('-inf'), qk - m_ij[:, None], float('-inf'))
            p = tl.exp2(qk_shift)
            l_ij = tl.sum(p, axis=1)

            # Load v
            v = tl.load(tl.advance(v_ptrs, (c, 0)), boundary_check=(0, 1), padding_option="zero")

            # Update acc_o
            acc_o_scale = tl.where(m_ij > float('-inf'), tl.exp2(m_i - m_ij), 1.0)
            acc_o = acc_o * acc_o_scale[:, None] + tl.dot(p.to(v.dtype), v)

            # Update m_i, lse_i
            lse_update = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
            lse_i = tl.where(m_ij > float('-inf'), lse_update, lse_i)
            m_i   = m_ij

    # Final scale
    exp_arg = tl.where(lse_i > float('-inf'), m_i - lse_i, float('-inf'))
    acc_o   = acc_o * tl.exp2(exp_arg)[:, None]

    # Write partial_o
    tl.store(
        partial_o_ptr
        + (pid_q_len_start + cf_idx[:, None]) * stride_pol
        + (pid_q_heads     + h_idx[:, None])  * stride_poh
        + pid_split_k                          * stride_pospk
        + offs_d[None, :]                      * stride_pod,
        acc_o.to(partial_o_ptr.dtype.element_ty),
        mask=row_valid[:, None],
    )

    # Write partial_lse [num_q_heads, total_q_len, SPLIT_K]
    tl.store(
        partial_lse_ptr
        + (pid_q_heads     + h_idx)  * stride_plh
        + (pid_q_len_start + cf_idx) * stride_pll
        + pid_split_k                * stride_plspk,
        lse_i,
        mask=row_valid,
    )


@triton.jit
def combine_kernel_split_k(
    partial_o_ptr, # [total_q_len, num_q_heads, SPLIT_K, head_dim]
    partial_lse_ptr, # [num_q_heads, total_q_len, SPLIT_K]
    o_ptr, # [total_q_len, num_q_heads, head_dim]
    lse_ptr, # [num_q_heads, total_q_len]

    # strides
    stride_pol, stride_poh, stride_pospk, stride_pod,
    stride_plh, stride_pll, stride_plspk,
    stride_ol, stride_oh, stride_od,
    stride_lh, stride_ll,

    # meta parameters
    head_dim: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Process pids
    pid_q = tl.program_id(0) # [0, total_q_len)
    pid_h = tl.program_id(1) # [0, num_q_heads)

    # Common offsets
    offs_k = tl.arange(0, SPLIT_K)
    offs_d = tl.arange(0, head_dim)

    # Load SPLIT_K lse
    lse_ptrs = \
        partial_lse_ptr + \
        pid_h * stride_plh + \
        pid_q * stride_pll + \
        offs_k * stride_plspk
    lse = tl.load(lse_ptrs) # [SPLIT_K]

    # Compute global LSE
    max_lse = tl.max(lse, axis=0) # [1]
    w = tl.exp2(lse - max_lse) # [SPLIT_K]
    sum_w = tl.sum(w, axis=0) # [1]
    global_lse = max_lse + tl.math.log2(sum_w)
    scale = tl.exp2(lse - global_lse) # [SPLIT_K]

    # Load SPLIT_K o
    partial_o_ptrs = \
        partial_o_ptr + \
        pid_q * stride_pol + \
        pid_h * stride_poh + \
        offs_k[:, None] * stride_pospk + \
        offs_d[None, :] * stride_pod
    partial_o = tl.load(partial_o_ptrs) # [SPLIT_K, HEAD_DIM]
    o = tl.sum(partial_o * scale[:, None], axis=0)

    o_ptrs = o_ptr + pid_q * stride_ol + pid_h * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty))

    lse_out_ptr = lse_ptr + pid_h * stride_lh + pid_q * stride_ll
    tl.store(lse_out_ptr, global_lse)


def _topk_sparse_attention_fwd(
    q: torch.Tensor,  # [total_q_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_k_len, num_k_heads, head_dim]
    v: torch.Tensor,  # [total_k_len, num_k_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len, topk], value: [-1, max_seqlen_k/block_size)
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    attention_mask: torch.Tensor = None,
    SPLIT_K: int = 32,
    CFACTOR: int = 4,
    topk_idx_coarsen_scale: int = 2,
):
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert block_size in {32, 64, 128, 256}

    # shape
    total_q_len, num_q_heads, head_dim = q.shape
    total_k_len, num_k_heads, head_dim = k.shape
    _, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == num_k_heads
    assert topk_idx.shape[1] == total_q_len
    assert is_power_of_two(block_size)
    assert is_power_of_two(head_dim)
    assert is_power_of_two(topk)
    assert is_power_of_two(CFACTOR)

    if attention_mask is not None:
        assert attention_mask.shape == (total_q_len, total_k_len)
        assert attention_mask.dtype == torch.float32

    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    assert is_power_of_two(num_share_q_heads)

    topk_coarsened = topk * topk_idx_coarsen_scale
    assert is_power_of_two(topk_coarsened), \
        f"topk_coarsened={topk_coarsened} must be power-of-2 for Triton constexpr arange"
    assert topk_coarsened % SPLIT_K == 0, \
        f"topk_coarsened ({topk_coarsened}) must be divisible by SPLIT_K ({SPLIT_K})"

    topk_idx_c, cu_seqlens_c = merge_topk_idx(
        topk_idx, cu_seqlens_q, topk_coarsened, CFACTOR,
    )
    # topk_idx_c: [num_kv_heads, total_q_len_coarsened, topk_coarsened], sentinel INT32_MAX
    # cu_seqlens_c: [batch_size + 1]

    # partial results
    partial_o = torch.zeros(
        total_q_len, num_q_heads, SPLIT_K, head_dim,
        dtype=q.dtype, device=q.device,
    )
    partial_lse = torch.full(
        (num_q_heads, total_q_len, SPLIT_K),
        float("-inf"),
        dtype=torch.float32, device=q.device,
    )

    has_mask = attention_mask is not None
    if not has_mask:
        attention_mask = q # dummy

    stride_mask_q = attention_mask.stride(0) if has_mask else 0
    stride_mask_k = attention_mask.stride(1) if has_mask else 0

    # warps and stages
    num_warps, num_stages = get_num_warps_stages(head_dim, block_size, IS_HOPPER_GPU)

    max_q_groups = triton.cdiv(max_seqlen_q, CFACTOR)
    grid = (batch_size * num_k_heads * max_q_groups, SPLIT_K)
    forward_kernel_split_k[grid](
        # tensors
        q, k, v,
        topk_idx_c,
        partial_o, partial_lse,
        cu_seqlens_q, cu_seqlens_k, cu_seqlens_c,
        attention_mask,

        # shapes
        total_q_len, total_k_len, num_k_heads,

        # scale
        sm_scale,

        # stride
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        topk_idx_c.stride(0), topk_idx_c.stride(1), topk_idx_c.stride(2),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        stride_mask_q, stride_mask_k,

        # meta parameters
        block_size_k=block_size,
        head_dim=head_dim,
        num_share_q_heads=num_share_q_heads,
        topk_coarsened=topk_coarsened,
        max_seqlen_q=max_seqlen_q,
        SPLIT_K=SPLIT_K,
        CFACTOR=CFACTOR,
        HAS_MASK=has_mask,
        INT32_MAX_VAL=INT32_MAX,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # final results
    o = torch.empty_like(q) # [total_q_len, num_q_heads, head_dim]
    lse = torch.empty(num_q_heads, total_q_len, dtype=torch.float32, device=q.device)

    # combine stage
    grid = (total_q_len, num_q_heads)
    combine_kernel_split_k[grid](
        # tensors
        partial_o, partial_lse,
        o, lse,

        # shapes
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0), lse.stride(1),

        # meta parameters
        head_dim=head_dim,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o, lse

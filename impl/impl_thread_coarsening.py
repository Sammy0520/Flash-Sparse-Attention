import torch
import triton
import triton.language as tl

from impl.utils import is_hopper_gpu, get_num_warps_stages, is_power_of_two

IS_HOPPER_GPU = is_hopper_gpu()


@triton.jit
def forward_kernel_split_k(
    # tensors
    q_ptr,           # [total_q_len, num_q_heads, head_dim]
    k_ptr,           # [total_k_len, num_k_heads, head_dim]
    v_ptr,           # [total_k_len, num_k_heads, head_dim]
    topk_idx_ptr,    # [num_kv_heads, total_q_len, topk]
    partial_o_ptr,   # [total_q_len, num_q_heads, SPLIT_K, head_dim]
    partial_lse_ptr, # [num_q_heads, total_q_len, SPLIT_K]
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    attention_mask_ptr,

    # shapes
    total_q_len,
    total_k_len,
    num_kv_heads,

    # scale
    sm_scale,

    # strides
    stride_ql, stride_qh, stride_qd,
    stride_kl, stride_kh, stride_kd,
    stride_vl, stride_vh, stride_vd,
    stride_th, stride_tl, stride_tk,
    stride_pol, stride_poh, stride_pospk, stride_pod,
    stride_plh, stride_pll, stride_plspk,
    stride_aql, stride_akl,

    # meta parameters
    block_size_k:      tl.constexpr,
    head_dim:          tl.constexpr,
    num_share_q_heads: tl.constexpr,
    topk:              tl.constexpr,
    max_seqlen_q:      tl.constexpr,
    SPLIT_K:           tl.constexpr,
    CFACTOR:           tl.constexpr,   # number of consecutive tokens per program
    HAS_MASK:          tl.constexpr,
):
    # ── pid decoding ──────────────────────────────────────────────────────
    # grid dim-0 = batch * num_kv_heads * (max_seqlen_q // CFACTOR)
    max_seqlen_q_cf: tl.constexpr = max_seqlen_q // CFACTOR

    pid_0       = tl.program_id(0)
    pid_split_k = tl.program_id(1)

    pid_batch       = pid_0 // (num_kv_heads * max_seqlen_q_cf)
    pid_k_heads     = (pid_0 % (num_kv_heads * max_seqlen_q_cf)) // max_seqlen_q_cf
    pid_seq_q_start = (pid_0 % max_seqlen_q_cf) * CFACTOR  # first seq-pos in this group
    pid_q_heads     = pid_k_heads * num_share_q_heads

    q_start = tl.load(cu_seqlens_q_ptr + pid_batch)
    q_len   = tl.load(cu_seqlens_q_ptr + pid_batch + 1) - q_start
    k_start = tl.load(cu_seqlens_k_ptr + pid_batch)
    k_len   = tl.load(cu_seqlens_k_ptr + pid_batch + 1) - k_start

    if pid_seq_q_start >= q_len:
        return

    # ── block-height constants ────────────────────────────────────────────
    # BLOCK_H    : head-dimension tile  (>= 16)
    # BLOCK_CF_H : CFACTOR tokens × BLOCK_H heads, flattened into one dim
    BLOCK_H:    tl.constexpr = num_share_q_heads if num_share_q_heads >= 16 else 16
    BLOCK_CF_H: tl.constexpr = CFACTOR * BLOCK_H

    pid_q_len_start = q_start + pid_seq_q_start  # absolute token offset of group start

    # ── topk from first token (approximation) ────────────────────────────
    topk_idx_base_ptr = topk_idx_ptr + pid_k_heads * stride_th + pid_q_len_start * stride_tl
    offs_topk = tl.arange(0, topk)
    topk_all  = tl.load(topk_idx_base_ptr + offs_topk * stride_tk)  # [topk]

    # Causal filter on first token
    real_topk = tl.sum(
        tl.where(
            (topk_all >= 0),
            1, 0,
        ),
        axis=0,
    )

    chunk_size: tl.constexpr = topk // SPLIT_K
    start_topk    = pid_split_k * chunk_size
    real_end_topk = tl.minimum(start_topk + chunk_size, real_topk)

    # ── row-index helpers for the BLOCK_CF_H flat layout ─────────────────
    # Row r  →  token group[cf_idx[r]],  head h_idx[r]
    offs_cf_h = tl.arange(0, BLOCK_CF_H)
    cf_idx    = offs_cf_h // BLOCK_H   # [0..CFACTOR-1] per BLOCK_H rows
    h_idx     = offs_cf_h % BLOCK_H    # [0..BLOCK_H-1] repeating
    offs_d    = tl.arange(0, head_dim)
    offs_bk   = tl.arange(0, block_size_k)

    # Validity mask: skip padding rows (out-of-sequence token or excess heads)
    row_valid = (
        ((pid_seq_q_start + cf_idx) < q_len) &
        (h_idx < num_share_q_heads)
    )  # [BLOCK_CF_H]

    # ── load Q: [BLOCK_CF_H, head_dim] ────────────────────────────────────
    # q layout: [total_q_len, num_q_heads, head_dim]
    # Row r → token (pid_q_len_start + cf_idx[r]), head (pid_q_heads + h_idx[r])
    q = tl.load(
        q_ptr
        + (pid_q_len_start + cf_idx[:, None]) * stride_ql
        + (pid_q_heads     + h_idx[:, None])  * stride_qh
        + offs_d[None, :]                      * stride_qd,
        mask=row_valid[:, None],
        other=0.0,
    )  # [BLOCK_CF_H, head_dim]

    # Per-row absolute seq position (for causal mask)
    seq_q_for_row = pid_seq_q_start + cf_idx  # [BLOCK_CF_H]

    # ── accumulators ──────────────────────────────────────────────────────
    m_i   = tl.full((BLOCK_CF_H,), float('-inf'), dtype=tl.float32)
    lse_i = tl.full((BLOCK_CF_H,), float('-inf'), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_CF_H, head_dim), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    # KV base pointers (advances applied inside loop)
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

    # ── sparse attention loop — single KV load serves all CFACTOR tokens ──
    for i in range(start_topk, real_end_topk):
        kv_idx = tl.load(topk_idx_base_ptr + i * stride_tk)

        if kv_idx >= 0:
            c = kv_idx * block_size_k  # KV offset relative to k_start

            k = tl.load(tl.advance(k_ptrs, (0, c)),
                        boundary_check=(1, 0), padding_option="zero")  # [head_dim, block_size_k]

            qk = tl.dot(q, k) * qk_scale  # [BLOCK_CF_H, block_size_k]

            # Causal mask — each row uses its own seq position
            qk += tl.where(
                seq_q_for_row[:, None] >= (kv_idx * block_size_k + offs_bk)[None, :],
                0, float("-inf"),
            )

            if HAS_MASK:
                # Use first token's mask row (consistent with topk approximation)
                mask_ptrs = tl.make_block_ptr(
                    base=attention_mask_ptr,
                    shape=(total_q_len, total_k_len),
                    strides=(stride_aql, stride_akl),
                    offsets=(pid_q_len_start, k_start + c),
                    block_shape=(1, block_size_k),
                    order=(1, 0),
                )
                attn_mask = tl.load(mask_ptrs, boundary_check=(0, 1), padding_option="zero")
                # broadcast (1, block_size_k) → (BLOCK_CF_H, block_size_k)
                qk = tl.where(attn_mask > 0, qk, float("-inf"))

            # ── online softmax ─────────────────────────────────────────────
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))  # [BLOCK_CF_H]

            # NaN guard for p: fully-masked block + m_i=-inf → m_ij=-inf
            #   exp2(qk - (-inf)) = exp2(nan).  Clamp to -inf → exp2(-inf)=0.
            qk_shift = tl.where(m_ij[:, None] > float('-inf'),
                                 qk - m_ij[:, None],
                                 float('-inf'))
            p    = tl.exp2(qk_shift)         # [BLOCK_CF_H, block_size_k]
            l_ij = tl.sum(p, axis=1)         # [BLOCK_CF_H]

            v = tl.load(tl.advance(v_ptrs, (c, 0)),
                        boundary_check=(0, 1), padding_option="zero")  # [block_size_k, head_dim]

            # NaN guard for acc_o_scale: exp2(-inf - (-inf)) = nan → use 1.0
            acc_o_scale = tl.where(m_ij > float('-inf'), tl.exp2(m_i - m_ij), 1.0)
            acc_o = acc_o * acc_o_scale[:, None] + tl.dot(p.to(v.dtype), v)

            # NaN guard for lse_i: skip update when block contributed nothing
            lse_update = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
            lse_i = tl.where(m_ij > float('-inf'), lse_update, lse_i)
            m_i   = m_ij

    # ── normalise partial output ───────────────────────────────────────────
    exp_arg = tl.where(lse_i > float('-inf'), m_i - lse_i, float('-inf'))
    acc_o   = acc_o * tl.exp2(exp_arg)[:, None]

    # ── write partial_o ────────────────────────────────────────────────────
    # Scatter BLOCK_CF_H rows back to their respective token positions.
    # partial_o layout: [total_q_len, num_q_heads, SPLIT_K, head_dim]
    tl.store(
        partial_o_ptr
        + (pid_q_len_start + cf_idx[:, None]) * stride_pol
        + (pid_q_heads     + h_idx[:, None])  * stride_poh
        + pid_split_k                          * stride_pospk
        + offs_d[None, :]                      * stride_pod,
        acc_o.to(partial_o_ptr.dtype.element_ty),
        mask=row_valid[:, None],
    )

    # ── write partial_lse ──────────────────────────────────────────────────
    # partial_lse layout: [num_q_heads, total_q_len, SPLIT_K]
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
    partial_o_ptr,   # [total_q_len, num_q_heads, SPLIT_K, head_dim]
    partial_lse_ptr, # [num_q_heads, total_q_len, SPLIT_K]
    o_ptr,           # [total_q_len, num_q_heads, head_dim]
    lse_ptr,         # [num_q_heads, total_q_len]

    stride_pol, stride_poh, stride_pospk, stride_pod,
    stride_plh, stride_pll, stride_plspk,
    stride_ol,  stride_oh,  stride_od,
    stride_lh,  stride_ll,

    head_dim: tl.constexpr,
    SPLIT_K:  tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_k = tl.arange(0, SPLIT_K)
    offs_d = tl.arange(0, head_dim)

    lse = tl.load(
        partial_lse_ptr
        + pid_h  * stride_plh
        + pid_q  * stride_pll
        + offs_k * stride_plspk,
    )  # [SPLIT_K]

    # Numerically stable combine; guard max_lse=-inf (all shards empty)
    max_lse      = tl.max(lse, axis=0)
    max_lse_safe = tl.where(max_lse == float('-inf'), 0.0, max_lse)

    w     = tl.exp2(lse - max_lse_safe)
    w     = tl.where(lse == float('-inf'), 0.0, w)
    sum_w = tl.sum(w, axis=0)

    global_lse = tl.where(
        sum_w > 0,
        max_lse_safe + tl.math.log2(sum_w),
        float('-inf'),
    )

    scale = tl.exp2(lse - global_lse)
    scale = tl.where(lse == float('-inf'), 0.0, scale)

    partial_o = tl.load(
        partial_o_ptr
        + pid_q           * stride_pol
        + pid_h           * stride_poh
        + offs_k[:, None] * stride_pospk
        + offs_d[None, :] * stride_pod,
    ).to(tl.float32)  # [SPLIT_K, head_dim]

    o = tl.sum(partial_o * scale[:, None], axis=0)

    tl.store(
        o_ptr + pid_q * stride_ol + pid_h * stride_oh + offs_d * stride_od,
        o.to(o_ptr.dtype.element_ty),
    )
    tl.store(lse_ptr + pid_h * stride_lh + pid_q * stride_ll, global_lse)


def _topk_sparse_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    attention_mask: torch.Tensor = None,
    SPLIT_K: int = 32,
    CFACTOR: int = 4,
):
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert block_size in {32, 64, 128, 256}

    total_q_len, num_q_heads, head_dim = q.shape
    total_k_len, num_k_heads, _        = k.shape
    _,           num_v_heads, _        = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    topk       = topk_idx.shape[-1]

    assert topk_idx.shape[0] == num_k_heads
    assert topk_idx.shape[1] == total_q_len
    assert is_power_of_two(block_size)
    assert is_power_of_two(head_dim)
    assert is_power_of_two(topk)
    assert topk % SPLIT_K == 0
    assert is_power_of_two(CFACTOR)
    assert max_seqlen_q % CFACTOR == 0, \
        f"max_seqlen_q ({max_seqlen_q}) must be divisible by CFACTOR ({CFACTOR})"

    if attention_mask is not None:
        assert attention_mask.shape == (total_q_len, total_k_len)
        assert attention_mask.dtype == torch.float32

    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    assert is_power_of_two(num_share_q_heads)

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
        attention_mask = q  # dummy pointer

    stride_mask_q = attention_mask.stride(0) if has_mask else 0
    stride_mask_k = attention_mask.stride(1) if has_mask else 0

    num_warps, num_stages = get_num_warps_stages(head_dim, block_size, IS_HOPPER_GPU)

    grid = (batch_size * num_k_heads * (max_seqlen_q // CFACTOR), SPLIT_K)
    forward_kernel_split_k[grid](
        q, k, v,
        topk_idx,
        partial_o, partial_lse,
        cu_seqlens_q, cu_seqlens_k,
        attention_mask,
        total_q_len, total_k_len, num_k_heads,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        stride_mask_q, stride_mask_k,
        block_size_k=block_size,
        head_dim=head_dim,
        num_share_q_heads=num_share_q_heads,
        topk=topk,
        max_seqlen_q=max_seqlen_q,
        SPLIT_K=SPLIT_K,
        CFACTOR=CFACTOR,
        HAS_MASK=has_mask,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    o   = torch.empty_like(q)
    lse = torch.empty(num_q_heads, total_q_len, dtype=torch.float32, device=q.device)

    grid = (total_q_len, num_q_heads)
    combine_kernel_split_k[grid](
        partial_o, partial_lse,
        o, lse,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0), lse.stride(1),
        head_dim=head_dim,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o, lse

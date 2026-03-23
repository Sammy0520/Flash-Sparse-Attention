import torch
import triton
import triton.language as tl

from impl.utils import get_num_warps_stages, is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()

# ---------------------------------------------------------------------------
# Forward kernel  (split-K)
# Grid: (batch * num_q_heads, num_q_blocks, SPLIT_K)
# Each program handles one Q-block × one KV shard.
# ---------------------------------------------------------------------------

@triton.jit
def forward_kernel_decode_split_k(
    q_ptr, k_ptr, v_ptr,
    partial_o_ptr,   # [total_q_len, num_q_heads, SPLIT_K, head_dim]
    partial_lse_ptr, # [num_q_heads, total_q_len, SPLIT_K]
    kernel_size,
    kernel_stride,
    cu_seqlens_q,
    cu_seqlens_k,
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    sm_scale,
    query_start_index,
    mask_ptr,
    total_q_len,
    compressed_k_len,
    stride_mask_q, stride_mask_k,
    stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_pol, stride_poh, stride_pospk, stride_pod,
    stride_plh, stride_pll, stride_plspk,
    chunk_size,       # KV tokens per shard (aligned to BLOCK_SIZE_K)
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    HAS_MASK:     tl.constexpr,
):
    # ── decode pid ────────────────────────────────────────────────────────
    pid_bh      = tl.program_id(0)
    pid_q_block = tl.program_id(1)
    pid_split_k = tl.program_id(2)

    num_q_heads = NUM_KV_HEADS * NUM_SHARE_Q_HEADS
    pid_b  = pid_bh // num_q_heads
    pid_h  = pid_bh % num_q_heads
    pid_kh = pid_h // NUM_SHARE_Q_HEADS

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len   = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len   = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    q_block_start = pid_q_block * BLOCK_SIZE_Q
    if q_block_start >= q_len:
        return

    # ── causal upper bound on KV (same logic as original) ─────────────────
    # The last query token in this block determines how far into K we can look.
    q_start_in_seq = query_start_index + kernel_size - 1
    hi = tl.minimum(
        k_len,
        (q_start_in_seq + q_block_start + BLOCK_SIZE_Q - kernel_size) // kernel_stride + 1,
    )

    # ── this shard's KV slice ──────────────────────────────────────────────
    shard_lo  = pid_split_k * chunk_size
    shard_hi  = tl.minimum(shard_lo + chunk_size, hi)   # also cap at causal hi

    # ── load Q ────────────────────────────────────────────────────────────
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(q_block_start, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    # causal index vectors (same as baseline)
    off_q = tl.arange(0, BLOCK_SIZE_Q) + q_start_in_seq + q_block_start
    off_k = tl.arange(0, BLOCK_SIZE_K) * kernel_stride + kernel_size - 1

    # ── accumulators ──────────────────────────────────────────────────────
    qk_scale = sm_scale * 1.44269504
    m_i   = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)

    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, shard_lo),           # start at shard boundary
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(shard_lo, 0),           # start at shard boundary
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    # ── KV loop over this shard's slice ───────────────────────────────────
    for i in range(shard_lo, shard_hi, BLOCK_SIZE_K):
        k = tl.load(k_ptrs, boundary_check=(1, 0), padding_option="zero")
        qk  = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(off_q[:, None] >= (i * kernel_stride + off_k)[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * qk_scale

        if HAS_MASK:
            mask_ptrs = tl.make_block_ptr(
                base=mask_ptr,
                shape=(total_q_len, compressed_k_len),
                strides=(stride_mask_q, stride_mask_k),
                offsets=(q_start + q_block_start, k_start + i),
                block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_K),
                order=(1, 0),
            )
            mask_block = tl.load(mask_ptrs, boundary_check=(0, 1), padding_option="zero")
            qk = tl.where(mask_block > 0, qk, float("-inf"))

        # ── online softmax ────────────────────────────────────────────────
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))

        # NaN guard for p:
        #   If this block is fully masked (qk all -inf) and no block has been
        #   processed yet (m_i=-inf), m_ij=-inf and exp2(qk-m_ij)=exp2(nan).
        #   Clamp shift to -inf → exp2(-inf)=0 (no contribution).
        qk_shift = tl.where(m_ij[:, None] > float("-inf"),
                             qk - m_ij[:, None],
                             float("-inf"))
        p    = tl.exp2(qk_shift)
        l_ij = tl.sum(p, axis=1)

        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")

        # NaN guard for acc_o_scale:
        #   Same scenario: exp2(m_i - m_ij)=exp2(nan). acc_o=0 here, so 1.0 is safe.
        acc_o_scale = tl.where(m_ij > float("-inf"),
                                tl.exp2(m_i - m_ij),
                                1.0)
        acc_o = acc_o * acc_o_scale[:, None] + tl.dot(p.to(v.dtype), v)

        # NaN guard for lse_i update:
        #   m_ij=-inf → exp2(lse_i-m_ij)=exp2(nan) inside log2. Skip.
        lse_update = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
        lse_i = tl.where(m_ij > float("-inf"), lse_update, lse_i)

        m_i = m_ij
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))

    # ── normalise partial output ───────────────────────────────────────────
    # Guard: empty shard → m_i=lse_i=-inf → exp2(nan). acc_o stays 0.
    exp_arg = tl.where(lse_i > float("-inf"), m_i - lse_i, float("-inf"))
    acc_o   = acc_o * tl.exp2(exp_arg)[:, None]

    # ── write partial_o ────────────────────────────────────────────────────
    po_ptrs = tl.make_block_ptr(
        base=partial_o_ptr
            + (q_start + q_block_start) * stride_pol
            + pid_h        * stride_poh
            + pid_split_k  * stride_pospk,
        shape=(q_len - q_block_start, HEAD_DIM),
        strides=(stride_pol, stride_pod),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(po_ptrs, acc_o.to(partial_o_ptr.dtype.element_ty), boundary_check=(0, 1))

    # ── write partial_lse ──────────────────────────────────────────────────
    q_row = tl.arange(0, BLOCK_SIZE_Q)
    pl_ptrs = (
        partial_lse_ptr
        + pid_h                           * stride_plh
        + (q_start + q_block_start)       * stride_pll
        + q_row                           * stride_pll
        + pid_split_k                     * stride_plspk
    )
    tl.store(pl_ptrs, lse_i, mask=q_row < (q_len - q_block_start))


# ---------------------------------------------------------------------------
# Combine kernel  (identical logic to sparse split-K combine)
# Grid: (total_q_len, num_q_heads)
# ---------------------------------------------------------------------------

@triton.jit
def combine_kernel_decode_split_k(
    partial_o_ptr,   # [total_q_len, num_q_heads, SPLIT_K, head_dim]
    partial_lse_ptr, # [num_q_heads, total_q_len, SPLIT_K]
    o_ptr,           # [total_q_len, num_q_heads, head_dim]
    lse_ptr,         # [num_q_heads, total_q_len]
    stride_pol, stride_poh, stride_pospk, stride_pod,
    stride_plh, stride_pll, stride_plspk,
    stride_ol,  stride_oh,  stride_od,
    stride_lh,  stride_ll,
    HEAD_DIM: tl.constexpr,
    SPLIT_K:  tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_k = tl.arange(0, SPLIT_K)
    offs_d = tl.arange(0, HEAD_DIM)

    lse = tl.load(
        partial_lse_ptr
        + pid_h  * stride_plh
        + pid_q  * stride_pll
        + offs_k * stride_plspk,
    )  # [SPLIT_K]

    # numerically-stable reduce; guard all-empty (max_lse=-inf)
    max_lse      = tl.max(lse, axis=0)
    max_lse_safe = tl.where(max_lse == float("-inf"), 0.0, max_lse)

    w     = tl.exp2(lse - max_lse_safe)
    w     = tl.where(lse == float("-inf"), 0.0, w)
    sum_w = tl.sum(w, axis=0)

    global_lse = tl.where(
        sum_w > 0,
        max_lse_safe + tl.math.log2(sum_w),
        float("-inf"),
    )

    scale = tl.exp2(lse - global_lse)
    scale = tl.where(lse == float("-inf"), 0.0, scale)

    partial_o = tl.load(
        partial_o_ptr
        + pid_q           * stride_pol
        + pid_h           * stride_poh
        + offs_k[:, None] * stride_pospk
        + offs_d[None, :] * stride_pod,
    ).to(tl.float32)  # [SPLIT_K, HEAD_DIM]

    o = tl.sum(partial_o * scale[:, None], axis=0)  # [HEAD_DIM]

    tl.store(
        o_ptr + pid_q * stride_ol + pid_h * stride_oh + offs_d * stride_od,
        o.to(o_ptr.dtype.element_ty),
    )
    tl.store(lse_ptr + pid_h * stride_lh + pid_q * stride_ll, global_lse)


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------

def _compressed_attention_fwd_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q,
    max_seqlen_k,
    sm_scale: float,
    query_start_index: int,
    attention_mask: torch.Tensor = None,
    SPLIT_K: int = 32,
):
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32

    total_q_len, num_q_heads, head_dim = q.shape
    total_k_len, num_k_heads, _        = k.shape
    _,           num_v_heads, _        = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    if attention_mask is not None:
        assert attention_mask.shape == (total_q_len, total_k_len), (
            f"attention_mask shape {attention_mask.shape} != ({total_q_len}, {total_k_len})"
        )
        assert attention_mask.dtype == torch.float32

    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads

    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    max_seqlen_q_val = max_seqlen_q.item() if hasattr(max_seqlen_q, "item") else max_seqlen_q
    max_seqlen_k_val = max_seqlen_k.item() if hasattr(max_seqlen_k, "item") else max_seqlen_k

    # chunk_size: KV tokens per shard, aligned up to BLOCK_SIZE_K so each shard
    # starts at a block boundary and the range [shard_lo, shard_hi) is a multiple
    # of BLOCK_SIZE_K (last shard may be shorter, handled by boundary_check).
    chunk_size = triton.cdiv(triton.cdiv(max_seqlen_k_val, SPLIT_K), BLOCK_SIZE_K) * BLOCK_SIZE_K

    num_q_blocks = triton.cdiv(max_seqlen_q_val, BLOCK_SIZE_Q)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)

    # ── partial tensors ────────────────────────────────────────────────────
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
    mask_ptr     = attention_mask if has_mask else q
    stride_mask_q = attention_mask.stride(0) if has_mask else 0
    stride_mask_k = attention_mask.stride(1) if has_mask else 0

    # ── forward kernel ─────────────────────────────────────────────────────
    grid_fwd = (batch_size * num_q_heads, num_q_blocks, SPLIT_K)
    forward_kernel_decode_split_k[grid_fwd](
        q, k, v,
        partial_o, partial_lse,
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        NUM_KV_HEADS=num_k_heads,
        NUM_SHARE_Q_HEADS=num_share_q_heads,
        HEAD_DIM=head_dim,
        sm_scale=sm_scale,
        query_start_index=query_start_index,
        mask_ptr=mask_ptr,
        total_q_len=total_q_len,
        compressed_k_len=total_k_len,
        stride_mask_q=stride_mask_q,
        stride_mask_k=stride_mask_k,
        stride_qn=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        stride_pol=partial_o.stride(0),
        stride_poh=partial_o.stride(1),
        stride_pospk=partial_o.stride(2),
        stride_pod=partial_o.stride(3),
        stride_plh=partial_lse.stride(0),
        stride_pll=partial_lse.stride(1),
        stride_plspk=partial_lse.stride(2),
        chunk_size=chunk_size,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        HAS_MASK=has_mask,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # ── combine kernel ─────────────────────────────────────────────────────
    o   = torch.empty_like(q)
    lse = torch.empty(num_q_heads, total_q_len, dtype=torch.float32, device=q.device)

    grid_comb = (total_q_len, num_q_heads)
    combine_kernel_decode_split_k[grid_comb](
        partial_o, partial_lse,
        o, lse,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0), lse.stride(1),
        HEAD_DIM=head_dim,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o, lse

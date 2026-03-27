import math
import torch
import triton
import triton.language as tl
from impl.utils import is_hopper_gpu, get_num_warps_stages, is_power_of_two


IS_HOPPER_GPU = is_hopper_gpu()
WINDOW_BIT = 1 << 30
TOPK_BIT = 1 << 29
INT32_MAX = 2147483647
BITS_MASK  = WINDOW_BIT | TOPK_BIT
NOT_BITS = INT32_MAX ^ BITS_MASK


def construct_hybrid_idx(
    topk_idx: torch.Tensor,      # [num_kv_heads, total_q_len, topk]
    cu_seqlens_k: torch.Tensor,  # [batch_size + 1]
    cu_seqlens_q: torch.Tensor,  # [batch_size + 1]
    block_size: int,
    window_size: int,
    CFACTOR: int = 1,
) -> torch.Tensor:
    num_kv_heads, total_q_len, topk = topk_idx.shape
    batch_size = len(cu_seqlens_k) - 1
    device = topk_idx.device

    num_w_blocks = math.ceil(window_size / block_size)
    max_fused_len = topk + num_w_blocks
    total_c_len = (total_q_len + CFACTOR - 1) // CFACTOR 
    
    hybrid_idx = torch.full(
        (num_kv_heads, total_c_len, max_fused_len),
        INT32_MAX, dtype=torch.int32, device=device
    )
    INV = torch.tensor(INT32_MAX, dtype=torch.int32, device=device)

    for b in range(batch_size):
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        q_len, k_len = q_end - q_start, k_end - k_start
        
        if q_len <= 0 or k_len <= 0:
            continue

        c_start = q_start // CFACTOR
        c_end = (q_end + CFACTOR - 1) // CFACTOR
        c_len = c_end - c_start
        num_k_blocks = math.ceil(k_len / block_size)
        
        # sliding window branch
        first_w = max(0, num_k_blocks - num_w_blocks)
        w_blocks = torch.arange(first_w, num_k_blocks, dtype=torch.int32, device=device)
        w_tagged = (w_blocks | WINDOW_BIT).view(1, 1, -1).expand(num_kv_heads, c_len, -1)

        # topk branch
        tk = topk_idx[:, q_start:q_end:CFACTOR, :]
        tk_valid = (tk >= 0) & (tk < num_k_blocks)
        tk_tagged = torch.where(tk_valid, tk | TOPK_BIT, INV)

        combined = torch.cat([tk_tagged, w_tagged], dim=-1)
        sort_key = torch.where(combined == INV, INV, combined & NOT_BITS)
        _, sort_ord = torch.sort(sort_key, dim=-1)
        
        sorted_raw = torch.gather(sort_key, -1, sort_ord)
        tags = torch.gather(combined, -1, sort_ord) & BITS_MASK

        # ── 4. 纯向量化去重与合并 (基于最大重复次数不超过2的特性) ──
        # 判断相邻元素是否为同一个合法的物理块
        is_dup = (sorted_raw[:, :, :-1] == sorted_raw[:, :, 1:]) & (sorted_raw[:, :, :-1] != INT32_MAX)
        
        # 左右偏移 mask 以执行 tag 的合并
        is_dup_prev = torch.cat([torch.zeros_like(is_dup[:, :, :1]), is_dup], dim=-1)
        is_dup_next = torch.cat([is_dup, torch.zeros_like(is_dup[:, :, :1])], dim=-1)
        shift_tags = torch.cat([torch.zeros_like(tags[:, :, :1]), tags[:, :, :-1]], dim=-1)

        # 核心逻辑：若该项是重复项的后者，则吸纳前项 tag；若是前者，则被判为 INV 废弃
        tags = torch.where(is_dup_prev, tags | shift_tags, tags)
        merged = torch.where(is_dup_next | (sorted_raw == INT32_MAX), INV, sorted_raw | tags)

        # ── 5. 最终规整：将所有的 INV 垫到张量末尾 ──
        _, sort_ord2 = torch.sort(merged, dim=-1)
        merged = torch.gather(merged, -1, sort_ord2)

        # 填入全局 hybrid_idx
        hybrid_idx[:, c_start:c_end, :] = merged

    return hybrid_idx


@triton.jit
def forward_kernel_unified(
    # tensors
    q_ptr, # [total_q_len, num_q_heads, head_dim]
    k_ptr, # [total_k_len, num_k_heads, head_dim]
    v_ptr, # [total_k_len, num_k_heads, head_dim]
    hybrid_idx_ptr, # [num_kv_heads, total_c_len, max_fused_len]
    o_ptr, # [total_q_len, num_q_heads, head_dim]
    cu_seqlens_q_ptr, # [batch_size + 1]
    cu_seqlens_k_ptr, # [batch_size + 1]
    gate_ptr, # [total_q_len, 3]
    attention_mask_ptr, # [total_q_len, total_k_len]

    # shapes
    total_q_len,
    total_k_len,
    num_kv_heads,
    fused_idx_cnt,

    # scale
    sm_scale,

    # stride
    stride_ql, stride_qh, stride_qd, # q_ptr, [total_q_len, num_q_heads, head_dim]
    stride_kl, stride_kh, stride_kd, # k_ptr, [total_k_len, num_k_heads, head_dim]
    stride_vl, stride_vh, stride_vd, # v_ptr, [total_k_len, num_k_heads, head_dim]
    stride_hh, stride_hl, stride_hf, # hybrid_idx_ptr, [num_kv_heads, total_c_len, fused_idx_cnt]
    stride_ol, stride_oh, stride_od, # o_ptr, [total_q_len, num_q_heads, head_dim]
    stride_gl, stride_g3, # gate_ptr, [total_q_len, 3]
    stride_aql, stride_akl, # attention_mask_ptr, [total_q_len, total_k_len]

    # meta parameters
    block_size_k: tl.constexpr,
    head_dim: tl.constexpr,
    num_share_q_heads: tl.constexpr, # num_q_heads // num_k_heads
    BLOCK_FUSED_IDX_CNT: tl.constexpr, # fused_idx_cnt rounded up
    max_seqlen_q: tl.constexpr,
    CFACTOR: tl.constexpr,
    HAS_MASK: tl.constexpr,
    INT32_MAX: tl.constexpr,
    WINDOW_BIT: tl.constexpr,
    TOPK_BIT: tl.constexpr,
    NOT_BITS: tl.constexpr,
):
    max_seqlen_q_cf: tl.constexpr = tl.cdiv(max_seqlen_q, CFACTOR)

    # Process pid 0
    pid_0 = tl.program_id(0)
    pid_batch = pid_0 // (num_kv_heads * max_seqlen_q_cf) # [0, batch_size)
    pid_k_heads = (pid_0 % (num_kv_heads * max_seqlen_q_cf)) // max_seqlen_q_cf # [0, num_k_heads)
    pid_seq_q_start = (pid_0 % max_seqlen_q_cf) * CFACTOR  # first seq-pos in this group
    pid_q_heads = pid_k_heads * num_share_q_heads

    # start and len of this batch
    q_start = tl.load(cu_seqlens_q_ptr + pid_batch)
    q_len = tl.load(cu_seqlens_q_ptr + pid_batch + 1) - q_start
    k_start = tl.load(cu_seqlens_k_ptr + pid_batch)
    k_len = tl.load(cu_seqlens_k_ptr + pid_batch + 1) - k_start

    if pid_seq_q_start >= q_len:
        return
    
    BLOCK_H: tl.constexpr = num_share_q_heads if num_share_q_heads >= 16 else 16
    BLOCK_CF_H: tl.constexpr = CFACTOR * BLOCK_H
    pid_q_len_start = q_start + pid_seq_q_start # [0, total_q_len)

    # Load hybrid_idx
    hybrid_idx_base_ptr = hybrid_idx_ptr + pid_k_heads * stride_hh + (pid_q_len_start // CFACTOR) * stride_hl
    offs_f = tl.arange(0, BLOCK_FUSED_IDX_CNT)
    mask_f = offs_f < fused_idx_cnt
    hybrid_idx = tl.load(hybrid_idx_base_ptr + offs_f * stride_hf, mask=mask_f, other=INT32_MAX)

    # Calculate real_fused_idx_cnt
    real_fused_idx_cnt = tl.sum(tl.where((hybrid_idx >= 0) & (hybrid_idx != INT32_MAX), 1, 0), axis=0)

    offs_cf_h = tl.arange(0, BLOCK_CF_H)
    cf_idx = offs_cf_h // BLOCK_H # [0..CFACTOR-1] per BLOCK_H rows
    h_idx = offs_cf_h % BLOCK_H # [0..BLOCK_H-1] repeating
    offs_d = tl.arange(0, head_dim)
    row_valid = ((pid_seq_q_start + cf_idx) < q_len) & (h_idx < num_share_q_heads) # [BLOCK_CF_H]

    # Load gate values
    gate_topk = tl.load(gate_ptr + (pid_q_len_start + cf_idx) * stride_gl + 1 * stride_g3, mask=row_valid, other=0.0)
    gate_win = tl.load(gate_ptr + (pid_q_len_start + cf_idx) * stride_gl + 2 * stride_g3, mask=row_valid, other=0.0)

    # Load Q: [BLOCK_CF_H, head_dim]
    q = tl.load(
        q_ptr
        + (pid_q_len_start + cf_idx[:, None]) * stride_ql
        + (pid_q_heads + h_idx[:, None]) * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=row_valid[:, None],
        other=0.0,
    )

    # Init states
    m_i = tl.full((BLOCK_CF_H,), float('-inf'), dtype=tl.float32)
    lse_i = tl.full((BLOCK_CF_H,), float('-inf'), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_CF_H, head_dim), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    # Init ptrs for k, v
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

    for i in range(0, real_fused_idx_cnt):
        hybrid_idx_cur = tl.load(hybrid_idx_base_ptr + i * stride_hf)
        raw_idx = hybrid_idx_cur & NOT_BITS
        is_topk = (hybrid_idx_cur & TOPK_BIT) != 0
        is_win = (hybrid_idx_cur & WINDOW_BIT) != 0

        # compute gate value and index
        gate_val = tl.where(is_topk, gate_topk, 0.0) + tl.where(is_win, gate_win, 0.0)
        c = raw_idx * block_size_k

        # load k and compute qk
        k = tl.load(tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero")
        qk = tl.dot(q, k) * qk_scale # [BLOCK_CF_H, block_size_k]

        # custom mask
        if HAS_MASK:
            mask_ptrs = tl.make_block_ptr(
                base=attention_mask_ptr, shape=(total_q_len, total_k_len),
                strides=(stride_aql, stride_akl), offsets=(pid_q_len_start, k_start + c),
                block_shape=(1, block_size_k), order=(1, 0),
            )
            attn_mask = tl.load(mask_ptrs, boundary_check=(0, 1), padding_option="zero")
            qk = tl.where(attn_mask > 0, qk, float("-inf"))

        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1)) # [BLOCK_CF_H]
        qk_shift = tl.where(m_ij[:, None] > float('-inf'), qk - m_ij[:, None], float('-inf')) # [BLOCK_CF_H, block_size_k]
        p = tl.exp2(qk_shift) # [BLOCK_CF_H, block_size_k]
        l_ij = tl.sum(p, axis=1) # [BLOCK_CF_H]

        # load v
        v = tl.load(tl.advance(v_ptrs, (c, 0)), boundary_check=(0, 1), padding_option="zero")

        # update acc_o
        p_gated = p * gate_val[:, None] # [BLOCK_CF_H, block_size_k]
        acc_o_scale = tl.where(m_ij > float('-inf'), tl.exp2(m_i - m_ij), 1.0)
        acc_o = acc_o * acc_o_scale[:, None] + tl.dot(p_gated.to(v.dtype), v)

        # update m_i, lse_i
        m_i = m_ij
        lse_update = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
        lse_i = tl.where(m_ij > float('-inf'), lse_update, lse_i)

    # final scale
    exp_arg = tl.where(lse_i > float('-inf'), m_i - lse_i, float('-inf'))
    acc_o = acc_o * tl.exp2(exp_arg)[:, None]

    # write output
    tl.store(
        o_ptr
        + (pid_q_len_start + cf_idx[:, None]) * stride_ol
        + (pid_q_heads + h_idx[:, None]) * stride_oh
        + offs_d[None, :] * stride_od,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=row_valid[:, None],
    )


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
    CFACTOR: int = 4,
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
    assert is_power_of_two(block_size)
    assert is_power_of_two(head_dim)

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

    # check CFACTOR
    assert is_power_of_two(CFACTOR)

    # gqa
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    assert is_power_of_two(num_share_q_heads)

    # softmax scale
    if sm_scale is None or sm_scale == 0.0:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # construct hybrid index [num_kv_heads, total_c_len, max_fused_len]
    hybrid_idx: torch.Tensor = construct_hybrid_idx(topk_idx, cu_seqlens_k, cu_seqlens_q, block_size, window_size)
    fused_idx_cnt = hybrid_idx.shape[-1]
    BLOCK_FUSED_IDX_CNT = triton.next_power_of_2(fused_idx_cnt)

    # output tensor
    o = torch.zeros_like(q)

    # launch kernel
    num_warps, num_stages = get_num_warps_stages(head_dim, block_size, IS_HOPPER_GPU)
    grid = (batch_size * num_k_heads * triton.cdiv(max_seqlen_q, CFACTOR),)
    forward_kernel_unified[grid](
        # tensors
        q,
        k,
        v,
        hybrid_idx,
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        gate,
        attention_mask,

        # shape
        total_q_len,
        total_k_len,
        num_k_heads,
        fused_idx_cnt,

        # scale
        sm_scale,

        # stride
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        hybrid_idx.stride(0), hybrid_idx.stride(1), hybrid_idx.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        gate.stride(0), gate.stride(1),
        stride_mask_q, stride_mask_k,

        # meta parameters
        block_size_k=block_size,
        head_dim=head_dim,
        num_share_q_heads=num_share_q_heads,
        BLOCK_FUSED_IDX_CNT=BLOCK_FUSED_IDX_CNT,
        max_seqlen_q=max_seqlen_q,
        CFACTOR=CFACTOR,
        HAS_MASK=has_mask,
        INT32_MAX=INT32_MAX,
        WINDOW_BIT=WINDOW_BIT,
        TOPK_BIT=TOPK_BIT,
        NOT_BITS=NOT_BITS,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o

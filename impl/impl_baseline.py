import torch
import triton
import triton.language as tl

from impl.utils import is_hopper_gpu, get_num_warps_stages

IS_HOPPER_GPU = is_hopper_gpu()


@triton.jit
def forward_kernel_orig(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    t_ptr,  # topk_idx: kh x n x k
    o_ptr,  # O: n x h x d
    lse_ptr,  # LSE: h x n
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    block_size,
    # sm_scale
    sm_scale,
    # optional mask (total_q_len, total_k_len), 1=attend 0=mask
    mask_ptr,
    total_q_len,
    total_k_len,
    stride_mask_q,
    stride_mask_k,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    # META parameters
    # q loop num
    num_q_loop: tl.constexpr,
    num_k_loop: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid = tl.program_id(0)

    Q = MAX_SEQ_LEN // num_q_loop
    HK = NUM_KV_HEADS // num_k_loop

    # 第几个 (b, kh_chunk, q_chunk)
    pid_b = pid // (HK * Q)
    pid_kh_chunk = (pid % (HK * Q)) // Q  # 每个block处理num_k_loop个KV head
    pid_q = pid % Q

    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    if pid_q * num_q_loop >= q_len:
        return
    real_q_loop = min(num_q_loop, q_len - pid_q * num_q_loop)

    for kh_offset in range(num_k_loop):
        pid_kh = pid_kh_chunk * num_k_loop + kh_offset
        pid_h = pid_kh * NUM_SHARE_Q_HEADS

        for j in range(real_q_loop):
            pid_q_j = pid_q * num_q_loop + j
            # init topk idx pointer
            off_t = tl.arange(0, BLOCK_SIZE_T)
            t_ptr_j = t_ptr + (q_start + pid_q_j) * stride_tn + pid_kh * stride_th
            topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < TOPK, other=-1)

            """Removed causal attention, which should be:
            real_topk = tl.sum(
                tl.where((topk_idx >= 0) & (topk_idx <= pid_q_j // block_size), 1, 0),
                axis=0,
            )
            """
            # real_topk = tl.sum(
            #     tl.where((topk_idx >= 0), 1, 0),
            #     axis=0,
            # )
            real_topk = tl.sum(
                tl.where((topk_idx >= 0) & (topk_idx <= pid_q_j // block_size), 1, 0),
                axis=0,
            )
            # init qkv pointer
            q_ptrs = tl.make_block_ptr(
                base=q_ptr + (q_start + pid_q_j) * stride_qn + pid_h * stride_qh,
                shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
                strides=(stride_qh, stride_qd),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
                order=(1, 0),
            )
            k_ptrs = tl.make_block_ptr(
                base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
                shape=(HEAD_DIM, k_len),
                strides=(stride_kd, stride_kn),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                order=(0, 1),
            )
            v_ptrs = tl.make_block_ptr(
                base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
                shape=(k_len, HEAD_DIM),
                strides=(stride_vn, stride_vd),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            # load q
            q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
            # init statistics
            off_h = tl.arange(0, BLOCK_SIZE_H)
            off_k = tl.arange(0, BLOCK_SIZE_K)
            m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
            # sparse attention
            for i in range(real_topk):
                # get current block start index
                c = tl.load(t_ptr_j).to(tl.int32) * BLOCK_SIZE_K
                t_ptr_j = t_ptr_j + stride_tk
                # load k
                k = tl.load(tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero")
                # compute qk
                qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
                qk += tl.where((pid_q_j >= c + off_k)[None, :], 0, float("-inf"))
                # [BLOCK_SIZE_H, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIZE_K] -> [BLOCK_SIZE_H, BLOCK_SIZE_K]
                qk += tl.dot(q, k) * qk_scale
                # optional tree/custom mask: 1=attend, 0=mask
                if HAS_MASK:
                    mask_ptrs = tl.make_block_ptr(
                        base=mask_ptr,
                        shape=(total_q_len, total_k_len),
                        strides=(stride_mask_q, stride_mask_k),
                        offsets=(q_start + pid_q_j, k_start + c),
                        block_shape=(1, BLOCK_SIZE_K),
                        order=(1, 0),
                    )
                    mask_block = tl.load(mask_ptrs, boundary_check=(0, 1), padding_option="zero")
                    # (1, BLOCK_SIZE_K) -> broadcast to (BLOCK_SIZE_H, BLOCK_SIZE_K)
                    qk = tl.where(mask_block > 0, qk, float("-inf"))
                # compute m_ij and l_ij
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp2(qk - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)
                # scale acc_o
                acc_o_scale = tl.exp2(m_i - m_ij)
                acc_o = acc_o * acc_o_scale[:, None]
                # load v and update acc_o
                v = tl.load(tl.advance(v_ptrs, (c, 0)), boundary_check=(0, 1), padding_option="zero")
                p = p.to(v.dtype)
                acc_o += tl.dot(p, v)
                # update statistics
                m_i = m_ij
                lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)

            # final scale
            acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
            # save output
            o_ptrs = tl.make_block_ptr(
                base=o_ptr + (q_start + pid_q_j) * stride_on + pid_h * stride_oh,
                shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
                strides=(stride_oh, stride_od),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
                order=(1, 0),
            )
            tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
            # save lse
            lse_ptrs = lse_ptr + (q_start + pid_q_j) * stride_ln + (pid_h + off_h) * stride_lh
            tl.store(lse_ptrs, lse_i, mask=off_h < NUM_SHARE_Q_HEADS)


def _topk_sparse_attention_fwd(
    q: torch.Tensor,  # [total_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_k_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_k_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    attention_mask: torch.Tensor = None,
):
    """attention_mask: optional (total_q_len, total_k_len), 1=attend 0=mask. Applied in kernel before softmax."""
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert block_size in {32, 64, 128, 256}
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    # assert q_len == k_len and k_len == v_len
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == num_k_heads
    assert topk_idx.shape[1] == q_len
    if attention_mask is not None:
        assert attention_mask.shape == (q_len, k_len), (
            f"attention_mask shape {attention_mask.shape} != (total_q_len, total_k_len) ({q_len}, {k_len})"
        )
        assert attention_mask.dtype == torch.float32, "attention_mask must be float32"
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # output tensor
    o = torch.zeros_like(q)

    lse = torch.zeros(num_q_heads, q_len, dtype=torch.float32, device=q.device)

    # launch kernel
    num_q_loop = num_k_loop = 1
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_H = max(16, triton.next_power_of_2(num_share_q_heads))
    BLOCK_SIZE_T = triton.next_power_of_2(topk)

    def grid(meta):
        grid = (
            batch_size * triton.cdiv(num_k_heads, num_k_loop) * triton.cdiv(max_seqlen_q, num_q_loop),
        )
        return grid

    has_mask = attention_mask is not None
    mask_ptr = attention_mask if has_mask else q
    stride_mask_q = attention_mask.stride(0) if has_mask else 0
    stride_mask_k = attention_mask.stride(1) if has_mask else 0

    num_warps, num_stages = get_num_warps_stages(head_dim, block_size, IS_HOPPER_GPU)
    forward_kernel_orig[grid](
        q,
        k,
        v,
        topk_idx,
        o,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        topk,
        block_size,
        sm_scale,
        mask_ptr,
        q_len,
        k_len,
        stride_mask_q,
        stride_mask_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        num_q_loop=num_q_loop,
        num_k_loop=num_k_loop,
        MAX_SEQ_LEN=max_seqlen_q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        HAS_MASK=has_mask,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse

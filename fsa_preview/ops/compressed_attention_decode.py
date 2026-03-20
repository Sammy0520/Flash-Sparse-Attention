import math
import warnings
from typing import Tuple

import torch
import triton
import triton.language as tl

from nsa_ref.ops.compressed_attention import _transform_score_kernel
from nsa_ref.ops.utils import get_num_warps_stages, is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()


def _compressed_attention_fwd_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    sm_scale: float,
    query_start_index: int,
    attention_mask: torch.Tensor = None,
):
    """attention_mask: optional (total_q_len, compressed_k_len), 1=attend 0=mask. Applied in kernel before softmax."""
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert k_len == v_len
    assert q_len >= 1  # decode supports multiple queries (e.g. speculative decoding)
    if attention_mask is not None:
        assert attention_mask.shape == (q_len, k_len), (
            f"attention_mask shape {attention_mask.shape} != (total_q_len, compressed_k_len) ({q_len}, {k_len})"
        )
        assert attention_mask.dtype == torch.float32, "attention_mask must be float32"
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # output tensor
    o = torch.zeros_like(q)
    lse = torch.full(
        (num_q_heads, q_len),
        fill_value=-torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    max_seqlen_q_val = max_seqlen_q.item() if hasattr(max_seqlen_q, "item") else max_seqlen_q
    num_q_blocks = triton.cdiv(max_seqlen_q_val, BLOCK_SIZE_Q)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
    debug = torch.zeros(16).to(torch.float32).cuda()
    has_mask = attention_mask is not None
    mask_ptr = attention_mask if has_mask else q  # dummy when HAS_MASK=0
    stride_mask_q = attention_mask.stride(0) if has_mask else 0
    stride_mask_k = attention_mask.stride(1) if has_mask else 0
    grid = lambda META: (batch_size, num_q_heads, num_q_blocks)
    forward_kernel_decode[grid](
        q, k, v, o, lse, debug,
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
        total_q_len=q_len,
        compressed_k_len=k_len,
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
        stride_on=o.stride(0),
        stride_oh=o.stride(1),
        stride_od=o.stride(2),
        stride_lh=lse.stride(0),
        stride_ln=lse.stride(1),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
        HAS_MASK=has_mask,
    )
    return o, lse


def _compressed_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    query_start_index: int = 0,
    attention_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention between query and compressed key and value. Compute attention output and topk block idx used in topk_sparse_attention.

    attention_mask: Optional (total_q_len, compressed_k_len), 1=attend 0=mask. For tree decoding (e.g. EAGLE); applied in kernel before softmax.

    Args:
        q (torch.Tensor): shape [total_q_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        kernel_size (int): kernel size in compress_key_value
        kernel_stride (int): stride of compress_key_value
        block_size (int): key value block size for topk sparse attention.
        topk (int): number of blocks for each query.
        cu_seqlens_q (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        cu_seqlens_k (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_k in flash_attn_func_varlen.
        max_seqlen_q (int): max q len of the batch.
        max_seqlen_k (int): max k len of the batch.
        sm_scale (float, optional): softmax scale. Defaults to None, means 1/sqrt(head_dim).
        init_blocks (int, optional): Number of init blocks for each query. Defaults to 1.
        local_blocks (int, optional): Number of local blocks for each query. Defaults to 2.
        parallel_topk_compute (str, optional): Only set it to False when the sequence length is too long. This can avoid a current bug.
            We'll fix this issue later. Defaults to auto, it will be set to False when the sequence length is greater than 32k and True otherwise.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: attention output and topk_idx used in topk_sparse_attention
    """
    if max_seqlen_q is None:
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    if max_seqlen_k is None:
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype and k.dtype == v.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    # softmax scale
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(q.shape[-1])

    attn_output, lse = _compressed_attention_fwd_decode(
        q,
        k,
        v,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
        query_start_index,
        attention_mask=attention_mask,
    )

    # do not select topk index
    if topk <= 0:
        warnings.warn("topk <= 0, returned topk_idx will be None")
        return attn_output, None

    assert topk >= init_blocks + local_blocks
    with torch.no_grad():
        # with cuda_timer("score kernels"):
        score = _get_attention_score_decode(
            q,
            k,
            lse,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
            query_start_index,
        )
        # transform score to block-wise score
        score = transform_score_decode(
                score,
                kernel_size,
                kernel_stride,
                block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                init_blocks,
                local_blocks,
            )

        # get topk
        topk = min(topk, score.shape[-1])
        topk_idx = score.topk(topk, dim=-1).indices
        topk_idx = topk_idx.to(torch.int32)

    return attn_output, topk_idx


def _get_attention_score_decode(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    lse: torch.Tensor,  # [num_q_heads, total_query_len]
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    query_start_index,
) -> torch.Tensor:
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert lse.dtype == torch.float32  # lse here is log2(sum(exp(qk*scale))), not log(sum(exp(qk*scale)))
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(head_dim)
    # gqa
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # init score
    score = torch.zeros(num_k_heads, q_len, max_seqlen_k, dtype=torch.float32, device=q.device)

    # launch kernel
    grid = lambda META: (
        batch_size * num_k_heads,
        1,
        triton.cdiv(max_seqlen_k, META["BLOCK_SIZE_K"]),
    )
    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    score_kernel_decode[grid](
        q,
        k,
        lse,
        score,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        query_start_index,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        lse.stride(0),
        lse.stride(1),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=4,
        num_stages=3,
    )
    return score


def transform_score_decode(
    score: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> torch.Tensor:
    num_k_heads, total_query_len, max_key_len = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    pad_len = kernel_size // kernel_stride - 1
    max_blocks = math.ceil(max_seqlen_k / block_size)

    block_score = torch.zeros(
        num_k_heads,
        total_query_len,
        max_blocks,
        dtype=torch.float32,
        device=score.device,
    )
    offs = (
        torch.arange(kernel_size // kernel_stride, device=score.device)[:, None]
        + torch.arange(block_size // kernel_stride, device=score.device)[None, :]
    ).view(-1)

    offs = torch.histc(offs, bins=offs.max() + 1, min=0, max=offs.max())

    num_offs = int(offs.shape[0])

    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_K = min(128, triton.next_power_of_2(max_blocks))
    BLOCK_SIZE_O = triton.next_power_of_2(num_offs)

    def grid(meta):
        grid = (
            num_k_heads * batch_size,
            triton.cdiv(total_query_len, BLOCK_SIZE_Q),
            triton.cdiv(max_blocks, BLOCK_SIZE_K),
        )
        return grid

    _transform_score_kernel[grid](
        score,
        block_score,
        offs,
        cu_seqlens_q,
        num_k_heads,
        offs.shape[0],
        max_key_len,
        max_blocks,
        pad_len,
        block_size,
        block_size // kernel_stride,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_score.stride(0),
        block_score.stride(1),
        block_score.stride(2),
        TOTAL_QUERY_LEN=total_query_len,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_O=BLOCK_SIZE_O,
        num_warps=4,
        num_stages=3,
    )

    return block_score


@triton.jit
def forward_kernel_decode(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    debug_ptr,
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
    stride_mask_q,
    stride_mask_k,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Per-Q-block kernel: grid (batch, heads, num_q_blocks), one program per Q block."""
    qk_scale = sm_scale * 1.44269504
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q_block = tl.program_id(2)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    q_block_start = pid_q_block * BLOCK_SIZE_Q
    if q_block_start >= q_len:
        return
    q_start_in_seq = query_start_index + kernel_size - 1

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(q_block_start, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
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
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    off_q = tl.arange(0, BLOCK_SIZE_Q) + q_start_in_seq + q_block_start
    off_k = tl.arange(0, BLOCK_SIZE_K) * kernel_stride + kernel_size - 1
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_D), 0, dtype=tl.float32)
    lo = 0
    hi = min(k_len, (q_start_in_seq + q_block_start + BLOCK_SIZE_Q - kernel_size) // kernel_stride + 1)
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        k = tl.load(k_ptrs, boundary_check=(1, 0), padding_option="zero")
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
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
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o_scale = tl.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))
    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(q_block_start, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    q_row = q_block_start + tl.arange(0, BLOCK_SIZE_Q)
    l_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh + q_row * stride_ln
    tl.store(l_ptrs, lse_i, mask=q_row < q_len)


@triton.jit
def score_kernel_decode(
    q_ptr,
    k_ptr,
    lse_ptr,
    s_ptr,
    kernel_size,
    kernel_stride,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    query_start_index,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_lh,
    stride_ln,
    stride_sh,
    stride_sq,
    stride_sk,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_bkh = tl.program_id(0)
    pid_b = pid_bkh // NUM_KV_HEADS
    pid_kh = pid_bkh % NUM_KV_HEADS
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if pid_q * BLOCK_SIZE_Q >= q_len or pid_k * BLOCK_SIZE_K >= k_len:
        return
    # init k pointer and load k
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    off_k = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
    causal_mask = off_q[:, None] + query_start_index >= (off_k * kernel_stride + kernel_size - 1)[None, :]
    # init score
    s = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_kh * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + q_start * stride_ln + pid_kh * stride_lh,
        shape=(q_len, 1),
        strides=(stride_ln, stride_lh),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1),
    )
    # load q and lse
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
    # compute qk
    qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
    qk += tl.dot(q, k) * qk_scale
    # compute score
    s += tl.where(causal_mask, tl.exp2(qk - lse), 0)
    # save output
    s_ptrs = tl.make_block_ptr(
        base=s_ptr + pid_kh * stride_sh + q_start * stride_sq,
        shape=(q_len, k_len),
        strides=(stride_sq, stride_sk),
        offsets=(pid_q * BLOCK_SIZE_Q, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_K),
        order=(1, 0),
    )
    tl.store(s_ptrs, s.to(s_ptr.dtype.element_ty), boundary_check=(0, 1))

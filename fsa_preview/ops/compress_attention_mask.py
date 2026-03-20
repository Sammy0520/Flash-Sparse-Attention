import torch
import triton
import triton.language as tl


@triton.jit
def compress_2d_attention_mask_kernel(
    attention_mask_ptr, # (total_q_len, total_k_len)
    compressed_mask_ptr, # (total_q_len, compressed_k_len)
    stride_atql, stride_atkl,
    stride_cmtql, stride_cmckl,
    total_k_len, compressed_k_len,
    kernel_stride,
    kernel_size,
    KERNEL_SIZE_BLOCK: tl.constexpr,
):
    pid_ql = tl.program_id(0) # [0, total_q_len)
    pid_ckl = tl.program_id(1) # [0, compressed_k_len)

    attention_mask_ptr += stride_atql * pid_ql # (total_k_len,)
    compressed_mask_ptr += stride_cmtql * pid_ql + stride_cmckl * pid_ckl

    offs = pid_ckl * kernel_stride + tl.arange(0, KERNEL_SIZE_BLOCK)
    offs_attention_mask = offs * stride_atkl
    mask_attention_mask = (offs < total_k_len) & (tl.arange(0, KERNEL_SIZE_BLOCK) < kernel_size)
    attention_mask = tl.load(attention_mask_ptr + offs_attention_mask, mask=mask_attention_mask, other=0.0)

    tl.store(compressed_mask_ptr, tl.max(attention_mask))


def compress_attention_mask(
    attention_mask: torch.Tensor,
    compressed_k_len: int,
    kernel_stride: int,
    kernel_size: int,
):
    if attention_mask is None:
        return None

    assert attention_mask.dim() in (2, 3), "attention_mask must be 2D or 3D"

    if attention_mask.dim() == 2:
        mask_2d = attention_mask
    else:
        mask_2d = attention_mask.max(dim=0).values  # (total_q_len, total_k_len)

    total_q_len, total_k_len = mask_2d.shape

    compressed_mask = torch.empty(
        total_q_len, compressed_k_len, device=attention_mask.device, dtype=torch.float32
    )

    grid = (total_q_len, compressed_k_len)
    compress_2d_attention_mask_kernel[grid](
        mask_2d,
        compressed_mask,
        mask_2d.stride(0), mask_2d.stride(1),
        compressed_mask.stride(0), compressed_mask.stride(1),
        total_k_len, compressed_k_len,
        kernel_stride,
        kernel_size,
        KERNEL_SIZE_BLOCK=triton.next_power_of_2(kernel_size),
    )

    return compressed_mask


def compress_attention_mask_baseline(
    attention_mask: torch.Tensor,
    compressed_k_len: int,
    kernel_stride: int,
    kernel_size: int,
):
    attention_mask_compressed = None

    if attention_mask is not None:
        mask_2d = attention_mask if attention_mask.dim() == 2 else attention_mask.max(dim=0).values
        total_q_len, _ = mask_2d.shape
        # Block c covers uncompress indices [c*kernel_stride, c*kernel_stride+kernel_size)
        compressed_mask = torch.zeros(
            total_q_len, compressed_k_len, device=attention_mask.device, dtype=torch.float32
        )

        for c in range(compressed_k_len):
            start = c * kernel_stride
            end = min(start + kernel_size, mask_2d.shape[1])
            compressed_mask[:, c] = mask_2d[:, start:end].to(torch.float32).amax(dim=1)

        attention_mask_compressed = compressed_mask

    return attention_mask_compressed

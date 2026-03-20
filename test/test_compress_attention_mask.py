import torch
import triton
import math

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fsa_preview.ops.compress_attention_mask import (
    compress_attention_mask,
    compress_attention_mask_baseline,
)


def _compressed_k_len(total_k_len: int, kernel_stride: int) -> int:
    import math
    return math.ceil(total_k_len / kernel_stride)


def _run_both(mask_2d_or_3d, compressed_k_len, kernel_stride, kernel_size, atol=1e-5):
    tri = compress_attention_mask(mask_2d_or_3d, compressed_k_len, kernel_stride, kernel_size)
    ref = compress_attention_mask_baseline(mask_2d_or_3d, compressed_k_len, kernel_stride, kernel_size)
    assert tri is not None and ref is not None
    assert tri.shape == ref.shape, f"shape mismatch: triton={tri.shape}, ref={ref.shape}"
    max_err = (tri - ref).abs().max().item()
    assert max_err <= atol, (
        f"max abs error={max_err:.2e} > atol={atol}\n"
        f"triton:\n{tri}\nbaseline:\n{ref}"
    )
    return tri, ref


def test_none_input():
    assert compress_attention_mask(None, 4, 2, 4) is None
    assert compress_attention_mask_baseline(None, 4, 2, 4) is None
    print("✓ test_none_input")


def test_minimal():
    mask = torch.ones(1, 1, device="cuda", dtype=torch.float32)
    ckl = _compressed_k_len(1, 1)
    _run_both(mask, ckl, kernel_stride=1, kernel_size=1)
    print("✓ test_minimal")


def test_stride_equals_size():
    torch.manual_seed(0)
    total_q, total_k = 8, 16
    kernel_stride = kernel_size = 4
    ckl = _compressed_k_len(total_k, kernel_stride)
    mask = torch.randint(0, 2, (total_q, total_k), device="cuda").float()
    _run_both(mask, ckl, kernel_stride, kernel_size)
    print("✓ test_stride_equals_size")


def test_stride_less_than_size():
    torch.manual_seed(1)
    total_q, total_k = 6, 20
    kernel_stride, kernel_size = 2, 4
    ckl = _compressed_k_len(total_k, kernel_stride)
    mask = torch.randint(0, 2, (total_q, total_k), device="cuda").float()
    _run_both(mask, ckl, kernel_stride, kernel_size)
    print("✓ test_stride_less_than_size")


def test_stride_greater_than_size():
    torch.manual_seed(2)
    total_q, total_k = 5, 24
    kernel_stride, kernel_size = 6, 3
    ckl = _compressed_k_len(total_k, kernel_stride)
    mask = torch.randint(0, 2, (total_q, total_k), device="cuda").float()
    _run_both(mask, ckl, kernel_stride, kernel_size)
    print("✓ test_stride_greater_than_size")


def test_non_divisible_total_k():
    torch.manual_seed(3)
    total_q, total_k = 7, 13
    kernel_stride = kernel_size = 4
    ckl = _compressed_k_len(total_k, kernel_stride)
    mask = torch.randint(0, 2, (total_q, total_k), device="cuda").float()
    _run_both(mask, ckl, kernel_stride, kernel_size)
    print("✓ test_non_divisible_total_k")


def test_3d_input():
    torch.manual_seed(4)
    n_heads, total_q, total_k = 4, 8, 16
    kernel_stride = kernel_size = 4
    ckl = _compressed_k_len(total_k, kernel_stride)
    mask_3d = torch.randint(0, 2, (n_heads, total_q, total_k), device="cuda").float()
    _run_both(mask_3d, ckl, kernel_stride, kernel_size)
    print("✓ test_3d_input")


def test_all_zeros():
    mask = torch.zeros(4, 16, device="cuda", dtype=torch.float32)
    ckl = _compressed_k_len(16, 4)
    tri, ref = _run_both(mask, ckl, 4, 4)
    assert tri.max().item() == 0.0
    print("✓ test_all_zeros")


def test_all_ones():
    mask = torch.ones(4, 16, device="cuda", dtype=torch.float32)
    ckl = _compressed_k_len(16, 4)
    tri, ref = _run_both(mask, ckl, 4, 4)
    assert tri.min().item() == 1.0
    print("✓ test_all_ones")


def test_large():
    torch.manual_seed(5)
    total_q, total_k = 64, 512
    kernel_stride = kernel_size = 8
    ckl = _compressed_k_len(total_k, kernel_stride)
    mask = torch.randint(0, 2, (total_q, total_k), device="cuda").float()
    _run_both(mask, ckl, kernel_stride, kernel_size)
    print("✓ test_large")


def test_multi_block():
    torch.manual_seed(6)
    total_q, total_k = 16, 256
    kernel_stride = kernel_size = 2
    ckl = _compressed_k_len(total_k, kernel_stride)
    assert ckl == 128
    mask = torch.randint(0, 2, (total_q, total_k), device="cuda").float()
    _run_both(mask, ckl, kernel_stride, kernel_size)
    print("✓ test_multi_block")


def do_benchmark():
    SHAPE        = (4, 1003)
    KERNEL_STRIDE = 16
    KERNEL_SIZE   = 32
    COMPRESSED_K_LEN = math.ceil(SHAPE[1] / KERNEL_STRIDE)

    mask = torch.randint(0, 2, SHAPE, device="cuda", dtype=torch.float32)

    ms_triton = triton.testing.do_bench(
        lambda: compress_attention_mask(mask, COMPRESSED_K_LEN, KERNEL_STRIDE, KERNEL_SIZE),
        warmup=100,
        rep=500,
    )

    ms_baseline = triton.testing.do_bench(
        lambda: compress_attention_mask_baseline(mask, COMPRESSED_K_LEN, KERNEL_STRIDE, KERNEL_SIZE),
        warmup=20,
        rep=200,
    )

    us_triton   = ms_triton   * 1e3
    us_baseline = ms_baseline * 1e3

    print(f"{'':─<44}")
    print(f"  {'Triton kernel':<20} {us_triton:>8.2f}  µs")
    print(f"  {'PyTorch baseline':<20} {us_baseline:>8.2f}  µs")
    print(f"{'':─<44}")
    print(f"  Speedup  {us_baseline / us_triton:.2f}x  ({'faster' if us_triton < us_baseline else 'slower'})")
    print(f"{'':─<44}")


if __name__ == "__main__":
    test_none_input()
    test_minimal()
    test_stride_equals_size()
    test_stride_less_than_size()
    test_stride_greater_than_size()
    test_non_divisible_total_k()
    test_3d_input()
    test_all_zeros()
    test_all_ones()
    test_large()
    test_multi_block()
    print("\n All tests passed.")

    print("Starting benchmark...")
    do_benchmark()

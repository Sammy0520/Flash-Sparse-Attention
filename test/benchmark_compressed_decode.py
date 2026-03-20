#!/usr/bin/env python
"""Benchmark compressed_attention_decode kernel in isolation.

Usage:
  python test/benchmark_compressed_decode.py
  python test/benchmark_compressed_decode.py --seqlen 8000 --q-lens 4 16 32 64
  python test/benchmark_compressed_decode.py --full-sweep
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fsa_preview.ops.compressed_attention_decode import _compressed_attention_fwd_decode


def run_benchmark(
    seqlen: int,
    q_len: int,
    num_heads: int = 8,
    head_dim: int = 128,
    kernel_size: int = 32,
    kernel_stride: int = 16,
    dtype=torch.float16,
    warmup: int = 10,
    iters: int = 100,
):
    """Benchmark _compressed_attention_fwd_decode for given (seqlen, q_len)."""
    device = "cuda"
    compressed_k_len = (seqlen - 1 - kernel_size) // kernel_stride + 1
    num_k_blocks = (compressed_k_len + 127) // 128  # BLOCK_SIZE_K=128
    num_q_blocks = (q_len + 15) // 16  # BLOCK_SIZE_Q=16

    q = torch.randn(q_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(compressed_k_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(compressed_k_len, num_heads, head_dim, device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, compressed_k_len], device=device, dtype=torch.int32)
    sm_scale = 1 / math.sqrt(head_dim)

    for _ in range(warmup):
        _compressed_attention_fwd_decode(
            q, k, v,
            kernel_size, kernel_stride,
            cu_seqlens_q, cu_seqlens_k,
            q_len, compressed_k_len,
            sm_scale,
            query_start_index=0,
        )

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _compressed_attention_fwd_decode(
            q, k, v,
            kernel_size, kernel_stride,
            cu_seqlens_q, cu_seqlens_k,
            q_len, compressed_k_len,
            sm_scale,
            query_start_index=0,
        )
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms, compressed_k_len, num_k_blocks, num_q_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=1000)
    parser.add_argument("--seqlens", nargs="+", type=int, default=None,
                        help="Multiple seqlen to sweep (overrides --seqlen)")
    parser.add_argument("--q-lens", nargs="+", type=int, default=[4, 16, 32],
                        help="Q lengths to benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--full-sweep", action="store_true",
                        help="Sweep seqlen 1k,4k,8k,16k and q_len 4,16,32")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print full traceback on error")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if args.full_sweep:
        seqlens = [1000, 4000, 8000, 16000]
        q_lens = [4, 16, 32]
    else:
        seqlens = args.seqlens if args.seqlens else [args.seqlen]
        q_lens = args.q_lens

    print("=" * 75)
    print("Compressed Attention Decode Kernel Benchmark (isolated)")
    print("=" * 75)
    print("  seqlen -> compressed_k_len, num_k_blocks (BLOCK_K=128)")
    print("  q_len  -> num_q_blocks (BLOCK_Q=16)")
    print("=" * 75)

    results = []
    for seqlen in seqlens:
        for q_len in q_lens:
            try:
                ms, cmp_len, nk, nq = run_benchmark(
                    seqlen, q_len,
                    warmup=args.warmup, iters=args.iters, dtype=dtype,
                )
                results.append((seqlen, q_len, cmp_len, nk, nq, ms))
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                print(f"  seqlen={seqlen} q_len={q_len}  ERROR: {err_msg}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    # print by seqlen
    for seqlen in seqlens:
        row = [r for r in results if r[0] == seqlen]
        if not row:
            continue
        print(f"\nseqlen={seqlen} (cmp_k~{row[0][2]}, K_blocks={row[0][3]})")
        for r in row:
            _, q_len, _, _, nq, ms = r
            print(f"  q_len={q_len:3d}  Q_blocks={nq}  ->  {ms:.4f} ms")

    print("=" * 75)


if __name__ == "__main__":
    main()

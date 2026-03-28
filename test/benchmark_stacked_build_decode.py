#!/usr/bin/env python3
"""
Micro-benchmark: stacked tensor build inside _linear_compress_decode.

Compares:
  (A) Legacy: list of slices + torch.cat(dim=0)
  (B) New:    torch.empty + per-window copy_

This isolates the (1) optimization from Triton / linear_compress_with_pe.
For full _linear_compress_decode timing, run twice (git checkout before/after) or use
test/benchmark_compress_stage.py.

Usage:
  cd Flash-Sparse-Attention && PYTHONPATH=. python test/benchmark_stacked_build_decode.py
  PYTHONPATH=. python test/benchmark_stacked_build_decode.py --warmup 50 --iters 500
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _collect_all_tokens_and_window_spans(
    prev_total_len: int,
    new_tokens: torch.Tensor,
    token_buffer: Optional[torch.Tensor],
    kernel_size: int,
    kernel_stride: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Mirror fsa_preview.ops.linear_compress_decode._linear_compress_decode geometry (through span list)."""
    device = new_tokens.device
    num_new = new_tokens.shape[0]

    if token_buffer is None:
        all_tokens = new_tokens
        buffer_start_pos = prev_total_len
    else:
        all_tokens = torch.cat([token_buffer, new_tokens], dim=0)
        buffer_start_pos = prev_total_len - token_buffer.shape[0]

    new_total_len = prev_total_len + num_new
    prev_max_output_idx = (
        math.floor((prev_total_len - kernel_size) / kernel_stride)
        if prev_total_len >= kernel_size
        else -1
    )
    new_max_output_idx = (
        math.floor((new_total_len - kernel_size) / kernel_stride)
        if new_total_len >= kernel_size
        else -1
    )

    if new_max_output_idx <= prev_max_output_idx:
        return all_tokens, []

    window_spans: List[Tuple[int, int]] = []
    for out_idx in range(prev_max_output_idx + 1, new_max_output_idx + 1):
        window_start_abs = out_idx * kernel_stride
        window_end_abs = window_start_abs + kernel_size
        window_start_rel = window_start_abs - buffer_start_pos
        window_end_rel = window_end_abs - buffer_start_pos
        if window_start_rel >= 0 and window_end_rel <= all_tokens.shape[0]:
            window_spans.append((window_start_rel, window_end_rel))

    return all_tokens, window_spans


def stacked_build_cat(all_tokens: torch.Tensor, window_spans: List[Tuple[int, int]], kernel_size: int) -> torch.Tensor:
    """Legacy: same pattern as old windows_to_compute + torch.cat."""
    windows_to_compute = [all_tokens[s:e] for s, e in window_spans]
    return torch.cat(windows_to_compute, dim=0)


def stacked_build_prealloc(
    all_tokens: torch.Tensor,
    window_spans: List[Tuple[int, int]],
    kernel_size: int,
) -> torch.Tensor:
    """Current: empty + copy_."""
    device = all_tokens.device
    dtype = all_tokens.dtype
    num_windows = len(window_spans)
    _, num_heads, head_dim = all_tokens.shape
    stacked = torch.empty(
        (num_windows * kernel_size, num_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    for w, (s, e) in enumerate(window_spans):
        stacked[w * kernel_size : (w + 1) * kernel_size].copy_(all_tokens[s:e])
    return stacked


def cuda_bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark stacked build: cat vs prealloc+copy_")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=32)
    parser.add_argument("--kernel-stride", type=int, default=16)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["131072_4", "131072_32", "4096_8"],
        help="Each item: prev_total_len_Knew e.g. 131072_4 means prev_len=131072, K=4 new tokens",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = "cuda"
    H, D = args.kv_heads, args.head_dim
    ks, stride = args.kernel_size, args.kernel_stride

    print("=" * 72)
    print("Stacked build micro-benchmark (legacy cat vs prealloc+copy_)")
    print(f"  warmup={args.warmup} iters={args.iters} dtype={args.dtype} H={H} D={D} ks={ks} stride={stride}")
    print("=" * 72)

    for scen in args.scenarios:
        parts = scen.split("_")
        if len(parts) != 2:
            print(f"  skip bad scenario {scen!r}, expected PREV_K")
            continue
        prev_total_len, k_new = int(parts[0]), int(parts[1])
        buf_len = min(ks - 1, prev_total_len)
        new_tokens = torch.randn(k_new, H, D, device=device, dtype=dtype)
        token_buffer = (
            torch.randn(buf_len, H, D, device=device, dtype=dtype) if buf_len > 0 else None
        )

        all_tokens, window_spans = _collect_all_tokens_and_window_spans(
            prev_total_len, new_tokens, token_buffer, ks, stride
        )
        if not window_spans:
            print(f"\n[{scen}] num_windows=0 (no stacked build this step) — skip")
            continue

        num_windows = len(window_spans)
        y_cat = stacked_build_cat(all_tokens, window_spans, ks)
        y_new = stacked_build_prealloc(all_tokens, window_spans, ks)
        if not torch.equal(y_cat, y_new):
            raise RuntimeError(f"[{scen}] stacked mismatch: cat vs prealloc")

        # Benchmark: wrap in lambdas that rebuild (no reuse of tensor across iters for fairness)
        def run_cat():
            stacked_build_cat(all_tokens, window_spans, ks)

        def run_pre():
            stacked_build_prealloc(all_tokens, window_spans, ks)

        ms_cat = cuda_bench_ms(run_cat, args.warmup, args.iters)
        ms_pre = cuda_bench_ms(run_pre, args.warmup, args.iters)
        speedup = ms_cat / ms_pre if ms_pre > 0 else float("inf")

        print(f"\n[{scen}] prev_len={prev_total_len} K_new={k_new} num_windows={num_windows} | stacked {tuple(y_cat.shape)}")
        print(f"  legacy cat:     {ms_cat:.4f} ms/iter")
        print(f"  prealloc+copy: {ms_pre:.4f} ms/iter")
        print(f"  speedup (cat/pre): {speedup:.3f}x")

    print("\n" + "=" * 72)
    print("Interpretation: small absolute times are normal; compare speedup across runs.")
    print("Full decode A/B: checkout old commit vs new, same command on benchmark_compress_stage.py.")
    print("=" * 72)


if __name__ == "__main__":
    main()

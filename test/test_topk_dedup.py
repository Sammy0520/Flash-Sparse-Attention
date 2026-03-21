"""
Test the TopK block overlap (dedup) kernel against the reference kernel.
Tests both correctness and performance using real tensors from LLaMA decode.

Usage:
  python test/test_topk_dedup.py
  python test/test_topk_dedup.py --data test/real_decode_tensors.pt --iters 100
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsa_ref.ops.topk_sparse_attention import _topk_sparse_attention_fwd


# ── import the new dedup kernel (your colleague implements this) ───────
# TODO: replace with actual import once implemented
# from nsa_ref.ops.topk_sparse_attention_dedup import _topk_sparse_attention_fwd_dedup
_topk_sparse_attention_fwd_dedup = None   # placeholder


def run_ref(q, k, v, topk_idx, block_size, cu_q, cu_k, N, past_len):
    """Reference: original kernel, one CTA per token."""
    o, lse = _topk_sparse_attention_fwd(
        q, k, v, topk_idx,
        block_size=block_size,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=N,
        max_seqlen_k=past_len,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
    )
    return o


def run_dedup(q, k, v, topk_idx, block_size, cu_q, cu_k, N, past_len):
    """New dedup kernel: union of blocks, one CTA per head."""
    assert _topk_sparse_attention_fwd_dedup is not None, \
        "Please implement and import _topk_sparse_attention_fwd_dedup"
    o = _topk_sparse_attention_fwd_dedup(
        q, k, v, topk_idx,
        block_size=block_size,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=N,
        max_seqlen_k=past_len,
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
    )
    return o


def benchmark(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="test/real_decode_tensors.pt",
                        help="Path to .pt file from capture_real_tensors.py")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--atol",  type=float, default=1e-2)
    parser.add_argument("--rtol",  type=float, default=1e-2)
    parser.add_argument("--use-random", action="store_true",
                        help="Use random tensors instead of real data (degenerate overlap)")
    args = parser.parse_args()

    device = "cuda"

    # ── load real tensors ─────────────────────────────────────────────
    if args.use_random:
        print("Using RANDOM tensors (degenerate overlap, for kernel smoke test only)")
        N, H, D = 8, 8, 128
        past_len   = 32768
        block_size = 64
        num_blocks = past_len // block_size
        topk_k     = 16
        q        = torch.randn(N, H, D, device=device, dtype=torch.bfloat16)
        k_cache  = torch.randn(past_len, H, D, device=device, dtype=torch.bfloat16)
        v_cache  = torch.randn(past_len, H, D, device=device, dtype=torch.bfloat16)
        topk_idx = torch.randint(0, num_blocks, (H, N, topk_k),
                                 device=device, dtype=torch.int32)
    else:
        print(f"Loading real tensors from {args.data} ...")
        d = torch.load(args.data, map_location="cpu", weights_only=False)
        q        = d["q"].to(device=device)          # [N, H, D]
        k_cache  = d["k_cache"].to(device=device)    # [past_len, H, D]
        v_cache  = d["v_cache"].to(device=device)    # [past_len, H, D]
        topk_idx = d["topk_idx"].to(device=device)   # [H, N, topk] int32
        N, H, D  = q.shape
        past_len   = k_cache.shape[0]
        block_size = d["block_size"]
        num_blocks = d["num_blocks"]
        topk_k     = topk_idx.shape[-1]
        print(f"  q: {tuple(q.shape)}, k_cache: {tuple(k_cache.shape)}")
        print(f"  topk_idx: {tuple(topk_idx.shape)}, block_size={block_size}")

    # Show overlap stats of the loaded topk_idx
    ti = topk_idx.cpu()
    unions = [len(set(ti[h].reshape(-1).tolist())) for h in range(H)]
    avg_union = sum(unions) / H
    bw_save   = (1 - avg_union / (N * topk_k)) * 100
    print(f"\n  topk_idx overlap stats:")
    print(f"    union blocks (avg/head): {avg_union:.1f} / {N*topk_k} slots")
    print(f"    bandwidth saving:        {bw_save:.1f}%")

    cu_q = torch.tensor([0, N],        device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, past_len], device=device, dtype=torch.int32)

    # ── correctness test ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("CORRECTNESS TEST")
    print("=" * 55)

    with torch.no_grad():
        o_ref = run_ref(q, k_cache, v_cache, topk_idx,
                        block_size, cu_q, cu_k, N, past_len)

    if _topk_sparse_attention_fwd_dedup is None:
        print("  [SKIP] dedup kernel not yet imported.")
        print("  Implement _topk_sparse_attention_fwd_dedup and update the import.")
    else:
        with torch.no_grad():
            o_dedup = run_dedup(q, k_cache, v_cache, topk_idx,
                                block_size, cu_q, cu_k, N, past_len)
        diff = (o_ref - o_dedup).abs()
        print(f"  max |ref - dedup|:  {diff.max().item():.6f}")
        print(f"  mean |ref - dedup|: {diff.mean().item():.6f}")
        per_token = diff.amax(dim=-1).amax(dim=-1)  # [N]
        print(f"  per-token max diff: {per_token.tolist()}")
        try:
            torch.testing.assert_close(o_ref, o_dedup,
                                       rtol=args.rtol, atol=args.atol)
            print(f"  [OK] outputs match (atol={args.atol})")
        except AssertionError as e:
            print(f"  [FAIL] {e}")

    # ── performance test ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PERFORMANCE TEST")
    print("=" * 55)
    print(f"  iters={args.iters}, N={N}, topk={topk_k}, "
          f"past_len={past_len}, block_size={block_size}")

    # Baseline: N × 1-token calls
    q1   = q[:1]
    cu_q1 = torch.tensor([0, 1], device=device, dtype=torch.int32)
    ti1   = topk_idx[:, :1, :]  # [H, 1, K]
    t_1xN = benchmark(
        lambda: [run_ref(q1, k_cache, v_cache, ti1,
                         block_size, cu_q1, cu_k, 1, past_len)
                 for _ in range(N)],
        iters=args.iters,
    )

    # N-token once, reference kernel (no dedup)
    t_ref = benchmark(
        lambda: run_ref(q, k_cache, v_cache, topk_idx,
                        block_size, cu_q, cu_k, N, past_len),
        iters=args.iters,
    )

    print(f"\n  1-token × {N} calls:     {t_1xN:.3f} ms  (baseline)")
    print(f"  {N}-token ref kernel:     {t_ref:.3f} ms  "
          f"(speedup vs 1×N: {t_1xN/t_ref:.2f}×)")

    if _topk_sparse_attention_fwd_dedup is not None:
        t_dedup = benchmark(
            lambda: run_dedup(q, k_cache, v_cache, topk_idx,
                              block_size, cu_q, cu_k, N, past_len),
            iters=args.iters,
        )
        print(f"  {N}-token dedup kernel:   {t_dedup:.3f} ms  "
              f"(speedup vs 1×N: {t_1xN/t_dedup:.2f}×, "
              f"vs ref: {t_ref/t_dedup:.2f}×)")
        print(f"\n  Expected speedup from dedup: "
              f"{N*topk_k/avg_union:.2f}× (union={avg_union:.1f})")
    else:
        print(f"\n  [SKIP] dedup kernel not imported.")
        print(f"  Expected speedup from dedup: "
              f"{N*topk_k/avg_union:.2f}× (union={avg_union:.1f} blocks)")


if __name__ == "__main__":
    main()

"""
TopK Block Overlap Analysis for Linear Speculative Decoding.

Measures two things:
  1. Block overlap statistics: for N draft tokens in one forward pass,
     how many topK blocks do adjacent/all tokens share?
     → Quantifies the "block deduplication" savings for the TopK path.

  2. Performance: N-token-at-once vs N × 1-token, across seqlens and N values.

Usage:
  python test/analyze_topk_overlap.py                        # default sweep
  python test/analyze_topk_overlap.py --seqlens 4096 8192
  python test/analyze_topk_overlap.py --n-values 1 4 8 16 32
  python test/analyze_topk_overlap.py --perf-only
  python test/analyze_topk_overlap.py --overlap-only

NOTE on random vs real data:
  Random q/k gives a *lower bound* on block overlap.
  Real LLM hidden states have higher overlap because adjacent draft tokens
  tend to attend to similar context.
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsa_ref.module.rope import RopeConfig
from nsa_ref.ops import linear_compress
from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode
import fsa_preview.ops.compressed_attention_decode as _cad_mod


# ---------------------------------------------------------------------------
# Hook: patch _compressed_attention_decode to capture topk_idx
# ---------------------------------------------------------------------------

_captured_topk_idx = None  # global slot filled by hook

_orig_compressed_attention_decode = _cad_mod._compressed_attention_decode


def _hooked_compressed_attention_decode(*args, **kwargs):
    global _captured_topk_idx
    out, topk_idx = _orig_compressed_attention_decode(*args, **kwargs)
    _captured_topk_idx = topk_idx
    return out, topk_idx


def install_hook():
    import fsa_preview.ops as _ops_pkg
    import fsa_preview.ops.compressed_attention_decode as _cad
    _cad._compressed_attention_decode = _hooked_compressed_attention_decode
    _ops_pkg._compressed_attention_decode = _hooked_compressed_attention_decode
    # also patch the reference inside fsa_decode module
    import fsa_preview.module.fsa_decode as _fd
    import importlib
    # re-import to pick up patched symbol
    _fd_src = Path(_fd.__file__).read_text()
    # direct attribute patch on the module
    _fd._compressed_attention_decode = _hooked_compressed_attention_decode


def uninstall_hook():
    import fsa_preview.ops.compressed_attention_decode as _cad
    import fsa_preview.ops as _ops_pkg
    import fsa_preview.module.fsa_decode as _fd
    _cad._compressed_attention_decode = _orig_compressed_attention_decode
    _ops_pkg._compressed_attention_decode = _orig_compressed_attention_decode
    _fd._compressed_attention_decode = _orig_compressed_attention_decode


# ---------------------------------------------------------------------------
# Build model + caches
# ---------------------------------------------------------------------------

def make_model_and_caches(
    seqlen: int, N: int,
    hidden_size: int, kv_heads: int, head_dim: int,
    kernel_size: int, kernel_stride: int, block_size: int, topk: int,
    init_blocks: int, local_blocks: int, window_size: int,
    dtype: torch.dtype, device: str = "cuda",
):
    """Return (model, x, cu_seqlens_q, cu_seqlens_k, k_cache, v_cache, cmp_k, cmp_v, pos_ids)."""
    past_len = seqlen - N
    assert past_len > kernel_size, \
        f"past_len={past_len} must be > kernel_size={kernel_size}. Increase seqlen or decrease N."

    rope_config = RopeConfig(
        max_position_embeddings=131072,
        head_dim=head_dim,
        rope_theta=500000,
        rope_scaling={
            "factor": 8.0, "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
    )
    model = FlashSparseAttentionDecode(
        hidden_size=hidden_size,
        num_q_heads=kv_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        window_size=window_size,
        rope_config=rope_config,
    ).to(device=device, dtype=dtype)

    k_cache = torch.randn(past_len, kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(past_len, kv_heads, head_dim, device=device, dtype=dtype)

    cu_seqlens_k_raw = torch.tensor([0, past_len], device=device, dtype=torch.int32)
    cmp_k, _ = linear_compress(
        k_cache, model.compress_key, cu_seqlens_k_raw,
        kernel_size, kernel_stride, model.intra_block_pe,
    )
    cmp_v, _ = linear_compress(
        v_cache, model.compress_value, cu_seqlens_k_raw,
        kernel_size, kernel_stride, None,
    )

    cu_seqlens_q = torch.tensor([0, N], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, past_len], device=device, dtype=torch.int32)
    x = torch.randn(N, hidden_size, device=device, dtype=dtype)
    pos_ids = torch.arange(past_len, past_len + N, device=device, dtype=torch.long)

    return model, x, cu_seqlens_q, cu_seqlens_k, k_cache, v_cache, cmp_k, cmp_v, pos_ids


# ---------------------------------------------------------------------------
# Overlap statistics
# ---------------------------------------------------------------------------

def compute_overlap_stats(topk_idx: torch.Tensor):
    """
    topk_idx: [num_heads, N, effective_topk]

    Returns:
      pairwise_overlap  : mean |topk_i ∩ topk_{i+1}| / effective_topk  (adjacent, avg over heads)
      union_ratio       : |∪ topk_i| / (N * effective_topk)
      bw_saving_pct     : % bandwidth saved by loading shared blocks only once
      effective_topk    : actual topk (may be < configured topk if seqlen is short)
      per_pair          : per-adjacent-pair overlap ratios
    """
    num_heads, N, effective_topk = topk_idx.shape
    idx_cpu = topk_idx.cpu()

    per_pair = []
    for i in range(N - 1):
        pair_overlaps = []
        for h in range(num_heads):
            s1 = set(idx_cpu[h, i].tolist())
            s2 = set(idx_cpu[h, i + 1].tolist())
            pair_overlaps.append(len(s1 & s2) / effective_topk)
        per_pair.append(sum(pair_overlaps) / num_heads)

    pairwise_overlap = sum(per_pair) / len(per_pair) if per_pair else float("nan")

    union_sizes = []
    for h in range(num_heads):
        all_blocks = set()
        for i in range(N):
            all_blocks.update(idx_cpu[h, i].tolist())
        union_sizes.append(len(all_blocks))

    avg_union = sum(union_sizes) / num_heads
    total_slots = N * effective_topk
    union_ratio = avg_union / total_slots
    bw_saving_pct = (1.0 - union_ratio) * 100.0

    return dict(
        pairwise_overlap=pairwise_overlap,
        union_ratio=union_ratio,
        bw_saving_pct=bw_saving_pct,
        avg_union_blocks=avg_union,
        effective_topk=effective_topk,
        per_pair=per_pair,
    )


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def benchmark(model, x_N, x_1, cu_q_N, cu_q_1, cu_k,
              k_cache, v_cache, cmp_k, cmp_v, pos_N, pos_1,
              N: int, n_iters: int = 10, n_warmup: int = 4):
    def run_N():
        model(x_N, cu_q_N, cu_k, k_cache, v_cache, cmp_k, cmp_v,
              position_ids=pos_N)

    def run_1xN():
        for _ in range(N):
            model(x_1, cu_q_1, cu_k, k_cache, v_cache, cmp_k, cmp_v,
                  position_ids=pos_1)

    for _ in range(n_warmup):
        run_N(); run_1xN()
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(n_iters): run_N()
    e.record(); torch.cuda.synchronize()
    t_N = s.elapsed_time(e) / n_iters

    s.record()
    for _ in range(n_iters): run_1xN()
    e.record(); torch.cuda.synchronize()
    t_1xN = s.elapsed_time(e) / n_iters

    return t_N, t_1xN


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seqlens",      type=int, nargs="+", default=[32768, 65536])
    parser.add_argument("--n-values",     type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--topk",         type=int, default=16)
    parser.add_argument("--block-size",   type=int, default=64)
    parser.add_argument("--kernel-size",  type=int, default=32)
    parser.add_argument("--kernel-stride",type=int, default=16)
    parser.add_argument("--kv-heads",     type=int, default=8)
    parser.add_argument("--head-dim",     type=int, default=128)
    parser.add_argument("--hidden-size",  type=int, default=4096)
    parser.add_argument("--window-size",  type=int, default=512)
    parser.add_argument("--init-blocks",  type=int, default=1)
    parser.add_argument("--local-blocks", type=int, default=2)
    parser.add_argument("--overlap-seeds",type=int, default=8,
                        help="Random seeds to average overlap stats over.")
    parser.add_argument("--perf-iters",   type=int, default=10)
    parser.add_argument("--perf-warmup",  type=int, default=4)
    parser.add_argument("--dtype",        type=str,  default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--perf-only",    action="store_true")
    parser.add_argument("--overlap-only", action="store_true")
    args = parser.parse_args()

    DTYPE  = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda"

    # info header
    print("=" * 72)
    print("TopK Block Overlap & Performance Analysis")
    print("=" * 72)
    cmp_tokens_per_block = args.block_size // args.kernel_stride
    orig_tokens_per_block = args.block_size
    print(f"  block_size={args.block_size} (orig tokens/block)  "
          f"topk={args.topk}  kernel={args.kernel_size}/{args.kernel_stride}")
    print(f"  kv_heads={args.kv_heads}  head_dim={args.head_dim}  dtype={args.dtype}")
    for seqlen in args.seqlens:
        total_blk = seqlen // args.block_size
        sparsity  = args.topk / total_blk * 100 if total_blk > 0 else float("nan")
        print(f"  seqlen={seqlen}: total_blocks≈{total_blk}, "
              f"topk coverage≈{sparsity:.1f}%  "
              f"{'(dense, low overlap expected)' if sparsity > 50 else ''}")
    print()

    # install hook to capture topk_idx from real forward passes
    install_hook()

    # ---------------------------------------------------------------
    # Part 1: Overlap statistics
    # ---------------------------------------------------------------
    if not args.perf_only:
        print("─" * 72)
        print("PART 1: TopK Block Overlap Statistics")
        print("  (random hidden states → lower bound; real LLM data → higher overlap)")
        print("─" * 72)
        print(f"  {'seqlen':>8}  {'N':>4}  {'eff_topk':>8}  "
              f"{'adj_overlap':>12}  {'union_blks':>11}  {'bw_save%':>9}")
        print("  " + "-" * 60)

        for seqlen in args.seqlens:
            for N in args.n_values:
                if N < 2:
                    continue  # pairwise overlap needs N≥2
                try:
                    stats_list = []
                    for seed in range(args.overlap_seeds):
                        torch.manual_seed(seed)
                        global _captured_topk_idx
                        _captured_topk_idx = None

                        model, x, cu_q, cu_k, k_cache, v_cache, cmp_k, cmp_v, pos = \
                            make_model_and_caches(
                                seqlen, N,
                                args.hidden_size, args.kv_heads, args.head_dim,
                                args.kernel_size, args.kernel_stride,
                                args.block_size, args.topk,
                                args.init_blocks, args.local_blocks, args.window_size,
                                DTYPE, device,
                            )
                        with torch.no_grad():
                            model(x, cu_q, cu_k, k_cache, v_cache, cmp_k, cmp_v,
                                  position_ids=pos)

                        if _captured_topk_idx is None:
                            print(f"  WARNING: topk_idx not captured for seqlen={seqlen} N={N}")
                            break
                        stats_list.append(compute_overlap_stats(_captured_topk_idx))

                    if not stats_list:
                        continue
                    eff_topk = stats_list[0]["effective_topk"]
                    adj_ov   = sum(s["pairwise_overlap"]  for s in stats_list) / len(stats_list)
                    union_b  = sum(s["avg_union_blocks"]  for s in stats_list) / len(stats_list)
                    bw_save  = sum(s["bw_saving_pct"]     for s in stats_list) / len(stats_list)

                    print(f"  {seqlen:>8}  {N:>4}  {eff_topk:>8}  "
                          f"{adj_ov:>11.1%}  {union_b:>11.1f}  {bw_save:>8.1f}%")

                except Exception as e:
                    print(f"  {seqlen:>8}  {N:>4}  ERROR: {e}")

        print()
        print("  Legend:")
        print("  • eff_topk    : actual blocks selected (may be < topk if seqlen short)")
        print("  • adj_overlap : fraction of blocks shared between token_i and token_{i+1}")
        print("  • union_blks  : unique blocks to load for all N tokens combined")
        print("  • bw_save%    : HBM bandwidth saved by deduplicating shared blocks")
        print()
        if any((sl // args.block_size) <= args.topk for sl in args.seqlens):
            print("  ⚠  Some seqlens have total_blocks ≤ topk (dense regime).")
            print("     Try --seqlens 8192 16384 32768 for meaningful sparsity.")
        print()

    # ---------------------------------------------------------------
    # Part 2: Performance benchmark
    # ---------------------------------------------------------------
    if not args.overlap_only:
        print("─" * 72)
        print("PART 2: Performance  (N-token-at-once vs N × 1-token)")
        print("─" * 72)
        print(f"  {'seqlen':>8}  {'N':>4}  {'N-tok(ms)':>11}  "
              f"{'Nx1-tok(ms)':>13}  {'speedup':>8}  {'latency_ratio':>14}")
        print("  " + "-" * 65)

        for seqlen in args.seqlens:
            # build model once per seqlen (reuse across N values for fair comparison)
            try:
                rope_config = RopeConfig(
                    max_position_embeddings=131072,
                    head_dim=args.head_dim,
                    rope_theta=500000,
                    rope_scaling={
                        "factor": 8.0, "high_freq_factor": 4.0,
                        "low_freq_factor": 1.0,
                        "original_max_position_embeddings": 8192,
                        "rope_type": "llama3",
                    },
                )
                model = FlashSparseAttentionDecode(
                    hidden_size=args.hidden_size,
                    num_q_heads=args.kv_heads,
                    num_kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    kernel_size=args.kernel_size,
                    kernel_stride=args.kernel_stride,
                    block_size=args.block_size,
                    topk=args.topk,
                    init_blocks=args.init_blocks,
                    local_blocks=args.local_blocks,
                    window_size=args.window_size,
                    rope_config=rope_config,
                ).to(device=device, dtype=DTYPE)

                for N in args.n_values:
                    past_len = seqlen - N
                    if past_len <= args.kernel_size:
                        continue

                    k_cache = torch.randn(past_len, args.kv_heads, args.head_dim,
                                         device=device, dtype=DTYPE)
                    v_cache = torch.randn(past_len, args.kv_heads, args.head_dim,
                                         device=device, dtype=DTYPE)
                    cu_k_raw = torch.tensor([0, past_len], device=device, dtype=torch.int32)
                    cmp_k, _ = linear_compress(k_cache, model.compress_key, cu_k_raw,
                                               args.kernel_size, args.kernel_stride,
                                               model.intra_block_pe)
                    cmp_v, _ = linear_compress(v_cache, model.compress_value, cu_k_raw,
                                               args.kernel_size, args.kernel_stride, None)

                    cu_k  = torch.tensor([0, past_len], device=device, dtype=torch.int32)
                    cu_qN = torch.tensor([0, N], device=device, dtype=torch.int32)
                    cu_q1 = torch.tensor([0, 1], device=device, dtype=torch.int32)
                    x_N   = torch.randn(N, args.hidden_size, device=device, dtype=DTYPE)
                    x_1   = torch.randn(1, args.hidden_size, device=device, dtype=DTYPE)
                    pos_N = torch.arange(past_len, past_len + N,
                                        device=device, dtype=torch.long)
                    pos_1 = torch.arange(past_len, past_len + 1,
                                        device=device, dtype=torch.long)

                    t_N, t_1xN = benchmark(
                        model, x_N, x_1, cu_qN, cu_q1, cu_k,
                        k_cache, v_cache, cmp_k, cmp_v, pos_N, pos_1,
                        N=N,
                        n_iters=args.perf_iters,
                        n_warmup=args.perf_warmup,
                    )
                    speedup = t_1xN / t_N
                    latency_ratio = t_N / (t_1xN / N)  # how much slower per token vs 1-tok
                    print(f"  {seqlen:>8}  {N:>4}  {t_N:>11.3f}  "
                          f"{t_1xN:>13.3f}  {speedup:>7.1f}x  {latency_ratio:>12.2f}x")

            except Exception as e:
                import traceback
                print(f"  seqlen={seqlen}: ERROR: {e}")
                traceback.print_exc()

        print()
        print("  Legend:")
        print("  • speedup      : N × 1-tok latency / N-tok latency  (>1 = N-tok wins)")
        print("  • latency_ratio: N-tok latency / (1-tok latency)     (1.0 = perfect)")
        print()

    uninstall_hook()


if __name__ == "__main__":
    main()

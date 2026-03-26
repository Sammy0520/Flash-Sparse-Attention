#!/usr/bin/env python
"""Benchmark each measurable step of the compress stage in FSA decode.

Compress stage includes:
  1. linear_compress_decode(K) - compress new keys
  2. linear_compress_decode(V) - compress new values
  3. torch.cat - merge cache + new compressed
  4. RoPE on q
  5. RoPE on compressed_k
  6. _compressed_attention_fwd_decode - attention between q and compressed K/V
  7. _get_attention_score_decode - score kernel for topk selection
  8. transform_score_decode - block-wise score
  9. topk - select top-k blocks

Usage:
  python test/benchmark_compress_stage.py
  python test/benchmark_compress_stage.py --q-lens 1 4 8 16 32 --seqlen 4000
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from nsa_ref.module import RopeConfig
from nsa_ref.ops import linear_compress
from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode
from fsa_preview.ops import _linear_compress_decode, _compressed_attention_decode
from fsa_preview.ops.compressed_attention_decode import (
    _compressed_attention_fwd_decode,
    _get_attention_score_decode,
    transform_score_decode,
)


def cuda_bench(name, fn, warmup=10, iters=100):
    """Run fn() repeatedly and return avg ms."""
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


def run_benchmark(
    seqlen: int,
    q_len: int,
    kv_heads: int = 8,
    head_dim: int = 128,
    kernel_size: int = 32,
    kernel_stride: int = 16,
    block_size: int = 64,
    topk: int = 16,
    hidden_size: int = 4096,
    dtype=torch.bfloat16,
    warmup: int = 10,
    iters: int = 100,
):
    device = "cuda"
    # Setup cu_seqlens: batch=1, single sequence
    cu_seqlens_k = torch.tensor([0, seqlen - 1], dtype=torch.int32, device=device)
    cmp_len = (seqlen - 1 - kernel_size) // kernel_stride + 1

    # Build module for proj and rope
    rope_config = RopeConfig(
        max_position_embeddings=131072,
        head_dim=head_dim,
        rope_theta=500000,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
    )
    sparse_attn = FlashSparseAttentionDecode(
        hidden_size=hidden_size,
        num_q_heads=kv_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
        block_size=block_size,
        topk=topk,
        init_blocks=1,
        local_blocks=2,
        window_size=512,
        rope_config=rope_config,
    ).cuda().to(dtype)

    # Pre-compute cmp_k_cache, cmp_v_cache using linear_compress (prefill style)
    k_cache = torch.randn(seqlen - 1, kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(seqlen - 1, kv_heads, head_dim, device=device, dtype=dtype)
    cmp_k_cache, compressed_cu_seqlens = linear_compress(
        k_cache,
        sparse_attn.compress_key,
        cu_seqlens_k,
        kernel_size,
        kernel_stride,
        sparse_attn.intra_block_pe,
    )
    cmp_v_cache, _ = linear_compress(
        v_cache,
        sparse_attn.compress_value,
        cu_seqlens_k,
        kernel_size,
        kernel_stride,
        None,
    )

    # New tokens (decode input)
    x = torch.randn(q_len, hidden_size, device=device, dtype=dtype)
    k_new = sparse_attn.proj_k(x).view(-1, kv_heads, head_dim)
    v_new = sparse_attn.proj_v(x).view(-1, kv_heads, head_dim)
    q = sparse_attn.proj_q(x).view(-1, sparse_attn.num_q_heads, head_dim)

    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    max_seqlen_q = q_len
    max_seqlen_k = compressed_seqlens.max().item()
    sm_scale = 1.0 / math.sqrt(head_dim)
    query_start_index = k_cache.shape[0]

    # Buffer for linear_compress_decode
    buffer_size = min(kernel_size - 1, cmp_k_cache.shape[0])
    initial_buffer_k = cmp_k_cache[-buffer_size:] if buffer_size > 0 else None
    initial_buffer_v = cmp_v_cache[-buffer_size:] if buffer_size > 0 else None

    results = {}

    # 1. linear_compress_decode(K)
    def do_compress_k():
        return _linear_compress_decode(
            k_new,
            sparse_attn.compress_key,
            kernel_size,
            kernel_stride,
            sparse_attn.intra_block_pe,
            cmp_k_cache.shape[0],
            initial_buffer_k,
        )

    decode_k = do_compress_k()
    if decode_k is not None:
        results["1_linear_compress_K"] = cuda_bench("linear_compress_K", lambda: do_compress_k(), warmup, iters)
    else:
        results["1_linear_compress_K"] = 0.0

    # 2. linear_compress_decode(V)
    def do_compress_v():
        return _linear_compress_decode(
            v_new,
            sparse_attn.compress_value,
            kernel_size,
            kernel_stride,
            None,
            cmp_v_cache.shape[0],
            initial_buffer_v,
        )

    decode_v = do_compress_v()
    if decode_v is not None:
        results["2_linear_compress_V"] = cuda_bench("linear_compress_V", lambda: do_compress_v(), warmup, iters)
    else:
        results["2_linear_compress_V"] = 0.0

    # 3. torch.cat (negligible, but measurable)
    compressed_k = torch.cat([cmp_k_cache, decode_k], dim=0) if decode_k is not None else cmp_k_cache
    compressed_v = torch.cat([cmp_v_cache, decode_v], dim=0) if decode_v is not None else cmp_v_cache

    def do_cat():
        torch.cat([cmp_k_cache, decode_k], dim=0)
        torch.cat([cmp_v_cache, decode_v], dim=0)

    if decode_k is not None:
        results["3_torch_cat"] = cuda_bench("torch_cat", do_cat, warmup, iters)
    else:
        results["3_torch_cat"] = 0.0

    # 4. RoPE on q
    def do_rope_q():
        sparse_attn.rope(q.clone(), cu_seqlens_q)

    results["4_RoPE_q"] = cuda_bench("RoPE_q", do_rope_q, warmup, iters)

    # 5. RoPE on compressed_k
    def do_rope_compressed_k():
        sparse_attn.rope(compressed_k.clone(), compressed_cu_seqlens, start=0, stride=kernel_stride)

    results["5_RoPE_compressed_k"] = cuda_bench("RoPE_compressed_k", do_rope_compressed_k, warmup, iters)

    # Apply RoPE for downstream steps
    q_rope = sparse_attn.rope(q, cu_seqlens_q)
    compressed_k_rope = sparse_attn.rope(compressed_k, compressed_cu_seqlens, start=0, stride=kernel_stride)

    # 6. _compressed_attention_fwd_decode
    def do_compressed_attn_fwd():
        _compressed_attention_fwd_decode(
            q_rope,
            compressed_k_rope,
            compressed_v,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            compressed_cu_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
            query_start_index,
            attention_mask=None,
        )

    attn_out, lse = _compressed_attention_fwd_decode(
        q_rope,
        compressed_k_rope,
        compressed_v,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        compressed_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
        query_start_index,
        attention_mask=None,
    )
    results["6_compressed_attn_fwd"] = cuda_bench("compressed_attn_fwd", do_compressed_attn_fwd, warmup, iters)

    # 7. _get_attention_score_decode
    def do_score():
        _get_attention_score_decode(
            q_rope,
            compressed_k_rope,
            lse,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            compressed_cu_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
            query_start_index,
        )

    score = _get_attention_score_decode(
        q_rope,
        compressed_k_rope,
        lse,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        compressed_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
        query_start_index,
    )
    results["7_get_attention_score"] = cuda_bench("get_attention_score", do_score, warmup, iters)

    # 8. transform_score_decode
    init_blocks, local_blocks = 1, 2
    block_score = transform_score_decode(
        score,
        kernel_size,
        kernel_stride,
        block_size,
        cu_seqlens_q,
        compressed_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        init_blocks,
        local_blocks,
    )

    def do_transform():
        transform_score_decode(
            score,
            kernel_size,
            kernel_stride,
            block_size,
            cu_seqlens_q,
            compressed_cu_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            init_blocks,
            local_blocks,
        )

    results["8_transform_score"] = cuda_bench("transform_score", do_transform, warmup, iters)

    # 9. topk
    k_val = min(topk, block_score.shape[-1])

    def do_topk():
        block_score.topk(k_val, dim=-1).indices.to(torch.int32)

    results["9_topk"] = cuda_bench("topk", do_topk, warmup, iters)

    # Total compress pipeline (excluding topk_sparse_attn, sliding, gate, proj_o)
    def do_full_compress():
        dk = _linear_compress_decode(
            k_new,
            sparse_attn.compress_key,
            kernel_size,
            kernel_stride,
            sparse_attn.intra_block_pe,
            cmp_k_cache.shape[0],
            initial_buffer_k,
        )
        dv = _linear_compress_decode(
            v_new,
            sparse_attn.compress_value,
            kernel_size,
            kernel_stride,
            None,
            cmp_v_cache.shape[0],
            initial_buffer_v,
        )
        ck = torch.cat([cmp_k_cache, dk], dim=0) if dk is not None else cmp_k_cache
        cv = torch.cat([cmp_v_cache, dv], dim=0) if dv is not None else cmp_v_cache
        qr = sparse_attn.rope(q, cu_seqlens_q)
        ck_rope = sparse_attn.rope(ck, compressed_cu_seqlens, start=0, stride=kernel_stride)
        _, topk_idx = _compressed_attention_decode(
            qr,
            ck_rope,
            cv,
            kernel_size,
            kernel_stride,
            block_size,
            topk,
            cu_seqlens_q,
            compressed_cu_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
            init_blocks,
            local_blocks,
            query_start_index,
            None,
        )
        return topk_idx

    results["0_FULL_compress_pipeline"] = cuda_bench("full_compress", do_full_compress, warmup, iters)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=1000, help="KV context length (before new tokens)")
    parser.add_argument("--seqlens", nargs="+", type=int, default=None,
                        help="Multiple seqlen to sweep (overrides --seqlen)")
    parser.add_argument("--q-lens", nargs="+", type=int, default=[1, 4, 8, 16],
                        help="Query lengths (new tokens per step)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    seqlens = args.seqlens if args.seqlens else [args.seqlen]

    print("=" * 80)
    print("Compress Stage Breakdown Benchmark")
    print(f"  seqlens={seqlens}, q_lens={args.q_lens}, dtype={args.dtype}")
    print("=" * 80)

    all_results = []
    for seqlen in seqlens:
        for q_len in args.q_lens:
            try:
                results = run_benchmark(
                    seqlen=seqlen,
                    q_len=q_len,
                    warmup=args.warmup,
                    iters=args.iters,
                    dtype=dtype,
                )
                results["q_len"] = q_len
                results["seqlen"] = seqlen
                all_results.append(results)
            except Exception as e:
                print(f"  seqlen={seqlen} q_len={q_len} ERROR: {e}")
                import traceback
                traceback.print_exc()

    if not all_results:
        print("No results.")
        return

    # Print table: rows = steps, cols = (seqlen, q_len)
    steps = [
        "0_FULL_compress_pipeline",
        "1_linear_compress_K",
        "2_linear_compress_V",
        "3_torch_cat",
        "4_RoPE_q",
        "5_RoPE_compressed_k",
        "6_compressed_attn_fwd",
        "7_get_attention_score",
        "8_transform_score",
        "9_topk",
    ]

    print("\n" + "-" * 100)
    col_label = lambda r: f"s={r['seqlen']}/q={r['q_len']}"
    header = f"{'Step':<32}" + "".join(f" {col_label(r):>12} " for r in all_results)
    print(header)
    print("-" * 100)

    for step in steps:
        row = f"{step:<32}"
        for r in all_results:
            ms = r.get(step, 0.0)
            row += f" {ms:>10.3f} "
        print(row)

    # Sum of components vs FULL (sanity check)
    component_keys = [s for s in steps if s != "0_FULL_compress_pipeline"]
    print("-" * 100)
    print("\nSum of components (1-9) vs FULL pipeline:")
    for r in all_results:
        s = sum(r.get(k, 0) for k in component_keys)
        full = r.get("0_FULL_compress_pipeline", 0)
        print(f"  seqlen={r['seqlen']} q_len={r['q_len']}: sum={s:.3f} ms, FULL={full:.3f} ms (diff={full - s:.3f})")

    # Note for q_len=1
    if any(r["q_len"] == 1 for r in all_results):
        print("\n  [Note] For q_len=1, linear_compress_K/V may be 0: no new compressed block when only 1 token is appended.")

    print("=" * 100)


if __name__ == "__main__":
    main()

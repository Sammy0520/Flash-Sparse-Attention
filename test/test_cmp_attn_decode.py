import argparse

import torch

from fsa_preview.ops import _linear_compress_decode
from nsa_ref.ops import compressed_attention, linear_compress


def create_cu_seqlens(seqlen: int) -> torch.Tensor:
    """Create cumulative sequence lengths tensor for batch processing."""
    return torch.arange(0, 2 * seqlen, seqlen, dtype=torch.int32)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark topk sparse attention kernels")

    parser.add_argument("--seqlen", type=int, default=65536, help="Sequence length")
    parser.add_argument("--num-q-heads", type=int, default=64, help="Number of query heads")
    parser.add_argument("--num-k-heads", type=int, default=64, help="Number of key/value heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")

    # NSA specific configuration
    parser.add_argument("--block-size", type=int, default=64, help="Block size for sparse attention")
    parser.add_argument("--topk", type=int, default=16, help="Top-k blocks for each query")

    # Benchmark configuration
    parser.add_argument("--warm-up", type=int, default=5, help="Number of warm-up runs")
    parser.add_argument("--benchmark-iters", type=int, default=10, help="Number of benchmark runs")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    seqlen = args.seqlen
    num_q_heads = args.num_q_heads
    num_k_heads = args.num_k_heads
    head_dim = args.head_dim
    block_size = args.block_size
    topk = args.topk
    kernel_size = 32
    kernel_stride = 16

    # Benchmark parameters
    warm_up = args.warm_up
    benchmark_iters = args.benchmark_iters

    # Create test data
    torch.random.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    assert num_q_heads % num_k_heads == 0, "num_k_heads must be divisible by num_q_heads"
    num_share_q_heads = num_q_heads // num_k_heads

    # Generate random q, k, v tensors
    q = torch.randn(seqlen, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(seqlen, num_k_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen, num_k_heads, head_dim, device=device, dtype=dtype)

    # generate nsa parameters
    compress_key = torch.randn(num_k_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype)
    compress_value = torch.randn(num_k_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype)
    intra_block_pe = torch.randn(num_k_heads, kernel_size, head_dim, device=device, dtype=dtype)

    # Create cumulative sequence lengths
    cu_seqlens = create_cu_seqlens(seqlen).to(device)

    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"cu_seqlens: {cu_seqlens}")
    print(f"kernel_size: {kernel_size}, kernel_stride: {kernel_stride}")

    # Test linear_compress_decode function
    print("\n" + "="*60)
    print("TESTING LINEAR_COMPRESS_DECODE")
    print("="*60)

    # Test with different append sizes
    test_append_sizes = [1]
    test_append_sizes = [size for size in test_append_sizes if size <= seqlen//2]

    for append_size in test_append_sizes:
        split_point = seqlen - append_size
        if split_point <= kernel_size:  # Need enough tokens for at least one window
            continue

        print(f"\nTesting: prefill({seqlen}) vs prefill({split_point}) + decode({append_size})")

        # Method 1: Full sequence training (ground truth)
        full_compressed_k, full_compressed_cu_seqlen = linear_compress(
            k,
            compress_key,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )

        full_compressed_v, _ = linear_compress(
            v,
            compress_value,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            None,
        )

        compressed_seqlens_full = full_compressed_cu_seqlen[1:] - full_compressed_cu_seqlen[:-1]
        attn_output_full, topk_idx_full = compressed_attention(
            q=q,
            k=full_compressed_k,
            v=full_compressed_v,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=full_compressed_cu_seqlen,
            max_seqlen_q=seqlen,
            max_seqlen_k=compressed_seqlens_full.max().item(),
            sm_scale=None,
            init_blocks=1,
            local_blocks=2,
            parallel_topk_compute=False,
        )

        # Method 2: Split into prefill(n-k) + decode(k)
        # First part: training on first split_point tokens
        k_first_part = k[:split_point]
        v_first_part = v[:split_point]
        cu_seqlens_first = create_cu_seqlens(split_point).to(device)

        compressed_k_first, compressed_cu_seqlens_first = linear_compress(
            k_first_part,
            compress_key,
            cu_seqlens_first,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )

        compressed_v_first, _ = linear_compress(
            v_first_part,
            compress_value,
            cu_seqlens_first,
            kernel_size,
            kernel_stride,
            None,
        )

        # Second part: decode on last append_size tokens
        k_last_part = k[split_point:]
        v_last_part = v[split_point:]

        # Prepare initial buffer with some context from first part
        # Need last (kernel_size-1) tokens from first part for potential overlapping windows
        buffer_size = min(kernel_size - 1, split_point)
        initial_buffer_k = k_first_part[-buffer_size:] if buffer_size > 0 else None
        initial_buffer_v = v_first_part[-buffer_size:] if buffer_size > 0 else None

        # Decode the last part
        decode_k_output = _linear_compress_decode(
            k_last_part,
            compress_key,
            kernel_size,
            kernel_stride,
            intra_block_pe,
            split_point,
            initial_buffer_k,
        )

        decode_v_output = _linear_compress_decode(
            v_last_part,
            compress_value,
            kernel_size,
            kernel_stride,
            None,
            split_point,
            initial_buffer_v,
        )

        # Combine results
        if decode_k_output is not None and decode_k_output.shape[0] > 0:
            combined_compressed_k = torch.cat([compressed_k_first, decode_k_output], dim=0)
            combined_compressed_v = torch.cat([compressed_v_first, decode_v_output], dim=0)
        else:
            combined_compressed_k = compressed_k_first
            combined_compressed_v = compressed_v_first

        compressed_seqlens_full = full_compressed_cu_seqlen[1:] - full_compressed_cu_seqlen[:-1]
        attn_output_cur, topk_idx_cur = compressed_attention(
            q=q,
            k=combined_compressed_k,
            v=combined_compressed_v,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=full_compressed_cu_seqlen,
            max_seqlen_q=seqlen,
            max_seqlen_k=compressed_seqlens_full.max().item(),
            sm_scale=None,
            init_blocks=1,
            local_blocks=2,
            parallel_topk_compute=False,
        )

        # Check shapes match
        assert full_compressed_k.shape == combined_compressed_k.shape, \
            f"K shape mismatch: {full_compressed_k.shape} vs {combined_compressed_k.shape}"
        assert full_compressed_v.shape == combined_compressed_v.shape, \
            f"V shape mismatch: {full_compressed_v.shape} vs {combined_compressed_v.shape}"

        # Check values match
        torch.testing.assert_close(
            full_compressed_k, combined_compressed_k,
            rtol=1e-4, atol=1e-5,
            msg="K values don't match between full training and split+decode"
        )
        torch.testing.assert_close(
            full_compressed_v, combined_compressed_v,
            rtol=1e-4, atol=1e-5,
            msg="V values don't match between full training and split+decode"
        )
        torch.testing.assert_close(
            attn_output_full, attn_output_cur,
            rtol=1e-4, atol=1e-5,
            msg="Compressed attention values don't match between full training and split+decode"
        )

        print(f"  ✅ Test PASSED for decode_size={append_size}")

    # -------------------------------------------------------------------------
    # Equivalence test: K tokens in one call vs K calls with 1 token each
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("EQUIVALENCE: multi-token vs incremental _linear_compress_decode")
    print("="*60)

    for K in [1, 4, 32]:
        if K < 1:
            continue
        # prev_total_len: uncompressed sequence length before the K new tokens.
        # Use a value so that at least one new compressed block is produced.
        prev_total_len = 64
        buffer_len = kernel_size - 1
        device = "cuda"
        dtype = torch.bfloat16
        num_heads = num_k_heads

        torch.manual_seed(123 + K)
        initial_buffer = torch.randn(buffer_len, num_heads, head_dim, device=device, dtype=dtype)
        new_tokens = torch.randn(K, num_heads, head_dim, device=device, dtype=dtype)

        # Method A: incremental (one token per call), collect and concat outputs
        buffer = initial_buffer.clone()
        prev = prev_total_len
        outputs_A = []
        for i in range(K):
            out = _linear_compress_decode(
                new_tokens[i : i + 1],
                compress_key,
                kernel_size,
                kernel_stride,
                intra_block_pe,
                prev,
                buffer,
            )
            if out is not None:
                outputs_A.append(out)
            buffer = torch.cat([buffer, new_tokens[i : i + 1]], dim=0)[-buffer_len:]
            prev = prev + 1

        result_A = torch.cat(outputs_A, dim=0) if outputs_A else None

        # Method B: one call with all K tokens
        result_B = _linear_compress_decode(
            new_tokens,
            compress_key,
            kernel_size,
            kernel_stride,
            intra_block_pe,
            prev_total_len,
            initial_buffer,
        )

        if result_A is None and result_B is None:
            print(f"  K={K}: both None (no new blocks) ✅")
            continue
        assert result_A is not None, f"K={K}: method A returned None but B did not"
        assert result_B is not None, f"K={K}: method B returned None but A did not"
        assert result_A.shape == result_B.shape, (
            f"K={K}: shape mismatch A={result_A.shape} B={result_B.shape}"
        )
        torch.testing.assert_close(
            result_A,
            result_B,
            rtol=1e-2,
            atol=1e-2,
            msg=f"K={K}: multi-token output != incremental output",
        )
        print(f"  K={K}: shape {result_A.shape} ✅")

    print("EQUIVALENCE TEST COMPLETED")

    print("\n" + "="*60)
    print("DECODE TESTING COMPLETED")
    print("="*60)

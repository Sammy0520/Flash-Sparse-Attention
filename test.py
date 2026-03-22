import torch
import triton
import math

from impl.impl_baseline import _topk_sparse_attention_fwd as topk_sparse_attention_fwd_baseline
from impl.impl_splitk import _topk_sparse_attention_fwd as topk_sparse_attention_fwd_opt
#topk_sparse_attention_fwd_opt = topk_sparse_attention_fwd_baseline


def compute_error_stats(tensor_ref, tensor_opt, name=""):
    assert tensor_ref.shape == tensor_opt.shape, f"Shape mismatch for {name}: {tensor_ref.shape} vs {tensor_opt.shape}"

    ref_nan_count = torch.isnan(tensor_ref).sum().item()
    opt_nan_count = torch.isnan(tensor_opt).sum().item()

    abs_error = torch.abs(tensor_ref - tensor_opt)
    valid_mask = ~(torch.isnan(tensor_ref) | torch.isnan(tensor_opt) | torch.isinf(abs_error))

    if valid_mask.sum() == 0:
        max_abs_error = float('nan')
        max_rel_error = float('nan')
        mean_abs_error = float('nan')
    else:
        max_abs_error = abs_error[valid_mask].max().item()
        mean_abs_error = abs_error[valid_mask].mean().item()
        with torch.no_grad():
            rel_error = abs_error / (torch.abs(tensor_ref) + 1e-8)
            max_rel_error = rel_error[valid_mask].max().item()

    return {
        'name': name,
        'shape': tuple(tensor_ref.shape),
        'dtype': str(tensor_ref.dtype),
        'ref_nan_count': ref_nan_count,
        'opt_nan_count': opt_nan_count,
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'max_rel_error': max_rel_error,
    }


def print_error_stats(stats):
    print(f"\n  {stats['name']}:")
    print(f"    Shape: {stats['shape']}, Dtype: {stats['dtype']}")
    print(f"    Ref NaN count: {stats['ref_nan_count']}, Opt NaN count: {stats['opt_nan_count']}")
    print(f"    Max Abs Error: {stats['max_abs_error']:.6e}")
    print(f"    Mean Abs Error: {stats['mean_abs_error']:.6e}")
    print(f"    Max Relative Error: {stats['max_rel_error']:.6e}")


def generate_test_data(
    total_q_len: int,     # total number of query tokens; batch_size is always 1
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    context_len: int,     # KV sequence length for the single batch
    topk: int,
    block_size: int,
    device: str = "cuda",
    topk_idx_all_zero: bool = False,
):
    """
    batch_size = 1 always.
      Q  shape: [total_q_len, num_q_heads, head_dim]
      K/V shape: [context_len, num_kv_heads, head_dim]
      cu_seqlens_q: [0, total_q_len]
      cu_seqlens_k: [0, context_len]
    """
    dtype = torch.bfloat16

    q = torch.randn((total_q_len, num_q_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((context_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((context_len, num_kv_heads, head_dim), dtype=dtype, device=device)

    # batch_size = 1: one sequence of total_q_len queries and context_len keys
    cu_seqlens_q = torch.tensor([0, total_q_len], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, context_len], dtype=torch.int32, device=device)

    num_blocks = context_len // block_size

    if topk_idx_all_zero:
        topk_idx = torch.zeros(num_kv_heads, total_q_len, topk, dtype=torch.int32, device=device)
    else:
        # shared_topk = torch.randperm(num_blocks, device=device)[:topk].to(torch.int32)
        # shared_topk[0] = 0
        # topk_idx = shared_topk.view(1, 1, topk).expand(num_kv_heads, total_q_len, topk).contiguous()
        topk_idx = torch.randint(
            0, num_blocks,
            (num_kv_heads, total_q_len, topk),
            dtype=torch.int32, device=device,
        )
        # Force first topk entry to block 0 so every token has at least one valid block
        topk_idx[:, :, 0] = 0

    sm_scale = 1.0 / math.sqrt(head_dim)

    max_seqlen_q = total_q_len
    max_seqlen_k = context_len

    return q, k, v, topk_idx, cu_seqlens_q, cu_seqlens_k, sm_scale, max_seqlen_q, max_seqlen_k


def run_benchmark():
    total_q_len  = 32
    num_q_heads  = 8
    num_kv_heads = 2
    head_dim     = 128
    context_len  = 128 * 1024
    block_size   = 64
    device       = "cuda"

    print(f"--- Benchmark: total_q_len={total_q_len}, batch_size=1, "
          f"HeadDim={head_dim}, Context={context_len//1024}K ---")
    print(f"{'TopK':<10} | {'Baseline (ms)':<15} | {'Opt (ms)':<15} | {'Speedup':<10} | {'Correct'}")
    print("-" * 80)

    for topk in [64, 128, 256, 512, 1024, 2048]:
        (q, k, v, topk_idx,
         cu_q, cu_k,
         sm_scale, max_seqlen_q, max_seqlen_k) = generate_test_data(
            total_q_len, num_q_heads, num_kv_heads, head_dim,
            context_len, topk, block_size, device,
        )

        common_args = (
            q, k, v, topk_idx, block_size,
            cu_q, cu_k,
            max_seqlen_q, max_seqlen_k,
            sm_scale,
        )

        ref_o,  ref_lse  = topk_sparse_attention_fwd_baseline(*common_args)
        tri_o,  tri_lse  = topk_sparse_attention_fwd_opt(*common_args)

        is_correct = "PASS"
        try:
            torch.testing.assert_close(tri_o,   ref_o,   atol=1e-2, rtol=1e-2, equal_nan=True)
            torch.testing.assert_close(tri_lse,  ref_lse, atol=1e-2, rtol=1e-2, equal_nan=True)
        except Exception:
            is_correct = "FAIL"
            print(f"\n{'='*80}")
            print(f"FAIL: TopK={topk}")
            print(f"{'='*80}")
            print_error_stats(compute_error_stats(ref_o,   tri_o,   "Output (O)"))
            print_error_stats(compute_error_stats(ref_lse, tri_lse, "Log-Sum-Exp (LSE)"))
            print(f"{'='*80}\n")

        if is_correct == "PASS":
            ms_base = triton.testing.do_bench(lambda: topk_sparse_attention_fwd_baseline(*common_args))
            ms_opt  = triton.testing.do_bench(lambda: topk_sparse_attention_fwd_opt(*common_args, SPLIT_K=16))
            speedup = ms_base / ms_opt
            print(f"{topk:<10} | {ms_base:>13.4f} | {ms_opt:>13.4f} | {speedup:>9.2f}x | {is_correct}")


if __name__ == "__main__":
    run_benchmark()

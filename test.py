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


def generate_legal_decode_data(
    num_seq: int,         # Batch size
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    context_len: int,
    topk: int,
    block_size: int,
    device="cuda"
):
    dtype = torch.bfloat16
    total_q_len = num_seq 
    
    # Q: [total_q_len, num_q_heads, head_dim]
    q = torch.randn((total_q_len, num_q_heads, head_dim), dtype=dtype, device=device)
    
    # K/V: [num_seq * context_len, num_kv_heads, head_dim]
    total_k_len = num_seq * context_len
    k = torch.randn((total_k_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((total_k_len, num_kv_heads, head_dim), dtype=dtype, device=device)

    cu_seqlens_q = torch.arange(0, num_seq + 1, step=1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, (num_seq + 1) * context_len, step=context_len, dtype=torch.int32, device=device)

    num_blocks = context_len // block_size

    topk_idx = torch.randint(0, num_blocks, (num_kv_heads, total_q_len, topk), dtype=torch.int32, device=device)
    topk_idx[:, :, 0] = 0
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    return q, k, v, topk_idx, cu_seqlens_q, cu_seqlens_k, sm_scale, total_q_len, context_len


def run_benchmark():
    num_seq = 2
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128
    context_len = 128 * 1024
    block_size = 64
    
    device = "cuda"
    
    print(f"--- Legal Decode Benchmark: Seqs={num_seq}, HeadDim={head_dim}, Context={context_len//1024}K ---")
    print(f"{'TopK':<10} | {'Baseline (ms)':<15} | {'Opt (ms)':<15} | {'Speedup':<10} | {'Correct'}")
    print("-" * 80)

    for topk in [64, 128, 256, 512, 1024, 2048]:
        q, k, v, topk_idx, cu_q, cu_k, sm_scale, total_q, max_k = generate_legal_decode_data(
            num_seq, num_q_heads, num_kv_heads, head_dim, context_len, topk, block_size, device
        )

        common_args = (q, k, v, topk_idx, block_size, cu_q, cu_k, 1, max_k, sm_scale)
        
        ref_o, ref_lse = topk_sparse_attention_fwd_baseline(*common_args)
        tri_o, tri_lse = topk_sparse_attention_fwd_opt(*common_args)

        is_correct = "PASS"
        try:
            torch.testing.assert_close(tri_o, ref_o, atol=1e-2, rtol=1e-2, equal_nan=True)
            torch.testing.assert_close(tri_lse, ref_lse, atol=1e-2, rtol=1e-2, equal_nan=True)
        except Exception as e:
            is_correct = "FAIL"
            print(f"\n{'='*80}")
            print(f"FAIL: TopK={topk}")
            print(f"{'='*80}")

            o_stats = compute_error_stats(ref_o, tri_o, "Output (O)")
            lse_stats = compute_error_stats(ref_lse, tri_lse, "Log-Sum-Exp (LSE)")
            
            print_error_stats(o_stats)
            print_error_stats(lse_stats)
            print(f"{'='*80}\n")

        if is_correct == "PASS":
            ms_base = triton.testing.do_bench(lambda: topk_sparse_attention_fwd_baseline(*common_args))
            ms_opt = triton.testing.do_bench(lambda: topk_sparse_attention_fwd_opt(*common_args))

            speedup = ms_base / ms_opt
            print(f"{topk:<10} | {ms_base:>13.4f} | {ms_opt:>13.4f} | {speedup:>9.2f}x | {is_correct}")


if __name__ == "__main__":
    run_benchmark()
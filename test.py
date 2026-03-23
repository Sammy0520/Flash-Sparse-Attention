import torch
import triton
import math
from impl.impl_baseline import _compressed_attention_fwd_decode as compressed_attention_fwd_decode_baseline
from impl.impl_splitk import _compressed_attention_fwd_decode as compressed_attention_fwd_decode_splitk


def compute_error_stats(ref_o, opt_o, ref_lse, opt_lse):
    o_diff = (ref_o - opt_o).abs()
    o_max_err = o_diff.max().item()
    o_mean_err = o_diff.mean().item()

    lse_diff = (ref_lse - opt_lse).abs()
    lse_max_err = lse_diff.max().item()
    
    return o_max_err, o_mean_err, lse_max_err


def run_test():
    device = "cuda"
    dtype = torch.bfloat16
    
    total_q_len = 32
    num_q_heads = 8
    num_kv_heads = 8
    head_dim = 128

    compressed_k_len = 8193 
    kernel_size = 32
    kernel_stride = 16
    query_start_index = 131071
    
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn((total_q_len, num_q_heads, head_dim), dtype=dtype, device=device)
    k = torch.randn((compressed_k_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    v = torch.randn((compressed_k_len, num_kv_heads, head_dim), dtype=dtype, device=device)
    
    # Batch size = 1
    cu_seqlens_q = torch.tensor([0, total_q_len], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, compressed_k_len], dtype=torch.int32, device=device)
    
    max_seqlen_q = torch.tensor(total_q_len, device=device)
    max_seqlen_k = torch.tensor(compressed_k_len, device=device)

    print(f"--- Benchmark Configuration ---")
    print(f"Q_shape: {q.shape}, K_shape: {k.shape}")
    print(f"Context: {query_start_index} -> {query_start_index + total_q_len}")
    print("-" * 40)

    ref_o, ref_lse = compressed_attention_fwd_decode_baseline(
        q, k, v, kernel_size, kernel_stride, 
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        sm_scale, query_start_index
    )
    
    opt_o, opt_lse = compressed_attention_fwd_decode_splitk(
        q, k, v, kernel_size, kernel_stride, 
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        sm_scale, query_start_index
    )

    o_max, o_mean, lse_max = compute_error_stats(ref_o, opt_o, ref_lse, opt_lse)
    
    is_correct = o_max < 1e-2 and lse_max < 1e-2
    status = "PASS" if is_correct else "FAIL"
    
    print(f"Correctness: [{status}]")
    print(f"  O Max Err: {o_max:.6e}, O Mean Err: {o_mean:.6e}")
    print(f"  LSE Max Err: {lse_max:.6e}")

    if is_correct:
        for _ in range(10):
            compressed_attention_fwd_decode_baseline(
                q, k, v, kernel_size, kernel_stride, 
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                sm_scale, query_start_index
            )

        ms_base = triton.testing.do_bench(lambda: compressed_attention_fwd_decode_baseline(
            q, k, v, kernel_size, kernel_stride, 
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            sm_scale, query_start_index
        ))

        ms_opt_32 = triton.testing.do_bench(lambda: compressed_attention_fwd_decode_splitk(
            q, k, v, kernel_size, kernel_stride, 
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            sm_scale, query_start_index, SPLIT_K=32
        ))

        ms_opt_16 = triton.testing.do_bench(lambda: compressed_attention_fwd_decode_splitk(
            q, k, v, kernel_size, kernel_stride, 
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            sm_scale, query_start_index, SPLIT_K=16
        ))

        ms_opt_8 = triton.testing.do_bench(lambda: compressed_attention_fwd_decode_splitk(
            q, k, v, kernel_size, kernel_stride, 
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            sm_scale, query_start_index, SPLIT_K=8
        ))

        ms_opt_4 = triton.testing.do_bench(lambda: compressed_attention_fwd_decode_splitk(
            q, k, v, kernel_size, kernel_stride, 
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            sm_scale, query_start_index, SPLIT_K=4
        ))

        print(f"\nPerformance:")
        print(f"  Baseline: {ms_base:.4f} ms")
        print(f"  Split-K(32):  {ms_opt_32:.4f} ms")
        print(f"  Split-K(16):  {ms_opt_16:.4f} ms")
        print(f"  Split-K(8):  {ms_opt_8:.4f} ms")
        print(f"  Split-K(4):  {ms_opt_4:.4f} ms")
        print(f"  Speedup(32):  {ms_base / ms_opt_32:.2f}x")
        print(f"  Speedup(16):  {ms_base / ms_opt_16:.2f}x")
        print(f"  Speedup(8):  {ms_base / ms_opt_8:.2f}x")
        print(f"  Speedup(4):  {ms_base / ms_opt_4:.2f}x")


if __name__ == "__main__":
    run_test()

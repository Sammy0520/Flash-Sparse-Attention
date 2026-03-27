import torch
from flash_attn import flash_attn_varlen_func
from impl.topk_sparse_attention_decode import _topk_sparse_attention_decode
from impl.unified_sparse_attention_decode import _unified_sparse_attention_decode

def benchmark():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    total_q_len = 32
    context_len = 128 * 1024  # 128k
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128
    
    block_size = 64
    topk = 16
    window_size = 512
    cfactor = 4
    
    num_k_blocks = context_len // block_size

    q = torch.randn(total_q_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(context_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    topk_idx = torch.randint(0, num_k_blocks, (num_kv_heads, total_q_len, topk), device=device, dtype=torch.int32)

    gate = torch.randn(total_q_len, 3, device=device).softmax(dim=-1).to(torch.float32)
    
    cu_seqlens_q = torch.tensor([0, total_q_len], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, context_len], device=device, dtype=torch.int32)

    torch.cuda.synchronize()
    
    # 预热
    for _ in range(10):
        _ = _topk_sparse_attention_decode(q, k, v, topk_idx, block_size, cu_seqlens_q, cu_seqlens_k, total_q_len, context_len)
        _ = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, total_q_len, context_len, causal=False, window_size=(window_size, -1))

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        # Branch 1: Sparse Top-K
        out_sparse = _topk_sparse_attention_decode(
            q, k, v, topk_idx, block_size, 
            cu_seqlens_q, cu_seqlens_k, total_q_len, context_len
        )
        # Branch 2: Sliding Window
        out_sliding = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, 
            total_q_len, context_len, causal=False, window_size=(window_size, -1)
        )
        # Combine (只比较这两路)
        baseline_output = gate[:, 1:2, None] * out_sparse + gate[:, 2:3, None] * out_sliding
    end_event.record()
    torch.cuda.synchronize()
    baseline_ms = start_event.elapsed_time(end_event) / 100

    # -----------------------------------------------------------
    # 4. 运行 Unified Kernel
    # -----------------------------------------------------------
    # 预热
    for _ in range(10):
        _ = _unified_sparse_attention_decode(q, k, v, topk_idx, block_size, window_size, cu_seqlens_q, cu_seqlens_k, total_q_len, context_len, gate, CFACTOR=cfactor)

    start_event.record()
    for _ in range(100):
        unified_output = _unified_sparse_attention_decode(
            q, k, v, topk_idx, block_size, window_size,
            cu_seqlens_q, cu_seqlens_k, total_q_len, context_len,
            gate, CFACTOR=cfactor
        )
    end_event.record()
    torch.cuda.synchronize()
    unified_ms = start_event.elapsed_time(end_event) / 100

    # abs_diff = (baseline_output - unified_output).abs().mean().item()
    # rel_diff = (baseline_output - unified_output).abs().mean().item() / baseline_output.abs().mean().item()
    
    print(f"--- Benchmark Results (Context: {context_len//1024}k) ---")
    print(f"Baseline Latency: {baseline_ms:.4f} ms")
    print(f"Unified Latency:  {unified_ms:.4f} ms")
    print(f"Speedup:          {baseline_ms / unified_ms:.2f}x")
    # print(f"Mean Abs Diff:    {abs_diff:.6f}")
    # print(f"Mean Rel Diff:    {rel_diff:.6f}")
    
    # if rel_diff < 1e-2:
    #     print("SUCCESS: Unified output matches Baseline closely.")
    # else:
    #     print("WARNING: Large numerical difference. Check Softmax normalization logic.")

if __name__ == "__main__":
    benchmark()

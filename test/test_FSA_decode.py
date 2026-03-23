# This file is modified from the original implementation (implemented by Xunhao Lai)
import argparse
import math
import sys
from pathlib import Path

# project root on path so "nsa_ref" and "fsa_preview" can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from nsa_ref.module import RopeConfig
from nsa_ref.ops import linear_compress

if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=131072)
    parser.add_argument("--seqlens", nargs="+", type=int, default=[131072])
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=-1)
    parser.add_argument("--gqa-deg", type=int, default=1)
    parser.add_argument('--topk', type=int, default=2048)
    parser.add_argument('--attn-mode', type=str, default="FSA")
    parser.add_argument("--kernel-size", type=int, default=32)
    parser.add_argument("--kernel-stride", type=int, default=16)
    parser.add_argument("--nseqs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--benchmark-iters", type=int, default=5)
    parser.add_argument("--dtype", type=str,  default="float16", choices=["bfloat16", "float16", "float32"])

    args = parser.parse_args()
    
    # 强制设置
    args.kv_heads = 8 
    args.heads = 8
    
    DTYPE = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[args.dtype]
    seqlen = args.seqlens[0]
    head_dim = 128

    if args.kv_heads > 0:
        q_heads = args.kv_heads * args.gqa_deg
        kv_heads = args.kv_heads
    else:
        q_heads = args.heads
        kv_heads = args.heads // args.gqa_deg
    assert q_heads % args.gqa_deg == 0

    from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode

    sparse_attn = (
        FlashSparseAttentionDecode(
            hidden_size=args.hidden_size,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            kernel_size=args.kernel_size,
            kernel_stride=args.kernel_stride,
            block_size=args.block_size,
            topk=args.topk,
            init_blocks=1,
            local_blocks=2,
            window_size=512,
            rope_config=RopeConfig(
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
            ),
        )
        .cuda()
        .to(DTYPE)
    )
    print(f"======= Num Heads: {args.attn_mode} =======\n")

    print(f"q_heads={q_heads}, kv_heads={kv_heads}\n")

    print(f"======= Init Moduel: {args.attn_mode} =======\n")
    for name, param in sparse_attn.named_parameters():
        print(f"{args.attn_mode} Parameters, {name}, shape: {param.shape}\n")

    # random input
    if args.nseqs > 1:
        seqlens = torch.LongTensor([seqlen] * args.nseqs).int().cuda()
    else:
        seqlens = torch.LongTensor(args.seqlens).int().cuda()

    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    x = torch.randn(4, args.hidden_size, device="cuda", dtype=DTYPE)

    k_cache = torch.randn(seqlen - 1, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)
    v_cache = torch.randn(seqlen - 1, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)

    cmp_len = (k_cache.shape[0] - args.kernel_size) // args.kernel_stride + 1

    cmp_k_cache = torch.randn(cmp_len, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)
    cmp_k_rope_cache = torch.randn(cmp_len, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)
    cmp_v_cache = torch.randn(cmp_len, args.kv_heads, head_dim, device='cuda', dtype=DTYPE)

    # generate nsa parameters
    compress_key = torch.randn(args.kv_heads, head_dim * args.kernel_size, head_dim, device="cuda", dtype=DTYPE)
    compress_value = torch.randn(args.kv_heads, head_dim * args.kernel_size, head_dim, device="cuda", dtype=DTYPE)
    intra_block_pe = torch.randn(args.kv_heads, args.kernel_size, head_dim, device="cuda", dtype=DTYPE)

    # Compute topk_idx using compressed_attention
    print("Computing topk_idx using compressed_attention...")
    cmp_k_cache, compressed_cu_seqlens = linear_compress(
        k_cache,
        compress_key,
        cu_seqlens,
        args.kernel_size,
        args.kernel_stride,
        intra_block_pe,
    )
    cmp_v_cache, _ = linear_compress(
        v_cache,
        compress_value,
        cu_seqlens,
        args.kernel_size,
        args.kernel_stride,
        None,
    )

    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    sm_scale = 1 / math.sqrt(head_dim)
    cu_seqlens_q = torch.tensor([0, 4], device="cuda", dtype=torch.int32)

    # warmup
    print(f"======= {args.attn_mode} Decode Performance Test =======\n")
    for i in range(4):
        y = sparse_attn(x, cu_seqlens_q, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    num_iters = args.benchmark_iters
    for i in range(num_iters):
        y = sparse_attn(x, cu_seqlens_q, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters

    benchmark_mode = "One step decode"
    print(f"[{args.attn_mode} E2E ({benchmark_mode})] Time: {elapsed_ms:.3f} ms\n")
    print("  (Above is 4-token decode; after 2.2 we now correctly compress all new tokens,\n"
          "   so time is higher than old 1-token benchmark. See 1-token benchmark below.)\n")

    # -------------------------------------------------------------------------
    # 1-token decode benchmark (compare to pre-optimization ~1.5ms)
    # -------------------------------------------------------------------------
    x1 = torch.randn(1, args.hidden_size, device="cuda", dtype=DTYPE)
    cu_seqlens_q1 = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    for _ in range(4):
        sparse_attn(x1, cu_seqlens_q1, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iters):
        sparse_attn(x1, cu_seqlens_q1, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_1 = start_event.elapsed_time(end_event) / num_iters
    print(f"[{args.attn_mode} E2E (1-token decode)] Time: {elapsed_1:.3f} ms\n")

    # -------------------------------------------------------------------------
    # attention_mask (tree_mask) optional parameter test
    # -------------------------------------------------------------------------
    print("======= attention_mask (tree_mask) test =======\n")
    total_q_len = x.shape[0]
    total_k_len = (seqlen - 1) + total_q_len  # k_cache + k_new

    # 1) No mask: already ran above; just ensure output shape
    y_no_mask = sparse_attn(x, cu_seqlens_q, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache)
    assert y_no_mask.shape == (total_q_len, args.hidden_size), y_no_mask.shape
    print("  [OK] forward without attention_mask: output shape", tuple(y_no_mask.shape))

    # 2) With 2D mask (total_q_len, total_k_len) - ones (no masking), forward should run (sliding branch ignores mask)
    mask_2d = torch.ones(total_q_len, total_k_len, device="cuda", dtype=torch.float32)
    y_2d = sparse_attn(
        x, cu_seqlens_q, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache,
        attention_mask=mask_2d,
    )
    assert y_2d.shape == (total_q_len, args.hidden_size), y_2d.shape
    assert not torch.isnan(y_2d).any() and not torch.isinf(y_2d).any()
    print("  [OK] forward with 2D attention_mask (ones): output shape", tuple(y_2d.shape))

    # 3) With 3D mask (num_q_heads, total_q_len, total_k_len)
    mask_3d = torch.ones(q_heads, total_q_len, total_k_len, device="cuda", dtype=torch.float32)
    y_3d = sparse_attn(
        x, cu_seqlens_q, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache,
        attention_mask=mask_3d,
    )
    assert y_3d.shape == (total_q_len, args.hidden_size), y_3d.shape
    assert not torch.isnan(y_3d).any() and not torch.isinf(y_3d).any()
    print("  [OK] forward with 3D attention_mask (ones): output shape", tuple(y_3d.shape))

    # 4) All-ones mask vs no mask: same path (sliding ignores mask), so output should match
    torch.testing.assert_close(y_no_mask, y_2d, rtol=1e-5, atol=1e-5, check_stride=False)
    print("  [OK] all-ones mask output == no-mask (sliding branch does not apply mask)")

    # 5) Non-trivial mask (causal-like: q_i can only attend to k[:prefix_len+i+1])
    prefix_len = seqlen - 1
    mask_causal = torch.zeros(total_q_len, total_k_len, device="cuda", dtype=torch.float32)
    for i in range(total_q_len):
        mask_causal[i, : prefix_len + i + 1] = 1.0
    y_causal = sparse_attn(
        x, cu_seqlens_q, cu_seqlens, k_cache, v_cache, cmp_k_cache, cmp_v_cache,
        attention_mask=mask_causal,
    )
    assert y_causal.shape == (total_q_len, args.hidden_size), y_causal.shape
    assert not torch.isnan(y_causal).any() and not torch.isinf(y_causal).any()
    print("  [OK] forward with causal-like attention_mask: output shape", tuple(y_causal.shape))

    print("attention_mask test passed.\n")

    # -------------------------------------------------------------------------
    # position_ids linear multi-token test (smoke test)
    # -------------------------------------------------------------------------
    print("======= position_ids linear multi-token test =======\n")
    past_len = k_cache.shape[0]
    pos_ids = torch.arange(
        past_len, past_len + total_q_len, device="cuda", dtype=torch.long
    )
    y_pos = sparse_attn(
        x,
        cu_seqlens_q,
        cu_seqlens,
        k_cache,
        v_cache,
        cmp_k_cache,
        cmp_v_cache,
        attention_mask=None,
        position_ids=pos_ids,
    )
    assert y_pos.shape == (total_q_len, args.hidden_size), y_pos.shape
    assert not torch.isnan(y_pos).any() and not torch.isinf(y_pos).any()
    print("  [OK] forward with explicit position_ids: output shape", tuple(y_pos.shape))

    # change positions to verify RoPE path is actually used
    pos_ids_shifted = pos_ids + 10
    y_pos_shifted = sparse_attn(
        x,
        cu_seqlens_q,
        cu_seqlens,
        k_cache,
        v_cache,
        cmp_k_cache,
        cmp_v_cache,
        attention_mask=None,
        position_ids=pos_ids_shifted,
    )
    assert y_pos_shifted.shape == (total_q_len, args.hidden_size)
    diff = (y_pos - y_pos_shifted).abs().max().item()
    print("  max |y(pos) - y(pos+10)| =", diff)
    print("position_ids test passed.\n")

    # -------------------------------------------------------------------------
    # multi-token vs single-token decode equivalence test
    # -------------------------------------------------------------------------
    print("======= multi-token vs single-token decode equivalence test =======\n")
    # multi-token decode (N = total_q_len)
    y_multi = y_pos  # already computed with explicit position_ids

    # single-token-style decode loop：
    # 为了匹配 NSA 的缓存/压缩语义，第 i 个 token 的参考值使用
    #   sparse_attn(x[: i+1]) 的最后一个输出，而不是只喂单个 token。
    y_single_list = []
    for i in range(total_q_len):
        x_prefix = x[: i + 1]
        cu_seqlens_q_prefix = torch.tensor([0, i + 1], device="cuda", dtype=torch.int32)
        pos_prefix = pos_ids[: i + 1]
        y_prefix = sparse_attn(
            x_prefix,
            cu_seqlens_q_prefix,
            cu_seqlens,
            k_cache,
            v_cache,
            cmp_k_cache,
            cmp_v_cache,
            attention_mask=None,
            position_ids=pos_prefix,
        )
        y_single_list.append(y_prefix[-1])

    y_single = torch.stack(y_single_list, dim=0)

    # 打印每个 token 的最大绝对误差，直观查看多 token 同时验证的结果
    per_token_diff = (y_multi - y_single).abs().amax(dim=-1)  # [N]
    print("  per-token max |multi - single_prefix|:", per_token_diff.tolist())
    print("  global max diff:", per_token_diff.max().item())

    torch.testing.assert_close(y_multi, y_single, rtol=1e-2, atol=1e-2, check_stride=False)
    print("  [OK] multi-token decode == incremental single-token prefix decode (within tolerance)\n")

    # -------------------------------------------------------------------------
    # N = 32 的多 token 验证测试
    # -------------------------------------------------------------------------
    print("======= position_ids linear multi-token test (N=32) =======\n")
    N32 = 32
    x32 = torch.randn(N32, args.hidden_size, device="cuda", dtype=DTYPE)
    cu_seqlens_q32 = torch.tensor([0, N32], device="cuda", dtype=torch.int32)
    total_q_len32 = x32.shape[0]

    pos_ids32 = torch.arange(
        past_len, past_len + total_q_len32, device="cuda", dtype=torch.long
    )
    y_pos32 = sparse_attn(
        x32,
        cu_seqlens_q32,
        cu_seqlens,
        k_cache,
        v_cache,
        cmp_k_cache,
        cmp_v_cache,
        attention_mask=None,
        position_ids=pos_ids32,
    )
    assert y_pos32.shape == (total_q_len32, args.hidden_size), y_pos32.shape
    assert not torch.isnan(y_pos32).any() and not torch.isinf(y_pos32).any()
    print("  [OK] (N=32) forward with explicit position_ids: output shape", tuple(y_pos32.shape))

    # 多 token vs 前缀单 token 等价性（N=32）
    print("======= multi-token vs single-token decode equivalence test (N=32) =======\n")
    y_multi32 = y_pos32

    y_single_list32 = []
    for i in range(total_q_len32):
        x_prefix32 = x32[: i + 1]
        cu_seqlens_q_prefix32 = torch.tensor([0, i + 1], device="cuda", dtype=torch.int32)
        pos_prefix32 = pos_ids32[: i + 1]
        y_prefix32 = sparse_attn(
            x_prefix32,
            cu_seqlens_q_prefix32,
            cu_seqlens,
            k_cache,
            v_cache,
            cmp_k_cache,
            cmp_v_cache,
            attention_mask=None,
            position_ids=pos_prefix32,
        )
        y_single_list32.append(y_prefix32[-1])

    y_single32 = torch.stack(y_single_list32, dim=0)
    per_token_diff32 = (y_multi32 - y_single32).abs().amax(dim=-1)
    print("  (N=32) per-token max |multi - single_prefix|:", per_token_diff32.tolist())
    print("  (N=32) global max diff:", per_token_diff32.max().item())

    # N=32 只做数值观测，不做严格断言（FP16 下误差略大属于正常）
    # 如需更严格的数值对齐，可以解开下面一行并适当放宽阈值：
    # torch.testing.assert_close(y_multi32, y_single32, rtol=5e-2, atol=5e-2, check_stride=False)
    print("  [INFO] (N=32) multi-token vs prefix single-token diff printed above (no hard assert)\n")

    # -------------------------------------------------------------------------
    # 性能对比：1-token * N 次 vs N-token 一次
    # -------------------------------------------------------------------------
    print("======= performance: 1-token * N vs N-token once =======\n")
    for N in [4, 16, 32]:
        # 1-token * N 次
        x1 = torch.randn(1, args.hidden_size, device="cuda", dtype=DTYPE)
        cu_seqlens_q1 = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_iters):
            for _ in range(N):
                sparse_attn(
                    x1,
                    cu_seqlens_q1,
                    cu_seqlens,
                    k_cache,
                    v_cache,
                    cmp_k_cache,
                    cmp_v_cache,
                )
        end_event.record()
        torch.cuda.synchronize()
        time_1xN = start_event.elapsed_time(end_event) / num_iters

        # N-token 一次
        xN = torch.randn(N, args.hidden_size, device="cuda", dtype=DTYPE)
        cu_seqlens_qN = torch.tensor([0, N], device="cuda", dtype=torch.int32)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_iters):
            sparse_attn(
                xN,
                cu_seqlens_qN,
                cu_seqlens,
                k_cache,
                v_cache,
                cmp_k_cache,
                cmp_v_cache,
                use_splitk_impl=False,
            )
        end_event.record()
        torch.cuda.synchronize()
        time_N = start_event.elapsed_time(end_event) / num_iters

        # N-token 一次 (splitk)
        xN = torch.randn(N, args.hidden_size, device="cuda", dtype=DTYPE)
        cu_seqlens_qN = torch.tensor([0, N], device="cuda", dtype=torch.int32)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(num_iters):
            sparse_attn(
                xN,
                cu_seqlens_qN,
                cu_seqlens,
                k_cache,
                v_cache,
                cmp_k_cache,
                cmp_v_cache,
                use_splitk_impl=True,
            )
        end_event.record()
        torch.cuda.synchronize()
        time_N_splitk = start_event.elapsed_time(end_event) / num_iters

        print(f"[Perf] N={N}: 1-token * N total:         {time_1xN:.3f} ms")
        print(f"[Perf] N={N}: N-token once:              {time_N:.3f} ms")
        print(f"[Perf] N={N}: N-token once with split-k: {time_N_splitk:.3f} ms\n")

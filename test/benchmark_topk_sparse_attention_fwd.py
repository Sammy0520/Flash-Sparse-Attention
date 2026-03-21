"""
bench_topk_sparse_attn.py
对 _topk_sparse_attention_fwd 进行计时，相邻 token 共享相似的 topk_idx。
用法：
    python bench_topk_sparse_attn.py
    python bench_topk_sparse_attn.py --batch 4 --seqlen 2048 --topk 16 --block_size 64
"""

import argparse
import time
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nsa_ref.ops.topk_sparse_attention import _topk_sparse_attention_fwd

# ─────────────────────────────────────────────────────────────────────────────
# 数据生成
# ─────────────────────────────────────────────────────────────────────────────

def make_inputs(
    batch:      int   = 2,
    seqlen:     int   = 1024,   # 每条序列的长度（所有序列等长，便于对比）
    num_q_heads:int   = 16,
    num_kv_heads:int  = 4,
    head_dim:   int   = 128,
    topk:       int   = 8,
    block_size: int   = 64,
    group_size: int   = 4,      # 相邻多少个 token 共享同一组基准 topk
    noise_prob: float = 0.1,    # 组内 token 有多大概率对 topk 做轻微扰动
    dtype:      torch.dtype = torch.bfloat16,
    device:     str   = "cuda",
    seed:       int   = 42,
):
    """
    生成确定性的随机输入，保证：
    - 相邻 group_size 个 token 使用同一组基准 topk block
    - 以 noise_prob 概率对组内后续 token 做轻微替换（模拟真实场景）
    """
    torch.manual_seed(seed)

    total_len    = batch * seqlen
    num_kv_blocks = seqlen // block_size   # 每条序列有多少个 K block

    # ── Q / K / V ────────────────────────────────────────────────────────────
    scale = head_dim ** -0.5
    q = torch.randn(total_len, num_q_heads,  head_dim, dtype=dtype, device=device)
    k = torch.randn(total_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # ── cu_seqlens（等长序列）────────────────────────────────────────────────
    seqlens      = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = seqlens.cumsum(0)
    cu_seqlens_k[1:] = seqlens.cumsum(0)
    max_seqlen_q = max_seqlen_k = seqlen

    # ── topk_idx：相邻 token 共享基准 topk ──────────────────────────────────
    # shape: [num_kv_heads, total_len, topk]
    topk_idx = torch.full(
        (num_kv_heads, total_len, topk), -1, dtype=torch.int64, device=device
    )

    for b in range(batch):
        q_start = b * seqlen
        for qi in range(seqlen):
            abs_qi   = q_start + qi
            max_block = qi // block_size          # causal 上界（包含）
            if max_block < 0:
                continue                           # 第 0 个 block 还没结束，跳过

            # 可选的 block 集合：[0, max_block]
            n_avail  = max_block + 1
            real_topk = min(topk, n_avail)

            # ── 组内第一个 token 生成基准 topk ──────────────────────────
            is_group_leader = (qi % group_size == 0)

            for kv_h in range(num_kv_heads):
                if is_group_leader or qi == 0:
                    # 从可用 block 里随机不重复采样 real_topk 个
                    perm = torch.randperm(n_avail, device=device)[:real_topk]
                    base_topk = perm.sort().values   # 保持升序
                    # 存到当前 token
                    topk_idx[kv_h, abs_qi, :real_topk] = base_topk
                else:
                    # 继承组长的 topk
                    leader_abs = q_start + (qi // group_size) * group_size
                    inherited  = topk_idx[kv_h, leader_abs, :real_topk].clone()

                    # 以 noise_prob 随机替换一个 block（轻微扰动）
                    if noise_prob > 0 and real_topk > 1 and torch.rand(1).item() < noise_prob:
                        replace_pos = torch.randint(real_topk, (1,)).item()
                        new_block   = torch.randint(n_avail, (1,), device=device).item()
                        inherited[replace_pos] = new_block
                        inherited, _ = inherited.sort()

                    topk_idx[kv_h, abs_qi, :real_topk] = inherited

    sm_scale = scale
    return (q, k, v, topk_idx, block_size,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            sm_scale)


# ─────────────────────────────────────────────────────────────────────────────
# 计时工具
# ─────────────────────────────────────────────────────────────────────────────

def cuda_time_ms(fn, warmup: int = 5, repeat: int = 20) -> float:
    """用 CUDA Event 精确计时，返回平均毫秒数。"""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_evt   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_evt[i].record()
        fn()
        end_evt[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_evt, end_evt)]
    return sum(times) / len(times)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",       type=int,   default=2)
    parser.add_argument("--seqlen",      type=int,   default=1024)
    parser.add_argument("--num_q_heads", type=int,   default=16)
    parser.add_argument("--num_kv_heads",type=int,   default=4)
    parser.add_argument("--head_dim",    type=int,   default=128)
    parser.add_argument("--topk",        type=int,   default=8)
    parser.add_argument("--block_size",  type=int,   default=64)
    parser.add_argument("--group_size",  type=int,   default=4,
                        help="相邻多少个 token 共享同一组基准 topk")
    parser.add_argument("--noise_prob",  type=float, default=0.1,
                        help="组内后续 token 轻微扰动的概率")
    parser.add_argument("--warmup",      type=int,   default=5)
    parser.add_argument("--repeat",      type=int,   default=20)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 GPU"
    device = "cuda"

    print("=" * 60)
    print("  topk sparse attention benchmark")
    print("=" * 60)
    print(f"  batch        = {args.batch}")
    print(f"  seqlen       = {args.seqlen}")
    print(f"  num_q_heads  = {args.num_q_heads}")
    print(f"  num_kv_heads = {args.num_kv_heads}")
    print(f"  head_dim     = {args.head_dim}")
    print(f"  topk         = {args.topk}")
    print(f"  block_size   = {args.block_size}")
    print(f"  group_size   = {args.group_size}  (相邻 token 共享 topk)")
    print(f"  noise_prob   = {args.noise_prob}")
    print(f"  seed         = {args.seed}")
    print(f"  warmup/repeat= {args.warmup}/{args.repeat}")
    print("=" * 60)

    inputs = make_inputs(
        batch        = args.batch,
        seqlen       = args.seqlen,
        num_q_heads  = args.num_q_heads,
        num_kv_heads = args.num_kv_heads,
        head_dim     = args.head_dim,
        topk         = args.topk,
        block_size   = args.block_size,
        group_size   = args.group_size,
        noise_prob   = args.noise_prob,
        device       = device,
        seed         = args.seed,
    )

    q, k, v, topk_idx, block_size, cu_q, cu_k, max_q, max_k, sm_scale = inputs

    # 验证 topk 相邻相似性
    total_len    = args.batch * args.seqlen
    same_count   = 0
    check_pairs  = min(total_len - 1, 500)
    for i in range(check_pairs):
        a = set(topk_idx[0, i].tolist())
        b = set(topk_idx[0, i + 1].tolist())
        if len(a | b) > 0:
            jaccard = len(a & b) / len(a | b)
            same_count += jaccard
    print(f"  topk 相邻 Jaccard 相似度（前{check_pairs}对，kv_head=0）: "
          f"{same_count / check_pairs:.3f}  （越接近1越相似）")
    print("=" * 60)

    def run():
        _topk_sparse_attention_fwd(
            q, k, v, topk_idx, block_size,
            cu_q, cu_k, max_q, max_k, sm_scale,
        )

    avg_ms = cuda_time_ms(run, warmup=args.warmup, repeat=args.repeat)

    # 计算理论 flops（仅 QK 和 PV 两个 matmul）
    # 每个 token attend topk 个 block，每个 block 有 block_size 个 KV
    flops_per_token = 2 * args.num_q_heads * args.head_dim * args.topk * args.block_size * 2
    total_flops     = flops_per_token * total_len
    tflops          = total_flops / (avg_ms * 1e-3) / 1e12

    print(f"  平均耗时        : {avg_ms:.3f} ms")
    print(f"  理论 TFLOPS     : {tflops:.2f}  TFLOPS")
    print(f"  total_len       : {total_len}")
    print("=" * 60)


if __name__ == "__main__":
    main()

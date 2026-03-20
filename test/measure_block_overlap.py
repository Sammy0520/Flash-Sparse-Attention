"""
Block overlap measurement for speculative decoding.

Three experiments:
  1. Correlated hidden states (simulate adjacent draft tokens): vary noise level
  2. Pre-trained-like k cache (low-rank structure): more realistic than pure random
  3. (Optional) Load a real .pt dump if available

Run:
  python test/measure_block_overlap.py
  python test/measure_block_overlap.py --seqlen 32768 --n 4 --noise-levels 0.0 0.1 0.5 1.0 5.0
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsa_ref.module.rope import RopeConfig
from nsa_ref.ops import linear_compress
from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode
import fsa_preview.ops.compressed_attention_decode as _cad
import fsa_preview.module.fsa_decode as _fd

# -----------------------------------------------------------------------
# Hook to capture topk_idx
# -----------------------------------------------------------------------
_captured = {}
_orig_cad = _cad._compressed_attention_decode

def _hook(*a, **kw):
    out, ti = _orig_cad(*a, **kw)
    _captured["topk_idx"] = ti.detach().cpu()
    return out, ti

_cad._compressed_attention_decode = _hook
_fd._compressed_attention_decode  = _hook

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def overlap_stats(topk_idx):
    """topk_idx: [H, N, K] → (adj_overlap, union_blocks, bw_save_pct)"""
    H, N, K = topk_idx.shape
    # pairwise adjacent overlap
    if N >= 2:
        adjs = []
        for i in range(N - 1):
            per_h = [
                len(set(topk_idx[h, i].tolist()) & set(topk_idx[h, i+1].tolist())) / K
                for h in range(H)
            ]
            adjs.append(sum(per_h) / H)
        adj = sum(adjs) / len(adjs)
    else:
        adj = float("nan")

    # union across all N tokens
    unions = []
    for h in range(H):
        u = set()
        for i in range(N):
            u.update(topk_idx[h, i].tolist())
        unions.append(len(u))
    avg_u = sum(unions) / H
    bw_save = (1 - avg_u / (N * K)) * 100
    return adj, avg_u, bw_save


def make_structured_k_cache(past_len, kv_heads, head_dim, rank=32, device="cuda", dtype=torch.float16):
    """Low-rank k cache: simulates real attention patterns where some directions dominate."""
    # Base: low-rank structure (like real hidden states after many attention layers)
    U = torch.randn(past_len, rank, device=device, dtype=torch.float32)
    V = torch.randn(rank, kv_heads * head_dim, device=device, dtype=torch.float32)
    k_flat = (U @ V) / math.sqrt(rank)
    # Add small Gaussian noise
    k_flat = k_flat + 0.1 * torch.randn_like(k_flat)
    k_flat = F.layer_norm(k_flat, [kv_heads * head_dim])
    return k_flat.view(past_len, kv_heads, head_dim).to(dtype)


def make_correlated_queries(N, hidden_size, noise_scale, device="cuda", dtype=torch.float16):
    """
    N queries sharing a common base with added noise.
    noise_scale=0: all identical (max overlap)
    noise_scale=1: noise ≈ signal amplitude (moderate)
    noise_scale=10: noise dominates (nearly independent)
    """
    base = torch.randn(1, hidden_size, device=device, dtype=torch.float32)
    base = F.layer_norm(base, [hidden_size])
    noise = torch.randn(N, hidden_size, device=device, dtype=torch.float32) * noise_scale
    x = (base + noise)
    x = F.layer_norm(x, [hidden_size])
    return x.to(dtype)


def build_model_and_caches(seqlen, N, kv_heads, head_dim, hidden_size,
                           kernel_size, kernel_stride, block_size, topk,
                           init_blocks, local_blocks, window_size,
                           dtype, device, structured_k=False):
    rope_config = RopeConfig(
        max_position_embeddings=131072, head_dim=head_dim, rope_theta=500000,
        rope_scaling={"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                      "original_max_position_embeddings": 8192, "rope_type": "llama3"},
    )
    model = FlashSparseAttentionDecode(
        hidden_size=hidden_size, num_q_heads=kv_heads, num_kv_heads=kv_heads,
        head_dim=head_dim, kernel_size=kernel_size, kernel_stride=kernel_stride,
        block_size=block_size, topk=topk, init_blocks=init_blocks,
        local_blocks=local_blocks, window_size=window_size, rope_config=rope_config,
    ).to(device=device, dtype=dtype)

    past_len = seqlen - N
    if structured_k:
        k_cache = make_structured_k_cache(past_len, kv_heads, head_dim,
                                          rank=64, device=device, dtype=dtype)
    else:
        k_cache = torch.randn(past_len, kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(past_len, kv_heads, head_dim, device=device, dtype=dtype)

    cu_k_raw = torch.tensor([0, past_len], device=device, dtype=torch.int32)
    cmp_k, _ = linear_compress(k_cache, model.compress_key, cu_k_raw,
                                kernel_size, kernel_stride, model.intra_block_pe)
    cmp_v, _ = linear_compress(v_cache, model.compress_value, cu_k_raw,
                                kernel_size, kernel_stride, None)
    cu_k = torch.tensor([0, past_len], device=device, dtype=torch.int32)
    cu_q = torch.tensor([0, N], device=device, dtype=torch.int32)
    pos  = torch.arange(past_len, past_len + N, device=device, dtype=torch.long)
    return model, k_cache, v_cache, cmp_k, cmp_v, cu_k, cu_q, pos


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seqlen",       type=int, default=32768)
    parser.add_argument("--n",            type=int, default=4, help="Draft token count")
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
    parser.add_argument("--noise-levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--n-seeds",      type=int, default=16)
    parser.add_argument("--dtype",        type=str,  default="float16")
    parser.add_argument("--skip-dump",    action="store_true",
                        help="Skip loading the .pt dump files.")
    args = parser.parse_args()

    DTYPE  = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda"
    N      = args.n

    cmp_len    = (args.seqlen - args.kernel_size) // args.kernel_stride + 1
    max_blocks = math.ceil(cmp_len / args.block_size)
    eff_topk   = min(args.topk, max_blocks)

    print("=" * 68)
    print("Block Overlap Quantification for Speculative Decoding")
    print("=" * 68)
    print(f"  seqlen={args.seqlen}  N={N}  topk={args.topk}  block_size={args.block_size}")
    print(f"  max_blocks={max_blocks}  eff_topk={eff_topk}  "
          f"sparsity={eff_topk/max_blocks:.1%}")
    print(f"  init_blocks={args.init_blocks}  local_blocks={args.local_blocks}")
    print(f"  fixed_overlap_floor = "
          f"{(args.init_blocks + args.local_blocks)/eff_topk:.1%}"
          f"  (init+local always same for adjacent tokens)")
    print()

    # ------------------------------------------------------------------
    # Experiment 1: Correlated queries, random k cache
    # ------------------------------------------------------------------
    print("─" * 68)
    print("EXP 1: Correlated queries + RANDOM k cache")
    print("  (noise=0 → identical queries; noise→∞ → independent queries)")
    print("  x_i = LayerNorm(base + noise_scale * randn)")
    print("─" * 68)
    print(f"  {'noise_scale':>12}  {'adj_overlap':>12}  {'union_blks':>11}  {'bw_save%':>9}")
    print("  " + "-" * 50)

    for noise in args.noise_levels:
        adjs, unions, bws = [], [], []
        for seed in range(args.n_seeds):
            torch.manual_seed(seed)
            model, k_cache, v_cache, cmp_k, cmp_v, cu_k, cu_q, pos = build_model_and_caches(
                args.seqlen, N, args.kv_heads, args.head_dim, args.hidden_size,
                args.kernel_size, args.kernel_stride, args.block_size, args.topk,
                args.init_blocks, args.local_blocks, args.window_size,
                DTYPE, device, structured_k=False,
            )
            x = make_correlated_queries(N, args.hidden_size, noise, device, DTYPE)
            with torch.no_grad():
                model(x, cu_q, cu_k, k_cache, v_cache, cmp_k, cmp_v, position_ids=pos)
            adj, u, bw = overlap_stats(_captured["topk_idx"])
            adjs.append(adj); unions.append(u); bws.append(bw)

        print(f"  {noise:>12.2f}  {sum(adjs)/len(adjs):>11.1%}  "
              f"{sum(unions)/len(unions):>11.1f}  {sum(bws)/len(bws):>8.1f}%")

    # ------------------------------------------------------------------
    # Experiment 2: Correlated queries, STRUCTURED k cache (low-rank)
    # ------------------------------------------------------------------
    print()
    print("─" * 68)
    print("EXP 2: Correlated queries + STRUCTURED k cache (low-rank)")
    print("  (more realistic: real LLM k vectors are low-rank / clustered)")
    print("─" * 68)
    print(f"  {'noise_scale':>12}  {'adj_overlap':>12}  {'union_blks':>11}  {'bw_save%':>9}")
    print("  " + "-" * 50)

    for noise in args.noise_levels:
        adjs, unions, bws = [], [], []
        for seed in range(args.n_seeds):
            torch.manual_seed(seed)
            model, k_cache, v_cache, cmp_k, cmp_v, cu_k, cu_q, pos = build_model_and_caches(
                args.seqlen, N, args.kv_heads, args.head_dim, args.hidden_size,
                args.kernel_size, args.kernel_stride, args.block_size, args.topk,
                args.init_blocks, args.local_blocks, args.window_size,
                DTYPE, device, structured_k=True,
            )
            x = make_correlated_queries(N, args.hidden_size, noise, device, DTYPE)
            with torch.no_grad():
                model(x, cu_q, cu_k, k_cache, v_cache, cmp_k, cmp_v, position_ids=pos)
            adj, u, bw = overlap_stats(_captured["topk_idx"])
            adjs.append(adj); unions.append(u); bws.append(bw)

        print(f"  {noise:>12.2f}  {sum(adjs)/len(adjs):>11.1%}  "
              f"{sum(unions)/len(unions):>11.1f}  {sum(bws)/len(bws):>8.1f}%")

    # ------------------------------------------------------------------
    # Experiment 3: Load real .pt dump (if available)
    # ------------------------------------------------------------------
    if not args.skip_dump:
        print()
        print("─" * 68)
        print("EXP 3: Real data from .pt dump files")
        print("─" * 68)
        dump_files = list(Path(".").glob("tmp_topk_sample*.pt"))
        if not dump_files:
            print("  No .pt files found. To generate:")
            print("  Add this snippet after topk_idx is computed in fsa_decode.py:")
            print()
            print("    import os, torch")
            print("    _dump_path = os.environ.get('FSA_DUMP_TOPK_PATH', '')")
            print("    if _dump_path and not hasattr(_dump_once, '_done'):")
            print("        torch.save({'topk_idx': topk_idx.cpu(),")
            print("                    'seqlen': k_cache.shape[0],")
            print("                    'N': q.shape[0]}, _dump_path)")
            print("        _dump_once._done = True")
            print()
            print("  Then run: FSA_DUMP_TOPK_PATH=topk_real.pt python your_inference.py")
        else:
            for f in sorted(dump_files):
                d = torch.load(f, map_location="cpu", weights_only=False)
                if "topk_idx" not in d:
                    print(f"  {f.name}: no topk_idx key, skip")
                    continue
                ti = d["topk_idx"]
                if ti.dim() != 3:
                    print(f"  {f.name}: topk_idx shape {ti.shape} unexpected, skip")
                    continue
                H, Nq, K = ti.shape
                adj, u, bw = overlap_stats(ti)
                print(f"  {f.name}: shape={tuple(ti.shape)}  "
                      f"adj_overlap={adj:.1%}  union={u:.1f}  bw_save={bw:.1f}%")

    # ------------------------------------------------------------------
    # Summary: what overlap to expect in production
    # ------------------------------------------------------------------
    print()
    print("─" * 68)
    print("SUMMARY: How to interpret results")
    print("─" * 68)
    fixed = args.init_blocks + args.local_blocks
    print(f"  Theoretical floor  (random independent): "
          f"{(fixed/eff_topk + (eff_topk-fixed)/eff_topk * eff_topk/max_blocks):.1%}")
    print(f"  Theoretical ceiling (all identical q):   100.0%")
    print(f"  Fixed contribution  (init+local always): {fixed/eff_topk:.1%}")
    print()
    print("  For your optimization decision:")
    print(f"  • If overlap > 50%: block dedup saves > (N-1)/N × 50% = "
          f"{(N-1)/N*0.5*100:.0f}% of TopK bandwidth → WORTH IT")
    print(f"  • If overlap > 70%: saves > {(N-1)/N*0.7*100:.0f}% of TopK bandwidth → DEFINITELY WORTH IT")
    print()
    print("  Real LLM data will likely show overlap between EXP1(noise=0.3) and EXP2(noise=0.3).")
    print("  Use EXP3 (real dump) to confirm before implementing.")


if __name__ == "__main__":
    main()

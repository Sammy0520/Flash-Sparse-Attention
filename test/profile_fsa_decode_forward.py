#!/usr/bin/env python3
"""
Profile / benchmark one full FlashSparseAttentionDecode.forward (整步 decode).

1) CUDA Event: average wall time per forward (ms).
2) torch.profiler: top CUDA ops by self_cuda_time_total (看 compressed / sparse / flash 谁大).

Usage:
  cd Flash-Sparse-Attention
  PYTHONPATH=. python test/profile_fsa_decode_forward.py --q-len 8 --seqlen 131072
  PYTHONPATH=. python test/profile_fsa_decode_forward.py --q-len 8 --seqlen 131072 --attention-mask

Under nsys (avoid CUPTI conflict with torch.profiler — only CUDA Event + .nsys-rep timeline):
  nsys profile --trace=cuda,nvtx,osrt --output=bench_logs/nsys.nsys-rep -- \\
    python test/profile_fsa_decode_forward.py --q-len 8 --seqlen 131072 --no-torch-profiler
  PYTHONPATH=. python test/profile_fsa_decode_forward.py --q-len 8 --seqlen 4096 --export-chrome trace.json

Requires CUDA. Matches tensor layout pattern from test/test_FSA_decode.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode
from nsa_ref.module import RopeConfig
from nsa_ref.ops import linear_compress


def _branch_ms_per_forward(prof, key: str, profile_runs: int, ev_avg) -> float:
    """Inclusive CUDA ms per forward for a record_function key (PyTorch version–agnostic)."""
    runs = max(1, profile_runs)
    if ev_avg is not None:
        cnt = max(1, int(getattr(ev_avg, "count", runs)))
        for attr in ("device_time_total", "cuda_time_total"):
            v = getattr(ev_avg, attr, None)
            if v is not None and float(v) > 0:
                # v: microseconds summed over cnt invocations in profile window
                return (float(v) / 1000.0) / cnt
        sc = float(getattr(ev_avg, "self_cuda_time_total", 0) or 0)
        if sc > 0:
            return (sc / 1000.0) / cnt

    total_us = 0.0
    n_ev = 0
    for e in prof.events():
        if getattr(e, "name", None) != key:
            continue
        u = getattr(e, "device_time_total", None)
        if u is None:
            u = getattr(e, "cuda_time_total", None)
        if u is None:
            u = getattr(e, "self_cuda_time_total", None)
        if u is None:
            u = getattr(e, "cuda_time", 0)
        total_us += float(u or 0)
        n_ev += 1
    if total_us <= 0:
        return 0.0
    # One record_function event per forward per key
    denom = runs if n_ev == runs else max(1, n_ev)
    return total_us / 1000.0 / denom


def build_module_and_inputs(
    *,
    seqlen: int,
    q_len: int,
    hidden_size: int,
    kv_heads: int,
    q_heads: int,
    head_dim: int,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    dtype: torch.dtype,
    device: str,
    attention_mask: bool = False,
    inplace_kv_cmp_buffers: bool = False,
    return_state: bool = False,
):
    """
    When ``return_state`` is True, the return value includes an extra ``dict`` of tensors
    (``kv_*`` / ``cmp_s*`` only if ``inplace_kv_cmp_buffers``).
    """
    sparse_attn = (
        FlashSparseAttentionDecode(
            hidden_size=hidden_size,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
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
        .to(device)
        .to(dtype)
    )

    seqlens = torch.tensor([seqlen], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(seqlens, dim=0)],
        dim=0,
    ).to(torch.int32)

    x = torch.randn(q_len, hidden_size, device=device, dtype=dtype)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)

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

    am = None
    if attention_mask:
        # Exercises FSA_decode.mask_compressed (Python loop); values all 1 → no masking effect.
        am = torch.ones(q_len, seqlen, device=device, dtype=torch.float32)

    def run_forward_legacy():
        return sparse_attn(
            x,
            cu_seqlens_q,
            cu_seqlens_k,
            k_cache,
            v_cache,
            cmp_k_cache,
            cmp_v_cache,
            attention_mask=am,
        )

    if not inplace_kv_cmp_buffers:
        if return_state:
            st = {
                "k_cache": k_cache,
                "v_cache": v_cache,
                "cmp_k_cache": cmp_k_cache,
                "cmp_v_cache": cmp_v_cache,
                "cu_seqlens_k": cu_seqlens_k,
                "x": x,
                "cu_seqlens_q": cu_seqlens_q,
                "q_len": q_len,
            }
            return sparse_attn, run_forward_legacy, st
        return sparse_attn, run_forward_legacy

    # A1/A2: preallocated KV + compressed buffers (append via copy_, no torch.cat)
    past = seqlen - 1
    raw_total = past + q_len
    kv_k = torch.empty(raw_total, kv_heads, head_dim, device=device, dtype=dtype)
    kv_v = torch.empty(raw_total, kv_heads, head_dim, device=device, dtype=dtype)
    kv_k[:past].copy_(k_cache)
    kv_v[:past].copy_(v_cache)

    cmp_past = cmp_k_cache.shape[0]
    cmp_cap = cmp_past + q_len + kernel_size
    cmp_sk = torch.empty(cmp_cap, kv_heads, head_dim, device=device, dtype=dtype)
    cmp_sv = torch.empty(cmp_cap, kv_heads, head_dim, device=device, dtype=dtype)
    cmp_sk[:cmp_past].copy_(cmp_k_cache)
    cmp_sv[:cmp_past].copy_(cmp_v_cache)

    def run_forward_inplace():
        return sparse_attn(
            x,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            None,
            attention_mask=am,
            kv_storage_k=kv_k,
            kv_storage_v=kv_v,
            kv_past_len=past,
            cmp_storage_k=cmp_sk,
            cmp_storage_v=cmp_sv,
            cmp_past_len=cmp_past,
        )

    if return_state:
        st = {
            "k_cache": k_cache,
            "v_cache": v_cache,
            "cmp_k_cache": cmp_k_cache,
            "cmp_v_cache": cmp_v_cache,
            "cu_seqlens_k": cu_seqlens_k,
            "x": x,
            "cu_seqlens_q": cu_seqlens_q,
            "q_len": q_len,
            "kv_k": kv_k,
            "kv_v": kv_v,
            "cmp_sk": cmp_sk,
            "cmp_sv": cmp_sv,
            "kv_past_init": past,
            "cmp_past_init": cmp_past,
        }
        return sparse_attn, run_forward_inplace, run_forward_legacy, st
    return sparse_attn, run_forward_inplace, run_forward_legacy


def main():
    p = argparse.ArgumentParser(description="Profile full FSA decode forward")
    p.add_argument("--seqlen", type=int, default=4096, help="Varlen batch total K side length (same convention as test_FSA_decode)")
    p.add_argument("--q-len", type=int, default=8, help="Decode query tokens this step (speculative K)")
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--q-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--kernel-size", type=int, default=32)
    p.add_argument("--kernel-stride", type=int, default=16)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--bench-iters", type=int, default=20, help="CUDA event averaging iters")
    p.add_argument("--profile-runs", type=int, default=3, help="Forwards inside one profiler window (keep small)")
    p.add_argument("--topn", type=int, default=40, help="Rows in profiler table")
    p.add_argument("--export-chrome", type=str, default="", help="Optional Chrome trace path (json)")
    p.add_argument(
        "--attention-mask",
        action="store_true",
        help="Pass a dense (q_len, seqlen) all-ones mask to profile mask→compressed_mask Python loop",
    )
    p.add_argument(
        "--no-torch-profiler",
        action="store_true",
        help="Skip torch.profiler block; use when wrapping with nsys to avoid CUPTI multiple subscribers",
    )
    p.add_argument(
        "--inplace-kv-cmp-buffers",
        action="store_true",
        help="Use preallocated kv_storage_* / cmp_storage_* (A1/A2: avoid O(seq) cat on this step)",
    )
    p.add_argument(
        "--verify-inplace",
        action="store_true",
        help="Run one legacy vs inplace forward and assert outputs close (requires --inplace-kv-cmp-buffers path built separately)",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = "cuda"

    _built = build_module_and_inputs(
        seqlen=args.seqlen,
        q_len=args.q_len,
        hidden_size=args.hidden_size,
        kv_heads=args.kv_heads,
        q_heads=args.q_heads,
        head_dim=args.head_dim,
        kernel_size=args.kernel_size,
        kernel_stride=args.kernel_stride,
        block_size=args.block_size,
        topk=args.topk,
        dtype=dtype,
        device=device,
        attention_mask=args.attention_mask,
        inplace_kv_cmp_buffers=args.inplace_kv_cmp_buffers,
    )
    if args.inplace_kv_cmp_buffers:
        sparse_attn, run_forward, run_forward_legacy = _built
    else:
        sparse_attn, run_forward = _built

    if args.verify_inplace:
        if not args.inplace_kv_cmp_buffers:
            raise SystemExit("--verify-inplace requires --inplace-kv-cmp-buffers")
        torch.cuda.synchronize()
        y0 = run_forward_legacy()
        y1 = run_forward()
        torch.cuda.synchronize()
        torch.testing.assert_close(y0, y1, rtol=2e-2, atol=2e-2, check_stride=False)
        print("[verify-inplace] legacy vs preallocated buffers: OK (assert_close)")

    print("=" * 72)
    print("FSA decode — full forward profile")
    print(
        f"  seqlen={args.seqlen} q_len={args.q_len} hidden={args.hidden_size} "
        f"Hq={args.q_heads} Hkv={args.kv_heads} D={args.head_dim} dtype={args.dtype}"
    )
    if args.attention_mask:
        print("  attention_mask: dense all-ones (q_len × seqlen) — profiles mask_compressed path")
    if args.inplace_kv_cmp_buffers:
        print("  kv/cmp: preallocated buffers (A1/A2 — no torch.cat for kv_cat / cmp append)")
    print("=" * 72)

    for _ in range(args.warmup):
        run_forward()
    torch.cuda.synchronize()

    # --- CUDA events: full forward ---
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(args.bench_iters):
        out = run_forward()
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / args.bench_iters
    print(f"\n[Full forward] avg {ms:.4f} ms/iter  (CUDA events, {args.bench_iters} iters)")
    print(f"  output shape: {tuple(out.shape)}")

    if args.no_torch_profiler:
        print(
            "\n[Note] 已跳过 torch.profiler（--no-torch-profiler）。"
            "与 nsys 同时开 CUDA trace 时 CUPTI 只能有一方订阅，否则 GPU 计时会丢。"
            "\n  数值 baseline：以本段 [Full forward] CUDA Event 为准；"
            "时间线：用本次生成的 .nsys-rep 在 Nsight Systems 里看。"
        )
        print("=" * 72)
        return

    # --- torch.profiler ---
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        with_modules=True,
    ) as prof:
        for _ in range(args.profile_runs):
            run_forward()
        torch.cuda.synchronize()

    if args.export_chrome:
        prof.export_chrome_trace(args.export_chrome)
        print(f"\n[Chrome trace] written to {args.export_chrome}")

    print(f"\n[Profiler] top {args.topn} by CUDA self time (sum over {args.profile_runs} forwards):")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=args.topn,
        )
    )

    # --- Full forward: sequential record_function scopes (fsa_decode.forward) ---
    forward_stage_labels = (
        ("FSA_decode.compress_layout", "compress_layout (compressed_seqlens + compressed_cu_seqlens)"),
        ("FSA_decode.qkv_proj", "qkv_proj (Linear q/k/v + view)"),
        ("FSA_decode.kv_cat", "kv_cat (concat k/v cache with new tokens)"),
        ("FSA_decode.linear_compress", "linear_compress (_linear_compress_decode K/V + cat cmp_*)"),
        ("FSA_decode.mask_compressed", "mask_compressed (mask → compressed_mask; only if attention_mask)"),
        ("FSA_decode.rope_q_cmpk", "rope_q_cmpk (RoPE on q + compressed_k)"),
        ("FSA_decode.compressed_path", "compressed_path (_compressed_attention_decode)"),
        ("FSA_decode.rope_full_k", "rope_full_k (RoPE on full k)"),
        ("FSA_decode.topk_sparse_path", "topk_sparse_path"),
        ("FSA_decode.sliding_flash_path", "sliding_flash_path (flash_attn varlen)"),
        ("FSA_decode.gate_and_fuse", "gate_and_fuse (gate + 3-way weighted sum)"),
        ("FSA_decode.proj_o", "proj_o (rearrange + Linear out)"),
    )
    by_key = {e.key: e for e in prof.key_averages()}
    runs = max(1, args.profile_runs)
    print("\n" + "=" * 72)
    print(f"[Forward — sequential scopes] CUDA inclusive ms per forward, ~{runs} forwards in window:")
    stage_ms = []
    for key, desc in forward_stage_labels:
        ev = by_key.get(key)
        if ev is None and not any(
            getattr(e, "name", None) == key or getattr(e, "key", None) == key for e in prof.events()
        ):
            if key == "FSA_decode.mask_compressed" and not args.attention_mask:
                print(f"  {desc}: (skipped — no attention_mask; use --attention-mask to profile)")
            else:
                print(f"  {desc}: (not found)")
            stage_ms.append(0.0)
            continue
        ms_per_fwd = _branch_ms_per_forward(prof, key, runs, ev)
        stage_ms.append(ms_per_fwd)
        print(f"  {desc}: {ms_per_fwd:.4f} ms")
    s_stages = sum(stage_ms)
    if s_stages > 0:
        print("  --- share of sum(all stages above) ---")
        for (_key, desc), m in zip(forward_stage_labels, stage_ms):
            if m > 0:
                print(f"    {desc.split()[0]} … {100.0 * m / s_stages:.1f}%")
        print(
            f"  stages sum: {s_stages:.4f} ms  |  [Full forward] CUDA event: {ms:.4f} ms  |  "
            f"差值多为未打标片段（如 .item() 同步、assert 路径）与 profiler 口径差异"
        )

    # --- Inside FSA_decode.compressed_path (_compressed_attention_decode) ---
    sub_labels = (
        ("FSA_decode.compressed.fwd_attn", "  fwd: Q×compressed_KV attention (forward_kernel_decode)"),
        ("FSA_decode.compressed.score", "  score: per-token score from q,k,lse (score_kernel_decode)"),
        ("FSA_decode.compressed.transform_score", "  transform: token score → block score (_transform_score_kernel + setup)"),
        ("FSA_decode.compressed.block_topk", "  block_topk: aten::topk + int32 cast"),
    )
    by_key_sub = {e.key: e for e in prof.key_averages()}
    sub_ms = []
    for key, _desc in sub_labels:
        evs = by_key_sub.get(key)
        if evs is None and not any(getattr(e, "name", None) == key for e in prof.events()):
            sub_ms.append(0.0)
        else:
            sub_ms.append(_branch_ms_per_forward(prof, key, runs, evs))
    if sum(sub_ms) > 0:
        print("\n[Compressed path breakdown] inclusive ms per forward (nested under compressed_path):")
        for (_key, desc), m in zip(sub_labels, sub_ms):
            print(f"{desc}: {m:.4f} ms")
        cs = sum(sub_ms)
        print(f"  (四段之和 {cs:.4f} ms，应与上行 compressed_path 接近)")

    print("=" * 72)

    print("\nTip: 上表为顺序 scope 的 inclusive CUDA；详表 topN 按 self_cuda。")
    print("     Chrome: --export-chrome trace.json → chrome://tracing")
    print("=" * 72)


if __name__ == "__main__":
    main()

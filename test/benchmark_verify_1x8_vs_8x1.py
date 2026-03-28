#!/usr/bin/env python3
"""
Compare speculative-style verification cost:

  (A) One forward with q_len=8  — verify 8 tokens in one shot
  (B) Eight forwards with q_len=1 — verify one token per step, same module weights

For each mode, report legacy vs inplace (A1/A2 buffers) time and speedup.

Timed region (defaults match **paper-friendly** apples-to-apples on ``FSA_decode.forward``):

  • **1×8**: one ``forward`` (q_len=8), same as ``profile_fsa_decode_forward``.

  • **8×1** (default): sum of CUDA times for **eight** ``forward`` calls only; KV/cmp cache
    updates for the next step run **outside** the timer (legacy still does ``cat``/decode
    between steps, but that is not counted — same scope as timing 1×8).

  Optional ``--e2e-caller-append``: 8×1 timer includes naive caller ``torch.cat`` on KV/cmp
  plus ``_linear_compress_decode`` each step. That inflates legacy 8×1 cost and makes
  inplace speedup look larger than 1×8; **avoid for comparing batch vs serial decode**.

Usage (repo root):
  python test/benchmark_verify_1x8_vs_8x1.py --seqlen 131072 --warmup 5 --iters 10
  python test/benchmark_verify_1x8_vs_8x1.py --seqlen 131072 --e2e-caller-append   # old behavior
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

from fsa_preview.ops import _linear_compress_decode

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_prof_path = Path(__file__).resolve().parent / "profile_fsa_decode_forward.py"
_spec = importlib.util.spec_from_file_location("_profile_fsa_decode_forward", _prof_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_prof_path}")
_prof_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prof_mod)
build_module_and_inputs = _prof_mod.build_module_and_inputs


def num_new_compressed_tokens(
    prev_total_len: int,
    num_new: int,
    kernel_size: int,
    kernel_stride: int,
) -> int:
    """How many compressed rows ``_linear_compress_decode`` appends for this append (Python mirror)."""
    if num_new < 1:
        return 0
    buffer_len = min(kernel_size - 1, prev_total_len)
    buffer_start_pos = prev_total_len - buffer_len
    new_total_len = prev_total_len + num_new

    prev_max_output_idx = (
        math.floor((prev_total_len - kernel_size) / kernel_stride)
        if prev_total_len >= kernel_size
        else -1
    )
    new_max_output_idx = (
        math.floor((new_total_len - kernel_size) / kernel_stride)
        if new_total_len >= kernel_size
        else -1
    )

    if new_max_output_idx <= prev_max_output_idx:
        return 0

    cnt = 0
    all_tokens_len = buffer_len + num_new
    for out_idx in range(prev_max_output_idx + 1, new_max_output_idx + 1):
        window_start_abs = out_idx * kernel_stride
        window_end_abs = window_start_abs + kernel_size
        window_start_rel = window_start_abs - buffer_start_pos
        window_end_rel = window_end_abs - buffer_start_pos
        if window_start_rel >= 0 and window_end_rel <= all_tokens_len:
            cnt += 1
    return cnt


def bench_mean_ms(body, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        body()
    torch.cuda.synchronize()
    total_ms = 0.0
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        body()
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / float(iters)


def verify_count_matches_decode(device: str, dtype: torch.dtype, kv_heads: int, head_dim: int) -> None:
    """Sanity: num_new_compressed_tokens matches _linear_compress_decode row count."""
    torch.manual_seed(123)
    ks, st = 32, 16
    compress_key = torch.randn(kv_heads, head_dim * ks, head_dim, device=device, dtype=dtype)
    intra = torch.randn(kv_heads, ks, head_dim, device=device, dtype=dtype)
    for past in (0, 1, 15, 16, 17, 100, 1000, 131071):
        kn = torch.randn(1, kv_heads, head_dim, device=device, dtype=dtype)
        bs = min(ks - 1, past)
        prefix_k = torch.randn(past, kv_heads, head_dim, device=device, dtype=dtype) if past > 0 else None
        ibk = prefix_k[-bs:] if prefix_k is not None and bs > 0 else None
        dk = _linear_compress_decode(kn, compress_key, ks, st, intra, past, ibk)
        n = num_new_compressed_tokens(past, 1, ks, st)
        if dk is None:
            assert n == 0, (past, n)
        else:
            assert dk.shape[0] == n, (past, dk.shape[0], n)
    print("[verify] num_new_compressed_tokens matches _linear_compress_decode: OK")


def _legacy_append_one_step(
    m,
    *,
    k_c: torch.Tensor,
    v_c: torch.Tensor,
    cmp_k: torch.Tensor,
    cmp_v: torch.Tensor,
    kn: torch.Tensor,
    vn: torch.Tensor,
    past: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ks, st = m.kernel_size, m.kernel_stride
    bs = min(ks - 1, past)
    ibk = k_c[-bs:] if bs > 0 else None
    ibv = v_c[-bs:] if bs > 0 else None
    dk = _linear_compress_decode(kn, m.compress_key, ks, st, m.intra_block_pe, past, ibk)
    dv = _linear_compress_decode(vn, m.compress_value, ks, st, None, past, ibv)
    k_c = torch.cat([k_c, kn], dim=0)
    v_c = torch.cat([v_c, vn], dim=0)
    if dk is not None:
        cmp_k = torch.cat([cmp_k, dk], dim=0)
        cmp_v = torch.cat([cmp_v, dv], dim=0)
    return k_c, v_c, cmp_k, cmp_v


def sum_ms_eight_forwards_legacy_forward_only(
    m,
    *,
    xs: torch.Tensor,
    kns: list[torch.Tensor],
    vns: list[torch.Tensor],
    k0: torch.Tensor,
    v0: torch.Tensor,
    cmp_k0: torch.Tensor,
    cmp_v0: torch.Tensor,
    cu_q1: torch.Tensor,
) -> float:
    k_c = k0.clone()
    v_c = v0.clone()
    cmp_k = cmp_k0.clone()
    cmp_v = cmp_v0.clone()
    acc = 0.0
    for i in range(8):
        past = k_c.shape[0]
        cu_k = torch.tensor([0, past + 1], dtype=torch.int32, device=k_c.device)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        m(xs[i], cu_q1, cu_k, k_c, v_c, cmp_k, cmp_v)
        e1.record()
        torch.cuda.synchronize()
        acc += e0.elapsed_time(e1)
        k_c, v_c, cmp_k, cmp_v = _legacy_append_one_step(
            m, k_c=k_c, v_c=v_c, cmp_k=cmp_k, cmp_v=cmp_v, kn=kns[i], vn=vns[i], past=past
        )
    return acc


def sum_ms_eight_forwards_inplace_forward_only(
    m,
    *,
    xs: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    cmp_k0: torch.Tensor,
    cmp_v0: torch.Tensor,
    kv_k: torch.Tensor,
    kv_v: torch.Tensor,
    cmp_sk: torch.Tensor,
    cmp_sv: torch.Tensor,
    prefix: int,
    cmp_p0: int,
    cu_q1: torch.Tensor,
) -> float:
    kv_k[:prefix].copy_(k0)
    kv_v[:prefix].copy_(v0)
    cmp_sk[:cmp_p0].copy_(cmp_k0)
    cmp_sv[:cmp_p0].copy_(cmp_v0)
    kv_past = prefix
    cmp_past = cmp_p0
    ks, st = m.kernel_size, m.kernel_stride
    acc = 0.0
    for i in range(8):
        cu_k = torch.tensor([0, kv_past + 1], dtype=torch.int32, device=xs.device)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        m(
            xs[i],
            cu_q1,
            cu_k,
            None,
            None,
            None,
            None,
            kv_storage_k=kv_k,
            kv_storage_v=kv_v,
            kv_past_len=kv_past,
            cmp_storage_k=cmp_sk,
            cmp_storage_v=cmp_sv,
            cmp_past_len=cmp_past,
        )
        e1.record()
        torch.cuda.synchronize()
        acc += e0.elapsed_time(e1)
        cmp_past += num_new_compressed_tokens(kv_past, 1, ks, st)
        kv_past += 1
    return acc


def bench_mean_ms_8x1_forward_only(
    *,
    legacy: bool,
    m,
    xs: torch.Tensor,
    kns: list[torch.Tensor],
    vns: list[torch.Tensor],
    k0: torch.Tensor,
    v0: torch.Tensor,
    cmp_k0: torch.Tensor,
    cmp_v0: torch.Tensor,
    kv_k: torch.Tensor,
    kv_v: torch.Tensor,
    cmp_sk: torch.Tensor,
    cmp_sv: torch.Tensor,
    prefix: int,
    cmp_p0: int,
    cu_q1: torch.Tensor,
    warmup: int,
    iters: int,
) -> float:
    def one_trial() -> float:
        if legacy:
            return sum_ms_eight_forwards_legacy_forward_only(
                m,
                xs=xs,
                kns=kns,
                vns=vns,
                k0=k0,
                v0=v0,
                cmp_k0=cmp_k0,
                cmp_v0=cmp_v0,
                cu_q1=cu_q1,
            )
        return sum_ms_eight_forwards_inplace_forward_only(
            m,
            xs=xs,
            k0=k0,
            v0=v0,
            cmp_k0=cmp_k0,
            cmp_v0=cmp_v0,
            kv_k=kv_k,
            kv_v=kv_v,
            cmp_sk=cmp_sk,
            cmp_sv=cmp_sv,
            prefix=prefix,
            cmp_p0=cmp_p0,
            cu_q1=cu_q1,
        )

    for _ in range(warmup):
        one_trial()
    torch.cuda.synchronize()
    total = 0.0
    for _ in range(iters):
        total += one_trial()
    return total / float(iters)


def run_8x1_legacy_body(
    m,
    *,
    xs: torch.Tensor,
    kns: list[torch.Tensor],
    vns: list[torch.Tensor],
    k0: torch.Tensor,
    v0: torch.Tensor,
    cmp_k0: torch.Tensor,
    cmp_v0: torch.Tensor,
    cu_q1: torch.Tensor,
) -> None:
    k_c = k0.clone()
    v_c = v0.clone()
    cmp_k = cmp_k0.clone()
    cmp_v = cmp_v0.clone()
    ks, st = m.kernel_size, m.kernel_stride
    for i in range(8):
        past = k_c.shape[0]
        cu_k = torch.tensor([0, past + 1], dtype=torch.int32, device=k_c.device)
        m(xs[i], cu_q1, cu_k, k_c, v_c, cmp_k, cmp_v)
        kn, vn = kns[i], vns[i]
        bs = min(ks - 1, past)
        ibk = k_c[-bs:] if bs > 0 else None
        ibv = v_c[-bs:] if bs > 0 else None
        dk = _linear_compress_decode(kn, m.compress_key, ks, st, m.intra_block_pe, past, ibk)
        dv = _linear_compress_decode(vn, m.compress_value, ks, st, None, past, ibv)
        k_c = torch.cat([k_c, kn], dim=0)
        v_c = torch.cat([v_c, vn], dim=0)
        if dk is not None:
            cmp_k = torch.cat([cmp_k, dk], dim=0)
            cmp_v = torch.cat([cmp_v, dv], dim=0)


def run_8x1_inplace_body(
    m,
    *,
    xs: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    cmp_k0: torch.Tensor,
    cmp_v0: torch.Tensor,
    kv_k: torch.Tensor,
    kv_v: torch.Tensor,
    cmp_sk: torch.Tensor,
    cmp_sv: torch.Tensor,
    prefix: int,
    cmp_p0: int,
    cu_q1: torch.Tensor,
) -> None:
    kv_k[:prefix].copy_(k0)
    kv_v[:prefix].copy_(v0)
    cmp_sk[:cmp_p0].copy_(cmp_k0)
    cmp_sv[:cmp_p0].copy_(cmp_v0)
    kv_past = prefix
    cmp_past = cmp_p0
    ks, st = m.kernel_size, m.kernel_stride
    for i in range(8):
        cu_k = torch.tensor([0, kv_past + 1], dtype=torch.int32, device=xs.device)
        m(
            xs[i],
            cu_q1,
            cu_k,
            None,
            None,
            None,
            None,
            kv_storage_k=kv_k,
            kv_storage_v=kv_v,
            kv_past_len=kv_past,
            cmp_storage_k=cmp_sk,
            cmp_storage_v=cmp_sv,
            cmp_past_len=cmp_past,
        )
        d = num_new_compressed_tokens(kv_past, 1, ks, st)
        kv_past += 1
        cmp_past += d


def assert_final_cmp_len_matches(
    m,
    *,
    xs: torch.Tensor,
    kns: list[torch.Tensor],
    vns: list[torch.Tensor],
    k0: torch.Tensor,
    v0: torch.Tensor,
    cmp_k0: torch.Tensor,
    cmp_v0: torch.Tensor,
    kv_k: torch.Tensor,
    kv_v: torch.Tensor,
    cmp_sk: torch.Tensor,
    cmp_sv: torch.Tensor,
    prefix: int,
    cmp_p0: int,
    cu_q1: torch.Tensor,
) -> None:
    k_c = k0.clone()
    v_c = v0.clone()
    cmp_k = cmp_k0.clone()
    cmp_v = cmp_v0.clone()
    ks, st = m.kernel_size, m.kernel_stride
    for i in range(8):
        past = k_c.shape[0]
        cu_k = torch.tensor([0, past + 1], dtype=torch.int32, device=k_c.device)
        m(xs[i], cu_q1, cu_k, k_c, v_c, cmp_k, cmp_v)
        kn, vn = kns[i], vns[i]
        bs = min(ks - 1, past)
        ibk = k_c[-bs:] if bs > 0 else None
        ibv = v_c[-bs:] if bs > 0 else None
        dk = _linear_compress_decode(kn, m.compress_key, ks, st, m.intra_block_pe, past, ibk)
        dv = _linear_compress_decode(vn, m.compress_value, ks, st, None, past, ibv)
        k_c = torch.cat([k_c, kn], dim=0)
        v_c = torch.cat([v_c, vn], dim=0)
        if dk is not None:
            cmp_k = torch.cat([cmp_k, dk], dim=0)
            cmp_v = torch.cat([cmp_v, dv], dim=0)

    kv_k[:prefix].copy_(k0)
    kv_v[:prefix].copy_(v0)
    cmp_sk[:cmp_p0].copy_(cmp_k0)
    cmp_sv[:cmp_p0].copy_(cmp_v0)
    kv_past = prefix
    cmp_past = cmp_p0
    for i in range(8):
        cu_k = torch.tensor([0, kv_past + 1], dtype=torch.int32, device=xs.device)
        m(
            xs[i],
            cu_q1,
            cu_k,
            None,
            None,
            None,
            None,
            kv_storage_k=kv_k,
            kv_storage_v=kv_v,
            kv_past_len=kv_past,
            cmp_storage_k=cmp_sk,
            cmp_storage_v=cmp_sv,
            cmp_past_len=cmp_past,
        )
        cmp_past += num_new_compressed_tokens(kv_past, 1, ks, st)
        kv_past += 1

    assert cmp_k.shape[0] == cmp_past, (cmp_k.shape[0], cmp_past)


def main() -> None:
    p = argparse.ArgumentParser(
        description="1×8 vs 8×1 verify: legacy vs inplace speedup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--q-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--kernel-size", type=int, default=32)
    p.add_argument("--kernel-stride", type=int, default=16)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--verify-cmp-len",
        action="store_true",
        help="Assert legacy vs inplace 8×1 compressed length counter matches full legacy cat path",
    )
    p.add_argument(
        "--verify-decode-count",
        action="store_true",
        help="Run micro-tests: num_new_compressed_tokens vs _linear_compress_decode",
    )
    p.add_argument(
        "--e2e-caller-append",
        action="store_true",
        help=(
            "8×1: time forward + naive caller KV/cmp append each step (legacy cat + decode). "
            "Default is forward-only for fair comparison with 1×8."
        ),
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = "cuda"

    if args.verify_decode_count:
        verify_count_matches_decode(device, dtype, args.kv_heads, args.head_dim)
        return

    built = build_module_and_inputs(
        seqlen=args.seqlen,
        q_len=8,
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
        attention_mask=False,
        inplace_kv_cmp_buffers=True,
        return_state=True,
    )
    m, run_inplace_1x8, run_legacy_1x8, st = built

    prefix = st["kv_past_init"]
    cmp_p0 = st["cmp_past_init"]
    k0, v0 = st["k_cache"], st["v_cache"]
    cmp_k0, cmp_v0 = st["cmp_k_cache"], st["cmp_v_cache"]
    kv_k, kv_v = st["kv_k"], st["kv_v"]
    cmp_sk, cmp_sv = st["cmp_sk"], st["cmp_sv"]

    g = torch.Generator(device=device)
    g.manual_seed(args.seed + 2026)
    xs = torch.randn(8, 1, args.hidden_size, device=device, dtype=dtype, generator=g)
    cu_q1 = torch.tensor([0, 1], dtype=torch.int32, device=device)

    with torch.no_grad():
        kns = [m.proj_k(xs[i]).view(1, args.kv_heads, args.head_dim) for i in range(8)]
        vns = [m.proj_v(xs[i]).view(1, args.kv_heads, args.head_dim) for i in range(8)]

    if args.verify_cmp_len:
        assert_final_cmp_len_matches(
            m,
            xs=xs,
            kns=kns,
            vns=vns,
            k0=k0,
            v0=v0,
            cmp_k0=cmp_k0,
            cmp_v0=cmp_v0,
            kv_k=kv_k,
            kv_v=kv_v,
            cmp_sk=cmp_sk,
            cmp_sv=cmp_sv,
            prefix=prefix,
            cmp_p0=cmp_p0,
            cu_q1=cu_q1,
        )
        print("[verify-cmp-len] legacy final cmp rows == inplace cmp_past: OK")

    ms_1x8_l = bench_mean_ms(run_legacy_1x8, warmup=args.warmup, iters=args.iters)
    ms_1x8_i = bench_mean_ms(run_inplace_1x8, warmup=args.warmup, iters=args.iters)

    if args.e2e_caller_append:
        ms_8x1_l = bench_mean_ms(
            lambda: run_8x1_legacy_body(
                m,
                xs=xs,
                kns=kns,
                vns=vns,
                k0=k0,
                v0=v0,
                cmp_k0=cmp_k0,
                cmp_v0=cmp_v0,
                cu_q1=cu_q1,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        ms_8x1_i = bench_mean_ms(
            lambda: run_8x1_inplace_body(
                m,
                xs=xs,
                k0=k0,
                v0=v0,
                cmp_k0=cmp_k0,
                cmp_v0=cmp_v0,
                kv_k=kv_k,
                kv_v=kv_v,
                cmp_sk=cmp_sk,
                cmp_sv=cmp_sv,
                prefix=prefix,
                cmp_p0=cmp_p0,
                cu_q1=cu_q1,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        mode_line = "8×1 timing: e2e (forward + naive caller cat/decode each step)"
    else:
        ms_8x1_l = bench_mean_ms_8x1_forward_only(
            legacy=True,
            m=m,
            xs=xs,
            kns=kns,
            vns=vns,
            k0=k0,
            v0=v0,
            cmp_k0=cmp_k0,
            cmp_v0=cmp_v0,
            kv_k=kv_k,
            kv_v=kv_v,
            cmp_sk=cmp_sk,
            cmp_sv=cmp_sv,
            prefix=prefix,
            cmp_p0=cmp_p0,
            cu_q1=cu_q1,
            warmup=args.warmup,
            iters=args.iters,
        )
        ms_8x1_i = bench_mean_ms_8x1_forward_only(
            legacy=False,
            m=m,
            xs=xs,
            kns=kns,
            vns=vns,
            k0=k0,
            v0=v0,
            cmp_k0=cmp_k0,
            cmp_v0=cmp_v0,
            kv_k=kv_k,
            kv_v=kv_v,
            cmp_sk=cmp_sk,
            cmp_sv=cmp_sv,
            prefix=prefix,
            cmp_p0=cmp_p0,
            cu_q1=cu_q1,
            warmup=args.warmup,
            iters=args.iters,
        )
        mode_line = "8×1 timing: forward-only (sum of 8× forward; caller append untimed)"

    s1 = ms_1x8_l / ms_1x8_i if ms_1x8_i > 0 else float("inf")
    s8 = ms_8x1_l / ms_8x1_i if ms_8x1_i > 0 else float("inf")

    print(f"seqlen={args.seqlen} dtype={args.dtype} warmup={args.warmup} iters={args.iters} seed={args.seed}")
    print(mode_line)
    print("")
    print("=== 1×8 (one forward, q_len=8) — time per verification batch ===")
    print(f"  legacy:  {ms_1x8_l:.4f} ms")
    print(f"  inplace: {ms_1x8_i:.4f} ms")
    print(f"  speedup (legacy / inplace): {s1:.3f}x")
    print("")
    print("=== 8×1 (eight forwards, q_len=1) — total time for 8 tokens ===")
    print(f"  legacy:  {ms_8x1_l:.4f} ms")
    print(f"  inplace: {ms_8x1_i:.4f} ms")
    print(f"  speedup (legacy / inplace): {s8:.3f}x")
    print("")
    print("=== Paper-oriented: inplace speedup shape (not batching advantage) ===")
    print(f"  speedup(1×8) / speedup(8×1) = {s1 / s8:.3f}")
    print("    (<1 is common: serial had more *repeated* cat/buffer work to eliminate.)")
    print(f"  Throughput (8 tokens, inplace): 1×8 {8.0 / ms_1x8_i:.2f} tok/ms, 8×1 {8.0 / ms_8x1_i:.2f} tok/ms")
    print("")
    print("=== Cross (algorithmic): serial / batch cost for the same 8 tokens ===")
    r_l = ms_8x1_l / ms_1x8_l
    r_i = ms_8x1_i / ms_1x8_i
    print(f"  legacy  8×1 / 1×8: {r_l:.3f}x  (how much slower serial is vs one-shot)")
    print(f"  inplace 8×1 / 1×8: {r_i:.3f}x")
    print("")
    print("=== Why the batch/serial *ratio* can drop after inplace ===")
    print(
        "  Inplace mainly removes O(prefix) concat/copy-style work inside forward. "
        "That work appears once per forward — so 8×1 pays it eight times, 1×8 pays once. "
        "After inplace, much of that repeated cost is gone from the serial path, so serial "
        "catches up *in relative terms*; one-shot was already closer to the compute floor."
    )
    print(
        f"  Batching still wins on latency (inplace): {ms_8x1_i:.2f} ms (8×1) vs "
        f"{ms_1x8_i:.2f} ms (1×8) for the same 8 tokens — ~{r_i:.2f}× faster one-shot."
    )


if __name__ == "__main__":
    main()

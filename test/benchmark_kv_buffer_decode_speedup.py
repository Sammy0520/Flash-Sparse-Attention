#!/usr/bin/env python3
"""
Cumulative decode-step benchmark: same weights, same shapes each step.

Measures total GPU time for N identical FSA decode forwards (fixed past length
per step, as in profile_fsa_decode_forward). This answers "whole decode"
in the sense of: if each autoregressive step has the same context size
(e.g. speculative decode with fixed window / repeated profiling setup),
the end-to-end speedup is roughly N × (t_legacy − t_inplace) in saved time,
or T_legacy / T_inplace as a ratio.

For growing-prefix multi-step with correct cmp state updates, callers must
advance kv_past_len / cmp_past_len and refresh buffers between steps; this
script intentionally matches the single-step profile convention first.

Usage (from repo root; copy the whole line, do not type literal "..."):
  python test/benchmark_kv_buffer_decode_speedup.py --seqlen 131072 --q-len 8 --steps 50 --warmup 10
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same directory as this script — avoids `import test.*` (test/ is not a package;
# `python test/benchmark_...py` puts test/ on sys.path first and breaks that import).
_prof_path = Path(__file__).resolve().parent / "profile_fsa_decode_forward.py"
_spec = importlib.util.spec_from_file_location("_profile_fsa_decode_forward", _prof_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_prof_path}")
_prof_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prof_mod)
build_module_and_inputs = _prof_mod.build_module_and_inputs


def bench_loop(run, *, warmup: int, steps: int) -> float:
    """Return total elapsed seconds (CUDA events) for `steps` forwards after warmup."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Legacy vs inplace KV/cmp buffers: cumulative decode steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python test/benchmark_kv_buffer_decode_speedup.py "
            "--seqlen 131072 --q-len 8 --steps 50 --warmup 10\n\n"
            "Do not pass a literal ... (three dots) as arguments; it is not a flag."
        ),
    )
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--q-len", type=int, default=8)
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
    p.add_argument("--steps", type=int, default=50, help="Number of decode forwards per mode (cumulative timing)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = "cuda"

    sparse_attn, run_inplace, run_legacy = build_module_and_inputs(
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
        attention_mask=False,
        inplace_kv_cmp_buffers=True,
    )

    t_legacy = bench_loop(run_legacy, warmup=args.warmup, steps=args.steps)
    torch.cuda.synchronize()
    t_inplace = bench_loop(run_inplace, warmup=args.warmup, steps=args.steps)

    ms_per_legacy = (t_legacy / args.steps) * 1000.0
    ms_per_inplace = (t_inplace / args.steps) * 1000.0
    speedup = t_legacy / t_inplace if t_inplace > 0 else float("inf")

    print(
        f"seqlen={args.seqlen} q_len={args.q_len} steps={args.steps} "
        f"dtype={args.dtype} warmup={args.warmup}"
    )
    print(f"legacy:   total {t_legacy*1000:.3f} ms  ({ms_per_legacy:.4f} ms/step)")
    print(f"inplace:  total {t_inplace*1000:.3f} ms  ({ms_per_inplace:.4f} ms/step)")
    print(f"speedup (legacy / inplace): {speedup:.3f}x")
    print(f"time saved over {args.steps} steps: {(t_legacy - t_inplace)*1000:.3f} ms")


if __name__ == "__main__":
    main()

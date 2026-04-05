#!/usr/bin/env python3
"""
把 SimpleBooks-92 的 train.txt 读取 + tokenize 后存成单个 .pt 文件（1D int64），
供 distill_nsa.py --local-data 使用。

用法：
  PYTHONUNBUFFERED=1 python train/preprocess_simplebooks.py
  PYTHONUNBUFFERED=1 python train/preprocess_simplebooks.py --target-tokens 50000000
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from transformers import AutoTokenizer

LLAMA_SNAPSHOT = "/root/autodl-tmp/models/Llama-3.1-8B-Instruct"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=LLAMA_SNAPSHOT)
    p.add_argument("--input", default="/root/simplebooks/simplebooks-92-raw/train.txt")
    p.add_argument("--out", default=None,
                   help="输出路径，默认根据实际 token 数自动命名")
    p.add_argument("--target-tokens", type=int, default=100_000_000,
                   help="目标 token 数（默认 100M）")
    p.add_argument("--log-every-lines", type=int, default=10000)
    args = p.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("=== preprocess_simplebooks ===")
    log(f"Loading tokenizer from {args.model_path} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    log(f"Tokenizer OK in {time.time()-t0:.1f}s")

    log(f"Reading {args.input} ...")
    if not os.path.isfile(args.input):
        log(f"ERROR: file not found: {args.input}")
        return

    all_ids: list[int] = []
    log(f"Tokenizing, target={args.target_tokens/1e6:.0f}M tokens ...")
    t0 = time.time()
    last_log = time.time()
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ids = tok.encode(line, add_special_tokens=False)
            all_ids.extend(ids)
            if i > 0 and i % args.log_every_lines == 0:
                now = time.time()
                dt = now - last_log
                last_log = now
                rate = (args.log_every_lines / dt) if dt > 0 else 0
                log(
                    f"  lines={i:,} | tokens={len(all_ids)/1e6:.2f}M | "
                    f"~{rate:.0f} lines/s | elapsed={now-t0:.0f}s"
                )
            if len(all_ids) >= args.target_tokens:
                break

    out_ids = all_ids[: args.target_tokens]
    n_m = len(out_ids) // 1_000_000
    out_path = args.out or f"/root/autodl-tmp/simplebooks_{n_m}M.pt"
    log(f"Done tokenizing: {len(out_ids)/1e6:.2f}M tokens, saving to {out_path} ...")
    tensor = torch.tensor(out_ids, dtype=torch.long)
    torch.save(tensor, out_path)
    log(f"Saved. File size ~{os.path.getsize(out_path) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

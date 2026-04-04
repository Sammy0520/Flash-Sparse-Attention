#!/usr/bin/env python3
"""
把 TinyStories 流式下载 + tokenize 后存成单个 .pt 文件（1D int64），
供 distill_nsa.py --local-data 使用。

用法：
  PYTHONUNBUFFERED=1 python train/download_tinystories_tokens.py
  PYTHONUNBUFFERED=1 python train/download_tinystories_tokens.py --target-tokens 100000000

TinyStories 数据集特点：
  - 简单英文短故事（幼儿级别），每条 ~200 tokens
  - 总量 ~2.1M 条, ~440M tokens
  - 适合快速验证蒸馏流程是否正确
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.pop("HF_ENDPOINT", None)

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

LLAMA_SNAPSHOT = "/root/autodl-tmp/models/Llama-3.1-8B-Instruct"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=LLAMA_SNAPSHOT)
    p.add_argument("--out", default="/root/autodl-tmp/tinystories_128M.pt")
    p.add_argument("--target-tokens", type=int, default=128_000_000,
                   help="目标 token 数（默认 100M，TinyStories 全量约 440M）")
    p.add_argument("--log-every-samples", type=int, default=5000)
    args = p.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("=== download_tinystories_tokens ===")
    log(f"proxy http_proxy={os.environ.get('http_proxy', '(unset)')}")
    log(f"Loading tokenizer from {args.model_path} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    log(f"Tokenizer OK in {time.time()-t0:.1f}s")

    log("Loading TinyStories (streaming) ...")
    t0 = time.time()
    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    log(f"load_dataset returned in {time.time()-t0:.1f}s")

    all_ids: list[int] = []
    log(f"Streaming + tokenizing, target={args.target_tokens/1e6:.0f}M tokens ...")
    t0 = time.time()
    last_log = time.time()
    for i, sample in enumerate(ds):
        text = sample.get("text", "") or ""
        ids = tok.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if i > 0 and i % args.log_every_samples == 0:
            now = time.time()
            dt = now - last_log
            last_log = now
            rate = (args.log_every_samples / dt) if dt > 0 else 0
            log(
                f"  samples={i:,} | tokens={len(all_ids)/1e6:.2f}M | "
                f"~{rate:.0f} samples/s | elapsed={now-t0:.0f}s"
            )
        if len(all_ids) >= args.target_tokens:
            break

    out_ids = all_ids[: args.target_tokens]
    log(f"Done tokenizing: {len(out_ids)/1e6:.2f}M tokens, saving to {args.out} ...")
    tensor = torch.tensor(out_ids, dtype=torch.long)
    torch.save(tensor, args.out)
    log(f"Saved. File size ~{os.path.getsize(args.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

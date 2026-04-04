#!/usr/bin/env python3
"""
把 PG19（Project Gutenberg 19世纪书籍，deepmind/pg19）流式下载 + tokenize 后存成单个 .pt 文件，
供 distill_nsa.py --local-data 使用。

PG19 特点：
  - 维多利亚时代英文书籍，每本 ~100K tokens（长篇文章），总量 ~1.5B tokens
  - 适合长上下文（seqlen=4096+）训练，句法复杂度高于 TinyStories，低于 C4
  - train split ~28,000 本书

用法：
  export https_proxy=127.0.0.1:7897
  PYTHONUNBUFFERED=1 python train/download_pg19_tokens.py
  PYTHONUNBUFFERED=1 python train/download_pg19_tokens.py --target-tokens 50000000
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
    p.add_argument("--out", default="/root/autodl-tmp/pg19_100M.pt")
    p.add_argument("--target-tokens", type=int, default=100_000_000,
                   help="目标 token 数（默认 50M，PG19 train 全量约 1.5B）")
    p.add_argument("--split", default="train",
                   help="数据集分片：train / validation / test")
    p.add_argument("--log-every-books", type=int, default=50)
    args = p.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("=== download_pg19_tokens ===")
    log(f"proxy https_proxy={os.environ.get('https_proxy', '(unset)')}")
    log(f"Loading tokenizer from {args.model_path} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    log(f"Tokenizer OK in {time.time()-t0:.1f}s")

    log(f"Loading PG19 split={args.split!r} (streaming) ...")
    t0 = time.time()
    ds = load_dataset(
        "emozilla/pg19",
        split=args.split,
        streaming=True,
    )
    log(f"load_dataset returned in {time.time()-t0:.1f}s")

    all_ids: list[int] = []
    log(f"Streaming + tokenizing, target={args.target_tokens/1e6:.0f}M tokens ...")
    t0 = time.time()
    last_log = time.time()
    book_count = 0
    for i, sample in enumerate(ds):
        # emozilla/pg19 的文本字段为 text
        text = sample.get("text", "") or ""
        if not text.strip():
            continue
        ids = tok.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        book_count += 1

        if book_count % args.log_every_books == 0:
            now = time.time()
            dt = now - last_log
            last_log = now
            rate = args.log_every_books / dt if dt > 0 else 0
            title = sample.get("short_book_title", "?")[:40]
            log(
                f"  books={book_count:,} | tokens={len(all_ids)/1e6:.2f}M | "
                f"~{rate:.1f} books/s | elapsed={now-t0:.0f}s | last: {title!r}"
            )
        if len(all_ids) >= args.target_tokens:
            break

    out_ids = all_ids[: args.target_tokens]
    log(f"Done: {book_count} books, {len(out_ids)/1e6:.2f}M tokens → {args.out}")
    tensor = torch.tensor(out_ids, dtype=torch.long)
    torch.save(tensor, args.out)
    log(f"Saved. File size ~{os.path.getsize(args.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
把 C4 流式下载 + tokenize 后存成单个 .pt 文件（1D int64），供 distill_nsa.py --local-data 使用。

用法（建议 unbuffered）：
  PYTHONUNBUFFERED=1 python train/download_c4_tokens.py --target-tokens 500000000

需要 http_proxy 等环境变量已正确设置（与训练时一致）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 与 distill_nsa 一致：避免错误 HF_ENDPOINT
os.environ.pop("HF_ENDPOINT", None)

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

LLAMA_SNAPSHOT = (
    "/data1/models/Llama-3.1-8B-Instruct/snapshots"
    "/0e9e39f249a16976918f6564b8830bc894c89659"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=LLAMA_SNAPSHOT)
    p.add_argument("--out", default="/data1/zzy/c4_500M.pt")
    p.add_argument("--target-tokens", type=int, default=500_000_000)
    p.add_argument("--log-every-samples", type=int, default=500)
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--config", default="en")
    args = p.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("=== download_c4_tokens ===")
    log(f"proxy http_proxy={os.environ.get('http_proxy', '(unset)')}")
    log(f"Loading tokenizer from {args.model_path} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    log(f"Tokenizer OK in {time.time()-t0:.1f}s")

    log("Calling load_dataset (streaming, may take 1–3 min for Hub metadata) ...")
    t0 = time.time()
    ds = load_dataset(
        args.dataset,
        args.config,
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

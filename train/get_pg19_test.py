#!/usr/bin/env python3
"""
从 PG19 测试集中随机获取一段连续 2048 seqlen 的文本。
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# 默认模型路径，根据你的环境调整
LLAMA_SNAPSHOT = "/root/autodl-tmp/models/Llama-3.1-8B-Instruct"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=LLAMA_SNAPSHOT)
    p.add_argument("--seqlen", type=int, default=2400, help="连续 token 长度")
    p.add_argument("--split", default="test", help="强制使用测试集")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"Loading tokenizer from {args.model_path} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path)

    log(f"Loading PG19 split={args.split!r} (streaming) ...")
    # 使用 streaming 模式加载
    ds = load_dataset(
        "emozilla/pg19",
        split=args.split,
        streaming=True,
    )

    # 1. 随机打乱流式数据 (buffer_size 设为 50，因为测试集总共也就约 100 本书)
    # 2. 或者简单地跳过随机数量的书籍
    shuffled_ds = ds.shuffle(seed=args.seed, buffer_size=50)

    log(f"Searching for a segment of {args.seqlen} tokens...")
    
    target_text = ""
    for sample in shuffled_ds:
        text = sample.get("text", "") or ""
        if not text.strip():
            continue
        
        # 将整本书转换成 token IDs
        # 使用 add_special_tokens=False 保证获取的是纯文本内容
        ids = tok.encode(text, add_special_tokens=False)
        
        if len(ids) < args.seqlen:
            # 如果这本书太短，换下一本
            continue
        
        # 随机选择一个起始点
        max_start = len(ids) - args.seqlen
        start_idx = random.randint(0, max_start)
        selected_ids = ids[start_idx : start_idx + args.seqlen]
        
        # 解码回文本
        target_text = tok.decode(selected_ids, skip_special_tokens=True)
        
        log(f"Found segment in book: {sample.get('short_book_title', 'Unknown')}")
        log(f"Book total tokens: {len(ids)}")
        log(f"Extracted slice range: [{start_idx} : {start_idx + args.seqlen}]")
        break

    if target_text:
        print("\n" + "="*40 + " EXTRACTED TEXT " + "="*40)
        print(target_text)
        print("="*96 + "\n")
        
        # 可选：保存到文件
        with open("pg19_sample.txt", "w", encoding="utf-8") as f:
            f.write(target_text)
    else:
        log("Error: Could not find a book long enough.")

if __name__ == "__main__":
    main()
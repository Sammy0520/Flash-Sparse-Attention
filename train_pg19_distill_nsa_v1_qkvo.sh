#!/usr/bin/env bash
# PG19：旧版 distill_nsa.py，proxy 前向（不加 --real-fsa），seqlen=2048，训练 q/k/v/o + NSA
# 数据：/data1/sn/pg19_100M.pt
#
# 使用前请设置：
#   MODEL — Llama-3.1-8B 本地路径（可选，下面有默认）
#   INIT  — 上游 ckpt 目录（含 ckpt.pt），例如 simplebooks/final；不设置则从随机初始化 NSA（仅 meanpool init）

set -euo pipefail

MODEL="${MODEL:-/data1/models/Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
OUT="${OUT:-/data1/sn/nsa_ckpt_pg19_distill_nsa_v1_qkvo}"
INIT="${INIT:-}"

EXTRA=()
if [[ -n "$INIT" ]]; then
  EXTRA+=(--init-ckpt "$INIT")
fi

python train/distill_nsa.py \
  --model "$MODEL" \
  --seqlen 2048 \
  --batch 1 \
  --grad-accum 1 \
  --lr 1e-5 \
  --max-tokens 2114514 \
  --save-steps 500 \
  --save-dir "$OUT" \
  "${EXTRA[@]}" \
  --layers all \
  --topk 16 \
  --block-size 64 \
  --local-data /data1/sn/pg19_100M.pt \
  --train-qkvo

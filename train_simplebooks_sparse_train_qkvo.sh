#!/usr/bin/env bash
# SimpleBooks seqlen=1024，显式稀疏（示例：topk=16, block_size=16 → 每序列 64 块中取 16 块，约 25%）
# 训练 q/k/v/o（--train-qkvo）。数据：/data1/sn/simplebooks_94M.pt
#
# 可选：从单文件 ckpt 热启动（与从头训二选一）
#   export INIT=/data1/sn/nsa_ckpt_simplebooks_seqlen1024.pt
#
# 生成时须匹配 topk/block_size，且不要加 --force-llama-proj：
#   python test/test_nsa_generate.py --nsa-ckpt .../final --seqlen 1024 \
#     --topk 16 --block-size 16 ...

set -euo pipefail

MODEL="${MODEL:-/data1/models/Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
OUT="${OUT:-/data1/sn/nsa_ckpt_simplebooks_sparse_qkvo}"
INIT="${INIT:-}"

EXTRA=()
if [[ -n "$INIT" ]]; then
  EXTRA+=(--init-ckpt "$INIT")
fi

python train/distill_nsa_v2.py \
  --model "$MODEL" \
  --seqlen 1024 \
  --batch 1 \
  --grad-accum 1 \
  --lr 1.0e-5 \
  --warmup-steps 4 \
  --min-lr-ratio 0.08 \
  --logit-kl-weight 0.5 \
  --logit-kl-last-k 8 \
  --logit-temperature 1.6 \
  --max-tokens 2000000 \
  --save-steps 500 \
  --save-dir "$OUT" \
  "${EXTRA[@]}" \
  --layers all \
  --topk 16 \
  --block-size 16 \
  --local-data /data1/sn/simplebooks_94M.pt \
  --train-qkvo

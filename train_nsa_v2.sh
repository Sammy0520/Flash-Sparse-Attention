#!/bin/bash
# NSA Distillation Training v2
# ================================
# 修复：
#   1. 数据量 5M → 100M tokens（20x），充分利用 200M 数据集
#   2. logit-kl-last-k 8 → 128（覆盖 12.5% 位置，而非 0.8%）
#   3. logit-kl-weight 0.5 → 1.5（KL 是端到端损失，应占主导）
#   4. 有效 batch 2048 → 8192 tokens/step（更稳定梯度）
#   5. lr 适配更大 batch → 4e-5
#
# GPU: A800 80GB
# 预估显存: LLaMA bf16 ~16GB + NSA 参数+Adam ~2GB + 激活 ~12GB ≈ 30GB
# 预估总 steps: 12207, 预估耗时: ~8-12h（取决于 real-fsa 速度）

python train/distill_nsa_v2.py \
    --model /root/autodl-tmp/models/Llama-3.1-8B-Instruct \
    --seqlen 1024 \
    --batch 2 \
    --grad-accum 4 \
    --lr 4e-5 \
    --logit-kl-weight 1.5 \
    --logit-kl-last-k 128 \
    --logit-temperature 2.0 \
    --max-tokens 100000000 \
    --save-steps 500 \
    --save-dir /root/autodl-tmp/nsa_ckpt_v2 \
    --layers all \
    --topk 16 \
    --local-data /root/autodl-tmp/c4_200M.pt

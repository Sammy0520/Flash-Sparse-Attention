#!/bin/bash
# PG19 长上下文蒸馏训练（seqlen=4096）
# ======================================
# 目标：从 TinyStories 检查点继续，在 PG19 长篇书籍上适配长上下文压缩注意力
#
# 数据规模：50M tokens 下载，训练 10M tokens
# 训练配置：proxy 模式（flash_attn），seqlen=4096，batch=1，grad-accum=8
#   → 每 optimizer step = 1×4096×8 = 32,768 tokens
#   → 10M tokens ≈ 305 steps
#   → 预计耗时：~5h（proxy 模式，~550 tok/s）
#
# 初始化：从 TinyStories step_001024 权重热启动（compress_key/value/gate 已收敛）
#
# 参数说明：
#   --lr 1e-3        继续从已收敛权重训练，低于首次训练的 3.5e-3
#   --warmup-steps 20  从 0 缓慢爬升，避免已收敛权重被大梯度破坏
#   --min-lr-ratio 0.05  终点 lr = 5e-5，比 TinyStories 训练更彻底退火
#   --logit-kl-last-k 32  比 TinyStories 的 128 小，减少 seqlen=4096 下的 KL 开销
#   --save-steps 64   每 64 步保存一次，方便中途监控（~5个检查点）

python train/distill_nsa_v2.py \
    --model /root/autodl-tmp/models/Llama-3.1-8B-Instruct \
    --seqlen 2048 \
    --batch 1 \
    --grad-accum 1 \
    --lr 1.11451419e-5 \
    --warmup-steps 4 \
    --min-lr-ratio 0.081419198 \
    --logit-kl-weight 2.0 \
    --logit-kl-last-k 256 \
    --logit-temperature 2.0 \
    --max-tokens 2114514 \
    --save-steps 114514 \
    --save-dir /root/autodl-tmp/nsa_ckpt_pg19 \
    --init-ckpt /root/autodl-tmp/nsa_ckpt_pg19/final \
    --layers all \
    --topk 16 \
    --local-data /root/autodl-tmp/pg19_100M.pt

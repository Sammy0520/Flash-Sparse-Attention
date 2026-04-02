python train/distill_nsa.py   \
    --model /data1/models/Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --real-fsa   \
    --seqlen 2048   \
    --batch 1   \
    --grad-accum 4   \
    --lr 2.0919810114514e-5    \
    --logit-kl-weight 0.5   \
    --logit-kl-last-k 8   \
    --logit-temperature 2.0   \
    --max-tokens 11144514   \
    --save-steps 200   \
    --save-dir /data1/sn/nsa_ckpt_seqlen_2048   \
    --layers all   \
    --topk 64   \
    --local-data /data1/zzy/c4_500M.pt


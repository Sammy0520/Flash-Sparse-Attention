python train/distill_nsa.py   \
    --model /data1/models/Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --real-fsa   \
    --seqlen 512   \
    --batch 2   \
    --grad-accum 2   \
    --lr 2.0919810114e-5    \
    --logit-kl-weight 0.5   \
    --logit-kl-last-k 8   \
    --logit-temperature 2.0   \
    --max-tokens 10000000   \
    --save-steps 200   \
    --save-dir /data1/sn/nsa_ckpt   \
    --layers all   \
    --topk 64   \
    --local-data /data1/zzy/c4_500M.pt


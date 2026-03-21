"""
Capture real q, k_cache, v_cache tensors from LLaMA-3.1-8B decode steps.
These can then be used to test the TopK block overlap kernel for both
correctness and performance.

Usage:
  python test/capture_real_tensors.py           # saves to test/real_decode_tensors.pt
  python test/capture_real_tensors.py --layer 14 --n-tokens 8 --seqlen 32768
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    LLAMA_PATH = ("/data1/models/Llama-3.1-8B-Instruct/snapshots"
                  "/0e9e39f249a16976918f6564b8830bc894c89659")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=LLAMA_PATH)
    parser.add_argument("--seqlen",    type=int, default=32768)
    parser.add_argument("--n-tokens",  type=int, default=8)
    parser.add_argument("--layer",     type=int, default=14)
    parser.add_argument("--out",       default="test/real_decode_tensors.pt")
    args = parser.parse_args()

    device, dtype = "cuda", torch.bfloat16

    # ── load model ────────────────────────────────────────────────────
    print(f"Loading model ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device, trust_remote_code=True)
    model.eval()

    cfg          = model.config
    num_q_heads  = cfg.num_attention_heads   # 32
    num_kv_heads = cfg.num_key_value_heads   # 8
    head_dim     = cfg.head_dim if hasattr(cfg, "head_dim") \
                   else cfg.hidden_size // num_q_heads  # 128
    gqa_groups   = num_q_heads // num_kv_heads  # 4

    # ── long prompt ───────────────────────────────────────────────────
    para = ("Large language models have demonstrated strong capabilities across "
            "many tasks. Speculative decoding accelerates inference by verifying "
            "multiple draft tokens simultaneously. Sparse attention reduces the "
            "quadratic complexity of attention by selecting only relevant context. ")
    ids = tok.encode(para * (args.seqlen // len(tok.encode(para)) + 2),
                     return_tensors="pt")[:, :args.seqlen].to(device)
    print(f"Prompt: {ids.shape[1]} tokens")

    # ── hook: capture q, k, hidden at target layer ────────────────────
    captured_q      = []   # post q_proj, per decode step
    captured_k      = []   # post k_proj, per decode step
    captured_v      = []   # post v_proj, per decode step
    captured_hidden = []   # pre-proj hidden, per decode step
    k_cache_prefill = None
    v_cache_prefill = None
    _step = [0]

    attn = model.model.layers[args.layer].self_attn

    def _hook(module, args_in, kwargs_in, output):
        step = _step[0]
        hidden = args_in[0] if args_in else kwargs_in.get("hidden_states")
        if hidden is None:
            return
        with torch.no_grad():
            q = module.q_proj(hidden).view(1, -1, num_q_heads,  head_dim)
            k = module.k_proj(hidden).view(1, -1, num_kv_heads, head_dim)
            v = module.v_proj(hidden).view(1, -1, num_kv_heads, head_dim)
            if hasattr(module, 'q_norm') and module.q_norm is not None:
                q = module.q_norm(q)
            if hasattr(module, 'k_norm') and module.k_norm is not None:
                k = module.k_norm(k)

        nonlocal k_cache_prefill, v_cache_prefill
        if step == 0:
            # prefill: save full KV cache
            k_cache_prefill = k.squeeze(0).detach()  # [past_len, Hkv, D]
            v_cache_prefill = v.squeeze(0).detach()
        else:
            # decode: save q averaged over GQA groups → [Hkv, D]
            q_dec = q.squeeze(0).squeeze(0)  # [Hq, D]
            q_dec = q_dec.view(num_kv_heads, gqa_groups, head_dim).mean(dim=1)
            captured_q.append(q_dec.detach())
            captured_k.append(k.squeeze(0).squeeze(0).detach())  # new k: [Hkv, D]
            captured_v.append(v.squeeze(0).squeeze(0).detach())
            captured_hidden.append(hidden.squeeze(0).squeeze(0).detach())
        _step[0] += 1

    handle = attn.register_forward_hook(_hook, with_kwargs=True)

    # ── prefill ───────────────────────────────────────────────────────
    print("Prefill ...")
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past_kv    = out.past_key_values
    next_token = out.logits[:, -1:].argmax(-1)

    # ── N greedy decode steps ─────────────────────────────────────────
    print(f"Decoding {args.n_tokens} steps ...")
    cur = next_token
    for _ in range(args.n_tokens):
        with torch.no_grad():
            out = model(cur, past_key_values=past_kv, use_cache=True)
        past_kv    = out.past_key_values
        next_token = out.logits[:, -1:].argmax(-1)
        cur        = next_token
    handle.remove()

    N        = len(captured_q)
    past_len = k_cache_prefill.shape[0]

    # q_stacked: [N, Hkv, D]
    q_stacked = torch.stack(captured_q, dim=0)

    # Append new k/v from decode steps to cache
    new_k = torch.stack(captured_k, dim=0)  # [N, Hkv, D]
    new_v = torch.stack(captured_v, dim=0)
    k_full = torch.cat([k_cache_prefill, new_k], dim=0)  # [past_len+N, Hkv, D]
    v_full = torch.cat([v_cache_prefill, new_v], dim=0)

    # Compute oracle topk_idx (block selection via raw k mean)
    block_size = 64
    num_blocks = past_len // block_size
    k_blocks   = k_cache_prefill[:num_blocks * block_size].view(
        num_blocks, block_size, num_kv_heads, head_dim)
    k_mean     = k_blocks.mean(dim=1)  # [num_blocks, Hkv, D]
    scale      = head_dim ** -0.5
    scores     = torch.einsum("nhd,bhd->hnb",
                              q_stacked.float(), k_mean.float()) * scale  # [H, N, B]

    # init+local forced
    init_blocks, local_blocks = 1, 2
    topk_val = 16
    eff_topk = min(topk_val, num_blocks)
    oracle_idx = torch.zeros(num_kv_heads, N, eff_topk, dtype=torch.long, device=device)
    for h in range(num_kv_heads):
        for i in range(N):
            q_pos      = past_len + i
            local_start = max(0, q_pos // block_size - local_blocks)
            local_end   = min(num_blocks, q_pos // block_size + 1)
            forced = sorted(set(range(init_blocks)) |
                            set(range(local_start, local_end)))[:eff_topk]
            remaining = eff_topk - len(forced)
            sc = scores[h, i].clone()
            if remaining > 0:
                mask = torch.ones(num_blocks, dtype=torch.bool, device=device)
                for b in forced:
                    mask[b] = False
                sc[~mask] = -1e9
                _, top_rest = sc.topk(remaining)
                selected = sorted(forced + top_rest.tolist())
            else:
                selected = forced
            oracle_idx[h, i] = torch.tensor(selected[:eff_topk], device=device)

    oracle_idx = oracle_idx.to(torch.int32)

    # ── save ─────────────────────────────────────────────────────────
    save = {
        # inputs for TopK kernel
        "q":           q_stacked.cpu(),          # [N, Hkv, D]
        "k_cache":     k_cache_prefill.cpu(),    # [past_len, Hkv, D]
        "v_cache":     v_cache_prefill.cpu(),    # [past_len, Hkv, D]
        "k_full":      k_full.cpu(),             # [past_len+N, Hkv, D]
        "v_full":      v_full.cpu(),             # [past_len+N, Hkv, D]
        "topk_idx":    oracle_idx.cpu(),         # [Hkv, N, topk] int32

        # metadata
        "N":           N,
        "past_len":    past_len,
        "num_blocks":  num_blocks,
        "eff_topk":    eff_topk,
        "block_size":  block_size,
        "head_dim":    head_dim,
        "num_kv_heads": num_kv_heads,

        # hidden states (can be used as FSA input x)
        "hidden_states": torch.stack(captured_hidden, dim=0).cpu(),  # [N, H_size]
    }
    out_path = Path(args.out)
    torch.save(save, out_path)

    # print overlap stats
    ti = oracle_idx.cpu()
    H2, N2, K2 = ti.shape
    adjs = []
    for i in range(N2 - 1):
        per_h = [len(set(ti[h, i].tolist()) & set(ti[h, i+1].tolist())) / K2
                 for h in range(H2)]
        adjs.append(sum(per_h) / H2)
    unions = [len(set(ti[h].reshape(-1).tolist())) for h in range(H2)]

    print(f"\nSaved → {out_path}")
    print(f"  q:        {tuple(q_stacked.shape)}")
    print(f"  k_cache:  {tuple(k_cache_prefill.shape)}")
    print(f"  topk_idx: {tuple(oracle_idx.shape)}")
    print(f"  adj overlap:  {sum(adjs)/len(adjs):.1%}")
    print(f"  union blocks: {sum(unions)/H2:.1f} / {N2*eff_topk} slots")
    print(f"  bw_save:      {(1 - sum(unions)/H2/(N2*eff_topk))*100:.1f}%")


if __name__ == "__main__":
    main()

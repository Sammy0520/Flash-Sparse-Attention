"""
Real block overlap measurement using LLaMA-3.1-8B (or any HuggingFace model).

Strategy:
  1. Tokenize a long prompt → prefill → get real KV cache
  2. Run N greedy decode steps, capture real q and k at one attention layer
  3. Oracle block selection: divide k cache into blocks, compute per-block
     attention score directly (sum of max logits), select topk blocks.
     This simulates what a perfectly trained compression would achieve.
  4. Measure TopK block overlap across the N decode steps.

Why oracle instead of FSA compression?
  FSA's compress_key weights are randomly initialized — they produce uniform
  compressed k regardless of query content. The oracle bypasses this by using
  raw k vectors to score blocks directly, giving the true upper-bound overlap.

Usage:
  python test/real_topk_overlap.py
  python test/real_topk_overlap.py --seqlen 32768 --n-tokens 8 --topk 16
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM


# ── overlap stats ──────────────────────────────────────────────────────

def overlap_stats(topk_idx):
    """topk_idx: [H, N, K] on CPU"""
    H, N, K = topk_idx.shape
    if N >= 2:
        adjs = []
        for i in range(N - 1):
            per_h = [
                len(set(topk_idx[h, i].tolist()) & set(topk_idx[h, i+1].tolist())) / K
                for h in range(H)
            ]
            adjs.append(sum(per_h) / H)
        adj = sum(adjs) / len(adjs)
    else:
        adj = float("nan")

    unions, bws = [], []
    for h in range(H):
        u = set()
        for i in range(N):
            u.update(topk_idx[h, i].tolist())
        unions.append(len(u))
        bws.append((1 - len(u) / (N * K)) * 100)

    return adj, sum(unions) / H, sum(bws) / H


# ── oracle block selection ─────────────────────────────────────────────

def oracle_topk(q, k_cache, topk, block_size, init_blocks, local_blocks):
    """
    Oracle block selection using raw k vectors (no compression).
    
    q:       [N, H, D]
    k_cache: [past_len, H, D]
    returns: oracle_idx [H, N, topk_eff]  (block indices)
    """
    N, H, D = q.shape
    past_len = k_cache.shape[0]
    num_blocks = past_len // block_size
    usable_len = num_blocks * block_size

    # Block-level k: mean-pool each block
    k_blocks = k_cache[:usable_len].view(num_blocks, block_size, H, D)
    k_mean   = k_blocks.mean(dim=1)           # [num_blocks, H, D]

    scale  = D ** -0.5
    scores = torch.einsum("nhd,bhd->hnb",
                          q[:, :H].float(),
                          k_mean.float()) * scale    # [H, N, num_blocks]

    eff_topk = min(topk, num_blocks)
    oracle_idx = torch.zeros(H, N, eff_topk, dtype=torch.long, device=q.device)

    for h in range(H):
        for i in range(N):
            q_pos = past_len + i
            local_start = max(0, q_pos // block_size - local_blocks)
            local_end   = min(num_blocks, q_pos // block_size + 1)

            forced = set(range(init_blocks))
            forced.update(range(local_start, local_end))

            sc = scores[h, i].clone()
            forced_list = sorted(forced)[:eff_topk]
            remaining   = eff_topk - len(forced_list)

            if remaining > 0:
                mask = torch.ones(num_blocks, dtype=torch.bool, device=q.device)
                for b in forced_list:
                    mask[b] = False
                sc_masked = sc.clone()
                sc_masked[~mask] = -1e9
                _, top_rest = sc_masked.topk(remaining)
                selected = sorted(forced_list + top_rest.tolist())
            else:
                selected = forced_list

            oracle_idx[h, i] = torch.tensor(selected[:eff_topk], device=q.device)

    return oracle_idx


# ── main ───────────────────────────────────────────────────────────────

def main():
    LLAMA_PATH = ("/data1/models/Llama-3.1-8B-Instruct/snapshots"
                  "/0e9e39f249a16976918f6564b8830bc894c89659")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",         default=LLAMA_PATH)
    parser.add_argument("--seqlen",        type=int, default=32768,
                        help="Target prefill length")
    parser.add_argument("--n-tokens",      type=int, default=8,
                        help="Number of greedy decode steps = N draft tokens")
    parser.add_argument("--layer",         type=int, default=14,
                        help="Which layer to capture q/k from (0-indexed)")
    parser.add_argument("--topk",          type=int, default=16)
    parser.add_argument("--block-size",    type=int, default=64,
                        help="Raw token block size (NOT compressed)")
    parser.add_argument("--init-blocks",   type=int, default=1)
    parser.add_argument("--local-blocks",  type=int, default=2)
    parser.add_argument("--multi-layers",  action="store_true",
                        help="Measure overlap on all layers (slow)")
    args = parser.parse_args()

    device = "cuda"
    dtype  = torch.bfloat16

    # ── 1. Load model ──────────────────────────────────────────────────
    print(f"Loading {Path(args.model).name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    hf_model.eval()

    num_q_heads  = hf_model.config.num_attention_heads
    num_kv_heads = hf_model.config.num_key_value_heads
    head_dim     = (hf_model.config.head_dim
                    if hasattr(hf_model.config, 'head_dim')
                    else hf_model.config.hidden_size // num_q_heads)
    gqa_groups   = num_q_heads // num_kv_heads

    print(f"  num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, "
          f"head_dim={head_dim}, gqa_groups={gqa_groups}")

    # ── 2. Build long prompt ───────────────────────────────────────────
    paragraph = (
        "The study of artificial intelligence has evolved rapidly over the past decade, "
        "with large language models demonstrating remarkable capabilities in natural language "
        "understanding and generation. These models rely on transformer architectures with "
        "attention mechanisms that scale quadratically with sequence length. Efficient inference "
        "requires sparse attention methods that select only the most relevant context. "
        "Speculative decoding further accelerates generation by verifying multiple draft tokens "
        "in a single forward pass, reducing the number of model calls needed. "
    )
    tokens_per_para = len(tokenizer.encode(paragraph))
    repeats = math.ceil(args.seqlen / tokens_per_para) + 2
    input_ids = tokenizer.encode(paragraph * repeats, return_tensors="pt")
    input_ids = input_ids[:, :args.seqlen].to(device)
    actual_seqlen = input_ids.shape[1]
    print(f"Prompt length: {actual_seqlen} tokens")

    # ── 3. Hook: capture q and k at the target layer ───────────────────
    layers_to_probe = (list(range(hf_model.config.num_hidden_layers))
                       if args.multi_layers else [args.layer])

    captured_q_all = {l: [] for l in layers_to_probe}
    captured_k_all = {l: None for l in layers_to_probe}
    _step = {l: 0 for l in layers_to_probe}

    def make_hook(layer_idx):
        def _attn_hook(module, args_in, kwargs_in, output):
            step = _step[layer_idx]
            hidden = args_in[0] if args_in else kwargs_in.get("hidden_states")
            if hidden is None:
                return
            with torch.no_grad():
                q = module.q_proj(hidden)   # [1, seq, Hq*D]
                k = module.k_proj(hidden)   # [1, seq, Hkv*D]
                q = q.view(1, -1, num_q_heads,  head_dim)
                k = k.view(1, -1, num_kv_heads, head_dim)
                if hasattr(module, 'q_norm') and module.q_norm is not None:
                    q = module.q_norm(q)
                if hasattr(module, 'k_norm') and module.k_norm is not None:
                    k = module.k_norm(k)

            if step == 0:
                # Prefill: capture full k cache
                captured_k_all[layer_idx] = k.squeeze(0).detach()  # [past_len, Hkv, D]
            else:
                # Decode: capture q for this token, avg GQA groups → Hkv heads
                q_dec = q.squeeze(0).squeeze(0)                     # [Hq, D]
                q_dec = q_dec.view(num_kv_heads, gqa_groups, head_dim).mean(dim=1)  # [Hkv, D]
                captured_q_all[layer_idx].append(q_dec.detach())
            _step[layer_idx] += 1
        return _attn_hook

    hooks = []
    for l in layers_to_probe:
        h = hf_model.model.layers[l].self_attn.register_forward_hook(
            make_hook(l), with_kwargs=True
        )
        hooks.append(h)

    # ── 4. Prefill ─────────────────────────────────────────────────────
    print("Running prefill ...")
    with torch.no_grad():
        out = hf_model(input_ids, use_cache=True)
    past_kv    = out.past_key_values
    next_token = out.logits[:, -1:].argmax(-1)

    # ── 5. Greedy decode ───────────────────────────────────────────────
    print(f"Running {args.n_tokens} greedy decode steps ...")
    cur = next_token
    for _ in range(args.n_tokens):
        with torch.no_grad():
            out = hf_model(cur, past_key_values=past_kv, use_cache=True)
        past_kv    = out.past_key_values
        next_token = out.logits[:, -1:].argmax(-1)
        cur        = next_token

    for h in hooks:
        h.remove()

    # ── 6. Oracle block selection & overlap stats ──────────────────────
    print()
    print("=" * 68)
    print("ORACLE BLOCK OVERLAP  (LLaMA-3.1-8B, real decode)")
    print("=" * 68)

    results = {}
    for l in layers_to_probe:
        q_list = captured_q_all[l]
        k_full = captured_k_all[l]
        if k_full is None or len(q_list) < 2:
            continue

        N        = len(q_list)
        past_len = k_full.shape[0]
        num_blocks = past_len // args.block_size
        eff_topk   = min(args.topk, num_blocks)

        q_tensor = torch.stack(q_list, dim=0).to(device)   # [N, Hkv, D]
        k_tensor = k_full.to(device)

        with torch.no_grad():
            ti = oracle_topk(q_tensor, k_tensor,
                             args.topk, args.block_size,
                             args.init_blocks, args.local_blocks)

        adj, union_blks, bw_save = overlap_stats(ti.cpu())
        results[l] = dict(adj=adj, union=union_blks, bw_save=bw_save,
                          N=N, past_len=past_len, num_blocks=num_blocks,
                          eff_topk=eff_topk, ti=ti.cpu())

        if not args.multi_layers or l == args.layer:
            print(f"\n  Layer {l:2d}")
            print(f"  past_len={past_len}, num_blocks={num_blocks}, "
                  f"eff_topk={eff_topk}  (sparsity={eff_topk/num_blocks:.1%})")
            print(f"  N decode tokens: {N}")
            print()
            print(f"  Adjacent overlap:        {adj:.1%}")
            print(f"  Union blocks (avg/head): {union_blks:.1f}  "
                  f"(of {N*eff_topk} slots)")
            print(f"  Bandwidth saving:        {bw_save:.1f}%")
            fixed_pct = (args.init_blocks + args.local_blocks) / eff_topk
            print(f"  Fixed floor (init+local):{fixed_pct:.1%}")
            print(f"  Semantic contribution:   {adj - fixed_pct:.1%}")
            print()
            print(f"  Per-head adjacent overlap:")
            H_dim, _, K_dim = ti.shape
            for hh in range(H_dim):
                per_step = []
                for i in range(N - 1):
                    s = len(set(ti[hh, i].tolist()) & set(ti[hh, i+1].tolist())) / K_dim
                    per_step.append(s)
                avg_h = sum(per_step) / len(per_step) if per_step else float("nan")
                steps_str = " ".join(f"{x:.0%}" for x in per_step)
                print(f"    head {hh:2d}: {avg_h:.1%}  [{steps_str}]")

    if args.multi_layers and results:
        print()
        print("─" * 68)
        print("  Layer summary (adjacent overlap):")
        for l in sorted(results):
            r = results[l]
            print(f"  layer {l:2d}: adj={r['adj']:.1%}  "
                  f"bw_save={r['bw_save']:.1f}%  "
                  f"num_blocks={r['num_blocks']}  eff_topk={r['eff_topk']}")

    # Save
    if args.layer in results:
        save_path = Path(__file__).parent / "topk_oracle_llama.pt"
        torch.save(results[args.layer], save_path)
        print(f"\n  Saved to {save_path}")


if __name__ == "__main__":
    main()

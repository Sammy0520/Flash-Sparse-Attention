"""
NSA + Speculative Decoding 接通 Demo (Stage 1: 自验证，无需外部 draft 模型)

流程：
  1. LLaMA prefill 长 prompt → 拿到真实 KV cache
  2. LLaMA greedy decode N 步   → 这 N 个 token 的 hidden state 作为 "draft hidden"
  3. 用这 N 个 hidden → NSA 注意力层 (一次 forward)  → nsa_out
  4. 用这 N 个 hidden → LLaMA 原始注意力 (N 次 forward) → ref_out
  5. 比较误差 + 测速

用法：
  /data1/zzy/envs/flash_sparse/bin/python test/nsa_verify_demo.py   --seqlen 131072 --n-tokens 8 --layer 14 --topk 8
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from nsa_ref.module import RopeConfig
from nsa_ref.ops import linear_compress
from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode

LLAMA_PATH = ("/data1/models/Llama-3.1-8B-Instruct/snapshots"
              "/0e9e39f249a16976918f6564b8830bc894c89659")

# ─────────────────────────────────────────────────────────────────────────────
# NSA 注意力层封装（借用 LLaMA 的 q/k/v/o 权重）
# ─────────────────────────────────────────────────────────────────────────────

class LlamaNSALayer(nn.Module):
    """用 FlashSparseAttentionDecode 替换 LlamaAttention，q/k/v/o 权重从原始层复制。"""

    def __init__(self, llama_attn, cfg, topk=16, block_size=64,
                 kernel_size=32, kernel_stride=16, init_blocks=1,
                 local_blocks=2, window_size=512):
        super().__init__()
        num_q  = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_d = getattr(cfg, 'head_dim', cfg.hidden_size // num_q)

        rope_scaling = getattr(cfg, 'rope_scaling', None)
        rope_cfg = RopeConfig(
            max_position_embeddings=cfg.max_position_embeddings,
            head_dim=head_d,
            rope_theta=cfg.rope_theta,
            rope_scaling=rope_scaling,
        )

        self.fsa = FlashSparseAttentionDecode(
            hidden_size=cfg.hidden_size,
            num_q_heads=num_q,
            num_kv_heads=num_kv,
            head_dim=head_d,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            window_size=window_size,
            rope_config=rope_cfg,
        )

        # 复制 q/k/v/o 投影权重
        with torch.no_grad():
            self.fsa.proj_q.weight.copy_(llama_attn.q_proj.weight)
            self.fsa.proj_k.weight.copy_(llama_attn.k_proj.weight)
            self.fsa.proj_v.weight.copy_(llama_attn.v_proj.weight)
            self.fsa.proj_o.weight.copy_(llama_attn.o_proj.weight)

        self.kernel_size   = kernel_size
        self.kernel_stride = kernel_stride
        self.block_size    = block_size

    def forward(self, hidden, k_cache, v_cache, cmp_k, cmp_v,
                cu_seqlens_q, cu_seqlens_k, position_ids,
                use_dedup: bool = False):
        return self.fsa(hidden, cu_seqlens_q, cu_seqlens_k,
                        k_cache, v_cache, cmp_k, cmp_v,
                        attention_mask=None, position_ids=position_ids,
                        use_dedup=use_dedup)

    def build_compressed_cache(self, k_cache, v_cache, cu_seqlens):
        """从 raw KV cache 构建压缩 KV cache。"""
        cmp_k, _ = linear_compress(
            k_cache, self.fsa.compress_key, cu_seqlens,
            self.kernel_size, self.kernel_stride, self.fsa.intra_block_pe)
        cmp_v, _ = linear_compress(
            v_cache, self.fsa.compress_value, cu_seqlens,
            self.kernel_size, self.kernel_stride, None)
        return cmp_k, cmp_v


# ─────────────────────────────────────────────────────────────────────────────
# 工具：从 past_key_values 提取指定层的 KV cache
# ─────────────────────────────────────────────────────────────────────────────

def extract_kv(past_key_values, layer_idx):
    """
    HuggingFace format: past_key_values[layer] = (k, v)
      k shape: [batch, num_kv_heads, seq_len, head_dim]
    NSA format: [seq_len, num_kv_heads, head_dim]
    """
    k, v = past_key_values[layer_idx]
    k = k.squeeze(0).permute(1, 0, 2).contiguous()  # [seq, kv_heads, head_dim]
    v = v.squeeze(0).permute(1, 0, 2).contiguous()
    return k, v


# ─────────────────────────────────────────────────────────────────────────────
# 工具：benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=LLAMA_PATH)
    parser.add_argument("--seqlen",   type=int, default=8192,
                        help="Prefill 长度（prefill token 数）")
    parser.add_argument("--n-tokens", type=int, default=8,
                        help="Draft token 数 / verify batch size")
    parser.add_argument("--layer",    type=int, default=14,
                        help="替换哪一层注意力")
    parser.add_argument("--topk",     type=int, default=16)
    args = parser.parse_args()

    device, dtype = "cuda", torch.bfloat16
    N = args.n_tokens
    L = args.layer

    # ── Step 1: 加载 LLaMA ───────────────────────────────────────────
    print("Loading LLaMA ...")
    tok   = AutoTokenizer.from_pretrained(args.model)
    llama = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device)
    llama.eval()
    cfg = llama.config

    # ── Step 2: 构造 NSA 层 ──────────────────────────────────────────
    print(f"Building NSA layer {L} (topk={args.topk}) ...")
    nsa_layer = LlamaNSALayer(
        llama.model.layers[L].self_attn, cfg, topk=args.topk
    ).to(device, dtype)

    # ── Step 3: 构造 prompt，做 prefill ──────────────────────────────
    para = ("Speculative decoding accelerates inference by verifying multiple "
            "candidate tokens simultaneously. Native Sparse Attention reduces "
            "computation while maintaining model quality in long-context tasks. ")
    ids = tok.encode(para, return_tensors="pt").to(device)
    repeat = args.seqlen // ids.shape[1] + 2
    ids = ids.repeat(1, repeat)[:, :args.seqlen]
    print(f"Prefill: {ids.shape[1]} tokens ...")

    with torch.no_grad():
        # num_logits_to_keep=1: 只保留最后一个 token 的 logit，避免 131K × vocab OOM
        prefill_out = llama(ids, use_cache=True, num_logits_to_keep=1)
    past_kv    = prefill_out.past_key_values
    next_token = prefill_out.logits[:, -1:].argmax(-1)
    print("Prefill done.")

    # ── Step 4: 从 prefill KV cache 拿到第 L 层的 raw KV ────────────
    k_cache, v_cache = extract_kv(past_kv, L)
    past_len = k_cache.shape[0]
    print(f"Layer {L} KV cache: {past_len} tokens, "
          f"shape {k_cache.shape}")  # [past_len, kv_heads, head_dim]

    cu_k = torch.tensor([0, past_len], device=device, dtype=torch.int32)

    # ── Step 5: 构建压缩 KV cache ────────────────────────────────────
    print("Building compressed KV cache ...")
    cmp_k, cmp_v = nsa_layer.build_compressed_cache(k_cache, v_cache, cu_k)
    print(f"Compressed K: {cmp_k.shape}")

    # ── Step 6: Greedy decode N 步，捕获 layer L 的 hidden states ────
    # 用 hook 捕获"attention 层输入"，这就是从 x=residual+LN 来的 hidden
    draft_hiddens = []

    def capture_hook(module, args_in, kwargs_in, output):
        hidden = args_in[0] if args_in else kwargs_in.get("hidden_states")
        if hidden is not None:
            draft_hiddens.append(hidden.squeeze(0).squeeze(0).detach().clone())

    handle = llama.model.layers[L].self_attn.register_forward_hook(
        capture_hook, with_kwargs=True)

    print(f"Greedy decode {N} draft tokens ...")
    cur = next_token
    # past_kv 里已经有 prefill 的 KV，decode 时直接 append
    for step in range(N):
        with torch.no_grad():
            out = llama(cur, past_key_values=past_kv, use_cache=True)
        past_kv    = out.past_key_values
        next_token = out.logits[:, -1:].argmax(-1)
        cur        = next_token

    handle.remove()
    print(f"Captured {len(draft_hiddens)} hidden states.")

    # draft_hiddens[i]: [hidden_size]  →  stack → [N, hidden_size]
    hidden_batch = torch.stack(draft_hiddens, dim=0)  # [N, H]
    cu_q = torch.tensor([0, N], device=device, dtype=torch.int32)
    pos  = torch.arange(past_len, past_len + N, device=device, dtype=torch.long)

    # ── Step 7: NSA verify pass (N tokens once) ───────────────────────
    print(f"\n=== NSA verify: {N} tokens at once ===")
    with torch.no_grad():
        nsa_out = nsa_layer(hidden_batch, k_cache, v_cache, cmp_k, cmp_v,
                            cu_q, cu_k, position_ids=pos)
    print(f"  Output shape:  {nsa_out.shape}")
    print(f"  Output norm:   {nsa_out.norm(dim=-1).mean().item():.4f} (mean per token)")

    # ── Step 8: 参考输出（因果前缀法，与 test_toy_block_decode.py 一致）────
    # 正确做法：token i 的参考 = NSA([x_0,...,x_i], k_cache) 最后一个 token
    # 不能用 1-token×N，因为那样 token i 看不到 token 0..i-1 的新 key（少了因果依赖）
    print(f"\n=== Reference: NSA 因果前缀 (token i = NSA(x[0..i])[-1]) ===")
    ref_outs = []
    with torch.no_grad():
        for i in range(N):
            h_prefix = hidden_batch[:i+1]             # [i+1, H]
            cu_q_prefix = torch.tensor([0, i+1], device=device, dtype=torch.int32)
            pos_prefix  = pos[:i+1]
            o = nsa_layer(h_prefix, k_cache, v_cache, cmp_k, cmp_v,
                          cu_q_prefix, cu_k, position_ids=pos_prefix)
            ref_outs.append(o[-1])                    # 只取最后一个 token 的输出
    ref_out = torch.stack(ref_outs, dim=0)            # [N, H]

    # ── Step 9: 数值对比 ─────────────────────────────────────────────
    diff = (nsa_out - ref_out).abs()
    print(f"\n=== 数值对比 (N-token-once  vs  因果前缀参考) ===")
    print(f"  per-token max diff: {diff.amax(dim=-1).tolist()}")
    print(f"  global max diff:    {diff.max().item():.6f}")
    print(f"  global mean diff:   {diff.mean().item():.6f}")
    torch.testing.assert_close(nsa_out, ref_out, atol=5e-2, rtol=5e-2)
    print("  [OK] N-token-once 与 因果前缀参考 数值一致（NSA multi-token decode 正确）")

    # ── Step 10: Timing ──────────────────────────────────────────────
    import fsa_preview.ops.selected_attention_decode as _sad

    print(f"\n=== Timing (NSA 注意力层, 不含 LLaMA 其他层) ===")

    t_once = benchmark(
        lambda: nsa_layer(hidden_batch, k_cache, v_cache, cmp_k, cmp_v,
                          cu_q, cu_k, position_ids=pos))

    t_1xN = benchmark(
        lambda: [nsa_layer(hidden_batch[i:i+1], k_cache, v_cache, cmp_k, cmp_v,
                           torch.tensor([0, 1], device=device, dtype=torch.int32),
                           cu_k, position_ids=pos[i:i+1]) for i in range(N)])

    print(f"  {N}-token once (ref):  {t_once:.3f} ms")
    print(f"  1-token × {N} calls:  {t_1xN:.3f} ms")
    print(f"  Speedup (multi-tok):  {t_1xN / t_once:.2f}×")

    # dedup kernel timing（同伴完成 kernel 后取消注释并填入路径）
    # ----------------------------------------------------------------
    # from nsa_ref.ops.topk_sparse_attention_dedup import _topk_sparse_attention_fwd_dedup
    # _sad._topk_sparse_attention_fwd_dedup = _topk_sparse_attention_fwd_dedup
    #
    # t_dedup = benchmark(
    #     lambda: nsa_layer(hidden_batch, k_cache, v_cache, cmp_k, cmp_v,
    #                       cu_q, cu_k, position_ids=pos, use_dedup=True))
    # print(f"  {N}-token once (dedup): {t_dedup:.3f} ms")
    # print(f"  Speedup (dedup/ref):   {t_once / t_dedup:.2f}×")
    # print(f"  Speedup (dedup/1×N):   {t_1xN / t_dedup:.2f}×")
    # ----------------------------------------------------------------
    print(f"\n  [dedup timing: uncomment the block above after kernel is ready]")

    # ── Step 11: 摘要 ────────────────────────────────────────────────
    num_blocks = past_len // nsa_layer.block_size
    print(f"\n=== 配置摘要 ===")
    print(f"  past_len={past_len}, block_size={nsa_layer.block_size}")
    print(f"  num_blocks={num_blocks}, topk={args.topk}")
    print(f"  sparse ratio: {args.topk}/{num_blocks} = {args.topk/num_blocks:.1%}")
    print(f"\n  说明：")
    print(f"  · q/k/v/o 权重来自 LLaMA layer {L}（真实）")
    print(f"  · compress_key/value/intra_block_pe/gate 随机初始化（未训练）")
    print(f"  · block 选择基于随机 compress_key，非最优，实际误差会比真实 NSA 更大")
    print(f"  · 下一步：用 test_topk_dedup.py 的 oracle 索引测 block dedup kernel")


if __name__ == "__main__":
    main()

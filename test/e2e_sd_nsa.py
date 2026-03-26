"""
端到端 Speculative Decoding + NSA Target Model Demo

架构（推荐）：
  Draft model  : 小 LLaMA（如 Llama-3.2-1B）greedy decode，参数 --draft-model
  Target model : 大 LLaMA（如 8B），verify 时全部层注意力替换为 NSA（--model）

  若省略 --draft-model，则 draft 与 target 为同一模型（旧消融：同容量 dense vs NSA）。

流程（一个 SD step）：
  1. Prefill: Target 大模型 forward → past_key_values；t_0 = target 在位置 L 的 argmax
  2. （可选）Draft 小模型单独 prefill 同一 prompt，得到 past_kv_draft
  3. Build NSA cache: 从 **target** prefill KV 构建每层的 (k_raw, v_raw, cmp_k, cmp_v)
  4. Draft: 小模型从 t_0 起 greedy × N（大模型与 draft 不同参）
  5. Verify: NSA target 一次 forward(verify_ids) → logits
  6. Accept/Reject、与 dense target 对比时延

用法：
  # 1B draft + 8B NSA target（需本机有对应权重或 HF 缓存）
  python test/e2e_sd_nsa.py --draft-model meta-llama/Llama-3.2-1B-Instruct \\
      --model /path/to/Llama-3.1-8B-Instruct --nsa-ckpt ...

  python test/e2e_sd_nsa.py --seqlen 8192 --n-draft 8
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from nsa_ref.module import RopeConfig
from nsa_ref.ops import linear_compress
from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode

# Target（大模型）默认路径；Draft 小模型通过 --draft-model 指定（HF id 或本地目录）
LLAMA_PATH = ("/data1/models/Llama-3.1-8B-Instruct/snapshots"
              "/0e9e39f249a16976918f6564b8830bc894c89659")


# ─────────────────────────────────────────────────────────────────────────────
# 单层 NSA（复用 nsa_verify_demo.py 的 LlamaNSALayer）
# ─────────────────────────────────────────────────────────────────────────────

class LlamaNSALayer(nn.Module):
    def __init__(self, llama_attn, cfg, topk=16, block_size=64,
                 kernel_size=32, kernel_stride=16, init_blocks=1,
                 local_blocks=2, window_size=512):
        super().__init__()
        num_q  = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_d = getattr(cfg, 'head_dim', cfg.hidden_size // num_q)

        rope_cfg = RopeConfig(
            max_position_embeddings=cfg.max_position_embeddings,
            head_dim=head_d,
            rope_theta=cfg.rope_theta,
            rope_scaling=getattr(cfg, 'rope_scaling', None),
        )
        self.fsa = FlashSparseAttentionDecode(
            hidden_size=cfg.hidden_size,
            num_q_heads=num_q, num_kv_heads=num_kv, head_dim=head_d,
            kernel_size=kernel_size, kernel_stride=kernel_stride,
            block_size=block_size, topk=topk,
            init_blocks=init_blocks, local_blocks=local_blocks,
            window_size=window_size, rope_config=rope_cfg,
        )
        with torch.no_grad():
            self.fsa.proj_q.weight.copy_(llama_attn.q_proj.weight)
            self.fsa.proj_k.weight.copy_(llama_attn.k_proj.weight)
            self.fsa.proj_v.weight.copy_(llama_attn.v_proj.weight)
            self.fsa.proj_o.weight.copy_(llama_attn.o_proj.weight)
        self.kernel_size   = kernel_size
        self.kernel_stride = kernel_stride

    def forward(self, hidden, k_raw, v_raw, cmp_k, cmp_v,
                cu_q, cu_k, position_ids, use_dedup=False):
        # cmp_valid_len / 预分配 buffer 需在 FlashSparseAttentionDecode 内支持后再接
        return self.fsa(hidden, cu_q, cu_k,
                        k_raw, v_raw, cmp_k, cmp_v,
                        attention_mask=None, position_ids=position_ids)

    def build_compressed_cache(self, k_raw, v_raw, cu_k):
        cmp_k, _ = linear_compress(k_raw, self.fsa.compress_key, cu_k,
                                   self.kernel_size, self.kernel_stride,
                                   self.fsa.intra_block_pe)
        cmp_v, _ = linear_compress(v_raw, self.fsa.compress_value, cu_k,
                                   self.kernel_size, self.kernel_stride, None)
        return cmp_k, cmp_v


# ─────────────────────────────────────────────────────────────────────────────
# 全模型 NSA Target：通过 hook 拦截所有 32 层 attention
# ─────────────────────────────────────────────────────────────────────────────

class NSATargetModel:
    """
    wraps LLaMA，verify 时用 forward-method patching 替换所有层 attention。

    与 post-hook 的区别：method patch 直接跳过 LLaMA 标准 attention 计算，
    只跑 NSA；post-hook 会先跑 LLaMA attention 再替换输出（两条路都跑了）。
    """

    def __init__(self, llama, nsa_layers: list):
        self.llama      = llama
        self.nsa_layers = nsa_layers

        self.k_raw: list = [None] * len(nsa_layers)
        self.v_raw: list = [None] * len(nsa_layers)
        self.cmp_k: list = [None] * len(nsa_layers)
        self.cmp_v: list = [None] * len(nsa_layers)
        self.cmp_k_buffer: list = [None] * len(nsa_layers)
        self.cmp_v_buffer: list = [None] * len(nsa_layers)
        self.cmp_valid_len: list = [0] * len(nsa_layers)
        self.past_len    = 0

        self._orig_forwards = {}   # 保存原始 forward 以便恢复
        self._verify_mode   = False
        self._pos_ids       = None

    # ── cache 初始化 ───────────────────────────────────────────────────────
    def init_caches_from_prefill(self, past_key_values, past_len: int, n_draft: int = 0):
        """
        n_draft: 用于预分配 cmp buffer 的尾部空间，max_cmp_len = init_len + n_draft。
                当 n_draft > 0 时启用预分配 + 尾部写入，去掉每步 torch.cat。
        """
        device = self.nsa_layers[0].fsa.proj_q.weight.device
        dtype  = self.nsa_layers[0].fsa.proj_q.weight.dtype
        self.past_len = past_len

        print(f"  Building NSA caches for {len(self.nsa_layers)} layers ...", end=" ")
        for l, nsa in enumerate(self.nsa_layers):
            k, v  = past_key_values[l]
            k_raw = k.squeeze(0).permute(1, 0, 2).contiguous().to(dtype)
            v_raw = v.squeeze(0).permute(1, 0, 2).contiguous().to(dtype)
            cu_k  = torch.tensor([0, k_raw.shape[0]], device=device, dtype=torch.int32)
            cmp_k, cmp_v = nsa.build_compressed_cache(k_raw, v_raw, cu_k)
            self.k_raw[l] = k_raw
            self.v_raw[l] = v_raw
            self.cmp_k[l] = cmp_k
            self.cmp_v[l] = cmp_v

            if n_draft > 0:
                init_len = cmp_k.shape[0]
                max_cmp_len = init_len + n_draft
                num_heads, head_dim = cmp_k.shape[1], cmp_k.shape[2]
                self.cmp_k_buffer[l] = torch.empty(
                    max_cmp_len, num_heads, head_dim, device=device, dtype=dtype
                )
                self.cmp_v_buffer[l] = torch.empty(
                    max_cmp_len, num_heads, head_dim, device=device, dtype=dtype
                )
                self.cmp_k_buffer[l][:init_len].copy_(cmp_k)
                self.cmp_v_buffer[l][:init_len].copy_(cmp_v)
                self.cmp_valid_len[l] = init_len
        print("done.")

    # ── method patching ────────────────────────────────────────────────────
    def _make_nsa_forward(self, layer_idx: int):
        """返回一个替换 self_attn.forward 的函数，只跑 NSA，不跑 LLaMA attention。"""
        target = self
        nsa    = self.nsa_layers[layer_idx]

        def nsa_forward(hidden_states=None, *args, **kwargs):
            # verify 模式外退回原始 forward（理论上不会进入，但保险起见）
            if not target._verify_mode:
                return target._orig_forwards[layer_idx](hidden_states, *args, **kwargs)

            N           = hidden_states.shape[1]
            hidden_flat = hidden_states.squeeze(0)          # [N, H]
            device      = hidden_states.device
            # FlashSparseAttentionDecode: k = cat(k_cache, k_new)，总长必须为 past_len + N；
            # 若只传 past_len，compressed / flash_attn 的 cu_seqlens 与真实 K 长度不一致，长上下文 verify 会错。
            past_kv_len = target.k_raw[layer_idx].shape[0]
            total_k_len = past_kv_len + N
            cu_q  = torch.tensor([0, N], device=device, dtype=torch.int32)
            cu_k  = torch.tensor([0, total_k_len], device=device, dtype=torch.int32)

            cmp_k = target.cmp_k_buffer[layer_idx] if target.cmp_k_buffer[layer_idx] is not None else target.cmp_k[layer_idx]
            cmp_v = target.cmp_v_buffer[layer_idx] if target.cmp_v_buffer[layer_idx] is not None else target.cmp_v[layer_idx]
            nsa_out = nsa(
                hidden_flat,
                target.k_raw[layer_idx], target.v_raw[layer_idx],
                cmp_k, cmp_v,
                cu_q, cu_k, target._pos_ids,
            )  # [N, H]

            # LlamaAttention.forward 返回 (hidden_states, attn_weights, past_kv)
            return (nsa_out.unsqueeze(0), None, None)

        return nsa_forward

    def _patch_attentions(self):
        for l, layer in enumerate(self.llama.model.layers):
            attn = layer.self_attn
            self._orig_forwards[l] = attn.forward
            attn.forward = self._make_nsa_forward(l)

    def _restore_attentions(self):
        for l, layer in enumerate(self.llama.model.layers):
            layer.self_attn.forward = self._orig_forwards[l]
        self._orig_forwards.clear()

    # ── verify pass ────────────────────────────────────────────────────────
    @torch.no_grad()
    def verify(self, verify_ids: torch.Tensor, use_dedup=False, debug_gate=False):
        """
        verify_ids: [1, N]  ← 应为 [next_token, draft[0], ..., draft[N-2]]
        Returns logits: [N, vocab_size]  ← 对应 draft[0]...draft[N-1] + bonus

        debug_gate=True 时，在 self._gate_debug 中保存每层 gate 输出 [N,3]（CPU float）。
        """
        N      = verify_ids.shape[1]
        device = verify_ids.device
        self._pos_ids = torch.arange(
            self.past_len, self.past_len + N, device=device, dtype=torch.long)
        self._verify_mode = True
        self._patch_attentions()
        gate_per_layer = [None] * len(self.nsa_layers)
        hooks = []
        if debug_gate:
            for l, nsa in enumerate(self.nsa_layers):

                def make_hook(idx):
                    def hook(_module, _inp, out):
                        gate_per_layer[idx] = out.detach().float().cpu()

                    return hook

                hooks.append(nsa.fsa.gate.register_forward_hook(make_hook(l)))
        try:
            out = self.llama(verify_ids, use_cache=False, num_logits_to_keep=N)
        finally:
            for h in hooks:
                h.remove()
            self._restore_attentions()
            self._verify_mode = False
        if debug_gate:
            self._gate_debug = gate_per_layer
        return out.logits.squeeze(0)   # [N, vocab_size]


# ─────────────────────────────────────────────────────────────────────────────
# Accept / Reject（greedy 版：argmax 对比）
# ─────────────────────────────────────────────────────────────────────────────

def greedy_accept(draft_ids: torch.Tensor,
                  verify_logits: torch.Tensor):
    """
    draft_ids:     [N]
    verify_logits: [N, vocab_size]
    返回 (accepted_tokens, n_accepted)
    规则：accept token i 当 verify_logits[i].argmax() == draft_ids[i]
    一旦出现第一个不匹配就停止（标准贪心 SD 接受规则的确定性版本）。
    """
    verify_argmax = verify_logits.argmax(dim=-1)  # [N]
    n_accepted = 0
    for i in range(len(draft_ids)):
        if verify_argmax[i] == draft_ids[i]:
            n_accepted += 1
        else:
            break
    # bonus：第一个不匹配处 target 的预测；若全部接受则用最后一个 logit
    bonus_pos = min(n_accepted, len(verify_argmax) - 1)
    bonus = verify_argmax[bonus_pos]
    accepted = list(draft_ids[:n_accepted].tolist()) + [bonus.item()]
    return accepted, n_accepted


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def print_nsa_gate_debug(gate_per_layer: list, labels=("compress", "sparse(topk)", "sliding")):
    """gate_per_layer[l]: [N, 3] CPU float，每层 NSA forward 里 gate Sequential 的输出（已 sigmoid）。"""
    if not gate_per_layer or gate_per_layer[0] is None:
        print("  (无 gate 数据)")
        return
    L = len(gate_per_layer)
    means = torch.stack([g.mean(dim=0) for g in gate_per_layer])  # [L, 3]
    print(f"\n=== NSA gate 统计 (sigmoid 后三路权重) ===")
    print(f"  列顺序: {labels[0]}, {labels[1]}, {labels[2]}")
    for name, idx in [("layer 0", 0), (f"layer {L // 2}", L // 2), (f"layer {L - 1}", L - 1)]:
        m = means[idx].tolist()
        print(
            "  "
            + name
            + " 对 verify 各 token 平均: "
            + ", ".join(f"{labels[i]}={m[i]:.4f}" for i in range(3))
        )
    gmean = means.mean(dim=0)
    print(
        f"  **全层平均** (每层先对 token 均值，再对 layer 均值): "
        f"{labels[0]}={gmean[0]:.4f}, {labels[1]}={gmean[1]:.4f}, {labels[2]}={gmean[2]:.4f}"
    )
    sparse_stack = torch.stack([g[:, 1] for g in gate_per_layer], dim=0)  # [L, N]
    print(
        f"  sparse(topk) 路: 跨层+token min={sparse_stack.min().item():.4f}, "
        f"max={sparse_stack.max().item():.4f}, mean={sparse_stack.mean().item():.4f}"
    )
    print(
        "  若 sparse(topk) 均值接近 0，则调大 --topk 几乎不改变 verify logits（与 compressed/sliding 混合后主导）。"
    )


def benchmark(fn, warmup=5, iters=20):
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


def timing_params_for_ctx(past_len: int, warmup_override, iters_override):
    """
    长上下文下 Exact verify 每次带满 KV forward 极占显存；
    默认 5+20 次易 OOM，按 ctx 缩短 warmup/iters（可用 CLI 覆盖）。
    """
    if past_len >= 32768:
        auto_w, auto_i = 1, 3
    elif past_len >= 8192:
        auto_w, auto_i = 2, 8
    else:
        auto_w, auto_i = 5, 20
    w = auto_w if warmup_override is None else warmup_override
    i = auto_i if iters_override is None else iters_override
    return w, i


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=LLAMA_PATH,
        help="Target 大模型（NSA verify / dense baseline），HF id 或本地目录",
    )
    parser.add_argument(
        "--draft-model", default=None,
        help="Draft 小模型路径或 HF id（如 Llama-3.2-1B）；不填则与 --model 相同",
    )
    parser.add_argument("--seqlen",  type=int, default=8192)
    parser.add_argument("--n-draft", type=int, default=8)
    parser.add_argument("--topk",    type=int, default=16)
    parser.add_argument(
        "--nsa-ckpt", default=None,
        help="蒸馏 checkpoint：目录（含 ckpt.pt）或 ckpt.pt 文件路径；"
             "设置后加载 compress_key/value/gate 等，跳过 mean-pool 初始化",
    )
    parser.add_argument(
        "--debug-gate",
        action="store_true",
        help="首次 NSA verify 后打印每层 gate（compress / sparse(topk) / sliding）均值，用于解释 topk 是否生效",
    )
    parser.add_argument(
        "--timing-warmup", type=int, default=None,
        help="NSA/Exact 测速 warmup 次数；默认按 seqlen 自动减小（长 ctx 防 OOM）",
    )
    parser.add_argument(
        "--timing-iters", type=int, default=None,
        help="NSA/Exact 测速计时迭代次数；默认按 seqlen 自动减小",
    )
    parser.add_argument(
        "--skip-exact-timing",
        action="store_true",
        help="跳过 dense Exact verify 测速（仅打印 NSA 时延；长上下文 OOM 时用）",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="自定义 prefill 文本，会 tokenize 后重复拼到 --seqlen；不填则用内置英文段落",
    )
    args = parser.parse_args()

    device, dtype = "cuda", torch.bfloat16
    N = args.n_draft

    # ── 1. 加载 Target 大模型 + tokenizer ─────────────────────────────────
    print("Loading target LLaMA ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    target_llama = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device)
    target_llama.eval()
    cfg        = target_llama.config
    num_layers = cfg.num_hidden_layers
    print(f"  [target] layers={num_layers}, hidden={cfg.hidden_size}, "
          f"q_heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}")

    # ── 1b. Draft 小模型（可选）───────────────────────────────────────────
    if args.draft_model:
        print(f"Loading draft LLaMA from {args.draft_model!r} ...")
        draft_llama = AutoModelForCausalLM.from_pretrained(
            args.draft_model, torch_dtype=dtype, device_map=device)
        draft_llama.eval()
        dv = draft_llama.config.vocab_size
        tv = cfg.vocab_size
        if dv != tv:
            print(
                f"  WARNING: draft vocab_size={dv} != target vocab_size={tv}；"
                "接受率/对比可能无效，请换同词表系列模型。",
            )
        print(
            f"  [draft] layers={draft_llama.config.num_hidden_layers}, "
            f"hidden={draft_llama.config.hidden_size}",
        )
    else:
        draft_llama = target_llama
        print("  [draft] 与 target 相同（未指定 --draft-model）")

    # ── 2. 构建所有层的 NSA（仅挂在 target 上）────────────────────────────
    print(f"Building NSA layers on target (topk={args.topk}) ...")
    nsa_layers = [
        LlamaNSALayer(target_llama.model.layers[l].self_attn, cfg, topk=args.topk)
        .to(device, dtype)
        for l in range(num_layers)
    ]

    if args.nsa_ckpt:
        ckpt_path = args.nsa_ckpt
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "ckpt.pt")
        print(f"Loading NSA weights from {ckpt_path} ...")
        # 必须在 CPU 上 load：ckpt 含 optimizer 等，map_location=cuda 会把整份解压到 GPU，
        # 与已加载的 LLaMA 叠在一起易 OOM；仅 nsa 权重随后由 load_state_dict 拷到 GPU。
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        nsa_sd = ckpt["nsa"]
        for l_key, sd in nsa_sd.items():
            li = int(l_key)
            nsa_layers[li].fsa.load_state_dict(sd, strict=True)
        print(f"  Loaded {len(nsa_sd)} layers (distill step={ckpt.get('step', '?')}).")
    else:
        # ── 均值池化初始化 compress_key/value ────────────────────────────
        kernel_size = nsa_layers[0].kernel_size
        head_dim    = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        kv_heads    = cfg.num_key_value_heads
        print(f"Initializing compress_key/value with mean-pool "
              f"(kernel_size={kernel_size}, kv_heads={kv_heads}, head_dim={head_dim}) ...")
        eye = torch.eye(head_dim, device=device, dtype=dtype) / kernel_size
        with torch.no_grad():
            for nsa in nsa_layers:
                ck = nsa.fsa.compress_key
                cv = nsa.fsa.compress_value
                ck.zero_()
                cv.zero_()
                for i in range(kernel_size):
                    ck[:, i*head_dim:(i+1)*head_dim, :] = eye
                    cv[:, i*head_dim:(i+1)*head_dim, :] = eye
                nn.init.zeros_(nsa.fsa.gate[0].weight)
                nsa.fsa.gate[0].weight[1].fill_(0.0)
                nsa.fsa.gate[0].weight[0].fill_(-1e-2)
                nsa.fsa.gate[0].weight[2].fill_(-1e-2)
        print("  compress_key/value initialized to mean-pool, gate biased toward topk.")

    nsa_target = NSATargetModel(target_llama, nsa_layers)

    # ── 3. 构造 prompt，prefill ───────────────────────────────────────────
    if args.prompt:
        para = args.prompt
    else:
        para = ("Speculative decoding accelerates inference by verifying multiple "
                "candidate tokens simultaneously. Native Sparse Attention reduces "
                "computational cost while maintaining quality in long-context tasks. ")
    ids = tok.encode(para, return_tensors="pt").to(device)
    ids = ids.repeat(1, args.seqlen // ids.shape[1] + 2)[:, :args.seqlen]
    print(f"Prefill: {ids.shape[1]} tokens ...")

    with torch.no_grad():
        prefill_target = target_llama(ids, use_cache=True, num_logits_to_keep=1)
    past_kv_target = prefill_target.past_key_values
    past_len       = ids.shape[1]
    # t_0 一律用 **大模型** prefill 后的 argmax，作为投机分支与 verify 的公共起点
    next_token = prefill_target.logits[:, -1:].argmax(-1)
    print("Prefill target done.")

    if args.draft_model:
        with torch.no_grad():
            prefill_draft = draft_llama(ids, use_cache=True, num_logits_to_keep=1)
        past_kv_draft = prefill_draft.past_key_values
        print("Prefill draft done.")
    else:
        past_kv_draft = past_kv_target

    # ── 4. 构建 NSA cache（来自 target prefill）──────────────────────────
    # n_draft>0 会预分配 cmp buffer；需 FSA.forward 支持 cmp_valid_len 后才可启用，否则用 0
    nsa_target.init_caches_from_prefill(past_kv_target, past_len, n_draft=0)

    # ── 5. Draft：小模型（或同模型）greedy × N ───────────────────────────
    #   next_token = t_0：来自 **target** prefill
    #   t_1...t_N：draft 模型从 t_0 起贪心
    #   verify_input  = [t_0, t_1, ..., t_{N-1}]
    #   draft_to_check = [t_1, ..., t_N]
    draft_label = "draft LLaMA" if args.draft_model else "LLaMA (same as target)"
    print(f"\nDraft: {draft_label} greedy × {N} tokens (t_0 from target) ...")
    draft_tokens = []   # t_1 ... t_N
    draft_logits = []   # logits that produced t_1 ... t_N
    cur = next_token    # t_0（位置 L）
    prefill_kv = past_kv_target  # 仅 target，用于后面 dense verify
    past_kv = past_kv_draft
    for _ in range(N):
        with torch.no_grad():
            out = draft_llama(
                cur, past_key_values=past_kv, use_cache=True,
                num_logits_to_keep=1)
        past_kv = out.past_key_values
        logit   = out.logits.squeeze()   # 预测 t_{i+1} 的 logit
        tok_id  = logit.argmax()
        draft_tokens.append(tok_id)
        draft_logits.append(logit)
        cur = tok_id.reshape(1, 1)

    # verify_ids: [t_0, t_1, ..., t_{N-1}]，位置 [L, L+1, ..., L+N-1]
    t_list      = [next_token.squeeze()] + draft_tokens[:-1]
    verify_ids  = torch.stack(t_list).unsqueeze(0)             # [1, N]
    # 要验证的 token: [t_1, ..., t_N]
    draft_check = torch.stack(draft_tokens)                    # [N]
    draft_logits_t = torch.stack(draft_logits)                 # [N, vocab_size]

    print(f"  verify_ids (input to target): {verify_ids.squeeze().tolist()}")
    print(f"  draft tokens to check:        {draft_check.tolist()}")
    print(f"  Draft text: '{tok.decode(draft_check.tolist())}'")
    past_kv = prefill_kv  # 后面 exact verify 用

    # ── 6. Verify（NSA 全模型，N tokens once）────────────────────────────
    print(f"\nVerify: NSA target (all {num_layers} layers, {N} tokens once) ...")
    verify_logits = nsa_target.verify(
        verify_ids, debug_gate=args.debug_gate)   # [N, vocab_size]
    print(f"  Verify logits shape: {verify_logits.shape}")
    if args.debug_gate:
        print_nsa_gate_debug(nsa_target._gate_debug)

    # ── 7. Accept / Reject ───────────────────────────────────────────────
    # verify_logits[i] = 预测 t_{i+1}，与 draft_check[i] = t_{i+1} 比较
    accepted, n_acc = greedy_accept(draft_check, verify_logits)
    print(f"\n=== Accept / Reject ===")
    print(f"  draft_check (t_1..t_N):  {draft_check.tolist()}")
    print(f"  verify argmax:           {verify_logits.argmax(dim=-1).tolist()}")
    print(f"  Accepted:                {n_acc}/{N} ({n_acc/N:.0%})")
    print(f"  Bonus token:             {accepted[-1]}  ('{tok.decode([accepted[-1]])}')")
    print(f"  Total output tokens:     {accepted}")

    # ── 8. 逐 token 对比：draft logits vs verify logits ──────────────────
    print(f"\n=== Logit 近似质量（draft vs NSA target verify）===")
    for i in range(N):
        d_top = draft_logits_t[i].argmax().item()
        v_top = verify_logits[i].argmax().item()
        kl    = F.kl_div(
            F.log_softmax(verify_logits[i].float(), dim=-1),
            F.softmax(draft_logits_t[i].float(), dim=-1),
            reduction="sum").item()
        match = "✓" if d_top == v_top else "✗"
        print(f"  token {i}: draft={d_top:6d} nsa={v_top:6d} {match}  KL={kl:.3f}")

    # ── 9. Timing ────────────────────────────────────────────────────────
    print(f"\n=== Timing ===")
    tw, ti = timing_params_for_ctx(past_len, args.timing_warmup, args.timing_iters)
    print(f"  (benchmark warmup={tw}, iters={ti}；长 ctx 可再降或 --skip-exact-timing)")

    # ① NSA verify：method-patched，只跑 NSA（不跑 LLaMA standard attention）
    t_nsa = benchmark(lambda: nsa_target.verify(verify_ids), warmup=tw, iters=ti)
    print(f"  NSA verify  ({N} tok, {past_len} ctx): {t_nsa:.1f} ms")

    # 小 draft 与 NSA 权重并存占显存；测 target dense 前可先卸 draft
    if args.draft_model:
        del draft_llama
        gc.collect()
        torch.cuda.empty_cache()

    # ② Exact verify（公平版）：
    # 先释放 NSA 层参数以腾出显存，然后用 DynamicCache 带完整 KV 跑 target dense
    print(f"  释放 NSA 参数以腾出显存 ...", end=" ")
    # 先清空 nsa_target 内部的大 tensor 引用
    for l in range(len(nsa_layers)):
        nsa_target.k_raw[l] = None
        nsa_target.v_raw[l] = None
        nsa_target.cmp_k[l] = None
        nsa_target.cmp_v[l] = None
    del nsa_layers, nsa_target
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"done. free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

    from transformers import DynamicCache

    def make_cache():
        # 每次调用创建新 cache，防止 benchmark 迭代间 cache 累积增长
        c = DynamicCache()
        for li, (k, v) in enumerate(prefill_kv):
            c.update(k, v, li)
        return c

    if args.skip_exact_timing:
        print(f"  Exact verify: 已跳过 (--skip-exact-timing)")
        print(f"\n  NSA verify-only: {t_nsa:.1f} ms（无 speedup 对比）")
    else:
        try:
            t_exact = benchmark(
                lambda: target_llama(
                    verify_ids,
                    past_key_values=make_cache(),
                    use_cache=False,
                    num_logits_to_keep=N,
                ),
                warmup=tw,
                iters=ti,
            )
            print(f"  Exact verify ({N} tok, {past_len} ctx): {t_exact:.1f} ms")
            print(f"\n  NSA speedup: {t_exact / t_nsa:.2f}×  (>1 意味着 NSA 更快)")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                "  Exact verify: CUDA OOM（长 KV + 多次 benchmark 易炸）。\n"
                "  可改用: --skip-exact-timing 或 --timing-warmup 0 --timing-iters 1\n"
                "  或先 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            )
            print(f"\n  NSA verify-only: {t_nsa:.1f} ms（无 speedup 对比）")

    # ── 10. 摘要 ─────────────────────────────────────────────────────────
    head_dim   = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    num_blocks = past_len // 64
    print(f"\n=== 配置摘要 ===")
    print(f"  target (--model): {args.model}")
    if args.draft_model:
        print(f"  draft (--draft-model): {args.draft_model}")
    print(f"  seqlen={past_len}, n_draft={N}, topk={args.topk}/{num_blocks} blocks")
    print(f"  sparsity: {args.topk}/{num_blocks} = {args.topk/num_blocks:.1%}")
    if args.nsa_ckpt:
        print(f"\n  已加载蒸馏 NSA 权重，接受率 / KL 应以本次运行为准")
    else:
        print(f"\n  注意：未指定 --nsa-ckpt 时为 mean-pool 初始化")
        print(f"  加 --nsa-ckpt /data1/zzy/nsa_ckpt/final 可对比训练后效果")
    print(f"\n  接下来：")
    print(f"  · 小 draft：保持 --draft-model 指向 1B（或 EAGLE 等）")
    print(f"  · 同伴 dedup kernel 就绪后设 use_dedup=True 再测端到端加速")


if __name__ == "__main__":
    main()

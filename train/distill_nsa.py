"""
NSA Attention Distillation Training
=====================================
目标：用 LLaMA-3.1-8B 的 dense attention 输出作为 teacher，训练 NSA 的
      compress_key / compress_value / intra_block_pe / gate 参数（共 ~256M）。
      q/k/v/o 权重和 LLaMA 其他权重全程冻结。

Loss = 每层 MSE(NSA_attn_out, LLaMA_attn_out)，对所有层求和
      可选 + --logit-kl-weight × KL：整模 NSA 前向与 frozen LLaMA 的末尾 K 个 next-token logits 对齐
      （直接优化「draft / dense target」与「NSA target」的 logit 近似，利于投机解码接受率）

用法：
  python train/distill_nsa.py
  python train/distill_nsa.py \\
      --max-tokens 500000000 --seqlen 2048 --batch 4 --save-steps 500

显存估计（seqlen=2048, batch=4）：
  LLaMA frozen bf16  ≈ 16 GB
  NSA 参数 + Adam    ≈  2 GB
  激活（32 层）      ≈  8 GB
  合计               ≈ 26 GB  （H100 80GB 绰绰有余）

v2 改进：
  - forward_prefill 用 flash_attn 替换 decode Triton kernel（快 100x）
  - 用双 hook 同时捕获 attn 输入/输出，消除冗余的 LLaMA 32 层重跑
  - 每步只跑一次完整 LLaMA forward

二阶段 --real-fsa（与 e2e 推理一致）：
  走真实 FlashSparseAttentionDecode（稀疏 TopK + Triton），必须用短 seqlen（建议 ≤256）。
  典型：--init-ckpt /path/to/stage1/final --real-fsa --seqlen 128 --batch 1 --grad-accum 2 \\
        --lr 3e-5 --max-tokens 20000000 --local-data ... --save-dir checkpoints/nsa_stage2

logit 蒸馏（拉近 dense 与 NSA 的 verify logits）：
  在上式基础上加例如：--logit-kl-weight 0.5 --logit-kl-last-k 8 --logit-temperature 2
  （多一次整模 forward，建议 --layers all 与 e2e 一致）
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 只清除错误的 HF_ENDPOINT（镜像站），保留 http_proxy 不变
os.environ.pop("HF_ENDPOINT", None)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from nsa_ref.module import RopeConfig
from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode

LLAMA_PATH = ("/data1/models/Llama-3.1-8B-Instruct/snapshots"
              "/0e9e39f249a16976918f6564b8830bc894c89659")


# ─────────────────────────────────────────────────────────────────────────────
# NSA 注意力层（训练版）
# ─────────────────────────────────────────────────────────────────────────────

class NSATrainLayer(nn.Module):
    """
    prefill 场景的 NSA 注意力层，用于与 LLaMA dense attention 对齐。
    q/k/v/o 权重从 LLaMA 复制且冻结；compress_key/value/gate 参与训练。
    forward_prefill 使用 flash_attn 替换 decode Triton kernel。
    """

    def __init__(self, llama_attn, cfg,
                 topk=16, block_size=64, kernel_size=32, kernel_stride=16,
                 init_blocks=1, local_blocks=2, window_size=512):
        super().__init__()
        num_q  = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_d = getattr(cfg, 'head_dim', cfg.hidden_size // num_q)

        rope_cfg = RopeConfig(
            max_position_embeddings=cfg.max_position_embeddings,
            head_dim=head_d, rope_theta=cfg.rope_theta,
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

        # q/k/v/o 权重复制并冻结
        with torch.no_grad():
            self.fsa.proj_q.weight.copy_(llama_attn.q_proj.weight)
            self.fsa.proj_k.weight.copy_(llama_attn.k_proj.weight)
            self.fsa.proj_v.weight.copy_(llama_attn.v_proj.weight)
            self.fsa.proj_o.weight.copy_(llama_attn.o_proj.weight)
        for p in [self.fsa.proj_q.parameters(),
                  self.fsa.proj_k.parameters(),
                  self.fsa.proj_v.parameters(),
                  self.fsa.proj_o.parameters()]:
            for param in p:
                param.requires_grad_(False)

        self.kernel_size   = kernel_size
        self.kernel_stride = kernel_stride
        self.head_dim      = head_d
        self.window_size   = window_size
        # 保存 LLaMA 的 rotary_emb 用于 RoPE 计算
        self.rotary_emb    = llama_attn.rotary_emb

    def _apply_rope(self, q, k, B, seqlen):
        """用 LLaMA 的 rotary_emb 对 q/k 应用 RoPE。"""
        H_q  = q.shape[1]
        H_kv = k.shape[1]
        D    = q.shape[2]
        total_len = q.shape[0]

        position_ids = torch.arange(seqlen, device=q.device).unsqueeze(0).expand(B, -1)
        # rotary_emb 第一个参数只用于获取 device/dtype，传 k 的任意 4-D 变形
        dummy = k.view(B, seqlen, H_kv, D).permute(0, 2, 1, 3)
        cos, sin = self.rotary_emb(dummy, position_ids)

        q_4d = q.view(B, seqlen, H_q,  D).permute(0, 2, 1, 3)
        k_4d = k.view(B, seqlen, H_kv, D).permute(0, 2, 1, 3)
        q_r, k_r = apply_rotary_pos_emb(q_4d, k_4d, cos, sin)
        q = q_r.permute(0, 2, 1, 3).reshape(total_len, H_q,  D)
        k = k_r.permute(0, 2, 1, 3).reshape(total_len, H_kv, D)
        return q, k

    def _compressed_attn_prefill(self, q, k, v, seqlens):
        """纯 PyTorch 压缩注意力，给 compress_key/value 提供梯度。"""
        fsa          = self.fsa
        H_q, H_kv, D = fsa.num_q_heads, fsa.num_kv_heads, self.head_dim
        ks, stride   = self.kernel_size, self.kernel_stride
        ratio        = H_q // H_kv
        scale        = D ** -0.5

        outputs = []
        offset  = 0
        for s in seqlens:
            s = int(s)
            q_s = q[offset:offset + s]  # [s, H_q, D]
            k_s = k[offset:offset + s]  # [s, H_kv, D]
            v_s = v[offset:offset + s]

            num_cmp = max(0, (s - ks) // stride + 1)
            if num_cmp > 0:
                # unfold: [num_cmp, H_kv, D, ks] → [num_cmp, H_kv, ks*D]
                k_win = k_s.unfold(0, ks, stride).permute(0, 1, 3, 2).reshape(num_cmp, H_kv, ks * D)
                v_win = v_s.unfold(0, ks, stride).permute(0, 1, 3, 2).reshape(num_cmp, H_kv, ks * D)

                # compress_key: [H_kv, ks*D, D]，用 f 表示 ks*D 合并维度
                ck = torch.einsum('n h f, h f d -> n h d', k_win, fsa.compress_key)
                cv = torch.einsum('n h f, h f d -> n h d', v_win, fsa.compress_value)

                # GQA 扩展
                ck_exp = ck.repeat_interleave(ratio, dim=1)  # [num_cmp, H_q, D]
                cv_exp = cv.repeat_interleave(ratio, dim=1)

                # 因果 attention scores: [H_q, s, num_cmp]
                scores = torch.einsum('s h d, n h d -> h s n', q_s, ck_exp) * scale

                # 因果掩码：compressed key n 覆盖 KV 位置 [n*stride, n*stride+ks-1]
                # 查询位置 i 只能看到满足 n*stride+ks-1 < i 的 compressed key
                q_pos   = torch.arange(s, device=q.device).float()
                k_end   = torch.arange(num_cmp, device=q.device).float() * stride + ks - 1
                visible = (q_pos[:, None] > k_end[None, :])           # [s, num_cmp]
                scores  = scores.masked_fill(~visible.unsqueeze(0), float('-inf'))

                attn_w  = F.softmax(scores, dim=-1)
                attn_w  = torch.nan_to_num(attn_w, nan=0.0)
                cmp_out = torch.einsum('h s n, n h d -> s h d', attn_w, cv_exp)
            else:
                cmp_out = torch.zeros(s, H_q, D, device=q.device, dtype=q.dtype)

            outputs.append(cmp_out)
            offset += s

        return torch.cat(outputs, dim=0)  # [total_len, H_q, D]

    def forward_prefill(self, hidden_flat, cu_seqlens):
        """
        训练专用 prefill forward（替换 decode Triton kernel）。
        使用 flash_attn 做 sliding window / topk proxy，
        用 PyTorch einsum 做 compressed path（提供 compress_key 梯度）。

        hidden_flat : [total_len, hidden_size]
        cu_seqlens  : [B+1], int32
        返回        : [total_len, hidden_size]
        """
        from flash_attn import flash_attn_varlen_func

        cu_seqlens = cu_seqlens.to(torch.int32)   # flash_attn 要求 int32
        total_len  = hidden_flat.shape[0]
        seqlens    = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        seqlen     = int(seqlens[0])   # 训练时所有序列等长
        B          = len(seqlens)

        fsa     = self.fsa
        H_q     = fsa.num_q_heads
        H_kv    = fsa.num_kv_heads
        D       = self.head_dim

        # QKV 投影
        q = fsa.proj_q(hidden_flat).view(total_len, H_q,  D)
        k = fsa.proj_k(hidden_flat).view(total_len, H_kv, D)
        v = fsa.proj_v(hidden_flat).view(total_len, H_kv, D)

        # 应用 RoPE（与 LLaMA 保持一致）
        q, k = self._apply_rope(q, k, B, seqlen)

        # ── compressed path（PyTorch，给 compress_key/value 梯度）─────
        cmp_out = self._compressed_attn_prefill(q, k, v, seqlens)
        # [total_len, H_q, D]

        # ── sliding window path（flash_attn）─────────────────────────
        sliding_out = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen,
            causal=True, window_size=(self.window_size, 0),
        )  # [total_len, H_q, D]

        # ── topk path proxy：训练时用 full causal attention 代替 ──────
        topk_out = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen,
            causal=True,
        )  # [total_len, H_q, D]

        # ── gate（给 gate 梯度）──────────────────────────────────────
        gate = fsa.gate(hidden_flat)  # [total_len, 3]，已过 sigmoid

        # ── 加权组合 ─────────────────────────────────────────────────
        H_flat = H_q * D
        combined = (gate[:, 0:1] * cmp_out.reshape(total_len, H_flat) +
                    gate[:, 1:2] * topk_out.reshape(total_len, H_flat) +
                    gate[:, 2:3] * sliding_out.reshape(total_len, H_flat))

        return fsa.proj_o(combined)  # [total_len, hidden_size]

    def forward_real_fsa(self, hidden_flat, cu_seqlens, position_ids_flat=None):
        """
        与 test/e2e_sd_nsa.py 一致：空 KV cache 下走完整 FlashSparseAttentionDecode
        （真实压缩路 + 稀疏 TopK + sliding），用于二阶段对齐推理算子。

        hidden_flat        : [total_len, hidden_size]
        cu_seqlens         : [B+1] int32
        position_ids_flat: [total_len]，每段内 0..seqlen-1；None 则用 FSA 内部 RoPE
        """
        fsa   = self.fsa
        empty = torch.empty(
            0, fsa.num_kv_heads, self.head_dim,
            device=hidden_flat.device, dtype=hidden_flat.dtype,
        )
        return fsa(
            hidden_flat,
            cu_seqlens.to(torch.int32),
            cu_seqlens.to(torch.int32),
            empty, empty, empty, empty, empty,
            attention_mask=None,
            position_ids=position_ids_flat,
        )

    def trainable_params(self):
        return [p for p in self.fsa.parameters() if p.requires_grad]


# ─────────────────────────────────────────────────────────────────────────────
# 数据集：流式加载文本，tokenize，按 seqlen 分块
# ─────────────────────────────────────────────────────────────────────────────

class StreamTokenDataset(IterableDataset):
    """从 HuggingFace datasets 流式加载，tokenize 后按 seqlen 分块。"""

    def __init__(self, tokenizer, seqlen=4096, dataset_name="allenai/c4",
                 dataset_split="train", dataset_config="en", max_tokens=None):
        self.tokenizer  = tokenizer
        self.seqlen     = seqlen
        self.max_tokens = max_tokens
        self.dataset_name   = dataset_name
        self.dataset_split  = dataset_split
        self.dataset_config = dataset_config

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(self.dataset_name, self.dataset_config,
                          split=self.dataset_split, streaming=True,
                          trust_remote_code=True)
        buf = []
        total = 0
        for sample in ds:
            text = sample.get("text", "") or sample.get("content", "")
            ids  = self.tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            while len(buf) >= self.seqlen:
                chunk = buf[:self.seqlen]
                buf   = buf[self.seqlen:]
                yield torch.tensor(chunk, dtype=torch.long)
                total += self.seqlen
                if self.max_tokens and total >= self.max_tokens:
                    return


class SyntheticTokenDataset(IterableDataset):
    """生成随机 token ID，用于在无网络时测试训练流程。"""

    def __init__(self, vocab_size, seqlen=2048, max_tokens=None):
        self.vocab_size = vocab_size
        self.seqlen     = seqlen
        self.max_tokens = max_tokens or 10**9

    def __iter__(self):
        total = 0
        while total < self.max_tokens:
            yield torch.randint(0, self.vocab_size, (self.seqlen,), dtype=torch.long)
            total += self.seqlen


class LocalTokenDataset(IterableDataset):
    """从本地 .pt 文件（1D token tensor）读取，按 seqlen 分块，循环使用。"""

    def __init__(self, path, seqlen=2048, max_tokens=None):
        try:
            self.data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            self.data = torch.load(path, map_location="cpu")  # 旧版 PyTorch
        self.seqlen     = seqlen
        self.max_tokens = max_tokens or 10**9
        print(f"  Local dataset: {len(self.data)/1e6:.0f}M tokens from {path}",
              flush=True)

    def __iter__(self):
        n     = len(self.data)
        total = 0
        i     = 0
        while total < self.max_tokens:
            if i + self.seqlen > n:
                i = 0   # 循环
            yield self.data[i:i + self.seqlen].long()
            i     += self.seqlen
            total += self.seqlen


def collate_fn(batch):
    return torch.stack(batch, dim=0)   # [B, seqlen]


# ─────────────────────────────────────────────────────────────────────────────
# Teacher hook：同时捕获每层 attention 的输入和输出（只需一次 forward）
# ─────────────────────────────────────────────────────────────────────────────

class AttentionCapture:
    """
    同时捕获 LLaMA 每层 self_attn 的输入（input_layernorm 之后）和输出。
    一次 teacher forward 即可获取所有层的 attn_input 和 attn_output，
    不需要再单独重跑 LLaMA 层。
    """

    def __init__(self):
        self.inputs  = {}   # layer_idx → [B, S, H] (layernorm 后)
        self.outputs = {}   # layer_idx → [B, S, H] (self_attn 输出)

    def register(self, llama_model):
        self._handles = []
        for l, layer in enumerate(llama_model.model.layers):
            # 输入 hook：pre_hook，支持位置参数和关键字参数两种调用方式
            def pre_hook(module, args, kwargs, l=l):
                hs = args[0] if args else kwargs.get('hidden_states')
                if hs is not None:
                    self.inputs[l] = hs.detach()
            self._handles.append(
                layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True))

            # 输出 hook：post_hook
            def post_hook(module, args, kwargs, output, l=l):
                h = output[0] if isinstance(output, tuple) else output
                self.outputs[l] = h
            self._handles.append(
                layer.self_attn.register_forward_hook(post_hook, with_kwargs=True))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self.inputs.clear()
        self.outputs.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Logit 蒸馏：全模型 self_attn → NSA，与 teacher logits 做 KL
# ─────────────────────────────────────────────────────────────────────────────

def make_nsa_self_attn_forward(nsa_tlayer: NSATrainLayer, use_real_fsa: bool):
    """
    返回可替换 LlamaAttention.forward 的函数。
    hidden_states 为 decoder 内 input_layernorm 之后，形状 [B, S, H]。
    """
    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if use_cache or past_key_value is not None:
            raise NotImplementedError(
                "logit KL 训练路径仅支持 use_cache=False 的整段 prefill。"
            )
        bsz, q_len, hsz = hidden_states.shape
        flat = hidden_states.reshape(-1, hsz)
        cu = torch.arange(
            0, (bsz + 1) * q_len, q_len,
            device=flat.device, dtype=torch.int32,
        )
        if position_ids is None:
            pos_flat = (
                torch.arange(q_len, device=flat.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(bsz, -1)
                .reshape(-1)
            )
        else:
            pos_flat = position_ids.reshape(-1).long()

        if use_real_fsa:
            out = nsa_tlayer.forward_real_fsa(flat, cu, position_ids_flat=pos_flat)
        else:
            out = nsa_tlayer.forward_prefill(flat, cu)

        out = out.reshape(bsz, q_len, hsz)
        return (out, None, None)

    return forward


def logit_kl_last_k_positions(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    last_k: int,
    temperature: float,
) -> torch.Tensor:
    """
    对序列末尾 K 个 next-token 位置做 KL(student || teacher)。
    logits[:, t, :] 预测 input_ids[:, t+1]。
    """
    S = student_logits.shape[1]
    if last_k <= 0 or last_k >= S:
        raise ValueError(f"logit_kl_last_k 须在 1..seqlen-1 之间，当前 last_k={last_k}, S={S}")
    # 位置 S-K-1 .. S-2 的 logits 对应标签 S-K .. S-1
    sl = student_logits[:, S - last_k - 1 : S - 1, :].float()
    tl = teacher_logits[:, S - last_k - 1 : S - 1, :].float()
    T = float(temperature)
    log_p = F.log_softmax(sl / T, dim=-1)
    q = F.softmax(tl / T, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# 均值池化初始化
# ─────────────────────────────────────────────────────────────────────────────

def init_meanpool(nsa_layer):
    """将 compress_key/value 初始化为均值池化，gate 偏向 topk 路。"""
    fsa         = nsa_layer.fsa
    kernel_size = nsa_layer.kernel_size
    head_dim    = nsa_layer.head_dim
    eye         = torch.eye(head_dim, device=fsa.compress_key.device,
                            dtype=fsa.compress_key.dtype) / kernel_size
    with torch.no_grad():
        fsa.compress_key.zero_()
        fsa.compress_value.zero_()
        for i in range(kernel_size):
            fsa.compress_key[:, i*head_dim:(i+1)*head_dim, :]   = eye
            fsa.compress_value[:, i*head_dim:(i+1)*head_dim, :] = eye
        nn.init.zeros_(fsa.gate[0].weight)
        fsa.gate[0].weight[1].fill_(0.0)   # topk 路 sigmoid(0)=0.5
        fsa.gate[0].weight[0].fill_(-1e-2) # compressed 路略低
        fsa.gate[0].weight[2].fill_(-1e-2) # sliding 路略低


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # nohup 重定向时 stdout 全缓冲，日志长时间不更新；改为行缓冲
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=LLAMA_PATH)
    parser.add_argument("--seqlen",     type=int, default=2048)
    parser.add_argument("--batch",      type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="梯度累积步数（有效 batch = batch × grad_accum）")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=100_000_000,
                        help="训练 token 总量")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-dir",   default="checkpoints/nsa_distill")
    parser.add_argument("--resume",     default=None,
                        help="从 checkpoint 目录恢复（含 optimizer + step）")
    parser.add_argument("--init-ckpt",  default=None,
                        help="二阶段：仅从该目录/ckpt.pt 加载 NSA 权重（不加载 optimizer，step 从 0）")
    parser.add_argument("--real-fsa",   action="store_true",
                        help="二阶段：学生前向走真实 FlashSparseAttentionDecode（慢，与 e2e 一致）")
    parser.add_argument("--dataset",    default="allenai/c4")
    parser.add_argument("--synthetic",   action="store_true",
                        help="用随机 token 替代真实数据集（测试用）")
    parser.add_argument("--local-data",  default=None,
                        help="本地 .pt token 文件路径，离线训练用（优先于 HF 数据集）")
    parser.add_argument("--layers",     type=str, default="all",
                        help="训练哪些层，all 或 0,1,2...（逗号分隔）")
    parser.add_argument(
        "--logit-kl-weight", type=float, default=0.0,
        help=">0：额外整模 NSA 前向，对序列末尾 K 个 next-token 做 KL(学生logits||教师logits)，"
             "直接拉近 dense 与 NSA 的分布（投机解码验收标准）。会多跑一次 llama forward，显存/时间增加。",
    )
    parser.add_argument(
        "--logit-kl-last-k", type=int, default=8,
        help="logit KL 覆盖的 next-token 数（建议与 --n-draft 一致，如 8）",
    )
    parser.add_argument(
        "--logit-temperature", type=float, default=2.0,
        help="logit 蒸馏温度 T，KL 按 Hinton KD 乘 T^2",
    )
    parser.add_argument(
        "--topk", type=int, default=16,
        help="稀疏分支选块数；须与 e2e --topk 一致，否则训推选块行为不同、接受率差",
    )
    parser.add_argument("--block-size", type=int, default=64, help="NSA KV block_size，与 e2e 一致")
    parser.add_argument(
        "--window-size", type=int, default=512,
        help="sliding 分支 window_size，与 FlashSparseAttentionDecode / e2e 一致",
    )
    parser.add_argument("--kernel-size", type=int, default=32)
    parser.add_argument("--kernel-stride", type=int, default=16)
    parser.add_argument("--init-blocks", type=int, default=1)
    parser.add_argument("--local-blocks", type=int, default=2)
    args = parser.parse_args()

    args.save_dir = (args.save_dir or "").strip()
    if not args.save_dir:
        print(
            "错误: --save-dir 未设置或为空（不要用未展开的 shell 变量）。\n"
            "示例: --save-dir /data1/zzy/nsa_ckpt_stage3_longctx",
            file=sys.stderr,
        )
        sys.exit(1)

    device, dtype = "cuda", torch.bfloat16
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. 加载 LLaMA（frozen teacher）────────────────────────────────
    print("Loading LLaMA (frozen teacher) ...")
    tok   = AutoTokenizer.from_pretrained(args.model)
    llama = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device)
    llama.eval()
    for p in llama.parameters():
        p.requires_grad_(False)
    cfg = llama.config
    num_layers = cfg.num_hidden_layers
    print(f"  Model: {num_layers} layers, hidden={cfg.hidden_size}")

    # ── 2. 构建 NSA 训练层 ────────────────────────────────────────────
    if args.layers == "all":
        train_layers = list(range(num_layers))
    else:
        train_layers = [int(x) for x in args.layers.split(",")]
    print(f"Training NSA compress/gate for layers: {train_layers}")

    nsa_layers = {}
    for l in train_layers:
        nsa_layers[l] = NSATrainLayer(
            llama.model.layers[l].self_attn,
            cfg,
            topk=args.topk,
            block_size=args.block_size,
            kernel_size=args.kernel_size,
            kernel_stride=args.kernel_stride,
            init_blocks=args.init_blocks,
            local_blocks=args.local_blocks,
            window_size=args.window_size,
        ).to(device, dtype)
        init_meanpool(nsa_layers[l])
    print(
        f"NSA hparams: topk={args.topk}, block_size={args.block_size}, "
        f"window_size={args.window_size}, kernel=({args.kernel_size},{args.kernel_stride}), "
        f"init/local blocks=({args.init_blocks},{args.local_blocks})",
        flush=True,
    )

    # 统计可训练参数
    trainable = []
    for l in train_layers:
        trainable.extend(nsa_layers[l].trainable_params())
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_params/1e6:.1f}M")

    # 二阶段：从阶段一 final 等加载 NSA（仅权重）
    if args.init_ckpt:
        raw = args.init_ckpt.rstrip("/")
        if os.path.isfile(raw):
            pth = raw
        elif os.path.isdir(raw):
            pth = os.path.join(raw, "ckpt.pt")
        else:
            # 既不是已有文件也不是已有目录：常见是 .../final 目录尚未生成
            cand = os.path.join(raw, "ckpt.pt")
            raise FileNotFoundError(
                "找不到 --init-ckpt 路径。\n"
                f"  传入: {args.init_ckpt!r}\n"
                "  请确认一阶段已保存，且使用以下之一：\n"
                "    · 目录（内含 ckpt.pt），例如 .../nsa_ckpt_stage1/final\n"
                "    · 或直接指向文件 .../final/ckpt.pt\n"
                f"  （当前若期望 {cand!r}，请 ls 检查该文件是否存在。）"
            )
        if not os.path.isfile(pth):
            raise FileNotFoundError(
                f"init checkpoint 文件不存在: {pth!r}\n"
                "若传入的是目录，请确保其中有 ckpt.pt。"
            )
        print(f"Loading init NSA weights from {pth} ...", flush=True)
        try:
            ic = torch.load(pth, map_location=device, weights_only=False)
        except TypeError:
            ic = torch.load(pth, map_location=device)
        for l in train_layers:
            nsa_layers[l].fsa.load_state_dict(ic["nsa"][l], strict=True)
        print(f"  init-ckpt loaded ({len(train_layers)} layers).", flush=True)

    if args.real_fsa:
        print("*** real-fsa mode: student = FlashSparseAttentionDecode (same as e2e) ***", flush=True)
        if args.seqlen > 256:
            print(f"  WARNING: seqlen={args.seqlen} is large; real-fsa is very slow. "
                  f"Recommend --seqlen 128 or 256.", flush=True)

    if args.logit_kl_weight > 0:
        print(
            f"*** logit KL: weight={args.logit_kl_weight}, last_k={args.logit_kl_last_k}, "
            f"T={args.logit_temperature} ***",
            flush=True,
        )
        if args.logit_kl_last_k >= args.seqlen:
            raise ValueError("--logit-kl-last-k 必须小于 --seqlen")
        if len(train_layers) != num_layers:
            print(
                "  WARNING: logit KL 下 student 仅替换 train_layers 中的注意力，"
                "其余层仍为 dense；与「全层 NSA」e2e 不完全一致，建议 --layers all。",
                flush=True,
            )

    # ── 3. Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # 从 checkpoint 恢复（同一阶段断点续训）
    start_step = 0
    if args.resume:
        ckpt = torch.load(os.path.join(args.resume, "ckpt.pt"),
                          map_location=device)
        for l in train_layers:
            nsa_layers[l].fsa.load_state_dict(ckpt["nsa"][l])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # ── 4. Teacher hook（双 hook：同时捕获输入和输出）─────────────────
    capture = AttentionCapture()
    capture.register(llama)

    # ── 5. Dataset ────────────────────────────────────────────────────
    if args.local_data:
        print(f"Dataset: local {args.local_data} | seqlen={args.seqlen} | "
              f"max_tokens={args.max_tokens/1e6:.0f}M", flush=True)
        dataset = LocalTokenDataset(args.local_data, seqlen=args.seqlen,
                                    max_tokens=args.max_tokens)
        loader = DataLoader(dataset, batch_size=args.batch,
                            collate_fn=collate_fn, num_workers=0)
    elif args.synthetic:
        print(f"Dataset: synthetic | seqlen={args.seqlen} | "
              f"max_tokens={args.max_tokens/1e6:.0f}M")
        dataset = SyntheticTokenDataset(
            vocab_size=llama.config.vocab_size,
            seqlen=args.seqlen, max_tokens=args.max_tokens)
        loader = DataLoader(dataset, batch_size=args.batch,
                            collate_fn=collate_fn, num_workers=0)
    else:
        print(f"Dataset: {args.dataset} | seqlen={args.seqlen} | "
              f"max_tokens={args.max_tokens/1e6:.0f}M")
        dataset = StreamTokenDataset(
            tok, seqlen=args.seqlen, dataset_name=args.dataset,
            max_tokens=args.max_tokens)
        loader  = DataLoader(dataset, batch_size=args.batch,
                             collate_fn=collate_fn, num_workers=2)

    # ── 6. Training loop ──────────────────────────────────────────────
    step        = start_step
    total_loss  = 0.0
    t0          = time.time()
    tokens_seen = step * args.batch * args.seqlen * args.grad_accum

    mode_s = "real-fsa (Triton decode)" if args.real_fsa else "proxy (flash_attn)"
    print(f"\nStart training [{mode_s}] (effective batch = {args.batch*args.grad_accum} "
          f"× {args.seqlen} = "
          f"{args.batch*args.grad_accum*args.seqlen/1e3:.0f}K tokens/step)\n",
          flush=True)

    optimizer.zero_grad()
    t_loop = time.time()
    for microstep, batch_ids in enumerate(loader):
        if microstep == 0:
            print("[progress] microstep 0: loading batch → GPU ...", flush=True)
        batch_ids = batch_ids.to(device)   # [B, seqlen]
        B         = batch_ids.shape[0]
        seqlens   = torch.full((B,), args.seqlen, device=device, dtype=torch.int32)
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            seqlens.cumsum(0),
        ]).to(torch.int32)   # cumsum 在部分版本会升到 int64
        position_ids = torch.arange(args.seqlen, device=device).unsqueeze(0).expand(B, -1)
        position_ids_flat = position_ids.reshape(-1)   # [B*S]，供 real-fsa RoPE

        # ── Teacher forward（一次 forward：logits + 双 hook 各层 attn_input/output）
        capture.clear()
        with torch.no_grad():
            out_teacher = llama(
                input_ids=batch_ids,
                position_ids=position_ids,
                use_cache=False,
                num_logits_to_keep=0,
            )
        teacher_logits = (
            out_teacher.logits.detach() if args.logit_kl_weight > 0 else None
        )
        if microstep == 0:
            print("[progress] microstep 0: teacher OK → NSA×32 forward + backward ...",
                  flush=True)

        # ── 每层 MSE：用 teacher hook，不重跑 LLaMA ──
        loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for l in train_layers:
            attn_input  = capture.inputs.get(l)   # [B, S, H]，layernorm 之后
            teacher_out = capture.outputs.get(l)  # [B, S, H]，self_attn 输出

            if attn_input is None or teacher_out is None:
                continue   # hook 未注册（不应发生）

            attn_input_flat = attn_input.reshape(-1, cfg.hidden_size)
            teacher_flat    = teacher_out.reshape(-1, cfg.hidden_size).detach()

            if args.real_fsa:
                nsa_out = nsa_layers[l].forward_real_fsa(
                    attn_input_flat, cu_seqlens, position_ids_flat=position_ids_flat)
            else:
                nsa_out = nsa_layers[l].forward_prefill(attn_input_flat, cu_seqlens)

            loss = loss + F.mse_loss(nsa_out.float(), teacher_flat.float())

        loss = loss / (len(train_layers) * args.grad_accum)

        # ── Logit KL：整模替换 train_layers 的 self_attn → 再跑一遍 llama ──
        if args.logit_kl_weight > 0:
            saved_forwards = {}
            for l in train_layers:
                attn_mod = llama.model.layers[l].self_attn
                saved_forwards[l] = attn_mod.forward
                attn_mod.forward = make_nsa_self_attn_forward(
                    nsa_layers[l], use_real_fsa=args.real_fsa)

            out_student = llama(
                input_ids=batch_ids,
                position_ids=position_ids,
                use_cache=False,
                num_logits_to_keep=0,
            )

            for l in train_layers:
                llama.model.layers[l].self_attn.forward = saved_forwards[l]

            lk = logit_kl_last_k_positions(
                out_student.logits,
                teacher_logits,
                last_k=args.logit_kl_last_k,
                temperature=args.logit_temperature,
            )
            loss = loss + (args.logit_kl_weight * lk) / args.grad_accum

        loss.backward()
        total_loss += loss.item() * args.grad_accum
        if microstep < args.grad_accum:
            print(f"[progress] microstep {microstep} backward done "
                  f"({time.time()-t_loop:.0f}s since loop start)", flush=True)

        if (microstep + 1) % args.grad_accum == 0:
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step        += 1
            tokens_seen += args.batch * args.seqlen * args.grad_accum

            if step == 1:
                elapsed = time.time() - t0
                print(f"step={step:6d} | loss_sum={total_loss:.4f} (仅第1个optimizer步) | "
                      f"tokens={tokens_seen/1e6:.1f}M | "
                      f"elapsed={elapsed/60:.1f}min", flush=True)
            if step % 10 == 0:
                elapsed   = time.time() - t0
                tok_per_s = (args.batch * args.seqlen * args.grad_accum * 10) / elapsed
                print(f"step={step:6d} | loss={total_loss/10:.4f} | "
                      f"tok/s={tok_per_s:.0f} | "
                      f"tokens={tokens_seen/1e6:.1f}M / {args.max_tokens/1e6:.0f}M | "
                      f"elapsed={elapsed/3600:.2f}h", flush=True)
                total_loss = 0.0
                t0         = time.time()

            if step % args.save_steps == 0:
                ckpt_path = os.path.join(args.save_dir, f"step_{step:06d}")
                os.makedirs(ckpt_path, exist_ok=True)
                torch.save({
                    "step": step,
                    "nsa": {l: nsa_layers[l].fsa.state_dict() for l in train_layers},
                    "optimizer": optimizer.state_dict(),
                    "train_layers": train_layers,
                    "real_fsa": args.real_fsa,
                    "nsa_hparams": {
                        "topk": args.topk,
                        "block_size": args.block_size,
                        "window_size": args.window_size,
                        "kernel_size": args.kernel_size,
                        "kernel_stride": args.kernel_stride,
                        "init_blocks": args.init_blocks,
                        "local_blocks": args.local_blocks,
                    },
                }, os.path.join(ckpt_path, "ckpt.pt"))
                print(f"  Saved checkpoint: {ckpt_path}")

            if tokens_seen >= args.max_tokens:
                break

    # 最终保存
    final_path = os.path.join(args.save_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    torch.save({
        "step": step,
        "nsa": {l: nsa_layers[l].fsa.state_dict() for l in train_layers},
        "optimizer": optimizer.state_dict(),
        "train_layers": train_layers,
        "real_fsa": args.real_fsa,
        "nsa_hparams": {
            "topk": args.topk,
            "block_size": args.block_size,
            "window_size": args.window_size,
            "kernel_size": args.kernel_size,
            "kernel_stride": args.kernel_stride,
            "init_blocks": args.init_blocks,
            "local_blocks": args.local_blocks,
        },
    }, os.path.join(final_path, "ckpt.pt"))
    print(f"\nTraining done. Final checkpoint: {final_path}")
    print(f"Total tokens: {tokens_seen/1e6:.1f}M, steps: {step}")


if __name__ == "__main__":
    main()

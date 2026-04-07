"""
Sparse Attention Distillation Training (v2)
=============================================
目标：用 LLaMA-3.1-8B 的 dense attention 输出作为 teacher，训练稀疏注意力的
      compress_key / compress_value / intra_block_pe / gate 参数（共 ~256M）。
      q/k/v/o 权重和 LLaMA 其他权重全程冻结。

Loss = 每层 MSE(sparse_attn_out, LLaMA_attn_out)，对所有层求和
      可选 + --logit-kl-weight × KL：整模稀疏注意力前向与 frozen LLaMA 的末尾 K 个
      next-token logits 对齐（直接优化 logit 近似，利于投机解码接受率）

--fsa 开关控制使用 FlashSparseAttention（FSA）还是 NativeSparseAttention（NSA，默认）。
两者接口一致，均使用压缩注意力 + 稀疏 TopK + 滑动窗口 + 门控组合。

v2 改进（相对 distill_nsa.py）：
  - 移除 --real-fsa 二阶段逻辑，统一用 FSA / NSA 的训练 forward
  - 移除手工 forward_prefill（RoPE + flash_attn proxy），直接调用模块的 forward(x, cu_seqlens)
  - 用 --fsa 开关在 FlashSparseAttention 与 NativeSparseAttention 之间切换

用法：
  # NSA（默认）
  python train/distill_nsa_v2.py --local-data data/tokens.pt
  # FSA
  python train/distill_nsa_v2.py --fsa --local-data data/tokens.pt
  python train/distill_nsa_v2.py \\
      --fsa --max-tokens 500000000 --seqlen 2048 --batch 4 --save-steps 500

显存估计（seqlen=2048, batch=4）：
  LLaMA frozen bf16  ≈ 16 GB
  稀疏注意力参数+Adam ≈  2 GB
  激活（32 层）      ≈  8 GB
  合计               ≈ 26 GB

logit 蒸馏（拉近 dense 与稀疏注意力的 verify logits）：
  --logit-kl-weight 0.5 --logit-kl-last-k 8 --logit-temperature 2
  （多一次整模 forward，建议 --layers all 与 e2e 一致）
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

# 只清除错误的 HF_ENDPOINT（镜像站），保留 http_proxy 不变
os.environ.pop("HF_ENDPOINT", None)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import math

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from nsa_ref.module import RopeConfig

LLAMA_PATH = ("/data1/models/Llama-3.1-8B-Instruct/snapshots"
              "/0e9e39f249a16976918f6564b8830bc894c89659")


# ─────────────────────────────────────────────────────────────────────────────
# 稀疏注意力训练层
# ─────────────────────────────────────────────────────────────────────────────

class SparseAttnTrainLayer(nn.Module):
    """
    稀疏注意力训练层，用于与 LLaMA dense attention 对齐。
    q/k/v/o 默认从 LLaMA 复制并冻结；若 freeze_qkvo=False 则 proj_q/k/v/o 可训练。
    直接调用 FlashSparseAttention / NativeSparseAttention 的 forward(x, cu_seqlens)。
    """

    def __init__(self, llama_attn, cfg, use_fsa=False,
                 topk=16, block_size=64, kernel_size=32, kernel_stride=16,
                 init_blocks=1, local_blocks=2, window_size=512,
                 freeze_qkvo=True):
        super().__init__()
        num_q  = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_d = getattr(cfg, 'head_dim', cfg.hidden_size // num_q)

        rope_cfg = RopeConfig(
            max_position_embeddings=cfg.max_position_embeddings,
            head_dim=head_d, rope_theta=cfg.rope_theta,
            rope_scaling=getattr(cfg, 'rope_scaling', None),
        )

        # 根据 --fsa 选择 FlashSparseAttention 或 NativeSparseAttention
        if use_fsa:
            from fsa.module.fsa import FlashSparseAttention
            sparse_cls = FlashSparseAttention
        else:
            from nsa_ref.module import NativeSparseAttention
            sparse_cls = NativeSparseAttention

        self.fsa = sparse_cls(
            hidden_size=cfg.hidden_size,
            num_q_heads=num_q, num_kv_heads=num_kv, head_dim=head_d,
            kernel_size=kernel_size, kernel_stride=kernel_stride,
            block_size=block_size, topk=topk,
            init_blocks=init_blocks, local_blocks=local_blocks,
            window_size=window_size, rope_config=rope_cfg,
        )

        # q/k/v/o：从 LLaMA 复制初始化；是否冻结由 freeze_qkvo 控制
        with torch.no_grad():
            self.fsa.proj_q.weight.copy_(llama_attn.q_proj.weight)
            self.fsa.proj_k.weight.copy_(llama_attn.k_proj.weight)
            self.fsa.proj_v.weight.copy_(llama_attn.v_proj.weight)
            self.fsa.proj_o.weight.copy_(llama_attn.o_proj.weight)
        for proj in [self.fsa.proj_q, self.fsa.proj_k,
                     self.fsa.proj_v, self.fsa.proj_o]:
            for param in proj.parameters():
                param.requires_grad_(not freeze_qkvo)

        self.kernel_size   = kernel_size
        self.kernel_stride = kernel_stride
        self.head_dim      = head_d
        self.window_size   = window_size

    def forward(self, hidden_flat, cu_seqlens):
        """
        hidden_flat : [total_len, hidden_size]
        cu_seqlens  : [B+1], int32
        返回        : [total_len, hidden_size]
        """
        return self.fsa(hidden_flat, cu_seqlens.to(torch.int32))

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
    """从本地 .pt 文件（1D token tensor）读取，按 seqlen 随机采样，循环使用。"""
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
        while total < self.max_tokens:
            i = torch.randint(0, n - self.seqlen, (1,)).item()
            yield self.data[i:i + self.seqlen].long()
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
        self.inputs  = {}   # layer_idx -> [B, S, H] (layernorm 后)
        self.outputs = {}   # layer_idx -> [B, S, H] (self_attn 输出)

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
# Logit 蒸馏：全模型 self_attn -> 稀疏注意力，与 teacher logits 做 KL
# ─────────────────────────────────────────────────────────────────────────────

def make_sparse_self_attn_forward(sparse_layer: SparseAttnTrainLayer):
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
        out = sparse_layer(flat, cu)
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
    返回每个位置的平均 KL 散度。
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
    # batchmean 只除以 B，不除以 last_k，需要手动除以 last_k 得到每位置平均 KL
    return F.kl_div(log_p, q, reduction="batchmean") * (T ** 2) / last_k


# ─────────────────────────────────────────────────────────────────────────────
# 均值池化初始化
# ─────────────────────────────────────────────────────────────────────────────

def init_meanpool(sparse_layer):
    """将 compress_key/value 初始化为均值池化，gate 偏向 topk 路。"""
    fsa         = sparse_layer.fsa
    kernel_size = sparse_layer.kernel_size
    head_dim    = sparse_layer.head_dim
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
                        help="梯度累积步数（有效 batch = batch x grad_accum）")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=100_000_000,
                        help="训练 token 总量")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-dir",   default="checkpoints/nsa_distill")
    parser.add_argument("--resume",     default=None,
                        help="从 checkpoint 目录恢复（含 optimizer + step）")
    parser.add_argument("--init-ckpt",  default=None,
                        help="仅从该目录/ckpt.pt 加载稀疏注意力权重（不加载 optimizer，step 从 0）")
    parser.add_argument("--fsa",        action="store_true",
                        help="使用 FlashSparseAttention（默认为 NativeSparseAttention）")
    parser.add_argument("--dataset",    default="allenai/c4")
    parser.add_argument("--synthetic",  action="store_true",
                        help="用随机 token 替代真实数据集（测试用）")
    parser.add_argument("--local-data", default=None,
                        help="本地 .pt token 文件路径，离线训练用（优先于 HF 数据集）")
    parser.add_argument("--layers",     type=str, default="all",
                        help="训练哪些层，all 或 0,1,2...（逗号分隔）")
    parser.add_argument(
        "--logit-kl-weight", type=float, default=0.0,
        help=">0：额外整模前向，对序列末尾 K 个 next-token 做 KL(学生logits||教师logits)，"
             "直接拉近 dense 与稀疏注意力的分布。会多跑一次 llama forward，显存/时间增加。",
    )
    parser.add_argument(
        "--logit-kl-last-k", type=int, default=8,
        help="logit KL 覆盖的 next-token 数（建议与 --n-draft 一致，如 8）",
    )
    parser.add_argument(
        "--logit-temperature", type=float, default=2.0,
        help="logit 蒸馏温度 T，KL 按 Hinton KD 乘 T^2",
    )
    parser.add_argument("--topk",          type=int, default=16,
                        help="稀疏分支选块数；须与 e2e --topk 一致")
    parser.add_argument("--block-size",    type=int, default=64)
    parser.add_argument("--window-size",   type=int, default=512)
    parser.add_argument("--kernel-size",   type=int, default=32)
    parser.add_argument("--kernel-stride", type=int, default=16)
    parser.add_argument("--init-blocks",   type=int, default=1)
    parser.add_argument("--local-blocks",  type=int, default=2)
    parser.add_argument(
        "--train-qkvo",
        action="store_true",
        help="不冻结 q/k/v/o 线性层，与 compress/gate 等一起训练（显存与可训练参数量显著增加）",
    )
    parser.add_argument(
        "--force-logit-kl-with-qkvo",
        action="store_true",
        help="与 --train-qkvo 同时使用时仍计算 logit KL（极耗显存，易 OOM/segfault；默认会跳过 KL，仅训 MSE）",
    )
    parser.add_argument("--warmup-steps",  type=int, default=200,
                        help="线性 warmup 步数（从 0 升至 lr）")
    parser.add_argument("--min-lr-ratio",  type=float, default=0.05,
                        help="余弦退火终止 lr = lr x min_lr_ratio")
    args = parser.parse_args()

    # --train-qkvo 时整模 logit KL 会再建一条「32 层 NSA × 全 LLaMA」的巨大反传图，极易 OOM/segfault；默认关闭 KL。
    if args.train_qkvo and args.logit_kl_weight > 0 and not args.force_logit_kl_with_qkvo:
        print(
            "*** train-qkvo: 禁用 logit KL（仅 MSE）。若需 KL 请加 --force-logit-kl-with-qkvo，"
            "或去掉 --train-qkvo。***",
            flush=True,
        )
        args.logit_kl_weight = 0.0

    args.save_dir = (args.save_dir or "").strip()
    if not args.save_dir:
        print(
            "错误: --save-dir 未设置或为空（不要用未展开的 shell 变量）。\n"
            "示例: --save-dir /data1/zzy/nsa_ckpt_stage3_longctx",
            file=sys.stderr,
        )
        sys.exit(1)

    attn_mode = "FSA" if args.fsa else "NSA"
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

    # ── 2. 构建稀疏注意力训练层 ────────────────────────────────────────
    if args.layers == "all":
        train_layers = list(range(num_layers))
    else:
        train_layers = [int(x) for x in args.layers.split(",")]
    qkvo_note = "qkvo+sparse" if args.train_qkvo else "compress/gate only"
    print(f"Training {attn_mode} ({qkvo_note}) for layers: {train_layers}")

    sparse_layers = {}
    for l in train_layers:
        sparse_layers[l] = SparseAttnTrainLayer(
            llama.model.layers[l].self_attn,
            cfg,
            use_fsa=args.fsa,
            topk=args.topk,
            block_size=args.block_size,
            kernel_size=args.kernel_size,
            kernel_stride=args.kernel_stride,
            init_blocks=args.init_blocks,
            local_blocks=args.local_blocks,
            window_size=args.window_size,
            freeze_qkvo=not args.train_qkvo,
        ).to(device, dtype)
        init_meanpool(sparse_layers[l])
    print(
        f"{attn_mode} hparams: topk={args.topk}, block_size={args.block_size}, "
        f"window_size={args.window_size}, kernel=({args.kernel_size},{args.kernel_stride}), "
        f"init/local blocks=({args.init_blocks},{args.local_blocks})",
        flush=True,
    )

    # 统计可训练参数
    trainable = []
    for l in train_layers:
        trainable.extend(sparse_layers[l].trainable_params())
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_params/1e6:.1f}M")

    # 从已有 checkpoint 加载稀疏注意力权重（仅权重，不加载 optimizer）
    if args.init_ckpt:
        raw = args.init_ckpt.rstrip("/")
        if os.path.isfile(raw):
            pth = raw
        elif os.path.isdir(raw):
            pth = os.path.join(raw, "ckpt.pt")
        else:
            cand = os.path.join(raw, "ckpt.pt")
            raise FileNotFoundError(
                "找不到 --init-ckpt 路径。\n"
                f"  传入: {args.init_ckpt!r}\n"
                "  请确认已保存，且使用以下之一：\n"
                "    - 目录（内含 ckpt.pt），例如 .../nsa_ckpt/final\n"
                "    - 或直接指向文件 .../final/ckpt.pt\n"
                f"  （当前若期望 {cand!r}，请 ls 检查该文件是否存在。）"
            )
        if not os.path.isfile(pth):
            raise FileNotFoundError(
                f"init checkpoint 文件不存在: {pth!r}\n"
                "若传入的是目录，请确保其中有 ckpt.pt。"
            )
        print(f"Loading init weights from {pth} ...", flush=True)
        try:
            ic = torch.load(pth, map_location=device, weights_only=False)
        except TypeError:
            ic = torch.load(pth, map_location=device)
        for l in train_layers:
            sparse_layers[l].fsa.load_state_dict(ic["nsa"][l], strict=True)
        print(f"  init-ckpt loaded ({len(train_layers)} layers).", flush=True)

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
                f"  WARNING: logit KL 下 student 仅替换 train_layers 中的注意力，"
                f"其余层仍为 dense；与「全层 {attn_mode}」e2e 不完全一致，建议 --layers all。",
                flush=True,
            )

    # ── 3. Optimizer + LR Scheduler ────────────────────────────────
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # 总 step 数（用于余弦退火）
    tokens_per_step = args.batch * args.seqlen * args.grad_accum
    total_steps = max(1, args.max_tokens // tokens_per_step)

    def lr_lambda(current_step):
        """Warmup + cosine annealing with min_lr."""
        if current_step < args.warmup_steps:
            return current_step / max(1, args.warmup_steps)
        progress = (current_step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 从 checkpoint 恢复（断点续训）
    start_step = 0
    if args.resume:
        ckpt = torch.load(os.path.join(args.resume, "ckpt.pt"),
                          map_location=device)
        for l in train_layers:
            sparse_layers[l].fsa.load_state_dict(ckpt["nsa"][l])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        for _ in range(start_step):
            scheduler.step()
        print(f"Resumed from step {start_step}, lr={scheduler.get_last_lr()[0]:.2e}")

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

    print(f"\nStart training [{attn_mode}] "
          f"(effective batch = {args.batch*args.grad_accum} x {args.seqlen} = "
          f"{args.batch*args.grad_accum*args.seqlen/1e3:.0f}K tokens/step)")
    print(f"LR schedule: warmup {args.warmup_steps} steps -> cosine decay to "
          f"{args.lr * args.min_lr_ratio:.1e} over {total_steps} steps")
    print(f"Total steps: {total_steps}\n", flush=True)

    optimizer.zero_grad()
    acc_mse = 0.0   # 累计每层平均 MSE
    acc_kl  = 0.0   # 累计 KL loss（未乘 weight）
    acc_n   = 0     # 累计 microstep 数
    t_loop = time.time()
    _interrupted = False

    def _sigint_handler(sig, frame):
        nonlocal _interrupted
        print("\nCtrl-C received, will save after current step ...", flush=True)
        _interrupted = True

    signal.signal(signal.SIGINT, _sigint_handler)

    for microstep, batch_ids in enumerate(loader):
        if microstep == 0:
            print("[progress] microstep 0: loading batch -> GPU ...", flush=True)
        batch_ids = batch_ids.to(device)   # [B, seqlen]
        B         = batch_ids.shape[0]
        seqlens   = torch.full((B,), args.seqlen, device=device, dtype=torch.int32)
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            seqlens.cumsum(0),
        ]).to(torch.int32)
        position_ids = torch.arange(args.seqlen, device=device).unsqueeze(0).expand(B, -1)

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
            print(f"[progress] microstep 0: teacher OK -> {attn_mode}x{num_layers} "
                  f"forward + backward ...", flush=True)

        # ── 每层 MSE：用 teacher hook，不重跑 LLaMA ──
        # 按层分别 backward，避免 32 层 NSA 在同一张图中同时反传（峰值显存过大，易 OOM / segfault）。
        n_mse_layers = len(train_layers)
        scale_mse = 1.0 / (n_mse_layers * args.grad_accum)
        mse_sum = 0.0
        for l in train_layers:
            attn_input  = capture.inputs.get(l)   # [B, S, H]，layernorm 之后
            teacher_out = capture.outputs.get(l)  # [B, S, H]，self_attn 输出

            if attn_input is None or teacher_out is None:
                continue

            if microstep == 0 and l == train_layers[0]:
                print("  [mse] layer 0 forward ...", flush=True)

            attn_input_flat = attn_input.reshape(-1, cfg.hidden_size)
            teacher_flat    = teacher_out.reshape(-1, cfg.hidden_size).detach()

            nsa_out = sparse_layers[l](attn_input_flat, cu_seqlens)
            mse_l = F.mse_loss(nsa_out.float(), teacher_flat.float())
            mse_sum += float(mse_l.detach().item())

            if microstep == 0 and l == train_layers[0]:
                print("  [mse] layer 0 backward ...", flush=True)
            (mse_l * scale_mse).backward()
            if microstep == 0 and l == train_layers[0]:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print("  [mse] layer 0 backward OK", flush=True)

        mse_loss_avg = mse_sum / max(n_mse_layers, 1)
        if microstep == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("  [mse] all layers backward OK", flush=True)

        # ── Logit KL：整模替换 train_layers 的 self_attn -> 再跑一遍 llama ──
        kl_loss_val = 0.0
        if args.logit_kl_weight > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if microstep == 0:
                print("  [kl] student LLaMA forward (full graph) ...", flush=True)
            saved_forwards = {}
            for l in train_layers:
                attn_mod = llama.model.layers[l].self_attn
                saved_forwards[l] = attn_mod.forward
                attn_mod.forward = make_sparse_self_attn_forward(sparse_layers[l])

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
            kl_loss_val = lk.item()
            if microstep == 0:
                print("  [kl] backward ...", flush=True)
            (args.logit_kl_weight * lk / args.grad_accum).backward()
            if microstep == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print("  [kl] backward OK", flush=True)

        # 记录分项 loss（未经 grad_accum 缩放的「每层平均 MSE」）
        acc_mse += mse_loss_avg
        acc_kl  += kl_loss_val
        acc_n   += 1
        if microstep < args.grad_accum:
            print(f"[progress] microstep {microstep} backward done "
                  f"({time.time()-t_loop:.0f}s since loop start)", flush=True)

        if (microstep + 1) % args.grad_accum == 0:
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step        += 1
            tokens_seen += args.batch * args.seqlen * args.grad_accum
            cur_lr = scheduler.get_last_lr()[0]
            avg_mse = acc_mse / max(acc_n, 1)
            avg_kl  = acc_kl  / max(acc_n, 1)

            if step == 1:
                elapsed = time.time() - t0
                kl_s = f" kl={avg_kl:.4f}" if args.logit_kl_weight > 0 else ""
                print(f"step={step:6d}/{total_steps} | mse={avg_mse:.4f}{kl_s} | "
                      f"lr={cur_lr:.2e} | "
                      f"tokens={tokens_seen/1e6:.1f}M", flush=True)
            if step % 4 == 0:
                elapsed   = time.time() - t0
                tok_per_s = (args.batch * args.seqlen * args.grad_accum * 4) / elapsed
                remaining_tokens = max(0, args.max_tokens - tokens_seen)
                eta_min = (remaining_tokens / tok_per_s / 60) if tok_per_s > 0 else float('inf')
                kl_s = f" kl={avg_kl:.4f}" if args.logit_kl_weight > 0 else ""
                print(f"step={step:6d}/{total_steps} | mse={avg_mse:.4f}{kl_s} | "
                      f"lr={cur_lr:.2e} | tok/s={tok_per_s:.0f} | "
                      f"tokens={tokens_seen/1e6:.1f}M / {args.max_tokens/1e6:.0f}M | "
                      f"ETA {eta_min:.0f}min", flush=True)
                acc_mse = 0.0
                acc_kl  = 0.0
                acc_n   = 0
                t0      = time.time()

            if step % args.save_steps == 0:
                _save_checkpoint(args, sparse_layers, optimizer, train_layers,
                                 step, f"step_{step:06d}")

            if tokens_seen >= args.max_tokens:
                break
            if _interrupted:
                break

    # 最终保存
    _save_checkpoint(args, sparse_layers, optimizer, train_layers, step, "final")
    print(f"\nTraining done. Final checkpoint: {os.path.join(args.save_dir, 'final')}")
    print(f"Total tokens: {tokens_seen/1e6:.1f}M, steps: {step}")


def _save_checkpoint(args, sparse_layers, optimizer, train_layers, step, name):
    """保存 checkpoint 到 args.save_dir/name/ckpt.pt"""
    ckpt_path = os.path.join(args.save_dir, name)
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save({
        "step": step,
        "nsa": {l: sparse_layers[l].fsa.state_dict() for l in train_layers},
        "optimizer": optimizer.state_dict(),
        "train_layers": train_layers,
        "train_qkvo": getattr(args, "train_qkvo", False),
        "attn_mode": "FSA" if args.fsa else "NSA",
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


if __name__ == "__main__":
    main()

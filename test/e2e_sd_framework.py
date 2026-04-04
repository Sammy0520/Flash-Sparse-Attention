"""
Flash-Sparse-Attention end-to-end SD framework.

Implements three systems with a unified interface:
  1) DenseARRunner  : dense autoregressive baseline (1 token/step)
  2) DenseSDRunner  : draft+target dense speculative decoding
  3) NSASDRunner    : draft + NSA target speculative decoding

The script follows docs/e2e.md as closely as possible for:
  - multi-round SD loop
  - stochastic accept/reject sampling
  - cache growth/trim rules
  - unified metrics output
"""

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from nsa_ref.module import RopeConfig
from nsa_ref.ops import linear_compress
from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode
from fsa_preview.ops import _linear_compress_decode


LLAMA_8B = (
    "/root/autodl-tmp/models/Llama-3.1-8B-Instruct"
)


def clamp_token_id(token_id: int, vocab_size: int) -> int:
    return max(0, min(int(token_id), vocab_size - 1))


def clone_legacy_past_key_values(past_key_values):
    """Shallow clone per-layer K/V tensors for one-off dense forward (debug compare)."""
    return tuple((k.clone(), v.clone()) for k, v in past_key_values)


def sanitize_token_tensor(ids: torch.Tensor, vocab_size: int, strict: bool) -> torch.Tensor:
    """Clamp ids to [0, vocab_size-1] or (if strict) raise before embedding lookup."""
    bad = (ids < 0) | (ids >= vocab_size)
    if bad.any():
        if strict:
            bad_idx = bad.nonzero(as_tuple=False)
            raise ValueError(
                f"token id out of range [0, {vocab_size}): "
                f"ids={ids}, bad positions={bad_idx.tolist()}"
            )
        ids = torch.clamp(ids, 0, vocab_size - 1)
    return ids


@dataclass
class GenerationStats:
    total_time: float
    total_generated_tokens: int
    tokens_per_second: float
    prefill_time: float
    decode_time: float
    total_drafted_tokens: int = 0
    total_accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    avg_accept_len: float = 0.0
    num_rounds: int = 0


def sample_from_logits(logits: torch.Tensor, temperature: float, vocab_size: int) -> int:
    if temperature <= 0:
        return clamp_token_id(logits.argmax(dim=-1).item(), vocab_size)
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(probs.sum().item())
    if s <= 0:
        return clamp_token_id(logits.argmax(dim=-1).item(), vocab_size)
    probs = probs / s
    return clamp_token_id(torch.multinomial(probs, 1).item(), vocab_size)


def cache_seq_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    k, _ = past_key_values[0]
    return int(k.shape[2])


def trim_legacy_cache(past_key_values, keep_len: int):
    trimmed = []
    for k, v in past_key_values:
        trimmed.append((k[:, :, :keep_len, :].contiguous(), v[:, :, :keep_len, :].contiguous()))
    return tuple(trimmed)


def build_prompt_ids(tokenizer, prompt: str, seqlen: int, device: str) -> torch.Tensor:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if ids.shape[1] >= seqlen:
        return ids[:, :seqlen]
    reps = seqlen // ids.shape[1] + 2
    return ids.repeat(1, reps)[:, :seqlen]


class SpeculativeSampler:
    """
    【推测采样验证器】根据草稿和目标logits，决定哪些token被接受
    
    核心原理(推测解码算法)：
    给定草稿模型生成的n个候选token，用目标模型的logits来验证和纠正。
    
    采样方案(最常用的接受方案)：
    对于每个位置i的候选token t_i：
      1. 计算接受概率: α = min(1, p_target(t_i) / p_draft(t_i))
         - p_target > p_draft时总是接受(概率=1)
         - p_target < p_draft时以一定概率拒绝
      2. 如果接受，继续到下一位置
      3. 如果拒绝，从差分分布中重新采样，然后停止验证
      4. 如果全部接受，从目标模型的下一项(第n+1位)bonus采样一个额外token
    
    这样既保持了草稿模型的高效性，也保证了最终输出遵循目标分布。
    """
    
    def __init__(self, temperature: float = 1.0, vocab_size: int = 128256):
        """
        参数：
        - temperature: 采样温度(softmax时的温度系数)
        - vocab_size: 词表大小
        """
        self.temperature = temperature
        self.vocab_size = vocab_size

    def verify(
        self,
        draft_ids: torch.Tensor,      # [N] 草稿生成的token ids
        target_logits: torch.Tensor,  # [N+1, vocab] 目标logits(包括bonus position)
        draft_logits: torch.Tensor,   # [N, vocab] 草稿logits
    ) -> Tuple[int, List[int]]:
        """
        【验证和采样】根据logits执行推测解码采样
        
        工作流程：
        1. 将logits非规范化为概率(softmax)
        2. 逐位置与候选token对比
        3. 根据概率比值决定接受/拒绝
        4. 如果全接受则bonus采样；否则拒绝位置进行修正采样
        
        参数：
        - draft_ids: [N] 草稿模型生成的候选token ids
        - target_logits: [N+1, vocab] 目标模型的logits
          - target_logits[0:N]用于验证前N个位置
          - target_logits[N]是bonus token的logits
        - draft_logits: [N, vocab] 草稿模型的logits
        
        返回：
        - (accepted_count, accepted_ids_list)
          - accepted_count: 被接受的token数(不含bonus)
          - accepted_ids_list: 接受的token id列表(可能包含修正的和bonus)
        """
        n = draft_ids.shape[0]
        vs = self.vocab_size
        
        # 【Logit清理】NSA可能产生NaN/Inf，需要在softmax前清理
        # 确保数值稳定性
        t_logits = torch.nan_to_num(target_logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        d_logits = torch.nan_to_num(draft_logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 【概率计算】logits -> 概率分布
        t_probs = torch.softmax(t_logits / max(self.temperature, 1e-6), dim=-1)
        d_probs = torch.softmax(d_logits / max(self.temperature, 1e-6), dim=-1)

        accepted = 0
        accepted_ids: List[int] = []
        
        # 【主验证循环】对每个位置的候选token进行验证
        for i in range(n):
            # 1. 获取第i个位置的候选token
            tok = clamp_token_id(int(draft_ids[i].item()), vs)
            p = float(t_probs[i, tok].item())  # 目标在该token的概率
            q = float(d_probs[i, tok].item())  # 草稿在该token的概率
            
            # 2. 计算接受概率: min(1, p/q)
            # 如果p > q，比值>1，所以α=1，总是接受
            # 如果p < q，比值<1，以此概率接受(实现"correction")
            acc_prob = min(1.0, p / max(q, 1e-8))
            
            # 3. 随机决定是否接受(以acc_prob的概率)
            if random.random() < acc_prob:
                # 接受这个token，继续验证下一个
                accepted += 1
                accepted_ids.append(tok)
                continue
            
            # 4. 拒绝这个token：需要进行修正采样
            # 修正分布 = max(0, p_target - p_draft)
            # 这确保修正后的分布的支撑完全被目标包含
            diff = torch.clamp(t_probs[i] - d_probs[i], min=0.0)
            diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            diff_sum = float(diff.sum().item())
            
            # 从修正分布采样一个新token作为第i位置的输出
            corrected = (
                clamp_token_id(int(torch.multinomial(diff / diff_sum, 1).item()), vs)
                if diff_sum > 0
                else clamp_token_id(int(t_probs[i].argmax().item()), vs)
            )
            accepted_ids.append(corrected)
            
            # 【提前终止】拒绝发生，验证到此torch.multinomial 是什么停止
            return accepted, accepted_ids

        # 【全部接受的情况】前n个都被接受了
        # bonus采样：从第n+1位置(超出草稿长度的)额外采样一个token
        bonus_row = torch.nan_to_num(t_probs[n], nan=0.0, posinf=0.0, neginf=0.0)
        bonus_sum = float(bonus_row.sum().item())
        bonus = (
            clamp_token_id(int(torch.multinomial(bonus_row / bonus_sum, 1).item()), vs)
            if bonus_sum > 0
            else clamp_token_id(int(t_logits[n].argmax().item()), vs)
        )
        accepted_ids.append(bonus)
        return accepted, accepted_ids


class LlamaNSALayer(nn.Module):
    """
    将Llama原始的全注意力机制替换为稀疏注意力(NSA)模块
    
    工作原理：
    - 用FlashSparseAttentionDecode实现高效的稀疏注意力计算
    - 从原Llama注意力层复制权重参数(Q、K、V、Output投影矩阵)
    - 支持在推测解码时使用稀疏模式加快计算
    """
    
    def __init__(
        self,
        llama_attn,
        cfg,
        topk=16,                 # 稀疏选择的top-k注意力头数
        block_size=64,           # 一个块的大小
        kernel_size=32,          # 压缩核的大小
        kernel_stride=16,        # 压缩核的步长
        init_blocks=1,           # 初始保留的块数
        local_blocks=2,          # 本地窗口保留的块数
        window_size=512,         # 本地窗口大小
    ):
        super().__init__()
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_d = getattr(cfg, "head_dim", cfg.hidden_size // num_q)
        rope_cfg = RopeConfig(
            max_position_embeddings=cfg.max_position_embeddings,
            head_dim=head_d,
            rope_theta=cfg.rope_theta,
            rope_scaling=getattr(cfg, "rope_scaling", None),
        )
        # 创建稀疏注意力模块(包括压缩、稀疏选择、RoPE等)
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
        
        # 【关键】从原Llama注意力层复制权重参数，确保语义相同
        # 这样NSA层可以直接替换原注意力层
        with torch.no_grad():
            self.fsa.proj_q.weight.copy_(llama_attn.q_proj.weight)
            self.fsa.proj_k.weight.copy_(llama_attn.k_proj.weight)
            self.fsa.proj_v.weight.copy_(llama_attn.v_proj.weight)
            self.fsa.proj_o.weight.copy_(llama_attn.o_proj.weight)
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride

    def build_compressed_cache(self, k_raw, v_raw, cu_k):
        """
        【缓存压缩】将原始KV缓存压缩成稀疏表示
        
        工作流程：
        1. 用线性压缩层(compress_key/compress_value)将原始KV压缩
        2. 压缩后的缓存更小，后续注意力计算会更快
        
        参数：
        - k_raw, v_raw: 原始未压缩的KV缓存 [seq_len, num_kv_heads, head_dim]
        - cu_k: cumulative length索引 [batch_start, batch_end]
        
        返回：
        - cmp_k, cmp_v: 压缩后的KV缓存
        """
        cmp_k, _ = linear_compress(
            k_raw,
            self.fsa.compress_key,      # 压缩线性变换矩阵
            cu_k,
            self.kernel_size,
            self.kernel_stride,
            self.fsa.intra_block_pe,    # 块内位置编码
        )
        cmp_v, _ = linear_compress(
            v_raw,
            self.fsa.compress_value,    # 压缩线性变换矩阵
            cu_k,
            self.kernel_size,
            self.kernel_stride,
            None,
        )
        return cmp_k, cmp_v

    def forward(self, hidden, k_raw, k_buffer, v_raw, cmp_k, cmp_v, cu_q, cu_k, position_ids, kv_commit_stash=None):
        """
        【NSA前向传播】稀疏注意力的计算
        
        工作流程：
        1. 接收当前token的隐藏状态
        2. 用压缩的KV缓存和top-k稀疏选择进行注意力计算
        3. 可选：将新的KV投影(k_new_rope, v_new)保存到stash，以便后续缓存更新
        
        参数：
        - hidden: 当前token的隐藏状态
        - k_raw, v_raw: 历史KV的原始形式(用于扩展)
        - cmp_k, cmp_v: 压缩的KV缓存
        - cu_q, cu_k: query/key的cumulative长度索引
        - position_ids: 当前token的位置编码id
        - kv_commit_stash: 可选的列表，用来保存新的KV张量供后续缓存更新
        """
        return self.fsa(
            x=hidden,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            k_cache=k_raw,
            k_buffer=k_buffer,
            v_cache=v_raw,
            cmp_k_cache=cmp_k,
            cmp_v_cache=cmp_v,
            attention_mask=None,
            position_ids=position_ids,
            kv_commit_stash=kv_commit_stash,
        )


class NSATargetModel:
    """
    【目标模型管理器】管理大型LLM(目标模型)的KV缓存和验证逻辑
    
    核心职责：
    1. 维护所有层的K/V缓存(原始和压缩形式)
    2. 在"验证模式"下用NSA替换原始注意力机制
    3. 处理缓存提交：将验证阶段产生的新KV追加到缓存
    
    关键理念：
    - 为了算力效率，验证多个候选token时用稀疏注意力而不是全稠密注意力
    - 通过Hook机制动态替换各层的前向传播函数
    """
    
    def __init__(self, llama, nsa_layers: List[LlamaNSALayer]):
        """
        初始化目标模型管理器
        
        参数：
        - llama: 原始的Llama模型(包含所有层)
        - nsa_layers: NSA替换层的列表(长度=num_hidden_layers)
        """
        self.llama = llama
        self.nsa_layers = nsa_layers
        
        # 【缓存存储】为每一层维护K/V的两种形式
        self.k_raw = [None] * len(nsa_layers)      # 原始未压缩的rope(K) [seq_len, num_kv_heads, head_dim]
        self.k_buffer = [None] * len(nsa_layers)   # 未压缩的K
        self.v_raw = [None] * len(nsa_layers)      # 原始未压缩的V
        self.cmp_k = [None] * len(nsa_layers)      # 压缩后的K缓存(用于稀疏注意力)
        self.cmp_v = [None] * len(nsa_layers)      # 压缩后的V缓存
        
        self.past_len = 0                          # 当前缓存中已有的token总数
        self._orig_forwards = {}                   # 保存各层原始的前向函数(用于恢复)
        self._verify_mode = False                  # 标志：当前是否在验证阶段
        self._hooks_patched = False                # 标志：Hook是否已安装
        self._pos_ids = None                       # 当前验证批次的position ids
        
        # 【关键】验证阶段产生的新KV投影，待提交到缓存
        # 结构: [(k_new_rope, v_new), ...] for each layer
        self._stash_kv_new: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(nsa_layers)
        
        # 【优化】复用cu张量，避免重复分配
        self._cu_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def init_from_prefill(self, past_key_values, k_nope_storage = None):
        """
        【初始化缓存】从预填充(prefill)阶段的密集KV缓存初始化NSA缓存
        
        工作流程：
        1. 提取预填充后的Llama缓存(HF格式: [batch=1, num_heads, seq_len, head_dim])
        2. 转换为NSA格式: [seq_len, num_heads, head_dim] (去掉batch维，转置序列和头维)
        3. 对每一层进行压缩，得到高效的稀疏缓存
        
        参数：
        - past_key_values: Llama模型返回的密集KV缓存(来自prefill阶段)
        """
        device = self.nsa_layers[0].fsa.proj_q.weight.device
        dtype = self.nsa_layers[0].fsa.proj_q.weight.dtype
        self.past_len = cache_seq_len(past_key_values)  # 记录预填充长度
        
        # 逐层处理KV缓存
        for l, nsa in enumerate(self.nsa_layers):
            k, v = past_key_values[l]  # HF格式: [1, num_heads, seq_len, head_dim]
            
            # 【转换】从HF格式转为NSA格式
            k_raw = k.squeeze(0).permute(1, 0, 2).contiguous().to(dtype)  
            # squeeze(0): 去掉batch维 
            # permute(1, 0, 2): [num_heads, seq_len, head_dim] -> [seq_len, num_heads, head_dim]
            
            v_raw = v.squeeze(0).permute(1, 0, 2).contiguous().to(dtype)
            
            # 创建cumulative索引: [0, k_raw.shape[0]] 表示整个序列
            cu_k = torch.tensor([0, k_raw.shape[0]], device=device, dtype=torch.int32)
            
            # 【压缩】对原始KV进行压缩
            if k_nope_storage is None:
                position_ids = torch.arange(0, k_raw.shape[0], device=k_raw.device)
                k_raw_nope = nsa.fsa.rope(
                    k_raw,
                    cu_k,
                    position_ids=-position_ids,
                )
            else:
                k_raw_nope = k_nope_storage[l]
                # 兼容两种格式:
                #   - 旧格式 (from HF hook): [1, seq_len, num_kv_heads * head_dim]
                #   - 新格式 (from NSA prefill): [seq_len, num_kv_heads, head_dim]
                if k_raw_nope.dim() == 2:
                    # 旧格式: [1, seq_len, nkv*hd] -> 先去 batch -> [seq_len, nkv*hd]
                    seq_len = k_raw_nope.shape[0]
                    k_raw_nope = k_raw_nope.view(seq_len, nsa.fsa.num_kv_heads, nsa.fsa.head_dim)
                elif k_raw_nope.dim() == 3 and k_raw_nope.shape[0] == 1:
                    # 旧格式 with batch dim: [1, seq_len, nkv*hd]
                    seq_len = k_raw_nope.shape[1]
                    k_raw_nope = k_raw_nope.view(seq_len, nsa.fsa.num_kv_heads, nsa.fsa.head_dim)
                # else: 已经是 [seq_len, num_kv_heads, head_dim]，无需变换

            cmp_k, cmp_v = nsa.build_compressed_cache(k_raw_nope, v_raw, cu_k)

            # 【保存】存储原始和压缩缓存
            # I guess k_raw here has rope applied.
            buffer_size = min(nsa.kernel_size - 1, k_raw.shape[0])
            # unapply rope
            self.k_buffer[l] = k_raw_nope[-buffer_size:].contiguous()

            self.k_raw[l] = k_raw
            self.v_raw[l] = v_raw
            self.cmp_k[l] = cmp_k
            self.cmp_v[l] = cmp_v

    def append_from_dense_cache(self, new_past_key_values, old_len: int, commit_len: int):
        """
        【可选的混合模式】将密集HF缓存的增量与NSA缓存对齐(调试/混合方案用)
        
        【用途】
        这个方法在纯NSA-SD流程中NOT被使用，主要用于：
        - 调试：对比NSA缓存和密集缓存是否一致
        - 混合方案：某些层用密集注意力，某些层用NSA
        
        【工作流程】
        1. 从new_past_key_values中提取增量部分[old_len : old_len+commit_len]
        2. 转换格式从HF [1, num_heads, seq, dim] 到 NSA [seq, num_heads, dim]
        3. 追加到k_raw/v_raw
        4. 重新压缩整个缓存得到完整的cmp_k/cmp_v
        
        参数：
        - new_past_key_values: 完整的HF格式缓存(包含历史和新部分)
        - old_len: 增量之前的缓存长度
        - commit_len: 要提交的新token数
        """
        if commit_len <= 0:
            return
        device = self.nsa_layers[0].fsa.proj_q.weight.device
        for l, nsa in enumerate(self.nsa_layers):
            k_full, v_full = new_past_key_values[l]
            # 【提取增量】只取新增的部分
            k_delta = k_full[:, :, old_len : old_len + commit_len, :]
            v_delta = v_full[:, :, old_len : old_len + commit_len, :]
            
            # 【转换格式】HF -> NSA格式
            k_raw_delta = k_delta.squeeze(0).permute(1, 0, 2).contiguous()
            v_raw_delta = v_delta.squeeze(0).permute(1, 0, 2).contiguous()

            # 【追加】如果有新KV，追加到缓存并重新压缩
            if k_raw_delta.shape[0] > 0:
                self.k_raw[l] = torch.cat([self.k_raw[l], k_raw_delta], dim=0)
                self.v_raw[l] = torch.cat([self.v_raw[l], v_raw_delta], dim=0)
                # 重新压缩整个缓存
                cu_full = torch.tensor([0, self.k_raw[l].shape[0]], device=device, dtype=torch.int32)
                cmp_k_full, cmp_v_full = nsa.build_compressed_cache(self.k_raw[l], self.v_raw[l], cu_full)
                self.cmp_k[l] = cmp_k_full
                self.cmp_v[l] = cmp_v_full
        self.past_len += commit_len

    def commit_after_verify(self, commit_len: int):
        """
        【缓存提交】将验证阶段产生的新KV追加到NSA缓存
        
        【背景】验证阶段(verify)时，NSA层通过Hook被调用，会产生新的KV投影。
        这些新KV被保存在_stash_kv_new中，现在需要官方提交到缓存系统。
        
        【工作流程】：
        1. 从stash中提取验证阶段产生的k_new_rope(带RoPE)和v_new(原始)
        2. 用incremental compression将新KV压缩(利用缓存的一部分作为context)
        3. 将压缩的新KV追加到cmp_k/cmp_v
        4. 将原始的新KV追加到k_raw/v_raw
        5. 更新缓存长度计数
        
        【参数】：
        - commit_len: 要提交的新token数(包括前一个token)
        """
        if commit_len <= 0:
            return
        dtype = self.k_raw[0].dtype
        
        # 逐层处理
        for l, nsa in enumerate(self.nsa_layers):
            # 【提取新KV】从验证阶段的stash中取出
            st = self._stash_kv_new[l]
            if st is None:
                raise RuntimeError("commit_after_verify: missing stashed KV; verify() did not run?")
            k_new_rope, k_new, v_new = st

            # 只取有效的部分(commit_len个token)
            k_new_rope = k_new_rope[:commit_len].contiguous().to(dtype)
            k_new = k_new[:commit_len].contiguous().to(dtype)
            v_new = v_new[:commit_len].contiguous().to(dtype)

            # 【准备增量压缩的context】
            # 用缓存末尾的kernel_size-1个KV作为context，确保压缩的连续性
            prev_raw_len = self.k_raw[l].shape[0]
            buffer_size = min(nsa.kernel_size - 1, prev_raw_len)
            if buffer_size > 0:
                v_buffer = self.v_raw[l][-buffer_size:]
                k_buffer = self.k_buffer[l]
                assert buffer_size <= k_buffer.shape[0]
            else:
                k_buffer = v_buffer = None

            # 【增量压缩】针对新KV apply incremental compression
            # _linear_compress_decode利用缓存context进行压缩
            decode_k = _linear_compress_decode(
                k_new,
                nsa.fsa.compress_key,
                nsa.kernel_size,
                nsa.kernel_stride,
                nsa.fsa.intra_block_pe,
                prev_raw_len,  # 告诉算子缓存的当前长度
                k_buffer,    # context KV
            )
            decode_v = _linear_compress_decode(
                v_new,
                nsa.fsa.compress_value,
                nsa.kernel_size,
                nsa.kernel_stride,
                None,
                prev_raw_len,
                v_buffer,
            )

            # 【追加压缩缓存】
            if decode_k is not None:
                self.cmp_k[l] = torch.cat([self.cmp_k[l], decode_k], dim=0) if self.cmp_k[l] is not None else decode_k
                self.cmp_v[l] = torch.cat([self.cmp_v[l], decode_v], dim=0) if self.cmp_v[l] is not None else decode_v

            # 【追加原始缓存】
            self.k_raw[l] = torch.cat([self.k_raw[l], k_new_rope], dim=0) if self.k_raw[l] is not None else k_new_rope
            self.v_raw[l] = torch.cat([self.v_raw[l], v_new], dim=0) if self.v_raw[l] is not None else v_new
            self.k_buffer[l] = torch.cat([self.k_buffer[l], k_new], dim=0) if self.k_buffer[l] is not None else k_new
            self.k_buffer[l] = self.k_buffer[l][-buffer_size:]
        
        # 【更新缓存长度】
        self.past_len += commit_len
        # 【清空stash】当前stash已被消费，准备下一轮
        self._stash_kv_new = [None] * len(self.nsa_layers)

    def _make_nsa_forward(self, layer_idx: int):
        """
        【Hook工厂】为某一层创建自定义前向函数，在验证模式下用NSA替换标准注意力
        
        核心思想：
        - 返回一个闭包(closure)，能访问自己的层索引和NSA层
        - 当模型前向传播时，该函数被调用替代原始注意力函数
        - verify_mode=True时：用NSA(稀疏)计算；=False时：调用原始注意力
        
        返回：
        - nsa_forward: 替换函数，签名与原始注意力层一致
        """
        target = self
        nsa = self.nsa_layers[layer_idx]

        def nsa_forward(hidden_states=None, *args, _lid=layer_idx, **kwargs):
            """
            自定义前向函数，在验证模式下用NSA稀疏注意力
            
            参数：
            - hidden_states: 当前层的输入 [batch=1, seq_len, hidden_size]
            - *args, **kwargs: 原始注意力函数的其他参数
            - _lid: 层索引(通过闭包传入)
            
            逻辑：
            1. 如果不在验证模式，调用原始注意力(回退到标准方案)
            2. 否则，用NSA计算：
               - 获取序列长度
               - 构建query/key的cumulative索引
               - 调用NSA层，并从中提取新的KV投影
               - 保存这些新KV供后续缓存提交
            """
            # 【回退】如果不在验证模式，直接使用原始注意力
            if not target._verify_mode:
                return target._orig_forwards[_lid](hidden_states, *args, **kwargs)
            
            # 【验证模式】用NSA计算，下面分解步骤：
            
            # 1. 获取输入序列长度
            n = hidden_states.shape[1]  # [batch=1, n, hidden_size] -> n个新token
            hidden_flat = hidden_states.squeeze(0)  # [n, hidden_size]
            
            # 2. 计算总的key长度 = 历史缓存长度 + 新token数
            past_kv_len = target.k_raw[_lid].shape[0]  # 历史缓存中K的序列长度
            total_k_len = past_kv_len + n              # 总长度
            
            # 3. 构建cumulative索引(用于压缩核和稀疏计算)
            cu_q = target._get_cu(hidden_states.device, n)              # query: [0, n]
            cu_k = target._get_cu(hidden_states.device, total_k_len)    # key: [0, total_k_len]
            
            # 4. 【关键】调用NSA层
            # kv_commit_stash是一个列表，NSA层会在其中放入新的KV投影
            stash: List = []
            out = nsa(
                hidden_flat,
                target.k_raw[_lid],      # 历史原始K缓存
                target.k_buffer[_lid],
                target.v_raw[_lid],      # 历史原始V缓存
                target.cmp_k[_lid],      # 历史压缩K缓存
                target.cmp_v[_lid],      # 历史压缩V缓存
                cu_q,
                cu_k,
                target._pos_ids,         # 当前验证批的position ids
                kv_commit_stash=stash,   # 接收新的KV投影
            )
            
            # 5. 提取新的KV投影(k_new_rope已带RoPE, v_new是原始值)
            if len(stash) != 1:
                raise RuntimeError(
                    f"kv_commit_stash expected 1 tuple, got {len(stash)}"
                )
            k_new_rope, k_new, v_new = stash[0]
            # 保存这些新KV，供后续commit_after_verify使用
            target._stash_kv_new[_lid] = (k_new_rope.detach(), k_new.detach(), v_new.detach())

            # 6. 返回格式转换：NSA返回[n, hidden_size]，需要转为[batch=1, n, hidden_size]和两个None(缓存)
            return (out.unsqueeze(0), None, None)

        return nsa_forward

    def _patch_attn(self):
        """
        【安装Hook】替换所有层的注意力函数为NSA版本
        
        机制：
        - 遍历Llama模型的每一层
        - 保存原始的注意力前向函数到_orig_forwards(用于恢复)
        - 用_make_nsa_forward创建的函数替换原函数
        
        这样当模型前向传播时，会使用NSA而不是标准注意力
        """
        for l, layer in enumerate(self.llama.model.layers):
            self._orig_forwards[l] = layer.self_attn.forward  # 保存原始函数
            layer.self_attn.forward = self._make_nsa_forward(l)  # 用NSA版本替换
        self._hooks_patched = True

    def _restore_attn(self):
        """
        【卸载Hook】恢复所有层的原始注意力函数
        
        用途：
        - 验证阶段结束后调用
        - 或者用于调试/对比实验
        """
        for l, layer in enumerate(self.llama.model.layers):
            layer.self_attn.forward = self._orig_forwards[l]
        self._orig_forwards.clear()
        self._hooks_patched = False

    def _get_cu(self, device: torch.device, end: int) -> torch.Tensor:
        """
        【获取或创建cumulative索引张量】用于压缩和稀疏计算
        
        cumulative长度张量表示："当前batch的起始=0，结束=end"
        用于告诉压缩算子和稀疏注意力"处理多少个token"

        参数：
        - device: 张量所在的设备
        - end: cumulative长度的终点值
        
        返回：
        - 张量[0, end]，数据类型int32
        
        优化：
        - 缓存已创建过的cu张量，避免重复分配
        """
        key = (int(end), device.index if device.type == "cuda" else -1)
        t = self._cu_cache.get(key)
        if t is None:
            t = torch.tensor([0, int(end)], device=device, dtype=torch.int32)
            self._cu_cache[key] = t
        return t

    def begin_verify(self):
        """【验证开始】安装Hook，准备进入验证模式"""
        if not self._hooks_patched:
            self._patch_attn()

    def end_verify(self):
        """【验证结束】卸载Hook，恢复正常注意力"""
        if self._hooks_patched:
            self._restore_attn()
        self._verify_mode = False

    @torch.no_grad()
    def verify(self, verify_ids: torch.Tensor) -> torch.Tensor:
        """
        【验证前向传播】用NSA稀疏注意力验证多个候选token的logits
        
        工作流程：
        1. 准备position ids：从past_len到past_len+n(新token的位置)
        2. 打开验证模式标志(verify_mode=True)
        3. 调用Llama模型的前向传播
           - 所有注意力层会使用Hook调用NSA版本而不是标准注意力
           - NSA版本返回稀疏注意力输出，显著降低计算量
           - NSA层会将新的KV投影保存到_stash_kv_new供后续提交
        4. 关闭验证模式(verify_mode=False)
        5. 返回最后一层的logits
        
        参数：
        - verify_ids: 验证输入，形状[1, n+1]
                   第一个是前一个token，后面n个是候选token
        
        返回：
        - logits: [n+1, vocab_size] 所有n+1个token的logits输出
        """
        n = verify_ids.shape[1]
        device = verify_ids.device
        
        # 【重置stash】准备接收新的KV投影
        self._stash_kv_new = [None] * len(self.nsa_layers)
        
        # 【构建position ids】
        # 从past_len(当前缓存长度)开始，生成n个连续的位置编码
        # 这告诉RoPE每个token的绝对位置，确保位置编码正确
        self._pos_ids = torch.arange(self.past_len, self.past_len + n, device=device, dtype=torch.long)
        
        # 【开启验证模式】标记Hook应该使用NSA而不是标准注意力
        self._verify_mode = True

        # 【前向传播】Llama模型的标准前向，但所有注意力被Hook替换为NSA
        # 传入正确的 position_ids，确保模型内部生成的 causal mask 和
        # position_embeddings 与真实位置一致（即使 NSA hook 自行计算 RoPE）。
        out = self.llama(
            verify_ids,
            position_ids=self._pos_ids.unsqueeze(0),
            use_cache=False,
            num_logits_to_keep=n,
        )
        
        # 【关闭验证模式】恢复正常
        self._verify_mode = False
        
        # 【返回logits】[1, n, vocab_size] -> [n, vocab_size]
        return out.logits.squeeze(0)


class DenseARRunner:
    def __init__(self, model, tokenizer, temperature: float = 1.0, strict_token_check: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self._vocab = model.config.vocab_size
        self.strict_token_check = strict_token_check

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill = self.model(input_ids, use_cache=True, num_logits_to_keep=1)
        cur = sample_from_logits(prefill.logits[:, -1, :], self.temperature, self._vocab)
        cache = prefill.past_key_values
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        generated = []
        for _ in range(max_new_tokens):
            generated.append(cur)
            inp = sanitize_token_tensor(
                torch.tensor([[cur]], device=input_ids.device, dtype=torch.long),
                self._vocab,
                self.strict_token_check,
            )
            out = self.model(inp, past_key_values=cache, use_cache=True, num_logits_to_keep=1)
            cache = out.past_key_values
            cur = sample_from_logits(out.logits[:, -1, :], self.temperature, self._vocab)

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        prefill_t = t1 - t0
        decode_t = t2 - t1
        total_t = t2 - t0
        stats = GenerationStats(
            total_time=total_t,
            total_generated_tokens=len(generated),
            tokens_per_second=len(generated) / max(decode_t, 1e-6),
            prefill_time=prefill_t,
            decode_time=decode_t,
        )
        return generated, stats


class DenseSDRunner:
    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        n_draft=8,
        temperature=1.0,
        strict_token_check: bool = False,
    ):
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.n_draft = n_draft
        self.temperature = temperature
        self.strict_token_check = strict_token_check
        self._vocab = min(
            int(target_model.config.vocab_size),
            int(draft_model.config.vocab_size),
        )
        self.sampler = SpeculativeSampler(temperature=temperature, vocab_size=self._vocab)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int):
        device = input_ids.device
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_prefill = self.target(input_ids, use_cache=True, num_logits_to_keep=1)
        d_prefill = self.draft(input_ids, use_cache=True, num_logits_to_keep=1)
        cur = sample_from_logits(t_prefill.logits[:, -1, :], self.temperature, self._vocab)
        target_cache = t_prefill.past_key_values
        draft_cache = d_prefill.past_key_values
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        generated = []
        drafted_total = 0
        accepted_total = 0
        rounds = 0

        while len(generated) < max_new_tokens:
            rounds += 1
            remain = max_new_tokens - len(generated)
            n = min(self.n_draft, remain)
            old_t_len = cache_seq_len(target_cache)
            old_d_len = cache_seq_len(draft_cache)

            draft_ids: List[int] = []
            draft_logits: List[torch.Tensor] = []
            d_cur = sanitize_token_tensor(
                torch.tensor([[cur]], device=device, dtype=torch.long),
                self._vocab,
                self.strict_token_check,
            )
            for _ in range(n):
                d_out = self.draft(d_cur, past_key_values=draft_cache, use_cache=True, num_logits_to_keep=1)
                draft_cache = d_out.past_key_values
                logit = d_out.logits[:, -1, :].squeeze(0).float()
                nxt = sample_from_logits(logit, self.temperature, self._vocab)
                draft_ids.append(nxt)
                draft_logits.append(logit)
                d_cur = sanitize_token_tensor(
                    torch.tensor([[nxt]], device=device, dtype=torch.long),
                    self._vocab,
                    self.strict_token_check,
                )

            draft_ids_t = torch.tensor(draft_ids, device=device, dtype=torch.long)
            draft_logits_t = torch.stack(draft_logits, dim=0)

            verify_input = sanitize_token_tensor(
                torch.tensor([[cur] + draft_ids], device=device, dtype=torch.long),
                self._vocab,
                self.strict_token_check,
            )
            t_out = self.target(
                verify_input,
                past_key_values=target_cache,
                use_cache=True,
                num_logits_to_keep=n + 1,
            )
            target_logits = t_out.logits.squeeze(0).float()
            full_target_cache = t_out.past_key_values

            accept_len, accepted_ids = self.sampler.verify(draft_ids_t, target_logits, draft_logits_t)
            commit_len = 1 + accept_len

            target_cache = trim_legacy_cache(full_target_cache, old_t_len + commit_len)
            if accept_len == n:
                extra = self.draft(
                    sanitize_token_tensor(
                        torch.tensor([[draft_ids[-1]]], device=device, dtype=torch.long),
                        self._vocab,
                        self.strict_token_check,
                    ),
                    past_key_values=draft_cache,
                    use_cache=True,
                    num_logits_to_keep=1,
                )
                draft_cache = extra.past_key_values
            draft_cache = trim_legacy_cache(draft_cache, old_d_len + commit_len)

            generated.extend(accepted_ids)
            drafted_total += n
            accepted_total += accept_len
            cur = clamp_token_id(accepted_ids[-1], self._vocab)
            if self.tokenizer.eos_token_id is not None and cur == self.tokenizer.eos_token_id:
                break

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        prefill_t = t1 - t0
        decode_t = t2 - t1
        total_t = t2 - t0
        stats = GenerationStats(
            total_time=total_t,
            total_generated_tokens=len(generated),
            tokens_per_second=len(generated) / max(decode_t, 1e-6),
            prefill_time=prefill_t,
            decode_time=decode_t,
            total_drafted_tokens=drafted_total,
            total_accepted_tokens=accepted_total,
            acceptance_rate=accepted_total / max(drafted_total, 1),
            avg_accept_len=accepted_total / max(rounds, 1),
            num_rounds=rounds,
        )
        return generated[:max_new_tokens], stats


class NSASDRunner(DenseSDRunner):
    """
    【NSA推测性解码运行器】结合NSA稀疏注意力的高效推测解码实现
    
    架构对比：
    - DenseSDRunner: 草稿(快速小模型) + 目标(全密集大模型)
    - NSASDRunner: 草稿(快速小模型) + 目标(NSA稀疏大模型) ← 本类
    
    核心优势：
    - 验证多个候选token时用稀疏注意力而不是全稠密注意力
    - 显著降低验证阶段的算力消耗，加快推测解码速度
    
    工作流程：
    1. 初始化NSA层(替换原Llama注意力为稀疏注意力)
    2. 预填充(prefill)后初始化NSA缓存
    3. 循环：草稿 → NSA验证 → 采样验证 → 缓存提交
    4. 输出生成的token序列
    """
    
    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        n_draft=8,
        temperature=1.0,
        topk=16,
        nsa_ckpt: Optional[str] = None,
        strict_token_check: bool = False,
        nsa_verify_logits_scale: float = 1.0,
        debug_logits_stats: bool = False,
    ):
        """
        初始化NSA推测解码器
        
        参数：
        - target_model: 目标大模型(Llama)
        - draft_model: 草稿小模型(快速生成候选token)
        - tokenizer: 分词器
        - n_draft: 每轮生成的草稿token数(通常8-16)
        - temperature: 采样温度(越大输出越随机)
        - topk: NSA稀疏注意力的top-k值
        - nsa_ckpt: NSA训练好的检查点路径(可选)
        - strict_token_check: 是否严格检查token id
        - nsa_verify_logits_scale: NSA logit的缩放因子(用于补偿NSA和dense logit尺度差异)
        - debug_logits_stats: 是否打印debug信息(首轮对比dense vs NSA logits)
        """
        # 【调用父类初始化】复用基础SD框架(采样、缓存管理等)
        super().__init__(
            target_model,
            draft_model,
            tokenizer,
            n_draft=n_draft,
            temperature=temperature,
            strict_token_check=strict_token_check,
        )
        self.nsa_verify_logits_scale = nsa_verify_logits_scale
        self.debug_logits_stats = debug_logits_stats
        
        # 【创建NSA层】为目标模型的每一层创建对应的NSA替换层
        cfg = target_model.config
        self.nsa_layers = [
            LlamaNSALayer(target_model.model.layers[l].self_attn, cfg, topk=topk).to(
                next(target_model.parameters()).device, next(target_model.parameters()).dtype
            )
            for l in range(cfg.num_hidden_layers)
        ]
        
        # 【加载或初始化NSA权重】
        if nsa_ckpt:
            # 从检查点加载训练好的NSA权重
            ckpt_path = nsa_ckpt
            if os.path.isdir(ckpt_path):
                ckpt_path = os.path.join(ckpt_path, "ckpt.pt")
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location="cpu")
            nsa_sd = ckpt["nsa"]  # 提取NSA权重字典
            for l_key, sd in nsa_sd.items():
                self.nsa_layers[int(l_key)].fsa.load_state_dict(sd, strict=False)

            # 【关键修复】checkpoint 的 load_state_dict 可能覆盖了 proj_q/k/v/o 权重。
            # 预填充阶段使用原始 Llama 的投影权重生成 KV 缓存，
            # 验证阶段的 NSA 模块也必须使用相同的投影权重，否则 Q/K 空间不匹配。
            # 因此在加载 checkpoint 后，强制从 Llama 原始注意力层复制投影权重。
            with torch.no_grad():
                for l_idx, nsa in enumerate(self.nsa_layers):
                    llama_attn = target_model.model.layers[l_idx].self_attn
                    nsa.fsa.proj_q.weight.copy_(llama_attn.q_proj.weight)
                    nsa.fsa.proj_k.weight.copy_(llama_attn.k_proj.weight)
                    nsa.fsa.proj_v.weight.copy_(llama_attn.v_proj.weight)
                    nsa.fsa.proj_o.weight.copy_(llama_attn.o_proj.weight)
        else:
            # 没有检查点时，用均值池初始化(相当于对所有KV求均值)
            self._init_mean_pool()
        
        # 【创建NSA目标模型管理器】负责缓存、验证、Hook等
        self.nsa_target = NSATargetModel(target_model, self.nsa_layers)

    def _init_mean_pool(self):
        """
        【均值池初始化】当没有预训练NSA检查点时，用简单的均值池策略初始化NSA压缩权重
        
        初始化策略：
        1. compress_key和compress_value初始化为单位矩阵(mean pooling)
           - 每个kernel_size of keys被等权重求均值
        2. gate (gating network)初始化为特定值
           - 第一个权重:=-1e-2 (倾向于不选择)
           - 其他权重:=-1e-2 (小的负值，弱化信号)
        
        这样NSA在训练前能表现得接近全注意力(求均值)，然后通过训练逐步学习稀疏模式
        """
        cfg = self.target.config
        device = next(self.target.parameters()).device
        dtype = next(self.target.parameters()).dtype
        kernel_size = self.nsa_layers[0].kernel_size
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        
        # 【单位矩阵】eye(head_dim) / kernel_size 相当于平均所有kernel_size个key
        eye = torch.eye(head_dim, device=device, dtype=dtype) / kernel_size
        
        with torch.no_grad():
            for nsa in self.nsa_layers:
                ck = nsa.fsa.compress_key      # [out_dim, kernel_size*head_dim]
                cv = nsa.fsa.compress_value    # [out_dim, kernel_size*head_dim]
                
                # 清零初始化
                ck.zero_()
                cv.zero_()
                
                # 【关键】设置为"均值"矩阵
                # 每个i对应的kernel_size个head_dims设为单位矩阵/kernel_size
                for i in range(kernel_size):
                    ck[:, i * head_dim : (i + 1) * head_dim, :] = eye
                    cv[:, i * head_dim : (i + 1) * head_dim, :] = eye
                
                # 【初始化gate (gating network)】用于学习每个位置的权重
                nn.init.zeros_(nsa.fsa.gate[0].weight)  # 先全零
                nsa.fsa.gate[0].weight[1].fill_(0.0)    # 第二行用0.0初始化
                nsa.fsa.gate[0].weight[0].fill_(-1e-2)  # 第一行用小负值初始化
                nsa.fsa.gate[0].weight[2].fill_(-1e-2)  # 第三行用小负值初始化

    def _nsa_prefill(self, input_ids: torch.Tensor):
        """
        【NSA 预填充】使用 FlashSparseAttentionDecode 替换所有层的注意力进行预填充。
        与 --real-fsa 训练路径完全一致（空缓存 + 完整序列），确保 hidden states
        经过 FSA decode 注意力处理，消除 train-test 分布偏差。

        返回:
        - out: 预填充输出 (包含 logits)
        - past_key_values: HF 格式的 KV 缓存
        - k_nope_storage: 各层非 RoPE 的 K 投影（用于初始化 NSA 缓存压缩）
        """
        device = input_ids.device
        dtype = next(self.target.parameters()).dtype

        orig_forwards = {}
        k_nope_storage = {}
        past_key_values_list = []

        for l, layer in enumerate(self.target.model.layers):
            orig_forwards[l] = layer.self_attn.forward
            fsa_layer = self.nsa_layers[l]

            def make_hook(_lid, _fsa_layer):
                def fsa_prefill_forward(hidden_states=None, *args, **kwargs):
                    bsz, seq_len, hsz = hidden_states.shape
                    hidden_flat = hidden_states.reshape(-1, hsz)
                    total_len = hidden_flat.shape[0]

                    cu_seqlens = torch.arange(
                        0, (bsz + 1) * seq_len, seq_len,
                        device=hidden_flat.device, dtype=torch.int32,
                    )
                    position_ids_flat = (
                        torch.arange(seq_len, device=hidden_flat.device, dtype=torch.long)
                        .unsqueeze(0).expand(bsz, -1).reshape(-1)
                    )

                    # 与 --real-fsa 训练路径完全一致：空缓存 + FlashSparseAttentionDecode
                    fsa = _fsa_layer.fsa
                    empty = torch.empty(
                        0, fsa.num_kv_heads, fsa.head_dim,
                        device=hidden_flat.device, dtype=hidden_flat.dtype,
                    )
                    stash = []
                    attn_out = fsa(
                        hidden_flat,
                        cu_seqlens, cu_seqlens,
                        empty, empty, empty, empty, empty,
                        attention_mask=None,
                        position_ids=position_ids_flat,
                        kv_commit_stash=stash,
                    )

                    # 从 stash 提取 K(rope), K(nope), V
                    k_rope_new, k_nope_new, v_new = stash[0]
                    k_nope_storage[_lid] = k_nope_new.detach()

                    # 构建 HF 格式缓存: [batch, num_heads, seq_len, head_dim]
                    # k_rope_new: [total_len, num_kv_heads, head_dim]
                    # 需要按 batch 拆分再 permute
                    k_hf = k_rope_new.view(bsz, seq_len, fsa.num_kv_heads, fsa.head_dim)
                    k_hf = k_hf.permute(0, 2, 1, 3).contiguous()
                    v_hf = v_new.view(bsz, seq_len, fsa.num_kv_heads, fsa.head_dim)
                    v_hf = v_hf.permute(0, 2, 1, 3).contiguous()
                    past_key_values_list.append((k_hf.detach(), v_hf.detach()))

                    return (attn_out.reshape(bsz, seq_len, hsz), None, None)
                return fsa_prefill_forward
            layer.self_attn.forward = make_hook(l, fsa_layer)

        try:
            out = self.target(input_ids, use_cache=False, num_logits_to_keep=1)
        finally:
            for l, layer in enumerate(self.target.model.layers):
                layer.self_attn.forward = orig_forwards[l]

        past_key_values = tuple(past_key_values_list)
        return out, past_key_values, k_nope_storage

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int):
        """
        【NSA推测解码主生成循环】
        
        工作流程总览：
        ┌──────────────────────────────────────────────────┐
        │1. 预填充(Prefill)                               │
        │   - 目标和草稿模型都生成初始token               │
        │   - 初始化NSA缓存(从密集HF缓存转换)             │
        └──────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────────────────┐
        │2. 推测解码循环(Speculative Decoding Loop)       │
        │   a) 草稿阶段：草稿模型快速生成n个候选token     │
        │   b) 验证阶段：NSA目标模型用稀疏注意验证        │
        │   c) 采样阶段：根据概率接受/拒绝候选            │
        │   d) 提交阶段：将接受的token追加到缓存          │
        │   重复上述直到生成足够token或遇到EOS            │
        └──────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────────────────────────────────┐
        │3. 返回结果                                      │
        │   - 生成的token列表                              │
        │   - 性能统计(TPS、接受率等)                      │
        └──────────────────────────────────────────────────┘
        """
        device = input_ids.device
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # ========== 预填充阶段 ==========
        # 【NSA 预填充】使用 NSA 注意力（而非 dense 注意力）进行预填充，
        # 确保 KV 缓存中的 hidden states 与训练时分布一致。
        t_prefill_out, prefill_target_kv, k_nope_storage = self._nsa_prefill(input_ids)
        self.nsa_target.init_from_prefill(prefill_target_kv, k_nope_storage)
        
        # 【草稿模型预填充】用草稿模型也处理同样的prompt
        d_prefill = self.draft(input_ids, use_cache=True, num_logits_to_keep=1)
        
        # 【采样首个token】从目标模型的预填充输出采样
        cur = sample_from_logits(t_prefill_out.logits[:, -1, :], self.temperature, self._vocab)
        
        # 【草稿模型缓存】继续用标准缓存
        draft_cache = d_prefill.past_key_values
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # ========== 初始化统计变量 ==========
        generated = []           # 最终输出token列表
        drafted_total = 0        # 总共草稿生成的token数
        accepted_total = 0       # 总共被接受的token数
        rounds = 0              # 推测解码的轮数
        
        # ========== 推测解码主循环 ==========
        # 【安装Hook】在验证模式下替换注意力为NSA版本
        self.nsa_target.begin_verify()
        try:
            while len(generated) < max_new_tokens:
                rounds += 1
                remain = max_new_tokens - len(generated)
                n = min(self.n_draft, remain)  # 本轮生成的草稿token数
                old_d_len = cache_seq_len(draft_cache)

                # ───────── 第一步：草稿生成 ─────────
                # 【思路】用快速小模型草稿地生成n个候选token
                draft_ids: List[int] = []        # 本轮生成的token ids
                draft_logits: List[torch.Tensor] = []  # 对应的logits(用于验证)
                d_cur = sanitize_token_tensor(
                    torch.tensor([[cur]], device=device, dtype=torch.long),
                    self._vocab,
                    self.strict_token_check,
                )
                
                # 逐个生成草稿token
                for _ in range(n):
                    # 草稿模型单步前向
                    d_out = self.draft(d_cur, past_key_values=draft_cache, use_cache=True, num_logits_to_keep=1)
                    draft_cache = d_out.past_key_values
                    logit = d_out.logits[:, -1, :].squeeze(0).float()
                    nxt = sample_from_logits(logit, self.temperature, self._vocab)
                    draft_ids.append(nxt)
                    draft_logits.append(logit)
                    d_cur = sanitize_token_tensor(
                        torch.tensor([[nxt]], device=device, dtype=torch.long),
                        self._vocab,
                        self.strict_token_check,
                    )

                # 转换为张量便于后续处理
                draft_ids_t = torch.tensor(draft_ids, device=device, dtype=torch.long)
                draft_logits_t = torch.stack(draft_logits, dim=0)
                
                # ───────── 第二步：目标模型验证 ─────────
                # 【思路】构建验证输入：[前一个token] + [n个候选token]
                verify_input = sanitize_token_tensor(
                    torch.tensor([[cur] + draft_ids], device=device, dtype=torch.long),
                    self._vocab,
                    self.strict_token_check,
                )

                # 【NSA验证】用稀疏注意力验证所有n+1个token
                # NSATargetModel会自动用Hook替换注意力为NSA版本
                target_logits = self.nsa_target.verify(verify_input).float()

                # 【调试】如果启用debug_logits_stats，第一轮打印dense vs NSA的logit对比
                if self.debug_logits_stats and rounds == 1:
                    dbg_cache = clone_legacy_past_key_values(prefill_target_kv)
                    with torch.no_grad():
                        dense_logits_dbg = self.target(
                            verify_input,
                            past_key_values=dbg_cache,
                            use_cache=True,
                            num_logits_to_keep=n + 1,
                        ).logits.squeeze(0).float()
                    print("\n[debug logits] round 1: dense vs NSA verify (same verify_input, same ctx length)")
                    for name, t in [("dense", dense_logits_dbg), ("nsa", target_logits)]:
                        print(
                            f"  {name}: max={t.max().item():.4f}  std={t.std().item():.4f}  "
                            f"mean={t.mean().item():.4f}  shape={tuple(t.shape)}"
                        )
                    # Per-position top-1 comparison
                    print("  Per-position argmax comparison:")
                    for pos in range(min(n + 1, dense_logits_dbg.shape[0], target_logits.shape[0])):
                        d_top = dense_logits_dbg[pos].argmax().item()
                        n_top = target_logits[pos].argmax().item()
                        match = "✓" if d_top == n_top else "✗"
                        d_tok = self.tokenizer.decode([d_top])
                        n_tok = self.tokenizer.decode([n_top])
                        print(f"    pos {pos}: dense={d_top}({d_tok!r}) nsa={n_top}({n_tok!r}) {match}")

                # 【logit缩放】可选的logit缩放用于补偿NSA和dense的尺度差异
                if self.nsa_verify_logits_scale != 1.0:
                    target_logits = target_logits * self.nsa_verify_logits_scale

                # ───────── 第三步：采样验证 ─────────
                # 【思路】根据目标和草稿的概率分布，决定哪些token被接受
                # 这是广为人知的speculative decoding采样方案
                accept_len, accepted_ids = self.sampler.verify(draft_ids_t, target_logits, draft_logits_t)
                commit_len = 1 + accept_len  # 包括前一个token
                
                # ───────── 第四步：缓存提交 ─────────
                # 【提交】将被接受的新token追加到NSA缓存
                self.nsa_target.commit_after_verify(commit_len)

                # 如果全部n个候选都被接受，需要生成一个额外的token供下一轮使用
                if accept_len == n:
                    extra = self.draft(
                        sanitize_token_tensor(
                            torch.tensor([[draft_ids[-1]]], device=device, dtype=torch.long),
                            self._vocab,
                            self.strict_token_check,
                        ),
                        past_key_values=draft_cache,
                        use_cache=True,
                        num_logits_to_keep=1,
                    )
                    draft_cache = extra.past_key_values
                
                # 【草稿缓存修剪】保持缓存大小一致
                draft_cache = trim_legacy_cache(draft_cache, old_d_len + commit_len)

                # ───────── 更新统计和状态 ─────────
                generated.extend(accepted_ids)
                drafted_total += n
                accepted_total += accept_len
                cur = clamp_token_id(accepted_ids[-1], self._vocab)
                
                # 【EOS检查】如果输出了结束符，提前终止
                if self.tokenizer.eos_token_id is not None and cur == self.tokenizer.eos_token_id:
                    break
        finally:
            # 【卸载Hook】无论是否正常结束，都要恢复原始注意力
            self.nsa_target.end_verify()

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        # ========== 计算性能统计 ==========
        prefill_t = t1 - t0
        decode_t = t2 - t1
        total_t = t2 - t0
        stats = GenerationStats(
            total_time=total_t,
            total_generated_tokens=len(generated),
            tokens_per_second=len(generated) / max(decode_t, 1e-6),
            prefill_time=prefill_t,
            decode_time=decode_t,
            total_drafted_tokens=drafted_total,
            total_accepted_tokens=accepted_total,
            acceptance_rate=accepted_total / max(drafted_total, 1),
            avg_accept_len=accepted_total / max(rounds, 1),
            num_rounds=rounds,
        )
        return generated[:max_new_tokens], stats


def print_stats(name: str, stats: GenerationStats, text: str):
    print(f"\n{'=' * 64}")
    print(f"{name}")
    print(f"  total_time            : {stats.total_time:.3f} s")
    print(f"  total_generated_tokens: {stats.total_generated_tokens}")
    print(f"  tokens_per_second     : {stats.tokens_per_second:.2f}")
    print(f"  prefill/decode        : {stats.prefill_time:.3f}s / {stats.decode_time:.3f}s")
    if stats.total_drafted_tokens > 0:
        print(f"  total_drafted_tokens  : {stats.total_drafted_tokens}")
        print(f"  total_accepted_tokens : {stats.total_accepted_tokens}")
        print(f"  acceptance_rate       : {stats.acceptance_rate:.3f}")
        print(f"  avg_accept_len        : {stats.avg_accept_len:.3f}")
        print(f"  num_rounds            : {stats.num_rounds}")
    print(f"  text (prefix)         : {text!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense-ar", "dense-sd", "nsa-sd", "all"], default="all")
    parser.add_argument("--model", default=LLAMA_8B, help="target model path / hf id")
    parser.add_argument("--draft-model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--nsa-ckpt", default=None, help="NSA ckpt dir or ckpt.pt")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-draft", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--strict-token-check",
        action="store_true",
        help="若 token id 超出词表则直接报错（默认会 clamp 到合法范围，避免 embedding 越界）",
    )
    parser.add_argument(
        "--nsa-verify-logits-scale",
        type=float,
        default=1.0,
        help="NSA verify 后、送入接受采样前对 target_logits 乘以该系数（缓解 NSA 与 dense 的 logit 尺度差，默认 1.0）",
    )
    parser.add_argument(
        "--debug-logits-stats",
        action="store_true",
        help="NSA-SD 首轮打印 dense vs NSA 的 verify logits 的 max/std/mean（会多跑一次 dense target 仅用于对比）",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "The afternoon sun filtered through the kitchen window, casting warm golden light on the polished wooden table. Dust motes danced lazily in the beam, floating above a half-finished ceramic mug painted with tiny sunflowers. Clara stood by the marble counter, her fingers wrapped around a steaming mug of hot milk, the warmth seeping through the ceramic to soothe her cold hands. She had just returned from a morning walk in the neighborhood park, where the cherry blossoms were in full bloom, their pale pink petals drifting down like snow whenever the wind blew. Her little cat, Mochi, had followed her home from the park that morning, weaving between her legs and purring so loudly that the sound vibrated through her jeans. Now, Mochi lay curled up on the soft linen sofa, her body tucked into a perfect circle, paws tucked under her chin, and tail wrapped gently around her paws. The cat had soft, cloud-like white fur that glowed in the sunlight, with a tiny pink nose and a pair of striking bright blue eyes that looked like fragments of the summer sky. When she blinked, her long white eyelashes fluttered, making her look even more like a fluffy snowball brought to life. "
            "Clara smiled softly, taking a slow sip of her hot milk, her eyes never leaving the sleeping cat. She had found Mochi three months ago, shivering under a park bench on a rainy evening, her fur matted and her tiny body trembling with cold. Clara had wrapped the kitten in her wool scarf, carried her home in her backpack, and spent the entire night drying her fur, feeding her warm milk, and making a cozy bed out of an old sweater and a cardboard box. At first, Mochi had been scared, hiding under the bed and refusing to eat, but slowly, she began to trust Clara. She started curling up on her lap while she read books, chasing small toys around the apartment, and greeting her at the door every morning with soft meows. Now, the little white cat was the heart of Clara’s small apartment, filling every quiet corner with warmth and joy, turning an empty house into a loving home. "
            "The apartment was small but cozy, filled with soft blankets, potted green plants that lined the windowsill, and shelves lined with old books and handmade trinkets. A vintage record player sat in the corner, softly playing a gentle jazz melody that floated through the air, mixing with the sweet scent of vanilla candles that Clara had lit that morning. Outside the window, the world was calm and quiet; a few birds chirped in the old oak tree, a bicycle bell rang faintly from the street below, and the distant hum of a passing car created a peaceful background noise. Inside, time seemed to slow down, wrapped in a blanket of comfort and tranquility, with no rush, no worries, just pure, simple happiness. "
            "Mochi stirred slightly in her sleep, letting out a tiny, quiet meow and twitching her paw as if chasing a butterfly in a dream. Clara chuckled quietly, careful not to wake her, and set her milk mug down on the table. She walked over to the sofa, knelt down beside it, and gently brushed a strand of soft fur from Mochi’s forehead. The cat leaned into her touch, purring softly even in her sleep, her body relaxing completely under Clara’s gentle hand. In that moment, Clara felt a deep sense of gratitude and peace wash over her. She had lived alone for years, focused on her work and her busy life, never realizing how much joy a small, loving companion could bring. Mochi wasn’t just a pet; she was a friend, a comfort, a little ray of sunshine that brightened every single day. "
            "As the sun began to dip lower in the sky, the golden light turned softer, painting the walls in warm orange and pink hues. Clara stood up, grabbed a soft knitted blanket from the armchair, and draped it gently over Mochi, making sure the cat stayed warm and cozy. She then walked back to the kitchen, refilled her mug with hot milk, and pulled a book from the shelf, settling into the armchair beside the sofa. She opened the book, but her eyes kept drifting back to Mochi, watching her sleep peacefully, her chest rising and falling in slow, steady breaths. The jazz music continued to play, the candle flickered softly, and the warm light filled every corner of the room. "
            "Clara thought about all the small, perfect moments she had shared with Mochi: the mornings they spent curled up in bed together, the afternoons playing with feather toys and crinkly balls, the evenings when Mochi sat on her lap while she worked, and the quiet nights when the cat slept at her feet, keeping her warm. She thought about how Mochi had changed her life, teaching her to slow down, to appreciate the little things, and to find happiness in the simplest moments. Before Mochi, her days were filled with deadlines and stress; now, her days were filled with purrs, soft meows, and endless affection. "
            "Outside, the sky turned a soft shade of purple, and the first stars began to twinkle in the evening sky. The neighborhood grew quieter, with only the occasional rustle of leaves and the distant hoot of an owl breaking the silence. Mochi woke up slowly, stretching her tiny body into a long, elegant arc, her front paws reaching forward and her back arching high. She let out a big yawn, showing her tiny pink tongue, then jumped down from the sofa and trotted over to Clara, rubbing her head against Clara’s leg and meowing softly for attention. Clara put down her book, bent down, and picked Mochi up, holding her close to her chest. The cat curled up in her arms, purring loudly, her warm body pressing against Clara’s chest, her soft fur tickling her skin. "
            "Clara carried Mochi to the window, looking out at the quiet street and the starry sky. She pressed a gentle kiss to the top of Mochi’s furry head, and the cat nuzzled her cheek, as if giving a kiss back. In that moment, surrounded by warmth, love, and quiet joy, Clara knew that this was exactly where she was meant to be. The small apartment, the gentle music, the soft candlelight, and the little white cat in her arms—this was her happy place, her safe haven, her perfect little world. No grand adventures, no fancy possessions, just a quiet life filled with love, comfort, and the unbreakable bond between a girl and her cat. As the night fell gently around them, Clara held Mochi close, grateful for every second of this peaceful, perfect happiness, knowing that more sweet moments were waiting to unfold with her fluffy, blue-eyed snowball by her side."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda"
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("Loading target model ...")
    target = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device).eval()
    if args.mode in ("dense-sd", "nsa-sd", "all"):
        print("Loading draft model ...")
        draft = AutoModelForCausalLM.from_pretrained(
            args.draft_model, torch_dtype=dtype, device_map=device
        ).eval()
    else:
        draft = None

    input_ids = build_prompt_ids(tok, args.prompt, args.seqlen, device=device)
    print(f"Prompt tokens: {input_ids.shape[1]}")

    dense_ar_tps = dense_sd_tps = None

    if args.mode in ("dense-ar", "all"):
        runner = DenseARRunner(
            target, tok, temperature=args.temperature, strict_token_check=args.strict_token_check
        )
        ids, stats = runner.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print_stats("Dense AR", stats, tok.decode(ids, skip_special_tokens=True))
        dense_ar_tps = stats.tokens_per_second

    if args.mode in ("dense-sd", "all"):
        runner = DenseSDRunner(
            target,
            draft,
            tok,
            n_draft=args.n_draft,
            temperature=args.temperature,
            strict_token_check=args.strict_token_check,
        )
        ids, stats = runner.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print_stats("Dense SD", stats, tok.decode(ids, skip_special_tokens=True))
        dense_sd_tps = stats.tokens_per_second
        if dense_ar_tps is not None:
            print(f"Dense-SD speedup over Dense-AR: {dense_sd_tps / max(dense_ar_tps, 1e-6):.3f}x")

    if args.mode in ("nsa-sd", "all"):
        runner = NSASDRunner(
            target,
            draft,
            tok,
            n_draft=args.n_draft,
            temperature=args.temperature,
            topk=args.topk,
            nsa_ckpt=args.nsa_ckpt,
            strict_token_check=args.strict_token_check,
            nsa_verify_logits_scale=args.nsa_verify_logits_scale,
            debug_logits_stats=args.debug_logits_stats,
        )
        ids, stats = runner.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print_stats("NSA SD", stats, tok.decode(ids, skip_special_tokens=True))
        if dense_sd_tps is not None:
            print(f"NSA-SD speedup over Dense-SD: {stats.tokens_per_second / max(dense_sd_tps, 1e-6):.3f}x")


if __name__ == "__main__":
    main()

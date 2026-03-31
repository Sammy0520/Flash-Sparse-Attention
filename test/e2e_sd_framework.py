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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from nsa_ref.module import RopeConfig
from nsa_ref.ops import linear_compress
from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode


LLAMA_8B = (
    "/data1/models/Llama-3.1-8B-Instruct/snapshots/"
    "0e9e39f249a16976918f6564b8830bc894c89659"
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
    def __init__(self, temperature: float = 1.0, vocab_size: int = 128256):
        self.temperature = temperature
        self.vocab_size = vocab_size

    def verify(
        self,
        draft_ids: torch.Tensor,      # [N]
        target_logits: torch.Tensor,  # [N+1, vocab]
        draft_logits: torch.Tensor,   # [N, vocab]
    ) -> Tuple[int, List[int]]:
        n = draft_ids.shape[0]
        vs = self.vocab_size
        # NSA verify may occasionally produce NaN/Inf on some rows.
        # Sanitize logits before softmax to keep SD loop robust.
        t_logits = torch.nan_to_num(target_logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        d_logits = torch.nan_to_num(draft_logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        t_probs = torch.softmax(t_logits / max(self.temperature, 1e-6), dim=-1)
        d_probs = torch.softmax(d_logits / max(self.temperature, 1e-6), dim=-1)

        accepted = 0
        accepted_ids: List[int] = []
        for i in range(n):
            tok = clamp_token_id(int(draft_ids[i].item()), vs)
            p = float(t_probs[i, tok].item())
            q = float(d_probs[i, tok].item())
            acc_prob = min(1.0, p / max(q, 1e-8))
            if random.random() < acc_prob:
                accepted += 1
                accepted_ids.append(tok)
                continue
            diff = torch.clamp(t_probs[i] - d_probs[i], min=0.0)
            diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            diff_sum = float(diff.sum().item())
            corrected = (
                clamp_token_id(int(torch.multinomial(diff / diff_sum, 1).item()), vs)
                if diff_sum > 0
                else clamp_token_id(int(t_probs[i].argmax().item()), vs)
            )
            accepted_ids.append(corrected)
            return accepted, accepted_ids

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
    def __init__(
        self,
        llama_attn,
        cfg,
        topk=16,
        block_size=64,
        kernel_size=32,
        kernel_stride=16,
        init_blocks=1,
        local_blocks=2,
        window_size=512,
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
        with torch.no_grad():
            self.fsa.proj_q.weight.copy_(llama_attn.q_proj.weight)
            self.fsa.proj_k.weight.copy_(llama_attn.k_proj.weight)
            self.fsa.proj_v.weight.copy_(llama_attn.v_proj.weight)
            self.fsa.proj_o.weight.copy_(llama_attn.o_proj.weight)
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride

    def build_compressed_cache(self, k_raw, v_raw, cu_k):
        cmp_k, _ = linear_compress(
            k_raw,
            self.fsa.compress_key,
            cu_k,
            self.kernel_size,
            self.kernel_stride,
            self.fsa.intra_block_pe,
        )
        cmp_v, _ = linear_compress(
            v_raw,
            self.fsa.compress_value,
            cu_k,
            self.kernel_size,
            self.kernel_stride,
            None,
        )
        return cmp_k, cmp_v

    def forward(self, hidden, k_raw, v_raw, cmp_k, cmp_v, cu_q, cu_k, position_ids, kv_commit_stash=None):
        return self.fsa(
            hidden,
            cu_q,
            cu_k,
            k_raw,
            v_raw,
            cmp_k,
            cmp_v,
            attention_mask=None,
            position_ids=position_ids,
            kv_commit_stash=kv_commit_stash,
        )


class NSATargetModel:
    def __init__(self, llama, nsa_layers: List[LlamaNSALayer]):
        self.llama = llama
        self.nsa_layers = nsa_layers
        self.k_raw = [None] * len(nsa_layers)
        self.v_raw = [None] * len(nsa_layers)
        self.cmp_k = [None] * len(nsa_layers)
        self.cmp_v = [None] * len(nsa_layers)
        self.past_len = 0
        self._orig_forwards = {}
        self._verify_mode = False
        self._pos_ids = None
        # Per-layer (k_new_rope, v_new) from the last verify forward; used for pure-NSA cache commit.
        self._stash_kv_new: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(nsa_layers)

    def init_from_prefill(self, past_key_values):
        device = self.nsa_layers[0].fsa.proj_q.weight.device
        dtype = self.nsa_layers[0].fsa.proj_q.weight.dtype
        self.past_len = cache_seq_len(past_key_values)
        for l, nsa in enumerate(self.nsa_layers):
            k, v = past_key_values[l]
            k_raw = k.squeeze(0).permute(1, 0, 2).contiguous().to(dtype)
            v_raw = v.squeeze(0).permute(1, 0, 2).contiguous().to(dtype)
            cu_k = torch.tensor([0, k_raw.shape[0]], device=device, dtype=torch.int32)
            cmp_k, cmp_v = nsa.build_compressed_cache(k_raw, v_raw, cu_k)
            self.k_raw[l] = k_raw
            self.v_raw[l] = v_raw
            self.cmp_k[l] = cmp_k
            self.cmp_v[l] = cmp_v

    def append_from_dense_cache(self, new_past_key_values, old_len: int, commit_len: int):
        """Optional: align NSA KV with dense HF cache (debug / hybrid). Not used in pure-NSA-SD."""
        if commit_len <= 0:
            return
        device = self.nsa_layers[0].fsa.proj_q.weight.device
        for l, nsa in enumerate(self.nsa_layers):
            k_full, v_full = new_past_key_values[l]
            k_delta = k_full[:, :, old_len : old_len + commit_len, :]
            v_delta = v_full[:, :, old_len : old_len + commit_len, :]
            k_raw_delta = k_delta.squeeze(0).permute(1, 0, 2).contiguous()
            v_raw_delta = v_delta.squeeze(0).permute(1, 0, 2).contiguous()

            if k_raw_delta.shape[0] > 0:
                self.k_raw[l] = torch.cat([self.k_raw[l], k_raw_delta], dim=0)
                self.v_raw[l] = torch.cat([self.v_raw[l], v_raw_delta], dim=0)
                cu_full = torch.tensor([0, self.k_raw[l].shape[0]], device=device, dtype=torch.int32)
                cmp_k_full, cmp_v_full = nsa.build_compressed_cache(self.k_raw[l], self.v_raw[l], cu_full)
                self.cmp_k[l] = cmp_k_full
                self.cmp_v[l] = cmp_v_full
        self.past_len += commit_len

    def commit_after_verify(self, commit_len: int):
        """
        Append committed raw KV from stashed NSA projections (same as inside FlashSparseAttentionDecode:
        k_new after RoPE, v_new without extra RoPE), then rebuild cmp_k/cmp_v from full raw cache.
        """
        if commit_len <= 0:
            return
        device = self.nsa_layers[0].fsa.proj_q.weight.device
        dtype = self.k_raw[0].dtype
        for l, nsa in enumerate(self.nsa_layers):
            st = self._stash_kv_new[l]
            if st is None:
                raise RuntimeError("commit_after_verify: missing stashed KV; verify() did not run?")
            k_new_rope, v_new = st
            kn = k_new_rope[:commit_len].contiguous().to(dtype)
            vn = v_new[:commit_len].contiguous().to(dtype)
            self.k_raw[l] = torch.cat([self.k_raw[l], kn], dim=0)
            self.v_raw[l] = torch.cat([self.v_raw[l], vn], dim=0)
            cu_full = torch.tensor([0, self.k_raw[l].shape[0]], device=device, dtype=torch.int32)
            cmp_k_full, cmp_v_full = nsa.build_compressed_cache(self.k_raw[l], self.v_raw[l], cu_full)
            self.cmp_k[l] = cmp_k_full
            self.cmp_v[l] = cmp_v_full
        self.past_len += commit_len
        self._stash_kv_new = [None] * len(self.nsa_layers)

    def _make_nsa_forward(self, layer_idx: int):
        target = self
        nsa = self.nsa_layers[layer_idx]

        def nsa_forward(hidden_states=None, *args, _lid=layer_idx, **kwargs):
            if not target._verify_mode:
                return target._orig_forwards[_lid](hidden_states, *args, **kwargs)
            n = hidden_states.shape[1]
            hidden_flat = hidden_states.squeeze(0)
            device = hidden_states.device
            past_kv_len = target.k_raw[_lid].shape[0]
            total_k_len = past_kv_len + n
            cu_q = torch.tensor([0, n], device=device, dtype=torch.int32)
            cu_k = torch.tensor([0, total_k_len], device=device, dtype=torch.int32)
            # Stash (k_new_rope, v_new) from *inside* FSA — same tensors as cat([cache, k_new]) in attention.
            stash: List = []
            out = nsa(
                hidden_flat,
                target.k_raw[_lid],
                target.v_raw[_lid],
                target.cmp_k[_lid],
                target.cmp_v[_lid],
                cu_q,
                cu_k,
                target._pos_ids,
                kv_commit_stash=stash,
            )
            if len(stash) != 1:
                raise RuntimeError(
                    f"kv_commit_stash expected 1 tuple, got {len(stash)} (position_ids must be set)"
                )
            kn, vn = stash[0]
            target._stash_kv_new[_lid] = (kn.clone(), vn.clone())
            return (out.unsqueeze(0), None, None)

        return nsa_forward

    def _patch_attn(self):
        for l, layer in enumerate(self.llama.model.layers):
            self._orig_forwards[l] = layer.self_attn.forward
            layer.self_attn.forward = self._make_nsa_forward(l)

    def _restore_attn(self):
        for l, layer in enumerate(self.llama.model.layers):
            layer.self_attn.forward = self._orig_forwards[l]
        self._orig_forwards.clear()

    @torch.no_grad()
    def verify(self, verify_ids: torch.Tensor) -> torch.Tensor:
        n = verify_ids.shape[1]
        device = verify_ids.device
        self._stash_kv_new = [None] * len(self.nsa_layers)
        self._pos_ids = torch.arange(self.past_len, self.past_len + n, device=device, dtype=torch.long)
        self._verify_mode = True
        self._patch_attn()
        try:
            out = self.llama(verify_ids, use_cache=False, num_logits_to_keep=n)
        finally:
            self._restore_attn()
            self._verify_mode = False
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
        cfg = target_model.config
        self.nsa_layers = [
            LlamaNSALayer(target_model.model.layers[l].self_attn, cfg, topk=topk).to(
                next(target_model.parameters()).device, next(target_model.parameters()).dtype
            )
            for l in range(cfg.num_hidden_layers)
        ]
        if nsa_ckpt:
            ckpt_path = nsa_ckpt
            if os.path.isdir(ckpt_path):
                ckpt_path = os.path.join(ckpt_path, "ckpt.pt")
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location="cpu")
            nsa_sd = ckpt["nsa"]
            for l_key, sd in nsa_sd.items():
                self.nsa_layers[int(l_key)].fsa.load_state_dict(sd, strict=True)
        else:
            self._init_mean_pool()
        self.nsa_target = NSATargetModel(target_model, self.nsa_layers)

    def _init_mean_pool(self):
        cfg = self.target.config
        device = next(self.target.parameters()).device
        dtype = next(self.target.parameters()).dtype
        kernel_size = self.nsa_layers[0].kernel_size
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        eye = torch.eye(head_dim, device=device, dtype=dtype) / kernel_size
        with torch.no_grad():
            for nsa in self.nsa_layers:
                ck = nsa.fsa.compress_key
                cv = nsa.fsa.compress_value
                ck.zero_()
                cv.zero_()
                for i in range(kernel_size):
                    ck[:, i * head_dim : (i + 1) * head_dim, :] = eye
                    cv[:, i * head_dim : (i + 1) * head_dim, :] = eye
                nn.init.zeros_(nsa.fsa.gate[0].weight)
                nsa.fsa.gate[0].weight[1].fill_(0.0)
                nsa.fsa.gate[0].weight[0].fill_(-1e-2)
                nsa.fsa.gate[0].weight[2].fill_(-1e-2)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int):
        device = input_ids.device
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_prefill = self.target(input_ids, use_cache=True, num_logits_to_keep=1)
        d_prefill = self.draft(input_ids, use_cache=True, num_logits_to_keep=1)
        cur = sample_from_logits(t_prefill.logits[:, -1, :], self.temperature, self._vocab)
        # One-time dense prefill to bootstrap k_raw/v_raw/cmp (same as e2e_sd_nsa); decode loop is pure NSA.
        prefill_target_kv = t_prefill.past_key_values
        self.nsa_target.init_from_prefill(prefill_target_kv)
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

            # Pure NSA target verify (no dense target forward in the loop)
            target_logits = self.nsa_target.verify(verify_input).float()

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

            if self.nsa_verify_logits_scale != 1.0:
                target_logits = target_logits * self.nsa_verify_logits_scale

            accept_len, accepted_ids = self.sampler.verify(draft_ids_t, target_logits, draft_logits_t)
            commit_len = 1 + accept_len
            self.nsa_target.commit_after_verify(commit_len)

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
    print(f"  text (prefix)         : {text[:512]!r}")


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
            "The afternoon sun filtered through the kitchen window, casting warm golden light on the wooden table. Clara stood by the counter, holding a mug of hot milk, watching her little cat curl up on the sofa. The cat, named Mochi, had soft white fur and a pair of bright blue eyes, looking like a fluffy snowball."
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

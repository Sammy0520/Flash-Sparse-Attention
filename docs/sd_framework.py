"""
GLA Speculative Decoding Framework
=====================================
基于真实模型结构实现，关键参数：
  340M: H=4, d_k=128, d_v=256, num_layers=24
  2.7B: H=5, d_k=256, d_v=512, num_layers=32
  recurrent_state shape: (B, H, d_k, d_v), dtype=float32
  use_gk=True -> gate 走 gk_proj (per-key gate, logsigmoid空间)
  Cache[layer] = dict, key='recurrent_state'

设计：TargetModelRunner 是唯一需要替换的插槽。
      换 kernel 只改 Runner，框架其他部分不动。

运行：
  # 先验证框架能跑通（只跑AR baseline，不加载draft model）
  python sd_framework.py --mode baseline --target-model 340M --max-tokens 50

  # 跑SD（n-gram draft + 2.7B target）
  python sd_framework.py --mode sd-m3 --target-model 2.7B --ngram-size 4
"""

import sys, os
import importlib.util
from functools import lru_cache

UPSTREAM_FLA_ROOT = "/data1/hxy/fla/flash-linear-attention"
LOCAL_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Keep the upstream FLA package first so `import fla` registers the custom
# Hugging Face architectures from flash-linear-attention.
if UPSTREAM_FLA_ROOT not in sys.path:
    sys.path.insert(0, UPSTREAM_FLA_ROOT)

# The local repo root is still useful for direct file-based imports below.
if LOCAL_REPO_ROOT not in sys.path:
    sys.path.append(LOCAL_REPO_ROOT)

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import time
from einops import rearrange, repeat


@lru_cache(maxsize=None)
def _load_local_module(module_name: str, relative_path: str):
    """
    Load a local speculative-decoding helper module by file path so it can
    coexist with the upstream `fla` package used for model registration.
    """
    module_path = os.path.join(LOCAL_REPO_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _encode_prompt(tokenizer, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GLAState:
    """
    所有 GLA 层的 recurrent state 封装。
    shape per layer: (B, H, d_k, d_v), dtype=float32
    """
    states: List[torch.Tensor]

    def clone(self) -> "GLAState":
        return GLAState(states=[s.clone() for s in self.states])

    @classmethod
    def from_fla_cache(cls, cache) -> "GLAState":
        """从 FLA Cache 对象提取所有层的 recurrent_state"""
        states = []
        for i in range(len(cache)):
            s = cache[i].get("recurrent_state", None)
            if s is not None:
                states.append(s.detach())
        return cls(states=states)

    def write_to_fla_cache(self, cache):
        """把 state 写回 FLA Cache（用于传给模型 forward）"""
        idx = 0
        for i in range(len(cache)):
            d = cache[i]
            if d.get("recurrent_state") is not None:
                d["recurrent_state"] = self.states[idx]
                idx += 1


@dataclass
class GenerationResult:
    generated_ids:   List[int]
    generated_text:  str
    total_time_s:    float
    tokens_per_sec:  float
    prefill_time_s:  float = 0.0
    decode_time_s:   float = 0.0
    total_drafted:   int   = 0
    total_accepted:  int   = 0
    num_rounds:      int   = 0
    acceptance_rate: float = 0.0
    avg_accept_len:  float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Draft 策略
# ═══════════════════════════════════════════════════════════════════════════════

class NGramDrafter:
    """
    用 history-based n-gram continuation 做 draft，不依赖 draft model。
    当找不到匹配的 n-gram 时，退化为重复上一个 token。
    """

    def __init__(self, vocab_size: int, ngram_size: int = 4):
        self.vocab_size = vocab_size
        self.ngram_size = max(2, ngram_size)

    def prefill(self, input_ids: torch.Tensor) -> List[int]:
        return input_ids[0].tolist()

    def draft(
        self,
        last_token_id: torch.Tensor,
        draft_len: int,
        draft_state: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        history = list(draft_state)
        ids_list = []

        for _ in range(draft_len):
            next_id = self._predict_next(history)
            history.append(next_id)
            ids_list.append(next_id)

        draft_ids = torch.tensor([ids_list], device=last_token_id.device, dtype=torch.long)
        draft_logits = torch.zeros(
            1, draft_len, self.vocab_size,
            device=last_token_id.device,
            dtype=torch.float32,
        )
        if ids_list:
            draft_logits[0, torch.arange(draft_len, device=last_token_id.device), draft_ids[0]] = 1.0
        return draft_ids, draft_logits, history

    def commit(self, draft_state: List[int], commit_input_ids: torch.Tensor) -> List[int]:
        return list(draft_state) + commit_input_ids[0].tolist()

    def _predict_next(self, history: List[int]) -> int:
        if not history:
            return 0
        max_n = min(self.ngram_size, len(history))
        for n in range(max_n, 0, -1):
            pattern = history[-n:]
            for i in range(len(history) - n - 1, -1, -1):
                if history[i:i + n] == pattern:
                    next_idx = i + n
                    if next_idx < len(history):
                        return history[next_idx]
        return history[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# 验证逻辑
# ═══════════════════════════════════════════════════════════════════════════════

class SpeculativeSampling:
    """
    标准 speculative sampling（Leviathan et al. 2023）。
    temperature=0 退化为贪心验证。
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def verify(
        self,
        draft_ids:     torch.Tensor,   # (1, draft_len)
        target_logits: torch.Tensor,   # (1, draft_len+1, vocab)
        draft_logits:  torch.Tensor,   # (1, draft_len, vocab)
    ) -> Tuple[int, List[int]]:
        """
        Returns:
          accept_len:   纯接受的 draft token 数
          accepted_ids: 含修正 token 的最终序列，长度 = accept_len + 1
        """
        draft_len  = draft_ids.shape[1]
        draft_list = draft_ids[0].tolist()

        if self.temperature <= 0:
            return self._greedy(draft_list, target_logits[0])

        t_probs = torch.softmax(target_logits[0] / self.temperature, dim=-1)
        d_probs = torch.softmax(draft_logits[0]  / self.temperature, dim=-1)

        accept_len = 0
        for i in range(draft_len):
            x   = draft_list[i]
            p_t = t_probs[i, x].item()
            p_d = d_probs[i, x].item()
            r   = torch.rand(1).item()
            if r < min(1.0, p_t / (p_d + 1e-8)):
                accept_len += 1
            else:
                diff = (t_probs[i] - d_probs[i]).clamp(min=0)
                s    = diff.sum()
                corrected = (torch.multinomial(diff / s, 1).item()
                             if s > 1e-8 else t_probs[i].argmax().item())
                return accept_len, draft_list[:accept_len] + [corrected]

        bonus = torch.multinomial(t_probs[draft_len], 1).item()
        return accept_len, draft_list[:accept_len] + [bonus]

    def _greedy(self, draft_list, target_logits):
        accept_len = 0
        for i, x in enumerate(draft_list):
            if target_logits[i].argmax().item() == x:
                accept_len += 1
            else:
                break
        corrected = target_logits[accept_len].argmax().item()
        return accept_len, draft_list[:accept_len] + [corrected]


# ═══════════════════════════════════════════════════════════════════════════════
# Target Model Runner —— 唯一需要替换的插槽
# ═══════════════════════════════════════════════════════════════════════════════

class TargetModelRunner(ABC):

    @abstractmethod
    def prefill(self, model, input_ids: torch.Tensor) -> Tuple[torch.Tensor, GLAState]:
        """返回 (last_logits (1,vocab), state)"""
        pass

    @abstractmethod
    def run_one_round(
        self,
        model,
        last_token_id: torch.Tensor,
        draft_ids:     torch.Tensor,
        current_state: GLAState,
        accept_len:    int,
        extra:         Optional[Dict],
    ) -> Tuple[torch.Tensor, GLAState, Optional[Dict]]:
        """返回 (logits (1,T+1,vocab), new_state, extra)"""
        pass

    # ── 公共工具 ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _prefill_common(self, model, input_ids):
        out    = model(input_ids, use_cache=True, return_dict=True)
        logits = out.logits[:, -1, :]
        state  = GLAState.from_fla_cache(out.past_key_values)
        return logits, state

    def _state_to_cache(self, model, state: GLAState, device):
        """GLAState → FLA Cache（用于传给模型 forward）"""
        dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(dummy, use_cache=True, return_dict=True)
        cache = out.past_key_values
        state.write_to_fla_cache(cache)
        return cache

    def _get_hidden(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """只过 embedding 层，得到 hidden states"""
        h = model.model.embeddings(input_ids)
        return h

    @torch.no_grad()
    def _exact_logits_from_state(self, model, state: GLAState, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Reference logits from the real model forward, starting from a provided recurrent state.
        Used for debugging runner/kernel alignment.
        """
        cache = self._state_to_cache(model, state, input_ids.device)
        out = model(input_ids, past_key_values=cache, use_cache=True, return_dict=True)
        return out.logits

    def _extract_qkvgk(
        self, attn, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从 GatedLinearAttention 提取 q/k/v/gk。
        gk 已转为 logsigmoid 空间（与 FLA kernel 约定一致）。
        """
        q = rearrange(attn.q_proj(hidden), "... (h d) -> ... h d", d=attn.head_k_dim)
        k = attn.k_proj(hidden)
        v = attn.v_proj(hidden)
        gk = attn.gk_proj(hidden)
        # gk_proj 是 Sequential，最后一层可能已经是 logsigmoid
        # 若不是，手动加：
        if attn.num_kv_groups > 1:
            k, gk = (
                repeat(x, "... (h d) -> ... (h g) d", g=attn.num_kv_groups, d=attn.head_k_dim)
                for x in (k, gk)
            )
            v = repeat(v, "... (h d) -> ... (h g) d", g=attn.num_kv_groups, d=attn.head_v_dim)
        else:
            k, gk = (rearrange(x, "... (h d) -> ... h d", d=attn.head_k_dim) for x in (k, gk))
            v = rearrange(v, "... (h d) -> ... h d", d=attn.head_v_dim)

        gk = F.logsigmoid(gk) / attn.gate_logit_normalizer
        if attn.clamp_min is not None:
            gk = torch.clamp_min(gk, attn.clamp_min)
        if attn.feature_map_fn is not None:
            q, k = map(attn.feature_map_fn, (q, k))
        return q, k, v, gk

    def _attn_output(self, attn, o: torch.Tensor, normed: torch.Tensor) -> torch.Tensor:
        """
        o: (B, T, H, d_v) → 经过 g_norm_swish_gate + o_proj → (B, T, d_model)
        """
        if attn.use_output_gate:
            g = attn.g_proj(normed)
            if attn.fuse_norm_and_gate:
                g = rearrange(g, "... (h d) -> ... h d", d=attn.head_v_dim)
                o = attn.g_norm_swish_gate(o, g)
                o = rearrange(o, "... h d -> ... (h d)")
            else:
                o = rearrange(attn.g_norm(o), "... h d -> ... (h d)")
                o = o * attn.gate_fn(g)
        else:
            o = rearrange(attn.g_norm(o), "... h d -> ... (h d)")
        o = o.to(attn.o_proj.weight.dtype)
        return attn.o_proj(o)


# ── Baseline Runner ────────────────────────────────────────────────────────────

class BaselineRunner(TargetModelRunner):
    """
    AR Baseline：每次只处理 1 个 token，不做 SD 优化。
    用模型自带 forward，最公平的对照组。
    """

    @torch.no_grad()
    def prefill(self, model, input_ids):
        return self._prefill_common(model, input_ids)

    @torch.no_grad()
    def run_one_round(self, model, last_token_id, draft_ids, current_state, accept_len, extra):
        """只处理第 1 个 token，忽略其余 draft"""
        cache = self._state_to_cache(model, current_state, draft_ids.device)
        out   = model(torch.cat([last_token_id, draft_ids[:, :1]], dim=1), past_key_values=cache,
                      use_cache=True, return_dict=True)
        logits    = out.logits[:, :2]                # (1, 2, vocab): draft_1 + bonus
        new_state = GLAState.from_fla_cache(out.past_key_values)
        return logits, new_state, None


# ── Method2 Runner ─────────────────────────────────────────────────────────────

class Method2Runner(TargetModelRunner):
    """
    Method2：fused_recurrent_verify_fwd_no_hbm
    前 MT 步 commit + 后 T-MT 步 verify，单次 kernel 调用。

    ★ 替换点：修改 self._kernel ★
    """

    def __init__(self):
        module = _load_local_module(
            "local_fla_ops_gla_fused_recurrent_sd",
            "fla/ops/gla/fused_recurrent_sd.py",
        )
        self._kernel = module.fused_recurrent_verify_fwd_no_hbm

    @torch.no_grad()
    def prefill(self, model, input_ids):
        return self._prefill_common(model, input_ids)

    @torch.no_grad()
    def run_one_round(self, model, last_token_id, draft_ids, current_state, accept_len, extra):
        input_ids = torch.cat([last_token_id, draft_ids], dim=1)
        hidden   = self._get_hidden(model, input_ids)   # (B, T+1, d_model)
        new_states = []
        new_extra = {}
        cur_hidden = hidden   # 逐层更新

        for layer_idx, layer in enumerate(model.model.layers):
            normed = layer.attn_norm(cur_hidden)
            q_cur, k_cur, v_cur, gk_cur = self._extract_qkvgk(layer.attn, normed)
            s = current_state.states[layer_idx].to(torch.float32)

            prev = extra[layer_idx] if extra and layer_idx in extra else None
            if prev is not None:
                match_tokens = min(prev["q"].shape[1], accept_len + 1)
                q = torch.cat([prev["q"][:, :match_tokens], q_cur], dim=1)
                k = torch.cat([prev["k"][:, :match_tokens], k_cur], dim=1)
                v = torch.cat([prev["v"][:, :match_tokens], v_cur], dim=1)
                gk = torch.cat([prev["gk"][:, :match_tokens], gk_cur], dim=1)
            else:
                match_tokens = 0
                q, k, v, gk = q_cur, k_cur, v_cur, gk_cur

            # ── kernel ────────────────────────────────────────────────────
            o, h_mt= self._kernel(
                q=q, k=k, v=v, gk=gk,
                initial_state=s,
                match_tokens=match_tokens,
            )
            # o: (B, T+1, H, d_v), positions correspond to
            #    [predict draft_1, ..., predict bonus]
            new_states.append(h_mt)
            new_extra[layer_idx] = {"q": q_cur, "k": k_cur, "v": v_cur, "gk": gk_cur}

            attn_out  = self._attn_output(layer.attn, o, normed)
            cur_hidden = cur_hidden + attn_out
            cur_hidden = cur_hidden + layer.mlp(layer.mlp_norm(cur_hidden))

        # 最终 norm + lm_head
        if hasattr(model.model, 'norm'):
            cur_hidden = model.model.norm(cur_hidden)
        logits = model.lm_head(cur_hidden)   # (B, T+1, vocab)

        new_state = GLAState(states=new_states)
        return logits, new_state, new_extra


# ── Method3 Runner ─────────────────────────────────────────────────────────────

class Method3Runner(TargetModelRunner):
    """
    Method3：gla_sd_fused_step（chunk commit+verify 融合）

    ★ 替换点：修改 self._kernel ★
    """

    def __init__(self, debug: bool = False, debug_rounds: int = 2):
        module = _load_local_module(
            "local_fla_ops_gla_kernel_v3",
            "fla/ops/gla/gla_kernel_v3.py",
        )
        self._kernel = module.gla_sd_fused_step
        self.debug = debug
        self.debug_rounds = debug_rounds
        self._debug_round_idx = 0

    @torch.no_grad()
    def prefill(self, model, input_ids):
        return self._prefill_common(model, input_ids)

    @torch.no_grad()
    def run_one_round(self, model, last_token_id, draft_ids, current_state, accept_len, extra):
        # Method3 follows a pipelined protocol:
        # - `current_state` is the committed base state from rounds < t-1
        # - `extra` holds the previous round projections for [last_token_{t-1}, draft_{t-1}]
        # - `accept_len` is the number of accepted draft tokens from the previous round
        # This round commits `[last_token_{t-1}] + accepted_prefix_{t-1}` and verifies
        # `[last_token_t] + draft_t`.
        input_ids = torch.cat([last_token_id, draft_ids], dim=1)
        hidden   = self._get_hidden(model, input_ids)
        new_states = []
        new_kv = {}
        cur_hidden = hidden

        for layer_idx, layer in enumerate(model.model.layers):
            normed = layer.attn_norm(cur_hidden)
            q_all, k_all, v_all, gk_all = self._extract_qkvgk(layer.attn, normed)
            s = current_state.states[layer_idx]

            prev = extra[layer_idx] if extra and layer_idx in extra else None
            if prev is not None:
                k_acc = prev["k"]
                v_acc = prev["v"]
                g_acc = prev["gk"]
                commit_len = min(k_acc.shape[1], accept_len + 1)
            else:
                k_acc = v_acc = g_acc = None
                commit_len = 0

            # ── kernel ────────────────────────────────────────────────────
            o_drf, new_s = self._kernel(
                k_acc=k_acc, v_acc=v_acc, g_acc=g_acc,
                accept_len=commit_len,
                q_drf=q_all, k_drf=k_all, v_drf=v_all, g_drf=gk_all,
                s_base=s,
            )
            # o_drf: (B, T+1, H, d_v), positions correspond to
            #        [predict draft_1, ..., predict bonus]
            new_states.append(new_s)
            new_kv[layer_idx] = {"k": k_all, "v": v_all, "gk": gk_all}

            attn_out  = self._attn_output(layer.attn, o_drf, normed)
            cur_hidden = cur_hidden + attn_out
            cur_hidden = cur_hidden + layer.mlp(layer.mlp_norm(cur_hidden))

        if hasattr(model.model, 'norm'):
            cur_hidden = model.model.norm(cur_hidden)
        logits = model.lm_head(cur_hidden)                        # (B, T+1, vocab)

        if self.debug and self._debug_round_idx < self.debug_rounds:
            exact_logits = self._exact_logits_from_state(model, current_state, input_ids)
            self._print_debug_compare(logits, exact_logits, draft_ids)
            self._debug_round_idx += 1

        new_state = GLAState(states=new_states)
        return logits, new_state, new_kv

    def _print_debug_compare(self, runner_logits, exact_logits, draft_ids):
        runner_top1 = runner_logits[0].argmax(dim=-1).tolist()
        exact_top1 = exact_logits[0].argmax(dim=-1).tolist()
        max_abs = (runner_logits - exact_logits).abs().amax(dim=-1)[0].tolist()
        draft_list = draft_ids[0].tolist()

        print("\n[M3 Debug]")
        print(f"  round: {self._debug_round_idx}")
        print(f"  draft: {draft_list}")
        for i, (rt, et, diff) in enumerate(zip(runner_top1, exact_top1, max_abs)):
            tag = "OK" if rt == et else "DIFF"
            print(f"  pos {i}: runner_top1={rt} exact_top1={et} max_abs={diff:.4f} [{tag}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 主框架
# ═══════════════════════════════════════════════════════════════════════════════

class SDFramework:
    """
    换 kernel 只需换 runner：
      SDFramework(target_model, tokenizer, Method3Runner(), drafter, draft_len=4)
    """

    def __init__(self, model, tokenizer, runner: TargetModelRunner,
                 drafter, draft_len: int = 4, temperature: float = 1.0):
        self.model      = model
        self.tokenizer  = tokenizer
        self.runner     = runner
        self.drafter    = drafter
        self.draft_len  = draft_len
        self.verifier   = SpeculativeSampling(temperature=temperature)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200) -> GenerationResult:
        device    = next(self.model.parameters()).device
        model_inputs = _encode_prompt(self.tokenizer, prompt, device)
        input_ids = model_inputs["input_ids"]

        # ── Prefill ───────────────────────────────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        first_logits, target_state = self.runner.prefill(self.model, input_ids)
        first_token = first_logits[0].argmax().item()
        draft_state = self.drafter.prefill(input_ids)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        prefill_time = t1 - t0

        generated_ids  = [first_token]
        last_token_id  = torch.tensor([[first_token]], device=device)

        # ── Decode ────────────────────────────────────────────────────────────
        total_drafted = total_accepted = num_rounds = 0
        accept_len = 0
        extra = None

        torch.cuda.synchronize()
        t_dec_start = time.perf_counter()

        while len(generated_ids) < max_new_tokens:
            remaining     = max_new_tokens - len(generated_ids)
            cur_draft_len = min(self.draft_len, remaining)
            if cur_draft_len <= 0:
                break

            # Draft
            draft_state_before_round = draft_state
            draft_ids, draft_logits, draft_state = self.drafter.draft(
                last_token_id, cur_draft_len, draft_state
            )

            # Target forward
            if isinstance(self.runner, (Method2Runner, Method3Runner)):
                target_logits, target_state, next_extra = self.runner.run_one_round(
                    self.model, last_token_id, draft_ids, target_state, accept_len, extra
                )
            else:
                target_logits, target_state, next_extra = self.runner.run_one_round(
                    self.model, last_token_id, draft_ids, target_state, 0, extra
                )

            # Verify
            accept_len, accepted_ids = self.verifier.verify(
                draft_ids, target_logits, draft_logits
            )

            commit_ids = [last_token_id]
            if accept_len > 0:
                accepted_prefix = torch.tensor(
                    [accepted_ids[:-1]],
                    dtype=torch.long,
                    device=device,
                )
                commit_ids.append(accepted_prefix)
            commit_input_ids = torch.cat(commit_ids, dim=1)

            # Keep draft synchronized to the prefix that ends right before the
            # new `last_token_id`. The new last token is carried separately.
            draft_state = self.drafter.commit(draft_state_before_round, commit_input_ids)
            extra = next_extra

            generated_ids.extend(accepted_ids)
            last_token_id = torch.tensor([[accepted_ids[-1]]], device=device)

            total_drafted  += cur_draft_len
            total_accepted += accept_len
            num_rounds     += 1

            if self.tokenizer.eos_token_id in accepted_ids:
                break

        torch.cuda.synchronize()
        t_dec_end   = time.perf_counter()
        decode_time = t_dec_end - t_dec_start
        total_time  = prefill_time + decode_time
        n           = len(generated_ids)

        return GenerationResult(
            generated_ids   = generated_ids,
            generated_text  = self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            total_time_s    = total_time,
            tokens_per_sec  = n / max(decode_time, 1e-6),
            prefill_time_s  = prefill_time,
            decode_time_s   = decode_time,
            total_drafted   = total_drafted,
            total_accepted  = total_accepted,
            num_rounds      = num_rounds,
            acceptance_rate = total_accepted / max(total_drafted, 1),
            avg_accept_len  = total_accepted / max(num_rounds, 1),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AR Baseline
# ═══════════════════════════════════════════════════════════════════════════════

class ARFramework:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200) -> GenerationResult:
        device    = next(self.model.parameters()).device
        model_inputs = _encode_prompt(self.tokenizer, prompt, device)
        input_ids = model_inputs["input_ids"]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        gen_ids = output[0, input_ids.shape[1]:].tolist()
        elapsed = t1 - t0
        return GenerationResult(
            generated_ids  = gen_ids,
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True),
            total_time_s   = elapsed,
            tokens_per_sec = len(gen_ids) / max(elapsed, 1e-6),
            decode_time_s  = elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 打印
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(r: GenerationResult, name: str):
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"  速度:      {r.tokens_per_sec:.2f} tokens/s")
    print(f"  生成:      {len(r.generated_ids)} tokens  ({r.total_time_s:.3f} s)")
    print(f"  Prefill:  {r.prefill_time_s:.3f} s  |  Decode: {r.decode_time_s:.3f} s")
    if r.total_drafted > 0:
        print(f"  接受率:   {r.acceptance_rate:.3f}  |  平均接受: {r.avg_accept_len:.2f}/round")
        print(f"  总轮数:   {r.num_rounds}")
    print(f"  文本:     {r.generated_text[:80]}...")


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, fla
    import fla.models.gla  # ensure `gla` is registered with Hugging Face Auto classes
    from transformers import AutoModelForCausalLM, AutoTokenizer

    PATHS = {
        "340M": "/data1/hxy/hf_cache/fla-hub-gla-340M-15B",
        "2.7B": "/data1/hxy/hf_cache/fla-hub-gla-2.7B-100B",
    }
    PROMPT = "The history of artificial intelligence begins in antiquity"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         choices=["baseline","sd-m2","sd-m3"], default="baseline")
    parser.add_argument("--target-model", choices=["340M","2.7B"], default="2.7B")
    parser.add_argument("--max-tokens",   type=int, default=100)
    parser.add_argument("--draft-len",    type=int, default=4)
    parser.add_argument("--ngram-size",   type=int, default=4)
    parser.add_argument("--debug-m3",     action="store_true")
    parser.add_argument("--debug-rounds", type=int, default=2)
    args = parser.parse_args()

    print(f"[加载 target: {args.target_model}]")
    target = AutoModelForCausalLM.from_pretrained(
        PATHS[args.target_model], torch_dtype=torch.bfloat16).cuda().eval()
    tok = AutoTokenizer.from_pretrained(PATHS[args.target_model])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # AR Baseline
    ar = ARFramework(target, tok)
    r_base = ar.generate(PROMPT, max_new_tokens=args.max_tokens)
    print_result(r_base, "AR Baseline")

    if args.mode == "baseline":
        exit(0)

    print(f"[使用 n-gram draft: n={args.ngram_size}]")
    drafter = NGramDrafter(vocab_size=target.config.vocab_size, ngram_size=args.ngram_size)

    runner = (
        Method2Runner()
        if args.mode == "sd-m2"
        else Method3Runner(debug=args.debug_m3, debug_rounds=args.debug_rounds)
    )
    label  = "SD-Method2" if args.mode == "sd-m2" else "SD-Method3"

    fw = SDFramework(target, tok, runner, drafter,
                     draft_len=args.draft_len, temperature=0.0)
    r_sd = fw.generate(PROMPT, max_new_tokens=args.max_tokens)
    print_result(r_sd, label)

    print(f"\n加速比: {r_sd.tokens_per_sec / r_base.tokens_per_sec:.2f}x")

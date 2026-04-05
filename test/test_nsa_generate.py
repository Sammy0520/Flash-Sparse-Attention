"""
NSA / FSA Autoregressive Generation Test
==========================================
加载训练好的稀疏注意力 checkpoint，使用普通（非投机解码）自回归方式续写 prompt，
验证 NSA / FSA 在非 SD 情境下是否工作正常。

核心流程：
  1. 加载 LLaMA + 训练好的稀疏注意力权重
  2. NSA prefill：用 FlashSparseAttentionDecode hook 替换所有层的 self_attn
  3. 逐 token 自回归 decode：每步跑一次 LLaMA forward（NSA hook 激活），
     从 logits 采样，commit 新 KV 到压缩缓存
  4. 打印生成文本 + 速度指标

用法：
  python test/test_nsa_generate.py --nsa-ckpt checkpoints/nsa_distill/final
  python test/test_nsa_generate.py --nsa-ckpt checkpoints/nsa_distill/final \\
      --prompt "Once upon a time" --max-new-tokens 128 --temperature 0.8
  python test/test_nsa_generate.py --nsa-ckpt /root/autodl-tmp/nsa_ckpt_simplebooks/final/ \\
      --temperature 0.8 --top-p 0.95 --seqlen 2048 --max-new-tokens 1024
"""

import argparse
import os
import sys
import time
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

LLAMA_8B = "/root/autodl-tmp/models/Llama-3.1-8B-Instruct"

# ── 默认 prompt（来自 e2e_sd_framework.py）───────────────────────────
DEFAULT_PROMPT = (
    "Sammy 's Flying Machine\n\n"
    "I\n\n"
    "Sammy Red Squirrel was sitting on the stone wall eating a nut . "
    "\" Caw , caw ! \" called Blacky Crow , as he flew over the field . "
    "\" Caw , caw , caw ! \" he called . \" What are you doing , Sammy ? \" "
    "Sammy stopped eating the nut , and looked up to see who was talking to him . "
    "He saw Blacky Crow sailing round and round over his head . "
    "\" I am eating my breakfast , \" he answered . \" Would you like to have a nut to eat , too ? \" "
    "\" Oh , no , \" answered Blacky Crow . \" I can find something better than that . "
    "I am going to the pasture now to get my breakfast . \" "
    "Then Blacky Crow flapped his big wings and flew far , far away . "
    "Sammy watched the crow fly over the tallest tree and out of sight . "
    "\" I wish I could fly , \" he said to himself . \" I know I could if I had some wings . \" "
    "Just then a flock of sparrows flew over head . "
    "\" Twitter , twitter ! \" they said . "
    "\" Twitter , twitter , twitter ! \" "
    "Sammy watched the sparrows flying until they were out of sight . "
    "\" I know I could fly , \" he said to himself again , \" if I had some wings . "
    "Perhaps I could make some wings , \" he thought . "
    "Just then something hit Sammy on the head . "
    "He looked up to see what it was , and there at his feet lay an oak leaf . "
    "He looked up in the top of the tree . "
    "West Wind flew by and shook the branches of the tree very gently . "
    "And another leaf floated softly down to the ground beside its brother . "
    "Sammy sat there watching the leaves for a few minutes . "
    "Then he jumped up and clapped his hands . "
    "\" I know what I can do , \" he said . \" I can make some wings for myself out of those oak leaves . "
    "I will ask all the other squirrels to come and watch me fly . \" "
    "Sammy hunted on the ground until he found two very large oak leaves . "
    "\" I can hold them out with my front paws , \" he said . \" I think they will look just like wings . \" "
    "Sammy put the two leaves on the ground and covered them with a stone . "
    "He was not going to let West Wind carry them away . "
    "Then he scampered off to tell all the other squirrels what he was going to do . "
    "He told all the red squirrels first . "
    "He told them he was going to fly from the big oak tree . "
    "\" If you wish to see me fly , \" he said , \" you must be at the tree in a few minutes . \" "
    "All the red squirrels scampered off to get the best seats among the branches of the oak tree . "
    "Sammy saw Bobby Gray Squirrel and told him to ask all the gray squirrels to come and see him fly . "
    "Then Sammy found Bunny Rabbit . "
    "When Bunny heard what Sammy was going to do , he wanted to try to fly , too . "
    "\" You are much too large for my wings , \" said Sammy . "
    "\" You would have to go to Mr. Man 's garden and ask him for some of the leaves from the rhubarb plants . \" "
    "Blacky Crow was flying over the field . He heard Sammy tell Bunny that he was going to fly . "
    "\" Ho , ho ! \" he laughed , \" I should like to see Sammy fly with those oak - leaf wings . "
    "I will fly to the oak tree this very minute . \" "
    "As he flew over the meadow he saw the sparrows and told them where he was going . "
    "They wanted to go , too . "
    "Every one wanted to go and watch Sammy fly . \n\n"
    "II\n\n"
    "When they were all seated , Sammy picked up the two leaves he had found and skipped gaily up the tree . "
    "He ran up the tree and out on one of the longest branches . "
    "\" Now , watch me ! \" he called to all his friends . "
    "\" See me fly just like a bird . \" "
    "Sammy took one leaf in each of his front paws and held them out as far as he could . "
    "He stood on the very end of the branch for just one minute . "
    "He saw that every one was watching him . "
    "\" You must flap your wings , \" called Blacky Crow . "
    "\" Hop off the branch , \" called one of the sparrows . "
    "So Sammy flapped his wings , and then he hopped off the branch . "
    "But , oh , dear me ! The wings would not hold Sammy up in the air . "
    "Sammy forgot to hold his wings out straight and they hung down at his side without a flutter . "
    "And down to the ground Sammy fell . "
    "Bump ! he came down at the foot of the oak tree . "
    "He almost fell on top of Bunny Rabbit . "
    "But Bunny saw him coming and jumped out of the way just in time . "
    "Sammy lay very still where he had fallen . "
    "All the squirrels ran down to see if he had hurt himself . "
    "Bobby Gray Squirrel ran to pick the fallen bird up from the ground . "
    "Sammy had given his nose such a bump that it was all black and blue . "
    "He had hurt his paw . And his make - believe wings were all crushed and broken . "
    "Sammy rubbed his nose and then he looked at his friends . "
    "\" I do n't believe oak leaves make good wings , \" he said . "
    "\" No , \" said the tiniest sparrow , \" the best wings are made of feathers . \" "
    "\" Caw , caw ! \" said Blacky Crow . \" My wings are made of feathers . See how I can fly . \" "
    "Then Blacky Crow flapped his big wings and flew away . "
    "The sparrows flew away , too . "
    "All the squirrels scampered off to hunt for nuts . "
    "And the rabbits went back to their home to take a nap . "
    "Sammy was left sitting alone on the old stone wall . "
    "Every few minutes he rubbed his poor little nose . "
    "And as he rubbed his nose he thought : "
    "\" Flying may be fun for birds , and swimming may be fun for ducks . "
    "But running and jumping among the branches of the big oak tree is more fun for squirrels . \"\n\n"
    "III\n\n"
    "The next morning Sammy woke up very early . "
    "His nose still hurt a little , and his paw was stiff and sore . "
    "But Sammy was not the kind of squirrel to give up so easily . "
    "He sat up in his nest and looked out at the sky . "
    "The sun was just coming up over the hills . "
    "The birds were beginning to sing in the tall trees . "
    "\" I will try again , \" said Sammy to himself . "
    "\" This time I will make better wings . \" "
    "He crept out of his nest and ran down the tree very slowly , for his paw still hurt him . "
    "Bobby Gray Squirrel was already up and sitting on a branch eating a nut . "
    "\" Good morning , Sammy , \" he said . \" How is your nose today ? \" "
    "\" It is better , thank you , \" said Sammy . "
    "\" I am going to make a new flying machine . \" "
    "Bobby nearly choked on his nut . "
    "\" Not another one ! \" he cried . "
    "\" Yes , \" said Sammy . \" But this time I will do it right . "
    "I watched the birds all yesterday afternoon , and I know what I did wrong . \" "
    "Bobby shook his head , but he followed Sammy down to the ground . "
    "Sammy looked all around the field and the edge of the woods . "
    "He found a long thin twig that had fallen from the oak tree . "
    "He found two more large oak leaves , bigger than the ones he had used before . "
    "\" I will tie the leaves to the twig , \" he said , \" so they cannot fold up when I jump . \" "
    "\" How will you tie them ? \" asked Bobby . "
    "Sammy thought for a moment . "
    "\" I will use some of the long grass by the brook , \" he said . "
    "So the two squirrels ran to the brook and pulled up several long blades of grass . "
    "Sammy worked for a long time . "
    "He tied one leaf to each end of the twig very carefully . "
    "Bobby sat and watched and ate two more nuts while Sammy worked . "
    "At last Sammy held up his new flying machine and looked at it proudly . "
    "\" That is much better , \" he said . "
    "\" The leaves will stay out straight now . \" "
    "Bobby looked at the flying machine . "
    "\" It does look more like wings , \" he said . "
    "\" But are you really going to jump out of the tree again ? \" "
    "\" Yes , \" said Sammy . \" But this time I will not jump from so high up . "
    "I will jump from the lowest branch first . \" "
    "Bunny Rabbit came hopping across the field just then . "
    "When he saw Sammy with the new flying machine , his eyes grew very wide . "
    "\" Oh , Sammy , \" he said , \" are you going to try to fly again ? "
    "I do not want to see you fall down and bump your nose again . \" "
    "\" I will not fall this time , \" said Sammy . "
    "But all the same , Bunny took a few steps back , just to be safe . "
    "Sammy ran up the oak tree and climbed out on the lowest branch . "
    "It was not very far from the ground . "
    "He held the twig in his front paws and spread the leaves out wide . "
    "The leaves stayed out straight just as he had hoped . "
    "\" Now , \" said Sammy , and he jumped . "
    "This time the wings did not fold up . "
    "For just one moment Sammy seemed to float a little in the air . "
    "Then he came down to the ground , but much more gently than before . "
    "He did not go bump at all . "
    "He landed softly on all four paws . "
    "Bobby stared at him with his mouth open . "
    "Bunny Rabbit sat up on his hind legs and clapped his front paws together . "
    "\" You did it ! \" cried Bobby . \" You almost flew ! \" "
    "\" I almost flew , \" said Sammy happily . "
)


def clamp_token_id(token_id: int, vocab_size: int) -> int:
    return max(0, min(int(token_id), vocab_size - 1))


def sample_from_logits(logits: torch.Tensor, temperature: float,
                       top_p: float, vocab_size: int) -> int:
    """从 logits 采样一个 token，支持 temperature + top-p (nucleus)。"""
    if temperature <= 0:
        return clamp_token_id(logits.argmax(dim=-1).item(), vocab_size)

    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    if 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        idx = torch.multinomial(sorted_probs, 1).item()
        return clamp_token_id(sorted_idx[idx].item(), vocab_size)

    s = float(probs.sum().item())
    if s <= 0:
        return clamp_token_id(logits.argmax(dim=-1).item(), vocab_size)
    probs = probs / s
    return clamp_token_id(torch.multinomial(probs, 1).item(), vocab_size)


def build_prompt_ids(tokenizer, prompt: str, seqlen: int,
                     device: str) -> torch.Tensor:
    """将 prompt tokenize 并填充/截断到 seqlen。"""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if ids.shape[1] >= seqlen:
        return ids[:, :seqlen]
    # 不够长则重复拼接
    reps = seqlen // ids.shape[1] + 2
    return ids.repeat(1, reps)[:, :seqlen]


# ─────────────────────────────────────────────────────────────────────────────
# LlamaNSALayer：包装 FlashSparseAttentionDecode，复制 LLaMA 权重
# ─────────────────────────────────────────────────────────────────────────────

class LlamaNSALayer(nn.Module):
    def __init__(self, llama_attn, cfg, topk=16,
                 block_size=64, kernel_size=32, kernel_stride=16,
                 init_blocks=1, local_blocks=2, window_size=512):
        super().__init__()
        num_q  = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_d = getattr(cfg, "head_dim", cfg.hidden_size // num_q)

        rope_cfg = RopeConfig(
            max_position_embeddings=cfg.max_position_embeddings,
            head_dim=head_d, rope_theta=cfg.rope_theta,
            rope_scaling=getattr(cfg, "rope_scaling", None),
        )
        self.fsa = FlashSparseAttentionDecode(
            hidden_size=cfg.hidden_size,
            num_q_heads=num_q, num_kv_heads=num_kv, head_dim=head_d,
            kernel_size=kernel_size, kernel_stride=kernel_stride,
            block_size=block_size, topk=topk,
            init_blocks=init_blocks, local_blocks=local_blocks,
            window_size=window_size, rope_config=rope_cfg,
        )
        # 复制 LLaMA 投影权重
        with torch.no_grad():
            self.fsa.proj_q.weight.copy_(llama_attn.q_proj.weight)
            self.fsa.proj_k.weight.copy_(llama_attn.k_proj.weight)
            self.fsa.proj_v.weight.copy_(llama_attn.v_proj.weight)
            self.fsa.proj_o.weight.copy_(llama_attn.o_proj.weight)

        self.kernel_size   = kernel_size
        self.kernel_stride = kernel_stride

    def build_compressed_cache(self, k_raw, v_raw, cu_k):
        cmp_k, _ = linear_compress(
            k_raw, self.fsa.compress_key, cu_k,
            self.kernel_size, self.kernel_stride, self.fsa.intra_block_pe,
        )
        cmp_v, _ = linear_compress(
            v_raw, self.fsa.compress_value, cu_k,
            self.kernel_size, self.kernel_stride, None,
        )
        return cmp_k, cmp_v

    def forward(self, hidden, k_raw, k_buffer, v_raw, cmp_k, cmp_v,
                cu_q, cu_k, position_ids, kv_commit_stash=None):
        return self.fsa(
            x=hidden,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            k_cache=k_raw, k_buffer=k_buffer, v_cache=v_raw,
            cmp_k_cache=cmp_k, cmp_v_cache=cmp_v,
            attention_mask=None, position_ids=position_ids,
            kv_commit_stash=kv_commit_stash,
        )


# ─────────────────────────────────────────────────────────────────────────────
# NSAGenerator：管理 prefill + autoregressive decode
# ─────────────────────────────────────────────────────────────────────────────

class NSAGenerator:
    """
    管理 LLaMA + NSA 层的 prefill 和 autoregressive decode。
    所有注意力层通过 hook 替换为 NSA 稀疏注意力。
    """

    def __init__(self, llama, nsa_layers: List[LlamaNSALayer]):
        self.llama      = llama
        self.nsa_layers = nsa_layers
        n = len(nsa_layers)

        # 每层 KV cache
        self.k_raw    = [None] * n   # rope(K)   [seq, nkv, hd]
        self.k_buffer = [None] * n   # raw K     (最近 kernel_size-1 个)
        self.v_raw    = [None] * n   # raw V     [seq, nkv, hd]
        self.cmp_k    = [None] * n   # 压缩 K
        self.cmp_v    = [None] * n   # 压缩 V

        self.past_len = 0
        self._orig_forwards: Dict[int, object] = {}
        self._verify_mode   = False
        self._pos_ids       = None
        self._stash_kv_new: List[Optional[Tuple]] = [None] * n
        self._cu_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    # ── helpers ────────────────────────────────────────────────────────

    def _get_cu(self, device, end):
        key = (int(end), device.index if device.type == "cuda" else -1)
        t = self._cu_cache.get(key)
        if t is None:
            t = torch.tensor([0, int(end)], device=device, dtype=torch.int32)
            self._cu_cache[key] = t
        return t

    # ── hook 管理 ──────────────────────────────────────────────────────

    def _make_nsa_forward(self, layer_idx: int):
        target = self
        nsa    = self.nsa_layers[layer_idx]

        def nsa_forward(hidden_states=None, *args, _lid=layer_idx, **kwargs):
            if not target._verify_mode:
                return target._orig_forwards[_lid](hidden_states, *args, **kwargs)

            n = hidden_states.shape[1]
            hidden_flat = hidden_states.squeeze(0)  # [n, H]
            past_kv_len = target.k_raw[_lid].shape[0]
            total_k_len = past_kv_len + n

            cu_q = target._get_cu(hidden_states.device, n)
            cu_k = target._get_cu(hidden_states.device, total_k_len)

            stash: List = []
            out = nsa(
                hidden_flat,
                target.k_raw[_lid], target.k_buffer[_lid],
                target.v_raw[_lid],
                target.cmp_k[_lid], target.cmp_v[_lid],
                cu_q, cu_k, target._pos_ids,
                kv_commit_stash=stash,
            )
            if len(stash) != 1:
                raise RuntimeError(f"kv_commit_stash expected 1 tuple, got {len(stash)}")
            k_new_rope, k_new, v_new = stash[0]
            target._stash_kv_new[_lid] = (
                k_new_rope.detach(), k_new.detach(), v_new.detach(),
            )
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

    # ── prefill ────────────────────────────────────────────────────────

    @torch.no_grad()
    def nsa_prefill(self, input_ids: torch.Tensor):
        """
        用 FlashSparseAttentionDecode hook 替换全部 self_attn 做 prefill，
        返回 (logits, past_key_values_hf, k_nope_storage)。
        """
        device = input_ids.device
        dtype  = next(self.llama.parameters()).dtype

        orig_forwards = {}
        k_nope_storage = {}
        past_kv_list   = []

        for l, layer in enumerate(self.llama.model.layers):
            orig_forwards[l] = layer.self_attn.forward
            fsa_layer = self.nsa_layers[l]

            def make_hook(_lid, _fsa):
                def fsa_prefill_forward(hidden_states=None, *args, **kwargs):
                    bsz, seq_len, hsz = hidden_states.shape
                    hidden_flat = hidden_states.reshape(-1, hsz)
                    cu = torch.arange(0, (bsz + 1) * seq_len, seq_len,
                                      device=hidden_flat.device, dtype=torch.int32)
                    pos_flat = (torch.arange(seq_len, device=hidden_flat.device,
                                            dtype=torch.long)
                                .unsqueeze(0).expand(bsz, -1).reshape(-1))

                    fsa = _fsa.fsa
                    empty = torch.empty(0, fsa.num_kv_heads, fsa.head_dim,
                                        device=hidden_flat.device,
                                        dtype=hidden_flat.dtype)
                    stash = []
                    attn_out = fsa(
                        hidden_flat, cu, cu,
                        empty, empty, empty, empty, empty,
                        attention_mask=None,
                        position_ids=pos_flat,
                        kv_commit_stash=stash,
                    )
                    k_rope_new, k_nope_new, v_new = stash[0]
                    k_nope_storage[_lid] = k_nope_new.detach()

                    k_hf = (k_rope_new
                            .view(bsz, seq_len, fsa.num_kv_heads, fsa.head_dim)
                            .permute(0, 2, 1, 3).contiguous())
                    v_hf = (v_new
                            .view(bsz, seq_len, fsa.num_kv_heads, fsa.head_dim)
                            .permute(0, 2, 1, 3).contiguous())
                    past_kv_list.append((k_hf.detach(), v_hf.detach()))
                    return (attn_out.reshape(bsz, seq_len, hsz), None, None)
                return fsa_prefill_forward
            layer.self_attn.forward = make_hook(l, fsa_layer)

        try:
            out = self.llama(input_ids, use_cache=False, num_logits_to_keep=1)
        finally:
            for l, layer in enumerate(self.llama.model.layers):
                layer.self_attn.forward = orig_forwards[l]

        return out, tuple(past_kv_list), k_nope_storage

    def init_from_prefill(self, past_key_values, k_nope_storage):
        """从 prefill 的 HF 格式 KV cache 初始化 NSA 缓存。"""
        device = self.nsa_layers[0].fsa.proj_q.weight.device
        dtype  = self.nsa_layers[0].fsa.proj_q.weight.dtype
        self.past_len = past_key_values[0][0].shape[2]  # [1, nkv, seq, hd]

        for l, nsa in enumerate(self.nsa_layers):
            k, v = past_key_values[l]
            k_raw = k.squeeze(0).permute(1, 0, 2).contiguous().to(dtype)
            v_raw = v.squeeze(0).permute(1, 0, 2).contiguous().to(dtype)
            cu_k  = torch.tensor([0, k_raw.shape[0]], device=device,
                                 dtype=torch.int32)

            k_raw_nope = k_nope_storage[l]
            if k_raw_nope.dim() == 2:
                seq_len = k_raw_nope.shape[0]
                k_raw_nope = k_raw_nope.view(
                    seq_len, nsa.fsa.num_kv_heads, nsa.fsa.head_dim)
            elif k_raw_nope.dim() == 3 and k_raw_nope.shape[0] == 1:
                seq_len = k_raw_nope.shape[1]
                k_raw_nope = k_raw_nope.view(
                    seq_len, nsa.fsa.num_kv_heads, nsa.fsa.head_dim)

            cmp_k, cmp_v = nsa.build_compressed_cache(k_raw_nope, v_raw, cu_k)

            buffer_size = min(nsa.kernel_size - 1, k_raw.shape[0])
            self.k_buffer[l] = k_raw_nope[-buffer_size:].contiguous()
            self.k_raw[l]    = k_raw
            self.v_raw[l]    = v_raw
            self.cmp_k[l]    = cmp_k
            self.cmp_v[l]    = cmp_v

    # ── decode step ────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_step(self, token_id: int) -> torch.Tensor:
        """
        用 NSA hook 跑一次 LLaMA forward（1 token），返回 logits [vocab_size]。
        同时 commit 新 KV 到缓存。
        """
        device = self.k_raw[0].device
        ids    = torch.tensor([[token_id]], device=device, dtype=torch.long)
        n      = 1

        self._stash_kv_new = [None] * len(self.nsa_layers)
        self._pos_ids = torch.tensor(
            [self.past_len], device=device, dtype=torch.long)
        self._verify_mode = True

        out = self.llama(
            ids,
            position_ids=self._pos_ids.unsqueeze(0),
            use_cache=False,
            num_logits_to_keep=1,
        )
        self._verify_mode = False

        # commit KV
        self._commit(commit_len=n)

        return out.logits.squeeze(0).squeeze(0)  # [vocab_size]

    def _commit(self, commit_len: int):
        """将 decode 阶段暂存的 KV 追加到缓存并增量压缩。"""
        if commit_len <= 0:
            return
        dtype = self.k_raw[0].dtype

        for l, nsa in enumerate(self.nsa_layers):
            st = self._stash_kv_new[l]
            if st is None:
                raise RuntimeError(
                    f"commit: layer {l} missing stashed KV")
            k_new_rope, k_new, v_new = st
            k_new_rope = k_new_rope[:commit_len].contiguous().to(dtype)
            k_new      = k_new[:commit_len].contiguous().to(dtype)
            v_new      = v_new[:commit_len].contiguous().to(dtype)

            prev_raw_len = self.k_raw[l].shape[0]
            buffer_size  = min(nsa.kernel_size - 1, prev_raw_len)
            if buffer_size > 0:
                v_buffer = self.v_raw[l][-buffer_size:]
                k_buffer = self.k_buffer[l]
            else:
                k_buffer = v_buffer = None

            decode_k = _linear_compress_decode(
                k_new, nsa.fsa.compress_key,
                nsa.kernel_size, nsa.kernel_stride,
                nsa.fsa.intra_block_pe, prev_raw_len, k_buffer,
            )
            decode_v = _linear_compress_decode(
                v_new, nsa.fsa.compress_value,
                nsa.kernel_size, nsa.kernel_stride,
                None, prev_raw_len, v_buffer,
            )

            if decode_k is not None:
                self.cmp_k[l] = (torch.cat([self.cmp_k[l], decode_k], dim=0)
                                 if self.cmp_k[l] is not None else decode_k)
                self.cmp_v[l] = (torch.cat([self.cmp_v[l], decode_v], dim=0)
                                 if self.cmp_v[l] is not None else decode_v)

            self.k_raw[l] = torch.cat([self.k_raw[l], k_new_rope], dim=0)
            self.v_raw[l] = torch.cat([self.v_raw[l], v_new], dim=0)
            self.k_buffer[l] = torch.cat(
                [self.k_buffer[l], k_new], dim=0)[-buffer_size:]

        self.past_len += commit_len
        self._stash_kv_new = [None] * len(self.nsa_layers)

    # ── 完整生成 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float, top_p: float,
                 tokenizer=None) -> Tuple[List[int], dict]:
        """
        完整的 NSA 自回归生成：prefill + token-by-token decode。
        返回 (generated_ids, stats_dict)。
        """
        vocab_size = self.llama.config.vocab_size

        # ── prefill ──
        t0 = time.time()
        out, past_kv, k_nope = self.nsa_prefill(input_ids)
        self.init_from_prefill(past_kv, k_nope)
        t_prefill = time.time() - t0
        prefill_len = input_ids.shape[1]
        print(f"  Prefill: {prefill_len} tokens in {t_prefill:.2f}s "
              f"({prefill_len / t_prefill:.0f} tok/s)")

        # ── 安装 decode hook ──
        self._patch_attn()

        # ── 采样第一个 token ──
        cur = sample_from_logits(
            out.logits[:, -1, :].squeeze(0).float(),
            temperature, top_p, vocab_size,
        )
        generated = [cur]

        # ── 逐 token decode ──
        t_decode_start = time.time()
        for step in range(1, max_new_tokens):
            logits = self.decode_step(cur)
            cur = sample_from_logits(logits.float(), temperature, top_p,
                                     vocab_size)
            generated.append(cur)

            # EOS 检测
            if tokenizer and cur == tokenizer.eos_token_id:
                break

            if (step + 1) % 32 == 0:
                elapsed = time.time() - t_decode_start
                print(f"  Decode: {step + 1}/{max_new_tokens} tokens "
                      f"({(step + 1) / elapsed:.1f} tok/s)", flush=True)

        t_decode = time.time() - t_decode_start
        self._restore_attn()

        stats = {
            "prefill_len":   prefill_len,
            "prefill_time":  t_prefill,
            "generated_len": len(generated),
            "decode_time":   t_decode,
            "decode_tok_s":  len(generated) / max(t_decode, 1e-6),
            "total_time":    t_prefill + t_decode,
        }
        return generated, stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NSA autoregressive generation test "
                    "(non-speculative, FlashSparseAttentionDecode)")
    parser.add_argument("--model", default=LLAMA_8B,
                        help="LLaMA model path")
    parser.add_argument("--nsa-ckpt", default=None,
                        help="训练好的稀疏注意力 checkpoint 目录或 ckpt.pt "
                             "（若不指定则用均值池初始化）")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--seqlen", type=int, default=1024,
                        help="prompt 填充/截断到此长度")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="nucleus sampling top-p（1.0 = 不启用）")
    parser.add_argument("--topk", type=int, default=16,
                        help="稀疏注意力 topk（须与训练一致）")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--kernel-size", type=int, default=32)
    parser.add_argument("--kernel-stride", type=int, default=16)
    parser.add_argument("--init-blocks", type=int, default=1)
    parser.add_argument("--local-blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--greedy", action="store_true",
                        help="贪心解码（temperature=0）")
    args = parser.parse_args()

    if args.greedy:
        args.temperature = 0.0

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device, dtype = "cuda", torch.bfloat16

    # ── 1. 加载 LLaMA ─────────────────────────────────────────────────
    print(f"Loading LLaMA from {args.model} ...")
    tok   = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    llama = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device).eval()
    for p in llama.parameters():
        p.requires_grad_(False)
    cfg = llama.config
    print(f"  {cfg.num_hidden_layers} layers, hidden={cfg.hidden_size}, "
          f"vocab={cfg.vocab_size}")

    # ── 2. 构建 NSA 层 ────────────────────────────────────────────────
    print("Building NSA layers (FlashSparseAttentionDecode) ...")
    nsa_layers = []
    for l in range(cfg.num_hidden_layers):
        nsa = LlamaNSALayer(
            llama.model.layers[l].self_attn, cfg,
            topk=args.topk, block_size=args.block_size,
            kernel_size=args.kernel_size, kernel_stride=args.kernel_stride,
            init_blocks=args.init_blocks, local_blocks=args.local_blocks,
            window_size=args.window_size,
        ).to(device, dtype)
        nsa_layers.append(nsa)

    # ── 3. 加载 checkpoint ────────────────────────────────────────────
    if args.nsa_ckpt:
        ckpt_path = args.nsa_ckpt
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "ckpt.pt")
        print(f"Loading NSA checkpoint from {ckpt_path} ...")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        nsa_sd = ckpt["nsa"]
        for l_key, sd in nsa_sd.items():
            nsa_layers[int(l_key)].fsa.load_state_dict(sd, strict=False)
        # 强制从 LLaMA 重新复制 proj 权重（ckpt 中可能保存了旧的）
        with torch.no_grad():
            for l_idx, nsa in enumerate(nsa_layers):
                la = llama.model.layers[l_idx].self_attn
                nsa.fsa.proj_q.weight.copy_(la.q_proj.weight)
                nsa.fsa.proj_k.weight.copy_(la.k_proj.weight)
                nsa.fsa.proj_v.weight.copy_(la.v_proj.weight)
                nsa.fsa.proj_o.weight.copy_(la.o_proj.weight)
        hparams = ckpt.get("nsa_hparams", {})
        attn_mode = ckpt.get("attn_mode", ckpt.get("real_fsa", "unknown"))
        print(f"  Checkpoint loaded: {len(nsa_sd)} layers, "
              f"step={ckpt.get('step', '?')}, mode={attn_mode}")
        if hparams:
            print(f"  Hparams: {hparams}")
    else:
        print("  No checkpoint specified; using default initialization.")

    # ── 4. 准备 prompt ────────────────────────────────────────────────
    input_ids = build_prompt_ids(tok, args.prompt, args.seqlen, device)
    prompt_text = tok.decode(input_ids[0, :512].tolist(), skip_special_tokens=True)
    print(f"\nPrompt ({input_ids.shape[1]} tokens): \"{prompt_text}...\"")
    print(f"Generating {args.max_new_tokens} tokens "
          f"(temp={args.temperature}, top_p={args.top_p}) ...\n")

    # ── 5. 生成 ──────────────────────────────────────────────────────
    generator = NSAGenerator(llama, nsa_layers)
    generated_ids, stats = generator.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tokenizer=tok,
    )

    # ── 6. 输出 ──────────────────────────────────────────────────────
    text = tok.decode(generated_ids, skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"Generated text ({len(generated_ids)} tokens):")
    print(f"{'='*60}")
    print(text)
    print(f"{'='*60}")
    print(f"\nStats:")
    print(f"  Prefill:  {stats['prefill_len']} tok in {stats['prefill_time']:.2f}s "
          f"({stats['prefill_len']/stats['prefill_time']:.0f} tok/s)")
    print(f"  Decode:   {stats['generated_len']} tok in {stats['decode_time']:.2f}s "
          f"({stats['decode_tok_s']:.1f} tok/s)")
    print(f"  Total:    {stats['total_time']:.2f}s")


if __name__ == "__main__":
    main()

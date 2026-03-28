# This file is modified from the original implementation (implemented by Xunhao Lai)
from typing import Optional

import torch
from einops import rearrange
from flash_attn import flash_attn_varlen_func

from fsa_preview.ops import (_compressed_attention_decode,
                             _linear_compress_decode,
                             _topk_sparse_attention_decode)
from nsa_ref.module.rope import RopeConfig, RotaryEmbedding


class FlashSparseAttentionDecode(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kernel_size: int,
        kernel_stride: int,
        block_size: int,
        topk: int,
        init_blocks: int,
        local_blocks: int,
        window_size: int,
        rope_config: RopeConfig,
    ):
        super().__init__()
        # configs
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.block_size = block_size
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.window_size = window_size
        self.rope_config = rope_config

        # qkv proj and o proj
        self.proj_q = torch.nn.Linear(self.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.proj_k = torch.nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.proj_v = torch.nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.proj_o = torch.nn.Linear(self.num_q_heads * self.head_dim, self.hidden_size, bias=False)

        # nsa parameteres
        self.compress_key = torch.nn.Parameter(
            torch.zeros(self.num_kv_heads, self.head_dim * self.kernel_size, self.head_dim)
        )
        self.compress_value = torch.nn.Parameter(
            torch.zeros(self.num_kv_heads, self.head_dim * self.kernel_size, self.head_dim)
        )
        self.intra_block_pe = torch.nn.Parameter(torch.zeros(self.num_kv_heads, self.kernel_size, self.head_dim))

        # gate function
        self.gate = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 3, bias=False), torch.nn.Sigmoid())

        # rope
        self.rope = RotaryEmbedding(self.rope_config)

    def forward(
        self,
        x: torch.Tensor,  # shape: [total_len, hidden_size]
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,  # shape: [batch_size + 1]
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        cmp_k_cache: torch.Tensor = None,
        cmp_v_cache: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        use_dedup: bool = False,
        # --- A1/A2: preallocated buffers (avoid O(seq) torch.cat each step) ---
        kv_storage_k: Optional[torch.Tensor] = None,
        kv_storage_v: Optional[torch.Tensor] = None,
        kv_past_len: Optional[int] = None,
        cmp_storage_k: Optional[torch.Tensor] = None,
        cmp_storage_v: Optional[torch.Tensor] = None,
        cmp_past_len: Optional[int] = None,
    ):
        """
        Args:
            attention_mask: Optional mask for q-to-k attention, shape (total_q_len, total_k_len)
                or (num_q_heads, total_q_len, total_k_len). Used for tree decoding (e.g. EAGLE):
                caller should pass a mask where tree positions can attend along their path.
                When provided, it is passed to compressed/topk (kernel support TODO). The sliding branch
                does not apply it (flash_attn has no custom mask API); for full tree semantics, add mask in compressed/topk kernels.
            position_ids: Optional absolute positions for current queries/keys, shape (total_q_len,).
                For linear multi-token decode, typically `arange(past_len, past_len + q_len)`.
            use_dedup: Reserved for duplicate-KV dedup kernel; currently ignored.
            kv_storage_k / kv_storage_v: Optional preallocated [max_seqlen, Hkv, D] tensors.
                When set, pass kv_past_len and set k_cache=v_cache=None; rows [0:kv_past_len) must
                already hold the prefix KV. This forward copies k_new/v_new into
                [kv_past_len : kv_past_len + q_len) and uses a slice as k/v (no cat).
            cmp_storage_k / cmp_storage_v: Optional preallocated compressed KV buffers.
                When set, pass cmp_past_len and set cmp_k_cache=cmp_v_cache=None; rows [0:cmp_past_len)
                hold the compressed prefix. New compressed tokens are copy_ appended (no cat).
        """
        _ = use_dedup  # TODO: wire to topk/compressed path when kernel is ready
        # dtype and shape check
        assert x.dtype == torch.bfloat16 or x.dtype == torch.float16
        assert x.shape[-1] == self.hidden_size
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()

        use_kv_storage = kv_storage_k is not None
        use_cmp_storage = cmp_storage_k is not None
        if use_kv_storage:
            assert kv_storage_v is not None and kv_past_len is not None, (
                "kv_storage_k requires kv_storage_v and kv_past_len"
            )
            assert k_cache is None and v_cache is None, (
                "With kv_storage_* use k_cache=v_cache=None; prefix must live in storage[:, :kv_past_len]"
            )
        else:
            assert k_cache is not None and v_cache is not None, "k_cache and v_cache required unless kv_storage_* is set"
        if use_cmp_storage:
            assert cmp_storage_v is not None and cmp_past_len is not None, (
                "cmp_storage_k requires cmp_storage_v and cmp_past_len"
            )
            assert cmp_k_cache is None and cmp_v_cache is None, (
                "With cmp_storage_* use cmp_k_cache=cmp_v_cache=None"
            )
        else:
            assert cmp_k_cache is not None and cmp_v_cache is not None, (
                "cmp_k_cache and cmp_v_cache required unless cmp_storage_* is set"
            )

        # qkv proj
        with torch.profiler.record_function("FSA_decode.qkv_proj"):
            q = self.proj_q(x).view(-1, self.num_q_heads, self.head_dim)
            k_new = self.proj_k(x).view(-1, self.num_kv_heads, self.head_dim)
            v_new = self.proj_v(x).view(-1, self.num_kv_heads, self.head_dim)
        q_len_new = k_new.shape[0]
        with torch.profiler.record_function("FSA_decode.kv_cat"):
            if use_kv_storage:
                past_raw_len = int(kv_past_len)
                assert kv_storage_k.shape[0] >= past_raw_len + q_len_new
                assert kv_storage_v.shape[0] >= past_raw_len + q_len_new
                kv_storage_k[past_raw_len : past_raw_len + q_len_new].copy_(k_new)
                kv_storage_v[past_raw_len : past_raw_len + q_len_new].copy_(v_new)
                k = kv_storage_k[: past_raw_len + q_len_new]
                v = kv_storage_v[: past_raw_len + q_len_new]
            else:
                past_raw_len = k_cache.shape[0]
                k = torch.cat([k_cache, k_new], dim=0)
                v = torch.cat([v_cache, v_new], dim=0)

        if position_ids is not None:
            # explicit absolute positions for the new queries/keys; only linear (single-batch) supported here
            assert position_ids.shape[0] == q.shape[0], (
                f"position_ids length {position_ids.shape[0]} != total_q_len {q.shape[0]}"
            )

        if attention_mask is not None:
            total_q_len, total_k_len = q.shape[0], k.shape[0]
            assert attention_mask.dim() in (2, 3), "attention_mask must be 2D or 3D"
            if attention_mask.dim() == 2:
                assert attention_mask.shape == (total_q_len, total_k_len), (
                    f"attention_mask shape {attention_mask.shape} != (total_q_len, total_k_len) ({total_q_len}, {total_k_len})"
                )
                mask_2d = attention_mask
            else:
                assert attention_mask.shape == (self.num_q_heads, total_q_len, total_k_len), (
                    f"attention_mask shape {attention_mask.shape} != (num_q_heads, total_q_len, total_k_len)"
                )
                mask_2d = attention_mask.max(dim=0).values  # (total_q_len, total_k_len)

        # compute seqlens after compression
        with torch.profiler.record_function("FSA_decode.compress_layout"):
            compressed_seqlens = torch.floor((seqlens_k - self.kernel_size) / self.kernel_stride) + 1
            # corner case: if sequence_length < kernel_size, no compression for this sequence
            compressed_seqlens[seqlens_k < self.kernel_size] = 0
            compressed_seqlens = compressed_seqlens.to(torch.int32)
            compressed_cu_seqlens = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device="cuda"),
                    torch.cumsum(compressed_seqlens, dim=0),
                ],
                dim=0,
            ).to(torch.int32)

        # compressed key and value before rope
        # _linear_compress_decode 使用「未压缩」序列坐标：prev_total_len 为追加 k_new 前的原始 KV
        # 长度；token_buffer 为边界重叠窗口，须为 **原始** K/V 尾部。误用 cmp_* 长度或已压缩向量
        # 会导致窗口索引错误或对已压缩特征再次 linear_compress（长上下文下偏差放大）。
        buffer_size = min(self.kernel_size - 1, past_raw_len)
        if use_kv_storage:
            initial_buffer_k = (
                kv_storage_k[past_raw_len - buffer_size : past_raw_len] if buffer_size > 0 else None
            )
            initial_buffer_v = (
                kv_storage_v[past_raw_len - buffer_size : past_raw_len] if buffer_size > 0 else None
            )
        else:
            initial_buffer_k = k_cache[-buffer_size:] if buffer_size > 0 else None
            initial_buffer_v = v_cache[-buffer_size:] if buffer_size > 0 else None

        with torch.profiler.record_function("FSA_decode.linear_compress"):
            # Decode the last part
            decode_k_output = _linear_compress_decode(
                k_new,
                self.compress_key,
                self.kernel_size,
                self.kernel_stride,
                self.intra_block_pe,
                past_raw_len,
                initial_buffer_k,
            )

            decode_v_output = _linear_compress_decode(
                v_new,
                self.compress_value,
                self.kernel_size,
                self.kernel_stride,
                None,
                past_raw_len,
                initial_buffer_v,
            )
            # Combine results
            if use_cmp_storage:
                cmp_p = int(cmp_past_len)
                if decode_k_output is not None:
                    n_cmp_k = decode_k_output.shape[0]
                    n_cmp_v = decode_v_output.shape[0]
                    assert n_cmp_k == n_cmp_v
                    assert cmp_storage_k.shape[0] >= cmp_p + n_cmp_k
                    assert cmp_storage_v.shape[0] >= cmp_p + n_cmp_v
                    cmp_storage_k[cmp_p : cmp_p + n_cmp_k].copy_(decode_k_output)
                    cmp_storage_v[cmp_p : cmp_p + n_cmp_v].copy_(decode_v_output)
                    compressed_k = cmp_storage_k[: cmp_p + n_cmp_k]
                    compressed_v = cmp_storage_v[: cmp_p + n_cmp_v]
                else:
                    compressed_k = cmp_storage_k[:cmp_p]
                    compressed_v = cmp_storage_v[:cmp_p]
            elif decode_k_output is not None:
                compressed_k = torch.cat([cmp_k_cache, decode_k_output], dim=0)
                compressed_v = torch.cat([cmp_v_cache, decode_v_output], dim=0)
            else:
                compressed_k = cmp_k_cache
                compressed_v = cmp_v_cache

        # Build compressed_mask from full attention_mask when present (for tree decoding)
        # compressed_mask[q, c] = 1 iff any key in block c is visible to q (1=attend, 0=mask)
        attention_mask_compressed = None
        if attention_mask is not None:
            with torch.profiler.record_function("FSA_decode.mask_compressed"):
                total_q_len, total_k_len = q.shape[0], k.shape[0]
                mask_2d = attention_mask if attention_mask.dim() == 2 else attention_mask.max(dim=0).values
                compressed_k_len = compressed_k.shape[0]
                # Block c covers uncompress indices [c*kernel_stride, c*kernel_stride+kernel_size)
                compressed_mask = torch.zeros(
                    total_q_len, compressed_k_len, device=x.device, dtype=torch.float32
                )
                for c in range(compressed_k_len):
                    start = c * self.kernel_stride
                    end = min(start + self.kernel_size, mask_2d.shape[1])
                    compressed_mask[:, c] = mask_2d[:, start:end].to(torch.float32).amax(dim=1)
                attention_mask_compressed = compressed_mask

        # do rope for query and compressed key
        with torch.profiler.record_function("FSA_decode.rope_q_cmpk"):
            if position_ids is not None:
                q = self.rope(q, cu_seqlens_q, position_ids=position_ids)
            else:
                q = self.rope(q, cu_seqlens_q)
            # compressed_k uses compressed_cu_seqlens + (start,stride) scheme as before
            compressed_k = self.rope(compressed_k, compressed_cu_seqlens, start=0, stride=self.kernel_stride)

        # attention between query and compressed key value
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        with torch.profiler.record_function("FSA_decode.compressed_path"):
            compressed_attn_output, topk_idx = _compressed_attention_decode(
                q,
                compressed_k,
                compressed_v,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                max_seqlen_q,
                compressed_seqlens.max().item(),
                None,
                self.init_blocks,
                self.local_blocks,
                query_start_index=past_raw_len,
                attention_mask=attention_mask_compressed,
            )

        # do rope for original key
        with torch.profiler.record_function("FSA_decode.rope_full_k"):
            if position_ids is not None:
                # apply RoPE only to new keys with explicit positions, keep cached part as-is
                new_pos = position_ids
                k_new_rope = self.rope(k_new, cu_seqlens_q, position_ids=new_pos)
                if use_kv_storage:
                    kv_storage_k[past_raw_len : past_raw_len + q_len_new].copy_(k_new_rope)
                    k = kv_storage_k[: past_raw_len + q_len_new]
                else:
                    k = torch.cat([k_cache, k_new_rope], dim=0)
            else:
                k = self.rope(k, cu_seqlens_k)

        # topk sparse attention
        with torch.profiler.record_function("FSA_decode.topk_sparse_path"):
            sparse_attn_output = _topk_sparse_attention_decode(
                q, k, v, topk_idx, self.block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                seqlens_k.max().item(),
                None,
                attention_mask=attention_mask,
            )

        # sliding window attention (flash_attn does not support custom mask; sliding branch never applies attention_mask)
        with torch.profiler.record_function("FSA_decode.sliding_flash_path"):
            sliding_attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                seqlens_k.max().item(),
                causal=False,
                window_size=(self.window_size, -1),
            )

        # gate + three-branch fusion
        with torch.profiler.record_function("FSA_decode.gate_and_fuse"):
            gate = self.gate(x)
            attn_output = (
                gate[:, 0:1, None] * compressed_attn_output
                + gate[:, 1:2, None] * sparse_attn_output
                + gate[:, 2:3, None] * sliding_attn_output
            )

        # rearrange and output proj
        with torch.profiler.record_function("FSA_decode.proj_o"):
            attn_output = rearrange(attn_output, "n h d -> n (h d)")
            attn_output = self.proj_o(attn_output)

        return attn_output

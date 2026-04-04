# This file is modified from the original implementation (implemented by Xunhao Lai)
import torch
from einops import rearrange
from flash_attn import flash_attn_varlen_func

from fsa_preview.ops import (_compressed_attention_decode,
                             _linear_compress_decode,
                             _topk_sparse_attention_decode)
from fsa_preview.ops.unified_sparse_attention_decode import _unified_sparse_attention_decode
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
        k_cache: torch.Tensor = None, # k_cache that stores rope(k)
        k_buffer: torch.Tensor = None, # k_buffer that stores min(self.kernel_size - 1, prev_raw_len) k
        v_cache: torch.Tensor = None,
        cmp_k_cache: torch.Tensor = None,
        cmp_v_cache: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        kv_commit_stash: list = None,
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
        """
        # dtype and shape check
        assert x.dtype == torch.bfloat16 or x.dtype == torch.float16
        assert x.shape[-1] == self.hidden_size
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()

        # qkv proj
        q = self.proj_q(x).view(-1, self.num_q_heads, self.head_dim) # [total_q_len, num_q_heads, head_dim]
        k_new = self.proj_k(x).view(-1, self.num_kv_heads, self.head_dim)
        v_new = self.proj_v(x).view(-1, self.num_kv_heads, self.head_dim)

        # apply rope to q and k_new
        total_q_len = q.shape[0]
        if position_ids is not None:
            assert position_ids.shape[0] == total_q_len, (
                f"position_ids length {position_ids.shape[0]} != total_q_len {q.shape[0]}"
            )
            q = self.rope(q, cu_seqlens_q, position_ids=position_ids)
            k_new_rope: torch.Tensor = self.rope(k_new, cu_seqlens_q, position_ids=position_ids)
        else:
            # TODO: This is incorrect for multiple batches. `start` parameter should be batched as well.
            q = self.rope(q, cu_seqlens_q, start=total_k_len - total_q_len)
            k_new_rope: torch.Tensor = self.rope(k_new, cu_seqlens_q, start=total_k_len - total_q_len)

        k = torch.cat([k_cache, k_new_rope], dim=0) # full roped k
        v = torch.cat([v_cache, v_new], dim=0)
        total_k_len = k.shape[0]

        if attention_mask is not None:
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
        prev_raw_len = k_cache.shape[0]
        buffer_size = min(self.kernel_size - 1, prev_raw_len)
        if k_buffer is not None:
            assert buffer_size <= k_buffer.shape[0]
        v_buffer = v_cache[-buffer_size:] if buffer_size > 0 else None

        # Decode the last part
        decode_k_output = _linear_compress_decode(
            k_new,
            self.compress_key,
            self.kernel_size,
            self.kernel_stride,
            self.intra_block_pe,
            prev_raw_len,
            k_buffer,
        )

        decode_v_output = _linear_compress_decode(
            v_new,
            self.compress_value,
            self.kernel_size,
            self.kernel_stride,
            None,
            prev_raw_len,
            v_buffer,
        )

        # Combine results
        if decode_k_output is not None:
            compressed_k = torch.cat([cmp_k_cache, decode_k_output], dim=0)
            compressed_v = torch.cat([cmp_v_cache, decode_v_output], dim=0)
        else:
            compressed_k = cmp_k_cache
            compressed_v = cmp_v_cache

        # Build compressed_mask from full attention_mask when present (for tree decoding)
        # compressed_mask[q, c] = 1 iff any key in block c is visible to q (1=attend, 0=mask)
        attention_mask_compressed = None
        if attention_mask is not None:
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

        # do rope for compressed key
        # compressed_k uses compressed_cu_seqlens + (start,stride) scheme as before
        # TODO: Make cmp_k_cache store rope(compressed_k)
        # TODO: Support position_ids???
        compressed_k = self.rope(compressed_k, compressed_cu_seqlens, start=0, stride=self.kernel_stride)

        # attention between query and compressed key value
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
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
            query_start_index=k_cache.shape[0],
            attention_mask=attention_mask_compressed,
            max_seqlen_k_original=seqlens_k.max().item(),
        )

        if kv_commit_stash is not None:
            kv_commit_stash.append((
                k_new_rope.detach(), # rope(k_new) will be appended to k_cache
                k_new.detach(), # k_new will be appended to k_buffer
                v_new.detach(), # v_new will be appended to v_cache
                # TODO: try to stash compressed_k and compressed_v to avoid extra _linear_compress_decode
            ))

        # compute gate
        gate = self.gate(x)

        # topk sparse attention and sliding window attention
        unified_attn_output = _unified_sparse_attention_decode(
            q,
            k,
            v,
            topk_idx,
            self.block_size,
            self.window_size,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            seqlens_k.max().item(),
            gate,
            causal=True,
        )

        attn_output = (
            gate[:, 0:1, None] * compressed_attn_output
            + unified_attn_output
        )

        # rearrange and output proj
        attn_output = rearrange(attn_output, "n h d -> n (h d)")
        attn_output = self.proj_o(attn_output)

        return attn_output

# This file is modified from the original implementation (implemented by Xunhao Lai)
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
    ):
        # dtype and shape check
        assert x.dtype == torch.bfloat16 or x.dtype == torch.float16
        assert x.shape[-1] == self.hidden_size
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

        # qkv proj
        q = self.proj_q(x).view(-1, self.num_q_heads, self.head_dim)
        k_new = self.proj_k(x).view(-1, self.num_kv_heads, self.head_dim)
        v_new = self.proj_v(x).view(-1, self.num_kv_heads, self.head_dim)
        k = torch.cat([k_cache, k_new], dim=0)
        v = torch.cat([v_cache, v_new], dim=0)

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
        # Prepare initial buffer with some context from first part
        # Need last (kernel_size-1) tokens from first part for potential overlapping windows
        buffer_size = min(self.kernel_size - 1, cmp_k_cache.shape[0])
        initial_buffer_k = cmp_k_cache[-buffer_size:] if buffer_size > 0 else None
        initial_buffer_v = cmp_v_cache[-buffer_size:] if buffer_size > 0 else None

        # Decode the last part
        decode_k_output = _linear_compress_decode(
            k_new,
            self.compress_key,
            self.kernel_size,
            self.kernel_stride,
            self.intra_block_pe,
            cmp_k_cache.shape[0],
            initial_buffer_k,
        )

        decode_v_output = _linear_compress_decode(
            v_new,
            self.compress_value,
            self.kernel_size,
            self.kernel_stride,
            None,
            cmp_v_cache.shape[0],
            initial_buffer_v,
        )
        # Combine results
        if decode_k_output is not None:
            compressed_k = torch.cat([cmp_k_cache, decode_k_output], dim=0)
            compressed_v = torch.cat([cmp_v_cache, decode_v_output], dim=0)
        else:
            compressed_k = cmp_k_cache
            compressed_v = cmp_v_cache

        # do rope for query and compressed key
        q = self.rope(q, cu_seqlens_q)
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
            1,
            compressed_seqlens.max().item(),
            None,
            self.init_blocks,
            self.local_blocks,
            query_start_index=k_cache.shape[0],
        )

        # do rope for original key
        k = self.rope(k, cu_seqlens_k)

        # topk sparse attention
        sparse_attn_output = _topk_sparse_attention_decode(
            q, k, v, topk_idx, self.block_size,
            # cu_seqlen_q and cu_seqlen_k
            cu_seqlens_q,
            cu_seqlens_k,
            # max_seqlen_q and max_seqlen_k
            1,
            seqlens_k.max().item(),
            None
        )

        # sliding window attention
        sliding_attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            1,
            seqlens_k.max().item(),
            causal=False,
            window_size=(self.window_size, -1),
        )

        # gate average
        gate = self.gate(x)
        attn_output = (
            gate[:, 0:1, None] * compressed_attn_output
            + gate[:, 1:2, None] * sparse_attn_output
            + gate[:, 2:3, None] * sliding_attn_output
        )

        # rearrange and output proj
        attn_output = rearrange(attn_output, "n h d -> n (h d)")
        attn_output = self.proj_o(attn_output)

        return attn_output

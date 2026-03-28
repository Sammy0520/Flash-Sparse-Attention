#!/usr/bin/env python3
"""Parity: FlashSparseAttentionDecode legacy cat vs preallocated kv / cmp buffers (A1/A2)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode
from nsa_ref.module import RopeConfig
from nsa_ref.ops import linear_compress


def test_inplace_matches_legacy(*, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(0)
    seqlen = 512
    q_len = 4
    hidden_size = 4096
    kv_heads = 8
    q_heads = 8
    head_dim = 128
    kernel_size = 32
    kernel_stride = 16
    block_size = 64
    topk = 16

    m = (
        FlashSparseAttentionDecode(
            hidden_size=hidden_size,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            window_size=512,
            rope_config=RopeConfig(
                max_position_embeddings=131072,
                head_dim=head_dim,
                rope_theta=500000,
                rope_scaling={
                    "factor": 8.0,
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                },
            ),
        )
        .to(device)
        .to(dtype)
    )

    cu_seqlens_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    x = torch.randn(q_len, hidden_size, device=device, dtype=dtype)

    k_cache = torch.randn(seqlen - 1, kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(seqlen - 1, kv_heads, head_dim, device=device, dtype=dtype)
    cmp_k, _ = linear_compress(
        k_cache, m.compress_key, cu_seqlens_k, kernel_size, kernel_stride, m.intra_block_pe
    )
    cmp_v, _ = linear_compress(
        v_cache, m.compress_value, cu_seqlens_k, kernel_size, kernel_stride, None
    )

    y0 = m(x, cu_seqlens_q, cu_seqlens_k, k_cache, v_cache, cmp_k, cmp_v)

    past = seqlen - 1
    raw_total = past + q_len
    kv_k = torch.empty(raw_total, kv_heads, head_dim, device=device, dtype=dtype)
    kv_v = torch.empty(raw_total, kv_heads, head_dim, device=device, dtype=dtype)
    kv_k[:past].copy_(k_cache)
    kv_v[:past].copy_(v_cache)

    cmp_past = cmp_k.shape[0]
    cmp_cap = cmp_past + q_len + kernel_size
    cmp_sk = torch.empty(cmp_cap, kv_heads, head_dim, device=device, dtype=dtype)
    cmp_sv = torch.empty(cmp_cap, kv_heads, head_dim, device=device, dtype=dtype)
    cmp_sk[:cmp_past].copy_(cmp_k)
    cmp_sv[:cmp_past].copy_(cmp_v)

    y1 = m(
        x,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        None,
        None,
        kv_storage_k=kv_k,
        kv_storage_v=kv_v,
        kv_past_len=past,
        cmp_storage_k=cmp_sk,
        cmp_storage_v=cmp_sv,
        cmp_past_len=cmp_past,
    )

    torch.testing.assert_close(y0, y1, rtol=1e-2, atol=1e-2, check_stride=False)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    test_inplace_matches_legacy()
    print("test_fsa_inplace_buffers: OK")

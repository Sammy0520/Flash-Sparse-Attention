import math
from pathlib import Path

import torch
import torch.nn as nn

# project root on path so "nsa_ref" and "fsa_preview" can be imported
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nsa_ref.module import RopeConfig  # noqa: E402
from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode  # noqa: E402
from nsa_ref.ops import linear_compress  # noqa: E402


class ToyBlock(nn.Module):
    """一个最小的 Transformer Block：LN -> NSA 注意力 -> 残差 -> LN + MLP -> 残差。"""

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
        rope_config: RopeConfig,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.attn = FlashSparseAttentionDecode(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            window_size=512,
            rope_config=rope_config,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,  # [total_len, hidden_size]
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cmp_k_cache: torch.Tensor,
        cmp_v_cache: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        h = self.ln1(x)
        h = self.attn(
            h,
            cu_seqlens_q,
            cu_seqlens_k,
            k_cache,
            v_cache,
            cmp_k_cache,
            cmp_v_cache,
            attention_mask=None,
            position_ids=position_ids,
        )
        x = x + h
        h2 = self.ln2(x)
        x = x + self.mlp(h2)
        return x


def build_toy_block_and_caches(
    seqlen: int = 1000,
    hidden_size: int = 4096,
    head_dim: int = 128,
    q_heads: int = 8,
    kv_heads: int = 8,
    kernel_size: int = 32,
    kernel_stride: int = 16,
    block_size: int = 64,
    topk: int = 16,
    dtype: torch.dtype = torch.float16,
):
    torch.manual_seed(42)

    rope_config = RopeConfig(
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
    )

    block = ToyBlock(
        hidden_size=hidden_size,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
        block_size=block_size,
        topk=topk,
        rope_config=rope_config,
    ).cuda().to(dtype)

    # 单 batch 的 KV cache（和 test_FSA_decode.py 一致）
    seqlens = torch.tensor([seqlen], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device="cuda"), torch.cumsum(seqlens, dim=0)],
        dim=0,
    )

    k_cache = torch.randn(seqlen - 1, kv_heads, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn(seqlen - 1, kv_heads, head_dim, device="cuda", dtype=dtype)

    cmp_len = (k_cache.shape[0] - kernel_size) // kernel_stride + 1
    cmp_k_cache = torch.randn(cmp_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    cmp_v_cache = torch.randn(cmp_len, kv_heads, head_dim, device="cuda", dtype=dtype)

    # 生成 NSA 的压缩参数，并用线性压缩更新 cmp_k/v_cache（和 test_FSA_decode 保持一致）
    compress_key = torch.randn(kv_heads, head_dim * kernel_size, head_dim, device="cuda", dtype=dtype)
    compress_value = torch.randn(kv_heads, head_dim * kernel_size, head_dim, device="cuda", dtype=dtype)
    intra_block_pe = torch.randn(kv_heads, kernel_size, head_dim, device="cuda", dtype=dtype)

    cmp_k_cache, compressed_cu_seqlens = linear_compress(
        k_cache, compress_key, cu_seqlens, kernel_size, kernel_stride, intra_block_pe
    )
    cmp_v_cache, _ = linear_compress(
        v_cache, compress_value, cu_seqlens, kernel_size, kernel_stride, None
    )

    return (
        block,
        cu_seqlens,
        k_cache,
        v_cache,
        cmp_k_cache,
        cmp_v_cache,
    )


def run_block_multi_vs_prefix_test(N: int = 4, dtype_str: str = "float16"):
    DTYPE = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32)[dtype_str]

    seqlen = 1000
    head_dim = 128
    q_heads = 8
    kv_heads = 8

    (
        block,
        cu_seqlens_k,
        k_cache,
        v_cache,
        cmp_k_cache,
        cmp_v_cache,
    ) = build_toy_block_and_caches(
        seqlen=seqlen,
        hidden_size=4096,
        head_dim=head_dim,
        q_heads=q_heads,
        kv_heads=kv_heads,
        kernel_size=32,
        kernel_stride=16,
        block_size=64,
        topk=16,
        dtype=DTYPE,
    )

    # 构造 N 个“新 token” 的输入 hidden
    x = torch.randn(N, 4096, device="cuda", dtype=DTYPE)
    cu_seqlens_q = torch.tensor([0, N], device="cuda", dtype=torch.int32)
    past_len = k_cache.shape[0]
    pos_ids = torch.arange(past_len, past_len + N, device="cuda", dtype=torch.long)

    print(f"======= ToyBlock multi-token decode test (N={N}) =======\n")

    # 1) 一次性 N-token 前向
    y_multi = block(
        x,
        cu_seqlens_q,
        cu_seqlens_k,
        k_cache,
        v_cache,
        cmp_k_cache,
        cmp_v_cache,
        position_ids=pos_ids,
    )
    assert y_multi.shape == (N, 4096)

    # 2) 前缀单 token（每次用 x[:i+1]，取最后一个 token 的输出）
    y_single_list = []
    for i in range(N):
        x_prefix = x[: i + 1]
        cu_seqlens_q_prefix = torch.tensor([0, i + 1], device="cuda", dtype=torch.int32)
        pos_prefix = pos_ids[: i + 1]
        y_prefix = block(
            x_prefix,
            cu_seqlens_q_prefix,
            cu_seqlens_k,
            k_cache,
            v_cache,
            cmp_k_cache,
            cmp_v_cache,
            position_ids=pos_prefix,
        )
        y_single_list.append(y_prefix[-1])

    y_single = torch.stack(y_single_list, dim=0)
    assert y_single.shape == (N, 4096)

    per_token_diff = (y_multi - y_single).abs().amax(dim=-1)
    print("  per-token max |multi - single_prefix|:", per_token_diff.tolist())
    print("  global max diff:", per_token_diff.max().item())

    # Block 里多了 LN+MLP，数值误差会比裸 NSA 稍大，阈值放宽一点
    torch.testing.assert_close(
        y_multi, y_single, rtol=5e-2, atol=5e-2, check_stride=False
    )
    print("  [OK] ToyBlock multi-token decode == prefix single-token decode (within tolerance)\n")


if __name__ == "__main__":
    # N 可以改，比如 4 / 16 / 32
    run_block_multi_vs_prefix_test(N=4, dtype_str="float16")
    run_block_multi_vs_prefix_test(N=16, dtype_str="float16")


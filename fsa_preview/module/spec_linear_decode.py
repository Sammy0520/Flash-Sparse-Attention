import torch

from fsa_preview.module.fsa_decode import FlashSparseAttentionDecode


@torch.no_grad()
def nsa_linear_validate(
    decoder: FlashSparseAttentionDecode,
    x_new: torch.Tensor,          # [N, hidden_size] 本次要验证的 N 个 token 的隐藏状态
    k_cache: torch.Tensor,        # [past_len, num_kv_heads, head_dim]
    v_cache: torch.Tensor,        # [past_len, num_kv_heads, head_dim]
    cmp_k_cache: torch.Tensor,    # [past_cmp_len, num_kv_heads, head_dim]
    cmp_v_cache: torch.Tensor,    # [past_cmp_len, num_kv_heads, head_dim]
    attention_mask: torch.Tensor = None,
):
    """单 batch 线性多 token 验证 helper（不改算子，只封装 cu_seqlens / position_ids 计算）。

    使用方式（伪代码）::

        # 假设你已经有 prefix 的 KV / 压缩 KV cache
        y = nsa_linear_validate(
            decoder=fsa_decoder,
            x_new=x_candidate,          # [N, hidden]
            k_cache=k_cache,
            v_cache=v_cache,
            cmp_k_cache=cmp_k_cache,
            cmp_v_cache=cmp_v_cache,
        )

    注意：
    - 这里只做“验证前向”：不会帮你更新外部的 k_cache / v_cache / cmp_k_cache / cmp_v_cache。
      你可以在接受某些 token 之后，用自己的一套 cache 管理逻辑来 append / 回滚。
    """
    assert x_new.dim() == 2, f"x_new shape must be [N, hidden], got {tuple(x_new.shape)}"
    device = x_new.device

    N = x_new.shape[0]
    past_len = k_cache.shape[0]

    # 单 batch：query 这次只有 N 个 token
    cu_seqlens_q = torch.tensor([0, N], device=device, dtype=torch.int32)
    # key 长度 = 之前的 past_len + 本次 N 个候选
    cu_seqlens_k = torch.tensor([0, past_len + N], device=device, dtype=torch.int32)

    # 显式绝对位置：[past_len, ..., past_len+N-1]
    position_ids = torch.arange(past_len, past_len + N, device=device, dtype=torch.long)

    out = decoder(
        x_new,
        cu_seqlens_q,
        cu_seqlens_k,
        k_cache=k_cache,
        v_cache=v_cache,
        cmp_k_cache=cmp_k_cache,
        cmp_v_cache=cmp_v_cache,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    return out


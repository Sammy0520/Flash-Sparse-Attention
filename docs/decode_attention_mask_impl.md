# Decode 中真正使用 attention_mask / tree_mask 的实现说明

当前 2.3 已在接口层接好 `attention_mask` 并下传到 compressed / topk 两路，但**两路 kernel 都还未使用**。本文说明如何把 mask 在 kernel 里用起来。

---

## 1. 语义与形状约定

- **调用方**传入的 `attention_mask` 形状为 `(total_q_len, total_k_len)` 或 `(num_q_heads, total_q_len, total_k_len)`，对应**未压缩**的 K/V 序列（即 `k = concat(k_cache, k_new)` 的长度）。
- 约定：**1 = 可 attend，0 = 必须 mask 掉**（在 score 上变为 -inf）。若你用的是 0/1 相反约定，在入口或 kernel 里取反即可。
- **Compressed 分支**的 K 是压缩后的，长度为 `compressed_k_len != total_k_len`，因此需要先把「未压缩 mask」约简成「按压缩块」的 mask 再传入 compressed kernel。

---

## 2. Compressed 分支：从 full mask 得到 compressed_mask

在 **fsa_decode.py** 里，在调用 `_compressed_attention_decode` 之前：

- 若 `attention_mask is not None`：
  - 若 mask 是 3D，先按需转为 2D（例如对 head 维取 max 或沿用第一维）。
  - 从 `attention_mask` 构造 **compressed_mask**，形状 `(total_q_len, compressed_k_len)`：
    - 压缩块 `c` 对应未压缩位置区间  
      `[c * kernel_stride, c * kernel_stride + kernel_size)`。
    - 定义：  
      `compressed_mask[q, c] = 1` 当且仅当在未压缩区间内**至少有一个**位置 `j` 满足 `attention_mask[q, j] == 1`（即该块内只要有一个 key 可见就整块可见）；否则 `compressed_mask[q, c] = 0`。
  - 将 **compressed_mask** 传给 `_compressed_attention_decode`（不再传原始的 full attention_mask，或两路分别传：compressed 用 compressed_mask，topk 用 full mask）。

示例（Python 伪代码）：

```python
# 在 fsa_decode.py 中，调用 _compressed_attention_decode 前
if attention_mask is not None:
    if attention_mask.dim() == 3:
        mask_2d = attention_mask.max(dim=0).values  # (total_q_len, total_k_len)
    else:
        mask_2d = attention_mask
    compressed_k_len = compressed_k.shape[0]
    compressed_mask = torch.zeros(total_q_len, compressed_k_len, device=x.device, dtype=torch.float32)
    for c in range(compressed_k_len):
        start = c * self.kernel_stride
        end = min(start + self.kernel_size, mask_2d.shape[1])
        compressed_mask[:, c] = mask_2d[:, start:end].float().max(dim=1).values
    attention_mask_compressed = compressed_mask
else:
    attention_mask_compressed = None
# 调用时传 attention_mask_compressed
_compressed_attention_decode(..., attention_mask=attention_mask_compressed)
```

可用 `unfold` / `stride_tricks` 或小 kernel 向量化，避免 Python 循环。

---

## 3. Compressed kernel：在 softmax 前加 mask

文件：**fsa_preview/ops/compressed_attention_decode.py**

### 3.1 `_compressed_attention_fwd_decode`

- 增加参数：`attention_mask: torch.Tensor = None`，形状 `(total_q_len, compressed_k_len)`（即上面的 compressed_mask）。
- 若 `attention_mask is not None`，将其传入 `forward_kernel_decode`（见下）；否则传 `None` 或空指针，kernel 内分支不加载 mask。

### 3.2 `forward_kernel_decode`（Triton）

- 增加可选参数：`mask_ptr`（可为 0 表示无 mask）、`stride_mask_q`、`stride_mask_k`（或 `MASK_PTR` 为 0 时忽略）。
- 在**当前**循环内，在已有因果项之后、softmax 之前，即：
  - 已有：`qk += tl.where(..., 0, -inf)`（因果）
  - 已有：`qk += tl.dot(q, k) * qk_scale`
  - **在这里**：若传入了 mask，则按当前 block 的 q 范围 `[q_start, q_start+BLOCK_SIZE_Q)` 和 k 范围 `[k_start+i, k_start+i+BLOCK_SIZE_K)` 从 `mask_ptr` 加载一块 `mask_block`，形状与 `qk` 一致 `(BLOCK_SIZE_Q, BLOCK_SIZE_K)`；然后：
  - `qk = tl.where(mask_block > 0, qk, float("-inf"))`  
  （或你的约定是 0 表示可见则用 `mask_block == 0` 时置 -inf。）
- 后面 `m_ij`、`p = exp2(qk - m_ij)` 等不变，这样被 mask 掉的位置不会参与 softmax 和输出。

加载 mask 块时注意：

- 用 `tl.load` 或 `tl.make_block_ptr` 按 `(q_start + 当前 q 偏移, k_start + i + 当前 k 偏移)` 与 `qk` 的 (BLOCK_Q, BLOCK_K) 对齐，并做 `boundary_check`，避免越界。
- 若 mask 是 float32，可直接加载；若为 bool/uint8，在 kernel 内转成 0/1 再用于 `tl.where`。

---

## 4. Topk sparse 分支 — 已实现

- 这里 q、k 都是**未压缩**的，因此直接使用**原始** `attention_mask`，形状 `(total_q_len, total_k_len)`。
- **nsa_ref** 的 `forward_kernel_orig` 已增加可选 `mask_ptr`、`total_q_len`、`total_k_len`、`stride_mask_q`、`stride_mask_k` 与 `HAS_MASK`；在每块算完 qk 后、softmax 前按块加载 mask 并 `qk = tl.where(mask_block > 0, qk, -inf)`。
- `_topk_sparse_attention_decode` 会将 3D mask 转为 2D（对 head 维取 max）、转为 float32 后传入 `_topk_sparse_attention_fwd`。

---

## 5. Sliding（flash_attn）分支 — 不应用自定义 mask

- `flash_attn_varlen_func` 当前**不支持**自定义 attention mask。
- **当前策略**：无论是否传入 `attention_mask`，sliding 分支**始终**调用 flash_attn（不传 mask），即 **sliding 一路不应用 tree_mask**。这样避免用 SDPA 做 fallback 时在某些环境下出现 NaN，且保持与无 mask 时相同的性能。
- 若需要三路都按 tree_mask 解码，需在 **compressed / topk 的 kernel 里支持 mask**（见上文）；sliding 一路在 flash-attn 或 FlashMask 等提供 varlen+自定义 mask 前，保持不应用 mask。

---

## 6. 实现顺序建议

1. **Compressed**：在 fsa_decode 里实现 full mask → compressed_mask 的约简，并传入 `_compressed_attention_decode`；在 `_compressed_attention_fwd_decode` 和 `forward_kernel_decode` 中增加可选 mask，在 softmax 前应用。
2. **测试**：构造简单 tree_mask（例如下三角 + 若干额外可见位置），对比「无 mask」与「有 mask」时 logits/输出差异，并做数值梯度检查（若需要）。
3. **Topk sparse**：在 nsa_ref 的 topk kernel 中增加可选 mask 支持，再在 `_topk_sparse_attention_decode` 中传入 `attention_mask`。

按上述顺序即可在 decode 中真正用上 EAGLE 风格的 tree_mask，并与计划 2.3 的接口一致。

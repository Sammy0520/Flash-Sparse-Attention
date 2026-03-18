# Linear Speculative Sampling 技术文档

## Flash-Sparse-Attention 线性投机采样实现方案

**文档版本**: 1.0  
**适用项目**: [Flash-Sparse-Attention](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)

---

## 一、背景与目标

### 1.1 线性投机采样概述

线性投机采样（Linear Speculative Sampling）是一种加速自回归解码的技术：

- **Draft 模型**：快速生成 K 个候选 token
- **Target 模型**：一次 forward 验证这 K 个 token
- **对比**：传统 decode 每步只验证 1 个 token，需要 K 次 forward

### 1.2 与 FSA 的适配目标

在 Flash-Sparse-Attention (FSA) 框架下，使 target 模型在**单次 forward 中验证 K 个 token**，同时保持：

- 压缩注意力（Compressed Attention）
- TopK 稀疏注意力（FSA/NSA）
- 滑动窗口注意力（Sliding Window）

的算法语义不变。

---

## 二、数据形状变化

### 2.1 输入输出对比

| 项目 | 当前 Decode (K=1) | 投机采样验证 (K>1) |
|------|-------------------|---------------------|
| 输入 x | `[1, hidden_size]` | `[K, hidden_size]` |
| Q | `[1, num_q_heads, head_dim]` | `[K, num_q_heads, head_dim]` |
| K/V 新增 | `[1, num_kv_heads, head_dim]` | `[K, num_kv_heads, head_dim]` |
| cu_seqlens_q | `[0, 1]` | `[0, K]` |
| cu_seqlens_k | `[0, past_len+1]` | `[0, past_len+K]` |
| topk_idx | `[num_heads, 1, topk]` | `[num_heads, K, topk]` |
| KV Cache 增长 | +1 token | +K tokens |

### 2.2 cu_seqlens 语义

- `cu_seqlens_q`：Q 的 batch 边界，单序列时为 `[0, K]`
- `cu_seqlens_k`：K/V 的 batch 边界，单序列时为 `[0, past_len+K]`

---

## 三、各模块修改清单

### 3.1 QKV 投影与 KV Cache 追加

**修改位置**：`fsa_preview/module/fsa_decode.py`

**修改内容**：

```python
# 当前
k_new = self.proj_k(x)  # x: [1, H]
v_new = self.proj_v(x)
k = torch.cat([k_cache, k_new], dim=0)  # [past_len+1, ...]

# 投机采样
k_new = self.proj_k(x)   # x: [K, H]
v_new = self.proj_v(x)
k = torch.cat([k_cache, k_new], dim=0)  # [past_len+K, ...]
```

**复杂度**：低，仅为 batch 维度扩展。

---

### 3.2 Linear Compress（压缩 KV）

**修改位置**：`fsa_preview/ops/linear_compress_decode.py`

**修改内容**：

- 支持输入 `k_new` / `v_new` 形状 `[K, num_heads, head_dim]`
- 保持与 `cmp_k_cache` 末 `(kernel_size-1)` 个 token 的重叠逻辑
- 压缩输出长度：`floor((K - kernel_size) / kernel_stride) + 1`（需处理 K < kernel_size 边界）

**复杂度**：中，需处理窗口边界和 K 较小时的退化情况。

---

### 3.3 Compressed Attention

**修改位置**：`fsa_preview/ops/compressed_attention_decode.py`

**修改内容**：

- 传入 `cu_seqlens_q = [0, K]`、`max_seqlen_q = K`
- 确保 `compressed_k` / `compressed_v` 包含 past + 新增压缩 token
- 校验 `query_start_index` 在 K-query 场景下的语义

**复杂度**：低，多为参数与接口调整。

---

### 3.4 TopK Sparse Attention（FSA 核心）

**修改位置**：`fsa/ops/FSA_topk_sparse_attention.py`

**修改内容**：

| 项目 | 当前 | 投机采样 | 说明 |
|------|------|----------|------|
| total_len | 1 | K | 单次处理的 query 数 |
| topk_idx 形状 | `[H, 1, topk]` | `[H, K, topk]` | 每个 query 的 topk blocks |
| valid_lens | 每个 block 0 或 1 | 0 ~ K | 每 block 的 attending query 数 |
| global_max_valid_tokens | 1 | K | 最大 attending query 数 |
| o_tiles_rest | `[H, B, 1, D]` | `[H, B, K, D]` | partial 输出 buffer |
| l_ij_rest | `[H, B, 1]` | `[H, B, K]` | softmax 统计 buffer |
| grid (token 维) | 1 | K | 并行 token 数 |

**Kernel 级修改**：

1. **block_to_token_kernel**：输入 `N_token = K`，逻辑不变，适配 K 个 query
2. **valid_lens 统计**：`valid_lens[b]` = 有多少个 query 的 topk 包含 block b，可能为 0~K
3. **forward_kernel_opt**：`selected_tokens_ptr` 指向的 query 索引数从 1 增至最多 K，内层循环需支持多 query
4. **reduce_kernel**：对每个 query 的 reduce 逻辑不变，遍历维度从 1 变为 K

**复杂度**：高，涉及 Triton kernel 形状和索引逻辑。

---

### 3.5 Sliding Window Attention

**修改位置**：对外调用 `flash_attn_varlen_func`

**修改内容**：

- Q: `[K, num_q_heads, head_dim]`
- K, V: `[past_len+K, num_kv_heads, head_dim]`
- `max_seqlen_q=K`, `max_seqlen_k=past_len+K`

**复杂度**：低，仅调整输入形状和序列长度参数。

---

## 四、性能收益与对应修改

### 4.1 Kernel 启动次数减少约 K 倍

| 项目 | 说明 |
|------|------|
| 收益 | K 次 forward → 1 次 forward，kernel 启动次数约减少 K 倍 |
| 对应修改 | 调度层：循环 K 次验证 → 一次传入 K 个 token 做单次 forward；FSA 入口支持 `x: [K, hidden_size]` |
| Kernel 修改 | 否，仅调度与接口变化 |

---

### 4.2 Past KV 读取带宽减少约 (K−1)/K

| 项目 | 说明 |
|------|------|
| 收益 | past KV 从被读取 K 次 → 读取 1 次，带宽节省约 (K−1)/K |
| 对应修改 | 数据流：一次 forward 内将 past KV 作为输入传给 Compressed / TopK / Sliding 三个模块，不再在外部多次读取 |
| Kernel 修改 | 否，由调用模式和数据流决定 |

---

### 4.3 GPU 并行度提高

| 项目 | 说明 |
|------|------|
| 收益 | 每 kernel 从处理 1 个 query → K 个 query，workload 增大，GPU 利用率提升 |
| 对应修改 | 各模块支持 K 个 query：FSA 的 `cur_max_valid_tokens`、`total_len`、grid 配置；Compressed / Sliding 的 `max_seqlen_q=K` |
| Kernel 修改 | 是，主要为形状与 grid 配置扩展 |

---

### 4.4 FSA Block 外循环带来的计算/读比提升

| 项目 | 说明 |
|------|------|
| 收益 | 每个 KV block 仍只 load 一次，但可被 K 个 query 复用，计算/读比提高 |
| 对应修改 | FSA TopK：`topk_idx`、`valid_lens`、`selected_tokens` 支持 K 个 query；`forward_kernel_opt` 内层对同一 block 处理多 query |
| Kernel 修改 | 是，涉及 FSA TopK Triton kernel 的 block-to-token 与多 query 支持 |

---

## 五、实现优先级建议

| 优先级 | 模块 | 说明 |
|--------|------|------|
| P0 | FSA TopK Sparse | 扩展 valid_lens、buffer、grid，支持 K 个 query，为核心性能路径 |
| P1 | Linear Compress Decode | 支持 K token 输入及边界处理 |
| P2 | Compressed Attn / Sliding | 多为接口与 cu_seqlens 调整 |

---

## 六、预期收益（量级估计）

假设 `past_len=4096`，`K=8`：

| 指标 | 估计 |
|------|------|
| Kernel 启动 | 约 8x 减少 |
| Past KV 读带宽 | 约 7/8 减少 |
| 端到端 decode 加速 | 约 2–4x（与 draft 模型开销相关） |

---

## 七、端到端测试指南

实现完成后，可按以下分层进行验证：

### 7.1 正确性测试（Correctness）

**目标**：验证 K-token 投机验证与 K 次单步 decode 的数值一致性。

**方法**：在相同输入和随机种子下，比较两种方式的输出：

| 方式 | 做法 | 输出 |
|------|------|------|
| Baseline | 循环 K 次，每次 `sparse_attn(x[i], cu_seqlens_q=[0,1], ...)`，逐次 append KV cache | `y_baseline`: K 个 token 的输出拼成 `[K, H, D]` |
| 投机采样 | 一次 `sparse_attn(x, cu_seqlens_q=[0,K], ...)`，K 个 token 的 hidden 一次性输入 | `y_speculative`: `[K, H, D]` |

**断言**：`torch.allclose(y_baseline, y_speculative, rtol=..., atol=...)`

**参考**：可沿用 `test/test_cmp_attn_decode.py` 中 prefill+decode 与全序列 prefill 的对比思路。

---

### 7.2 单元测试扩展

**修改 `test/test_FSA_decode.py`**：

1. 增加 `--num-verify-tokens`（或 `--K`）参数，默认 1，支持 2/4/8 等。
2. 当 `K > 1` 时：
   - `x` 改为 `[K, hidden_size]`
   - `cu_seqlens_q` 改为 `[0, K]`
   - `cu_seqlens_k` 改为 `[0, past_len+K]`
   - KV cache 预先填充 `past_len` 个 token；新增部分由 `proj_k/proj_v(x)` 得到，与 cache 拼接。

**运行**：

```bash
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python test/test_FSA_decode.py \
  --seqlens 4096 \
  --num-verify-tokens 8 \
  --kv-heads 4 --topk 16 --block-size 64 \
  --dtype bfloat16
```

---

### 7.3 性能基准测试

**脚本**：在 `test/test_FSA_decode.py` 或新建 `test/test_FSA_decode_speculative.py` 中：

1. **K=1 decode**：循环 N 次，每次 1 个 token，计总时延。
2. **K-token 投机**：循环 N/K 次，每次 K 个 token，计总时延。

**指标**：

- 单步延迟（ms）：K=1 vs K-token 的单次 forward
- 吞吐（tokens/s）：在相同 total token 数下的 tokens 每秒
- 加速比：K-token 相对 K×单步的加速

**运行示例**：

```bash
CUDA_VISIBLE_DEVICES=0 python test/test_FSA_decode_speculative.py \
  --past-len 4096 --num-verify-tokens 8 \
  --benchmark-iters 20
```

---

### 7.4 端到端集成测试（可选）

**目标**：在真实 LLM（如 Llama）上跑投机采样 pipeline。

**流程**：

1. Prefill：对 prompt 做一次 forward，得到 initial KV cache。
2. 投机验证：
   - Draft 模型生成 K 个候选 token；
   - Target 模型（FSA decode）一次验证 K 个 token；
   - 按 speculative decoding 算法接受/拒绝并更新 KV cache。

**参考**：`test/train.py` 中的 `SparseLlamaAttention` 与 `replace_llama_attention`，可据此增加投机 decode 的入口。

---

### 7.5 测试清单总结

| 层级 | 测试内容 | 通过标准 |
|------|----------|----------|
| 正确性 | K-token 输出 vs K 次单步 decode | `allclose` 在可接受误差内 |
| 单元 | 不同 K、past_len、seqlen 组合 | 无报错，输出 shape 正确 |
| 性能 | K=1 vs K-token 延迟与吞吐 | K-token 有明显加速 |
| 集成 | 真实模型投机 decode | 能完整跑通，生成 token 合理 |

---

### 7.6 快速验证命令汇总

```bash
# 正确性：K=1 vs K=8 输出对比（需在测试脚本中实现）
python test/test_FSA_decode_speculative.py --correctness --K 8

# 性能：decode 基准
bash scripts/run_unit_test_decode.sh

# 扩展：K-token 性能（需新增脚本或参数）
python test/test_FSA_decode.py --num-verify-tokens 8 --seqlens 4096
```

---

## 八、参考资料

- Flash-Sparse-Attention 论文：[arXiv:2508.18224](https://arxiv.org/abs/2508.18224)
- NSA 原始论文：[Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- 项目仓库：[Relaxed-System-Lab/Flash-Sparse-Attention](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)

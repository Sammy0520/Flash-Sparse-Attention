# `linear_compress_decode` 性能优化设计说明

**范围**：`fsa_preview/ops/linear_compress_decode.py`  
**目标**：降低多 token decode 路径上的 **Host 调度开销、小张量分配、重复 layout、kernel launch 并行度不足、HBM 往返**。

---

## 0. 现状简述（便于对齐）

- **`_linear_compress_decode`**：用 `windows_to_compute` 收集每个新窗口切片，再 `torch.cat` 拼成 `stacked`，并构造 `cu_seqlens` / `y_cu_seqlens`，最后调用 `linear_compress_with_pe`。
- **`linear_cmp_func`**：每次对 `w` 做 `reshape(num_heads, kernel_size, head_dim, head_dim).contiguous()`，以固定 tile 参数 launch `linear_compress_fwd_kernel`。
- **`linear_compress_fwd_kernel`**：grid 第二维当前固定为 `1`，但 kernel 内存在 `pid_k`（输出序列块）逻辑；长输出序列时并行度可能不足。

---

## 第一阶段：调用层优化（低风险）

### 1) 去掉 `windows_to_compute` + `torch.cat`，改为预分配 + 一次写入

**问题**

- Python list 存多个 `window_tokens` 视图/小 tensor，再 `cat`：增加 **中间对象、潜在额外拷贝、allocator 压力**。
- Host 端循环与 Python 调度成本在「每步多窗口」时更明显。

**改什么**

- 文件：`linear_compress_decode.py` 中 `_linear_compress_decode`（窗口收集与 `stacked` 构造段）。

**怎么改（推荐流程）**

1. **第一遍循环（只计数或收集元数据）**  
   - 遍历 `out_idx in [prev_max+1, new_max]`，用与现逻辑相同的 `window_start_rel` / `window_end_rel` 判定窗口是否完整落在 `all_tokens` 内。  
   - 统计 `num_windows`，或同时把合法的 `(window_start_rel, window_end_rel)` 记进一个小列表（仅存两个 int，比存 tensor 轻）。

2. **若 `num_windows == 0`**：与现逻辑一致，`return None`。

3. **预分配 `stacked`**  
   - `stacked = torch.empty((num_windows * kernel_size, num_heads, head_dim), dtype=..., device=...)`  
   - 形状与当前 `torch.cat` 结果一致：`[num_windows * kernel_size, num_heads, head_dim]`。

4. **第二遍循环（写入）**  
   - 对每个合法窗口：`stacked[w * kernel_size : (w + 1) * kernel_size].copy_(all_tokens[s:e])`  
   - 若使用 `non_blocking=True`，需保证后续 kernel 与 copy 在同一 stream 上且同步语义正确；一般同步 launch 足够。

**可选进阶（减少一次 Python 循环）**

- 若窗口在 `all_tokens` 中步长规律且可表达为 strided batch，可考虑 `as_strided` / `torch.index_select` 一次 gather。当前一般窗口是连续 `[s:e)`，**双循环 + `copy_` 已足够清晰且风险低**。

**收益**

- 去掉 `cat` 的合并拷贝路径；减少临时 tensor 数量。

**风险**

- 低；需保证写入区间与 `cu_seqlens` 语义一致（每段长度仍为 `kernel_size`）。

**验证**

- 与旧实现逐元素对比 `stacked`（小随机 `all_tokens` / 多种 `prev_total_len`、`K`）。  
- 端到端：压缩输出与旧版一致（给定相同 `w`、`pe`）。

---

### 2) `cu_seqlens` / `y_cu_seqlens` 缓存化

**问题**

- 每步 `torch.arange` 创建小张量；decode 高频时增加 **CUDA allocator 与周边开销**。

**改什么**

- `_linear_compress_decode` 中构造 `cu_seqlens`、`y_cu_seqlens` 的位置。  
- 可抽成小工具函数，例如模块级 LRU 或自定义 dict cache。

**怎么改**

- **Cache key** 建议：  
  - `cu_seqlens`：`(device_index, num_windows, kernel_size)`  
  - `y_cu_seqlens`：`(device_index, num_windows)`（与 `kernel_size` 无关）  
- **Value**：缓存 `torch.Tensor`，`requires_grad=False`。  
- **注意**：多 GPU 时 key 必须含 device；可对 `num_windows` 设 **LRU 容量上限**，避免 dict 无限增长。

**收益**

- 减少重复 `arange` 与分配。

**风险**

- 低；注意不要在错误 device 上复用 tensor。

**验证**

- 对比缓存命中前后 `cu_seqlens` 与 `y_cu_seqlens` 数值一致。

---

### 3) 预先 reshape / 持久化 `compress_weight`（`w`）

**问题**

- `linear_cmp_func` 每次调用执行：  
  `w = w.reshape(num_heads, kernel_size, head_dim, head_dim).contiguous()`  
  若 `w` 布局本已不连续，会触发 **额外拷贝**。

**改什么**

- **调用约定**：在模型初始化或第一次 decode 前，将 `compress_weight` 转为 kernel 布局并 `contiguous()`，之后 decode 路径只传该 tensor。  
- **代码**：`linear_cmp_func`（及 `LinearCompressDecode.forward` 若仍使用）中：若 `w` 已是 `[num_heads, kernel_size, head_dim, head_dim]` 且 `is_contiguous()`，则跳过 reshape/contiguous；或提供布局开关。

**`linear_compress_with_pe` 的注意点**

- PE 分支里有 `einsum(pe, w, "h D, h D d -> h d")`，这里 `w` 的语义是 **`[num_heads, kernel_size * head_dim, head_dim]`**（flatten 的 D 维）。  
- 若对外只保留 4D `w`，需要在 einsum 前 `w.view(num_heads, kernel_size * head_dim, head_dim)`（**view，不拷贝**）再算 bias。

**收益**

- 去掉热路径上的重复 contiguous 拷贝。

**风险**

- 低–中：需统一 **一处**「权威布局」，避免有的路径传 3D、有的传 4D 导致 assert 失败；文档化接口。

**验证**

- 对比优化前后 `y` 与 bias 路径输出一致。  
- Profiler 中 `reshape` / `contiguous` 时间应下降或消失。

---

## 第二阶段：Kernel 与 launch 配置（中风险）

### 4) 改 tile 参数：`BLOCK_*`、`num_warps`、`num_stages`

**问题**

- 当前 `block_output_seq_size`、`block_headD_dim`、`block_head_dim`（Hopper 分支）、`num_warps`、`num_stages` 多为固定值，对不同 `head_dim` / GPU 未必最优。

**怎么改**

1. 使用 **`triton.autotune`** 包装 `linear_compress_fwd_kernel`，定义多组 `triton.Config`，把上述参数放进 `kwargs`。  
2. `key` 建议包含：`kernel_size`、`head_dim`、架构（如 Hopper vs 非 Hopper）。  
3. 提供 **fallback**：保留当前固定配置作为默认。  
4. 在 CI 或本地脚本对典型 shape 跑 `triton.testing.do_bench` 生成缓存。

**收益**

- 在算力、寄存器、occupancy 之间找到更优点。

**风险**

- 中：错误配置可能导致变慢或边界错误；需用正确性测试约束。

**本仓库实测备注**

- 曾在 `fsa_preview/ops/linear_compress_decode.py` 上对 `linear_compress_fwd_kernel` 做有限 `triton.autotune`（仅 `BLOCK_HEADD_DIM` / `num_warps` / `num_stages` 等安全组合，`BLOCK_OUTPUT_SEQ_SIZE` 固定 64）。在典型 decode 形状（如 `total_len=2048, H=8, D=128, ks=32`）下 **多次 benchmark 未优于原固定参数**，且 autotune 偶发 **选到更慢配置**；部分 tile 组合还曾触发 Triton LLVM 断言。故 **decode 路径已回退为固定 launch**；若再尝试 autotune，建议 **手写更小候选集 + 每台机器固化最优 config**，而非依赖运行时全量搜索。

---

### 5) 改 grid 并行映射：让 `pid_k` 真正并行更多输出块

**问题**

- `linear_cmp_func` 中 grid 形如：  
  `(batch_size * num_heads, 1, cdiv(head_dim, BLOCK_HEADD_DIM))`  
  第二维恒为 `1`，但 kernel 使用 `pid_k` 与 `BLOCK_OUTPUT_SEQ_SIZE`；当某条样本压缩后长度 `y_len` 较大时，**输出序列维并行未打开**。

**怎么改**

1. 在 host 侧计算 **`max_y_len`**：对 `y_cu_seqlens` 相邻差分求各段 `y_len`，再取 `max`。  
2. 将 grid 第二维改为：  
   `triton.cdiv(max_y_len, META["BLOCK_OUTPUT_SEQ_SIZE"])`（或与 autotune 联动）。  
3. kernel 内已有 `if pid_k * BLOCK_OUTPUT_SEQ_SIZE >= y_len: return`，短序列会自动退出；若空转比例过高，可考虑 **按样本长度分桶** 多套 kernel。

**说明（decode 专用路径）**

- 对当前 `_linear_compress_decode` 构造的 `y_cu_seqlens`（每窗口输出 1 个 token），每段 `y_len = 1`，`max_y_len = 1`，**第二维并行收益为 0**，可保持 `1`。  
- 对 **通用** `linear_cmp_func` 调用（每段多个输出 token），此项更有意义。

**收益**

- 长 `y_len` 场景显著提高并行度。

**风险**

- 中：grid 变大带来 launch 数增加，短序列可能变慢，需 profile。

---

### 6) 改内核访存模式（W / X 读取与复用）

**方向 A：减少 X 的 HBM 往返**

- 调用层减少 `cat` 后，可进一步让 **kernel 直接从 ring buffer / `all_tokens` 按窗口索引读取**，不再物化 `stacked`（**大改**，需改 kernel 参数与地址计算）。

**方向 B：W 复用与布局**

- 保证 `W` 4D contiguous + 合理 `order`，减少 `tl.load` 次优访问。  
- 结合 autotune 调整 `BLOCK_HEADD_DIM`，使 `tl.dot` 路径更饱和。

**方向 C：融合 PE bias**

- 将 `linear_compress_with_pe` 里对 `y` 的 `+ bias` 移入 kernel 写回前（或单独小 fused kernel），减少一次对 `y` 的全量读写。

**风险**

- 中高：尤其「取消 stacked」会改 kernel 接口与地址计算。

---

## 建议实施顺序

1. **(1) 预分配 + `copy_`** — 逻辑清晰，易做对拍。  
2. **(3) w 持久化 + `linear_cmp_func` 短路** — 与 (1) 独立，收益稳定。  
3. **(2) seqlens 缓存** — 小改，注意 LRU。  
4. **(5) grid 第二维** — 对通用 `linear_cmp_func` 优先；decode 专用路径若 `y_len` 恒为 1 可暂缓。  
5. **(4) autotune** — 需 benchmark；**decode 上若实测无收益可跳过**（见上文「本仓库实测备注」），或改为 **离线测几组后把最优 tile 写死**。  
6. **(6) 访存/融合** — 按 profile 结果选一条深挖。

---

## 测试与对拍清单

- **正确性**：随机 `x` / `w` / `pe`，覆盖 `num_windows ∈ {0,1,2,5}`，`kernel_size ∈ {16,32,64,128}`，`bf16` / `fp16`。  
- **边界**：`prev_total_len < kernel_size`、窗口部分越界（应不产生或正确过滤）。  
- **性能**：`torch.profiler` / Nsight：关注 `cat`、`arange`、`reshape` / `contiguous`、kernel 时间、occupancy。  
- **回归**：与上游 `linear_compress_with_pe` 调用方对齐 `w` 形状约定（3D vs 4D）。

---

## 附录：代码锚点（便于 PR 对照）

以下行号以仓库中 `fsa_preview/ops/linear_compress_decode.py` 为准，合并代码后请以实际文件为准。

| 位置 | 内容 |
|------|------|
| `_linear_compress_decode` | 窗口收集、`torch.cat`、`cu_seqlens` / `y_cu_seqlens` |
| `linear_cmp_func` | `w.reshape(...).contiguous()`、grid、`linear_compress_fwd_kernel` launch |
| `linear_compress_fwd_kernel` | `pid_bh` / `pid_k` / `pid_D`、边界判断、`tl.dot` 累加写回 |
| `linear_compress_with_pe` | `linear_cmp_func` 调用、PE `einsum` 与 `y + bias` |

---

## 相关论文与仓库

- 论文：[Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/pdf/2502.11089)  
- 本文件仅描述 **实现侧优化方向**，不改变算法语义。

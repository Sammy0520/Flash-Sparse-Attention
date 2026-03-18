# FSA 接 Llama 与端到端测试文档

**文档版本**: 1.0  
**适用项目**: [Flash-Sparse-Attention](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)  
**配套文档**: [linear-speculative-sampling.md](./linear-speculative-sampling.md)

---

## 一、接 Llama 方案

### 1.1 当前项目状态

| 模块 | 用途 | 是否支持 Decode |
|------|------|-----------------|
| `fsa/` `FlashSparseAttention` | Prefill / 训练 | 否，无 KV cache |
| `fsa_preview/` `FlashSparseAttentionDecode` | 单步 decode | 是 |
| `test/train.py` `SparseLlamaAttention` | 替换 Llama attention | 仅 Prefill |

`FlashSparseAttentionDecode` 已实现，但尚未接入 Llama 的 decode 流程。

---

### 1.2 需要的 Llama 层改造

#### 1.2.1 `LlamaFSADecodeLayer`

用 `FlashSparseAttentionDecode` 替换单层 attention，并实现 decode 前向：

```python
class LlamaFSADecodeLayer(nn.Module):
    """单层 Llama + FSA Decode attention"""

    def __init__(self, config, layer_idx, args):
        super().__init__()
        self.self_attn = FlashSparseAttentionDecode(...)  # 与 fsa_decode 参数一致
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)
        self.layer_idx = layer_idx

    def forward_decode(
        self,
        x: torch.Tensor,  # [K, hidden_size]
        fsa_cache: dict,  # 该层的 k_cache, v_cache, cmp_k_cache, cmp_v_cache, cu_seqlens_k
    ):
        residual = x
        x = self.input_layernorm(x)
        attn_out = self.self_attn(
            x,
            cu_seqlens_q=fsa_cache["cu_seqlens_q"],
            cu_seqlens_k=fsa_cache["cu_seqlens_k"],
            k_cache=fsa_cache["k_cache"],
            v_cache=fsa_cache["v_cache"],
            cmp_k_cache=fsa_cache["cmp_k_cache"],
            cmp_v_cache=fsa_cache["cmp_v_cache"],
        )
        # 更新 fsa_cache（append k_new, v_new 等）
        x = residual + attn_out

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x, fsa_cache
```

---

#### 1.2.2 `fsa_cache` 结构

每一层的 cache 包含：

```python
fsa_cache[layer_idx] = {
    "k_cache": torch.Tensor,      # [past_len, num_kv_heads, head_dim]
    "v_cache": torch.Tensor,      # [past_len, num_kv_heads, head_dim]
    "cmp_k_cache": torch.Tensor,  # 压缩后的 K
    "cmp_v_cache": torch.Tensor,  # 压缩后的 V
    "cu_seqlens_k": torch.Tensor, # [0, past_len] 或 [0, past_len+K]
    "cu_seqlens_q": torch.Tensor, # [0, 1] 或 [0, K]
}
```

---

#### 1.2.3 `LlamaFSAForCausalLM` 与 `prefill_and_build_cache`

**Prefill**：对 prompt 做一次完整 forward，构建各层的 `fsa_cache`。

```python
def prefill_and_build_cache(self, input_ids: torch.Tensor):
    """对 prompt 做 prefill，构建 fsa_cache"""
    hidden_states = self.model.embed_tokens(input_ids)
    past_key_values = None  # 可选，若用 SparseLlamaAttention 的 prefill

    fsa_cache = [None] * self.config.num_hidden_layers

    for layer_idx, layer in enumerate(self.model.layers):
        # 使用 FlashSparseAttention (prefill) 或一次性 forward
        # 得到该层 k_cache, v_cache, cmp_k_cache, cmp_v_cache
        # 构造 cu_seqlens_k = [0, seq_len]
        fsa_cache[layer_idx] = {
            "k_cache": k_cache,
            "v_cache": v_cache,
            "cmp_k_cache": cmp_k_cache,
            "cmp_v_cache": cmp_v_cache,
            "cu_seqlens_k": cu_seqlens_k,
            "cu_seqlens_q": torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        }

    return fsa_cache
```

**Decode**：单次 forward 处理 K 个 token。

```python
def forward_decode(
    self,
    next_token_ids: torch.Tensor,  # [batch, K] 或 [K]
    fsa_cache: list,
):
    """Decode：对 K 个 token 做一次 forward"""
    x = self.model.embed_tokens(next_token_ids)  # [K, hidden_size]
    K = x.shape[0]

    # cu_seqlens_q 用于 K-token
    cu_seqlens_q = torch.tensor([0, K], device=x.device, dtype=torch.int32)

    for layer_idx, layer in enumerate(self.model.layers):
        fsa_cache[layer_idx]["cu_seqlens_q"] = cu_seqlens_q
        x, fsa_cache[layer_idx] = layer.forward_decode(x, fsa_cache[layer_idx])

    logits = self.lm_head(self.model.norm(x))
    return logits, fsa_cache
```

---

#### 1.2.4 cu_seqlens 构造（参考 train.py）

与 Flash Attention varlen 的 packed 格式一致：

```python
# Prefill 时，单序列
seq_len = input_ids.shape[1]
cu_seqlens = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

# 多序列时
seqlens = torch.LongTensor([len1, len2, ...]).cuda()
cu_seqlens = torch.cat([
    torch.zeros(1, dtype=torch.int32, device="cuda"),
    torch.cumsum(seqlens, dim=0),
], dim=0).to(torch.int32)
```

Decode 时：
- `cu_seqlens_q = [0, K]`（单次验证 K 个 token）
- `cu_seqlens_k = [0, past_len+K]`（KV 总长度）

---

### 1.3 实现检查清单

| 步骤 | 内容 |
|------|------|
| 1 | 实现 `LlamaFSADecodeLayer`，替换 `LlamaDecoderLayer` 的 attention |
| 2 | 实现 `prefill_and_build_cache`：用 prefill 路径构建各层 cache |
| 3 | 实现 `forward_decode`：embed → 各层 FSA decode → lm_head |
| 4 | 确保 `FlashSparseAttentionDecode` 支持 K>1（见 linear-speculative-sampling.md） |

---

## 二、层级 2 测试：完整 Llama Prefill + Decode 循环

### 2.1 测试目标

对比两种 decode 方式：
- **Baseline**：每步 1 个 token，循环 N 次
- **SD**：每步 K 个 token，循环 N/K 次（总 token 数相同）

验证 K-token 验证的加速效果，**不涉及 Draft 模型**。

---

### 2.2 前置条件

- 已实现 `LlamaFSAForCausalLM`、`prefill_and_build_cache`、`forward_decode`
- `forward_decode` 支持 `next_token_ids: [K]`（K≥1）

---

### 2.3 测试脚本结构

```python
# test/test_llama_fsa_decode_e2e.py（示意）

import torch
import copy
from transformers import AutoTokenizer

def benchmark_decode_loop(model, fsa_cache, num_steps, K=1, warmup=5, iters=20):
    """Decode 循环基准测试"""
    model.eval()
    total_tokens = 0

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            next_ids = torch.randint(0, 32000, (K,), device="cuda")
            _, fsa_cache = model.forward_decode(next_ids, copy.deepcopy(fsa_cache))

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    fsa_cache_work = copy.deepcopy(fsa_cache)
    start.record()
    with torch.no_grad():
        steps = num_steps // K
        for _ in range(steps):
            next_ids = torch.randint(0, 32000, (K,), device="cuda")
            _, fsa_cache_work = model.forward_decode(next_ids, fsa_cache_work)
            total_tokens += K
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms, total_tokens


def run_level2_test(model, tokenizer, past_len=4096, num_steps=256, K=8):
    # 1. Prefill：构造长度为 past_len 的 prompt
    prompt = "Hello world " * (past_len // 12)  # 约 past_len tokens
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :past_len].cuda()
    fsa_cache = model.prefill_and_build_cache(prompt_ids)

    # 2. Baseline: K=1，每步 1 个 token
    elapsed_b, tokens_b = benchmark_decode_loop(
        model, fsa_cache, num_steps, K=1, warmup=5, iters=20
    )
    tok_per_sec_baseline = tokens_b / (elapsed_b / 1000)

    # 3. SD: 每步 K 个 token
    elapsed_sd, tokens_sd = benchmark_decode_loop(
        model, fsa_cache, num_steps, K=K, warmup=5, iters=20
    )
    tok_per_sec_sd = tokens_sd / (elapsed_sd / 1000)

    print(f"Baseline (K=1): {tok_per_sec_baseline:.1f} tok/s")
    print(f"SD (K={K}):    {tok_per_sec_sd:.1f} tok/s")
    print(f"Speedup:       {tok_per_sec_sd / tok_per_sec_baseline:.2f}x")
    return tok_per_sec_baseline, tok_per_sec_sd
```

---

### 2.4 运行命令

```bash
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python test/test_llama_fsa_decode_e2e.py \
  --model-path meta-llama/Llama-3-8B \
  --past-len 4096 \
  --num-steps 256 \
  --K 8 \
  --warmup 5 \
  --iters 20
```

---

### 2.5 建议测试配置

| past_len | K | num_steps |
|----------|---|-----------|
| 4096 | 4 | 256 |
| 4096 | 8 | 256 |
| 8192 | 8 | 256 |
| 16384 | 8 | 256 |

---

### 2.6 输出格式示例

```
=== Level 2: Decode Loop Benchmark ===
Config: past_len=4096, K=8, steps=256

Baseline (K×1-step):  XX.X tok/s
SD (1×K-step):        XX.X tok/s
Speedup:              X.XXx
```

---

## 三、层级 3 测试：Draft + Target 完整投机采样

### 3.1 测试目标

端到端 speculative decoding：
- **Draft 模型**（如 Llama-1B）：快速生成 K 个候选 token
- **Target 模型**（FSA-Llama）：一次 forward 验证 K 个 token
- 按标准 SD 算法做 accept/reject，统计 TTFT、TPOT、acceptance rate

---

### 3.2 前置条件

- Target：已接 FSA decode 的 Llama
- Draft：标准 Llama（小模型，如 1B）
- 实现完整的 SD 主循环

---

### 3.3 投机采样主循环（伪代码）

```python
def run_speculative_generation(
    target_model,
    draft_model,
    tokenizer,
    prompt_ids,
    max_new_tokens=256,
    K=8,
):
    target_model.eval()
    draft_model.eval()

    # 1. Prefill：两个模型各做一次
    target_cache = target_model.prefill_and_build_cache(prompt_ids)
    draft_kv = draft_model(prompt_ids).past_key_values

    generated = prompt_ids.clone()
    total_target_forwards = 0
    total_tokens_accepted = 0

    with torch.no_grad():
        while generated.shape[1] < prompt_ids.shape[1] + max_new_tokens:
            # 2. Draft 生成 K 个 token
            draft_tokens = []
            draft_kv_i = draft_kv
            for _ in range(K):
                logits = draft_model(
                    generated[:, -1:], past_key_values=draft_kv_i
                ).logits[:, -1]
                next_token = logits.argmax(dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                generated = torch.cat([generated, next_token], dim=1)
                draft_kv_i = draft_model(
                    next_token, past_key_values=draft_kv_i
                ).past_key_values

            draft_tokens = torch.cat(draft_tokens, dim=1)  # [1, K]

            # 3. Target 一次验证 K 个 token
            target_logits, target_cache = target_model.forward_decode(
                draft_tokens.squeeze(0), target_cache
            )
            total_target_forwards += 1

            # 4. Accept/Reject（greedy 对比）
            accepted = 0
            for i in range(K):
                pred = target_logits[i].argmax().item()
                if pred == draft_tokens[0, i].item():
                    accepted += 1
                    total_tokens_accepted += 1
                else:
                    total_tokens_accepted += 1
                    generated = generated[:, : -(K - i)]
                    generated = torch.cat([
                        generated,
                        torch.tensor([[pred]], device=generated.device),
                    ], dim=1)
                    break
            else:
                total_tokens_accepted += K

            if accepted < K:
                break

    return generated, total_target_forwards, total_tokens_accepted
```

---

### 3.4 测试脚本要点

1. **Baseline**：不用 draft，target 自回归每步 1 个 token
2. **SD**：draft 生成 K 个 → target 一次验证 K 个
3. **指标**：
   - TTFT（Time To First Token）
   - TPOT（Time Per Output Token）或 tokens/s
   - Target forward 次数
   - Acceptance rate（近似：accepted / (K × num_target_forwards)）

---

### 3.5 运行命令

```bash
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python test/test_speculative_e2e.py \
  --target-model meta-llama/Llama-3-8B \
  --draft-model meta-llama/Llama-3.2-1B \
  --K 8 \
  --max-new-tokens 256 \
  --prompt-len 512 \
  --num-samples 10
```

---

### 3.6 建议测试配置

| Target | Draft | K | max_new_tokens | Prompt 长度 |
|--------|-------|---|----------------|-------------|
| Llama-8B+FSA | Llama-1B | 4 | 256 | 512 |
| Llama-8B+FSA | Llama-1B | 8 | 256 | 512 |
| Llama-8B+FSA | Llama-1B | 8 | 512 | 1024 |

---

### 3.7 输出格式示例

```
=== Level 3: Full Speculative Decoding ===
Config: target=Llama-8B, draft=Llama-1B, K=8

Baseline:       XX.X tok/s
SD:             XX.X tok/s
Speedup:        X.XXx
Accept rate:    XX.X%
Target forwards: XXX (vs baseline XXX)
```

---

## 四、测试执行顺序

| 顺序 | 层级 | 内容 | 依赖 |
|------|------|------|------|
| 1 | 单元 | `test_FSA_decode.py` 支持 `--num-verify-tokens K` | 无 |
| 2 | 接 Llama | 实现 `LlamaFSADecodeLayer`、`prefill_and_build_cache`、`forward_decode` | 单元 K>1 |
| 3 | 层级 2 | `test_llama_fsa_decode_e2e.py`：Baseline vs SD decode 循环 | 接 Llama |
| 4 | 层级 3 | `test_speculative_e2e.py`：Draft + Target 完整 SD | 层级 2 |

---

## 五、参考路径

| 文件 | 说明 |
|------|------|
| `fsa_preview/module/fsa_decode.py` | `FlashSparseAttentionDecode` 实现 |
| `test/train.py` | `SparseLlamaAttention`、`replace_llama_attention`、cu_seqlens 构造 |
| `test/test_FSA_decode.py` | 单步 decode 测试（当前 K=1） |
| `docs/linear-speculative-sampling.md` | SD 设计、cu_seqlens 语义、模块修改清单 |

---

## 六、与 linear-speculative-sampling 的关系

本测试文档对应 `linear-speculative-sampling.md` 中：

- **7.1 正确性测试**：在单元层（`test_FSA_decode.py`）验证 K-token 与 K 次单步数值一致
- **7.2 单元测试扩展**：增加 `--num-verify-tokens` 参数
- **7.3 性能基准**：层级 2 的 decode 循环基准
- **7.4 端到端集成测试**：层级 3 的 Draft + Target 完整 SD

实现并跑通本文档的测试流程后，即可支撑 SD 方案的完整验证。

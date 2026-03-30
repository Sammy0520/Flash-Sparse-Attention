一、要实现的三个系统
1. Baseline：dense-llama（逐 token）
模型：Llama-3.1-8B（dense attention）
每次只生成 1 个 token
使用 KV cache（标准 autoregressive）
2. dense-llama-sd（标准 SD）
draft model：Llama-3.2-1B
target model：Llama-3.1-8B（dense）
每轮：
draft 生成 N 个 token
target 一次 forward 验证
sampling 接受部分 token
必须是多轮循环
3. nsa-llama-sd（你的方法）
draft model：同上（1B）
target model：Llama-3.1-8B + NSA attention
其他流程和 dense-SD 完全一致
唯一差别：target forward 用 NSA
二、统一接口（必须这样组织）

实现三个类：

DenseARRunner
DenseSDRunner
NSASDRunner

统一接口：

generate(input_ids, max_new_tokens) -> generated_ids, stats
三、核心：multi-round SD 主循环（必须严格按这个写）
state:
  generated_ids
  draft_kv_cache
  target_kv_cache
  current_token

while len(generated_ids) < max_new_tokens:

  # 1. draft阶段
  从 current_token 开始
  用 draft model autoregressive 生成 N 个 token
  同时保存：
    draft_ids: [t1 ... tN]
    draft_logits: [N, vocab]
    draft_kv_cache（增长）

  # 2. target验证（一次 forward）
  输入：
    current_token + draft_ids
  输出：
    verify_logits: [N, vocab]

  # 3. sampling accept（关键）
  对 i=1..N：
    计算：
      p = target_prob(token_i)
      q = draft_prob(token_i)
      接受概率 = min(1, p/q)

    随机采样决定是否接受
    如果拒绝：
      从 target 分布重新采样一个 token
      停止

  得到：
    accept_len = k
    accepted_ids

  # 4. commit（必须只提交 accepted 部分）
  generated_ids += accepted_ids

  # 5. KV cache 更新（核心）
    draft_kv_cache：只保留前 k 个新增
    target_kv_cache：只追加前 k 个对应 KV

  # 6. 更新 current_token
    current_token = accepted_ids[-1]
四、KV cache 规则（必须严格遵守）
dense-SD
draft cache
正常 append
如果 reject：
丢弃未接受部分
target cache
只 append accepted 部分
不允许保留未接受 token
NSA-SD（重点）

每层有：

k_raw, v_raw
cmp_k, cmp_v
每轮：
raw cache：
k_raw = cat(old, new[:k])
v_raw = cat(old, new[:k])
compressed cache：
必须只对新增部分做压缩，然后拼接
new_cmp = compress(new[:k])
cmp_k = cat(old_cmp, new_cmp)

禁止：

重新压缩整个序列（会导致 O(L^2)）
五、sampling accept（必须替换你现在的 greedy）

对每个 token i：

p = softmax(target_logits[i])[draft_token]
q = softmax(draft_logits[i])[draft_token]

accept_prob = min(1, p / q)
采样 u ~ Uniform(0,1)
如果 u < accept_prob：接受
否则：
从 target 分布重新采样一个 token
停止本轮
六、实验设置（必须统一）
输入
固定 prompt（长度可以 512 / 2k / 8k）
tokenizer 必须一致
参数
max_new_tokens = 256（或 512）
n_draft = 4 / 8 / 16
temperature = 1.0
七、必须记录的指标

每个系统都输出：

total_time
total_generated_tokens
tokens_per_second = total_generated_tokens / total_time

SD 系统额外记录：

total_drafted_tokens
total_accepted_tokens
acceptance_rate = accepted / drafted
avg_accept_len = accepted / rounds
num_rounds
八、你最终要验证的结论

必须同时成立：

dense-llama-sd  的 tokens/s > dense-llama
nsa-llama-sd    的 tokens/s > dense-llama-sd
九、实现顺序（必须按这个顺序做）
实现 DenseARRunner（最简单）
实现 DenseSDRunner（multi-round + sampling + KV cache）
验证：
acceptance > 0（通常 0.5~0.8）
dense-SD 比 dense 快
替换 target 为 NSA（先不优化 cache，保证正确）
最后实现 NSA cache 增量更新
十、完成标准（agent必须满足）

你的实现是“正确”的，当且仅当：

acceptance rate 明显 > 0（不是全 0）
多轮运行（num_rounds > 1）
KV cache 长度随生成增长（不是每轮重算）
dense-SD 比 baseline 快
NSA-SD 在合理参数下进一步更快
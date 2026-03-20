下面是之前关于 EAGLE3 的总结，方便你在 NSA 新项目里对照使用。

一、EAGLE3 多 token 投机采样 — 总览
目标：用「一次前向」同时验证多条候选路径上的多个 token，加速解码。
做法：Draft 模型按 top-k 树 生成多分支候选 → 目标模型用 树状 attention mask 一次前向算整棵树的 logits → 按路径取 logits，做验证 → 只接受一条路径的一段 token，更新 KV，再建下一棵树，循环。
二、主要流程（简要）
1. EaModel.eagenerate（入口）
准备：logits_processor、KV cache、reset tree 等。
Prefill + 建第一棵树：initialize_tree。
循环：设 tree_mask → tree_decoding（一次前向）→ evaluate_posterior（多路径多 token 验证）→ update_inference_inputs（接受 token、更新 KV、建下一棵树）→ 检查 eos / max_new_tokens / max_length，直到结束。
2. utils.py
initialize_tree：用 prefix 前向一次，采一个根 token，调 ea_layer.topK_genrate 建第一棵树，返回 draft_tokens / retrieve_indices / tree_mask / tree_position_ids 等。
tree_decoding：把整棵树 + position_ids 喂给目标模型一次前向，用 retrieve_indices 从 logits 里取出每条路径的 logits。
evaluate_posterior：比较 target logits 与 draft candidates，得到 best_candidate、accept_length、sample_p（greedy 或采样分支）。
update_inference_inputs：只取 best_candidate 路径上前 accept_length+1 个 token 拼到 input_ids；在 KV 里把对应位置拷到新前缀；用接受路径的 hidden + sample_p 采 1 个新 token，再调 topK_genrate 建下一棵树；更新 new_token 并返回。
3. cnets.py — Draft 模型
用目标模型的 hidden 做输入，按 top-k 树 扩展多分支候选。
Model.topK_genrate：前缀前向取根 hidden → 根做 top-k → 按 depth 循环扩展（每层前向、路径分数、全局留 top-k）、扩展 tree_mask → 按分数取 top total_tokens 个节点 → 建 tree_mask / tree_position_ids / retrieve_indices → 返回 draft_tokens 等供一次前向验证。
4. modeling_llama_kv.py — Base 模型（LLaMA）
LlamaModel._prepare_decoder_attention_mask：先建因果 mask，若有 tree_mask 则把「树段对树段」那一块用 tree_mask 覆盖（tree_mask==0 的位置置为 -inf），这样树内按路径可见。
LlamaModel.forward：用上面这份 mask 和 position_ids、past_key_values 做一次前向，得到整棵树的 hidden。
LlamaForCausalLM：对 hidden 做 lm_head 得到 logits；utils 里用 logits[0, retrieve_indices] 按路径取。
LlamaAttention.forward：标准 Q/K/V + RoPE + attn_weights + attention_mask 再 softmax；EAGLE 只是传入「带 tree 的 mask」，没有改 attention 公式。
三、概念速查
logits：模型对「下一个 token」的 logits，形状 [..., vocab_size]；softmax 后为概率。
vocab：词表大小，logits 的最后一维。
hidden (hidden_states)：某一层、某一位置的向量表示，在 lm_head 之前；EAGLE 的 draft 用目标模型给的 hidden 做输入。
前向 / 后向：前向 = 从输入算到 logits/loss；后向 = 反向传播算梯度；解码时只用前向。
tree_mask：树状注意力掩码，树段内每个节点只能看自己路径上的祖先，使「一次前向」= 多条路径并行。
retrieve_indices：每条「根→叶」路径在 draft 序列中的位置下标，用来从整棵树 logits 里按路径取 [num_paths, path_len, vocab]。
position_ids：树解码时为 prefix_len + tree_position_ids（树内深度），保证路径上位置连续。
四、树结构（简图）
根：1 个节点（sample_token）。
扩展：每层从当前 k 个节点各做 top-k 扩展，按路径分数全局保留 top-k，共 depth 层。
剪枝：从整棵展开的树里按分数保留 total_tokens 个节点，得到最终子树；draft_tokens / tree_mask / retrieve_indices 都基于这棵子树。
五、和 NSA 的迁移要点
一次前向验证多 token = 合适的候选结构 + 对应的 attention mask + 按路径取 logits + evaluate_posterior 式验证 + KV 只保留接受路径。
NSA 若要做多 token 验证：定义好「路径」与「路径在序列中的下标」→ 设计 NSA 的 mask（谁可以看谁）→ 一次前向 → 用类似 retrieve_indices 的方式按路径取 logits → 复用或改写 evaluate_posterior / update_inference_inputs 的思路。
本项目 没有自定义算子，全是 PyTorch 标准 op + mask/索引；NSA 可先在同一层面改逻辑，再考虑是否写 CUDA kernel。
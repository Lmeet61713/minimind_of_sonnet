# 04 · Agent RL 与推理部署

> 这一篇是整个仓库最「工程」的部分：**多轮工具调用**打破了单轮 RL 的所有假设
> ——序列不再是「prompt + 一段回答」，而是「模板 + 模型输出 + 工具返回 + 模板 + 模型输出 + …」交错。
> 怎么在这种结构上做策略梯度，是 `train_agent.py` 要回答的全部问题。

---

## 1. 从单轮到多轮：三个新问题

单轮 GRPO 的世界很干净：

```
[ prompt tokens ][ completion tokens ]
                   └── 全部算 loss ──┘
```

多轮 Agent 的世界：

```
<|im_start|>system ... tools ...<|im_end|>
<|im_start|>user 北京天气?<|im_end|>
<|im_start|>assistant
我需要调用工具 <tool_call>{"name":"get_current_weather",...}</tool_call><|im_end|>   ← 模型生成，算 loss
<|im_start|>tool
{"city":"北京","temperature":"28°C",...}<|im_end|>                                  ← 工具返回，不算 loss
<|im_start|>assistant
北京今天 28°C，晴。<|im_end|>                                                        ← 模型生成，算 loss
```

三个新问题：

1. **mask 不再连续**：模型自己的 token 和工具的 token 交错，`completion_mask` 必须是 0/1 的拼接。
2. **对齐不再显然**：整条序列是分好几轮拼出来的，`logp_pos` 怎么算？
3. **模板 token 混进来了**：工具结果不是直接用 `tokenizer(tool_result)` 拼进去的，
   中间还夹着 `<|im_start|>tool\n`、`<|im_end|>` 这些模板 token。

---

## 2. marker 技巧：本仓库最值得学的一段

`rollout_single` 里这几行：

```python
marker = f"<|agent_observation_{id(messages)}_{len(response_ids)}|>"
assistant_message["content"] += marker                 # ① 在 assistant 正文末尾塞一个唯一标记
marked_context = tokenizer.apply_chat_template(messages, tokenize=False, ...)   # ② 整体渲染
assistant_message["content"] = new_text                # ③ 立刻还原，不污染后续
_, found, observation = marked_context.partition(marker)                        # ④ 按 marker 切开
obs_delta = tokenizer(observation, add_special_tokens=False)["input_ids"]       # ⑤ 只 tokenize 后半段
```

**它要解决的问题**：我们已经有一条 `response_ids`（模型前面几轮生成的 token），
现在要把「工具返回 + 下一轮的 `<|im_start|>assistant\n` 模板头」**精确地**追加到它后面。

**为什么不能直接 `tokenizer(json.dumps(result))`？**

- 会漏掉模板 token（`<|im_start|>tool` / `<|im_end|>`）；
- BPE 是**上下文相关**的：`"28°C"` 单独 tokenize 和放在长文本末尾 tokenize 的切分可能不同，
  拼起来的结果和「整段重新 tokenize」不一致，模型看到的 `input_ids` 就和你以为的不一样。

**marker 方案的正确性**：`marker` 卡在「assistant 正文」和「模板插入的 `<|im_end|>`」之间，
所以按它切开后的后半段，正好是**从这一轮结束到下一轮开始**的全部新增文本（含模板 token）。
再 tokenize 这一段，拼回 `response_ids`，就与「整段重新 tokenize」逐 token 相同。

还有两个细节：

```python
if new_ids and new_ids[-1] == eos and obs_delta[:1] == [eos]:
    obs_delta = obs_delta[1:]          # 去掉重复的 eos
```

- 模型本轮输出以 `<eos>` 结尾，而模板渲染又会补一个 `<|im_end|>` 的 eos。
  两个条件都满足才去掉一个——**撞上 `max_new_tokens` 被硬截断时不能删**（那时 `new_ids[-1] != eos`）。
- `marker` 里带 `id(messages)` 和当前长度：同一进程内每次都不一样，
  不会撞上模型输出或工具返回里的已有文本。

`response_mask` 的构造：

```python
response_mask.extend([int(t != eos) for t in new_ids])   # 模型生成的 token → 1（eos 除外）
response_mask.extend([0] * len(obs_delta))               # 工具/模板 token → 0
```

> 只看 `mask == 1` 的位置算 loss，工具返回的内容永远不会被「学」。

---

## 3. Reward 设计：让规则教会模型「怎么用工具」

`calculate_rewards` 有两条分支。

### 3.1 没有调用工具（模型直接回答）

| 项 | 分值 | 作用 |
|---|---|---|
| 回答长度在 [5, 800] | +0.5 / −0.5 | 防太短（敷衍）和太长（灌水） |
| thinking 长度在 [20, 300] | +1.0 / −0.5 | 鼓励有效思考，不要长篇大论 |
| `</think>` 恰好出现 1 次 | +0.25 / −0.25 | 格式正确性 |
| reward model 打分 | ±3 | 内容质量 |
| 3-gram 重复率惩罚 | −(0~0.5) | 防复读 |
| 总分 | clip 到 ±3 | 防单条样本主导 |

### 3.2 调用了工具

| 项 | 分值 | 作用 |
|---|---|---|
| 工具数量与 GT 对齐 | `0.5` / `−0.5·gap` | `gap = |valid_call_count − len(gt)| + 多余调用数` |
| GT 命中率 | `2.5 × #verified / #gt` | 主奖励：最终答案里有没有正确答案 |
| 未完成（轮数用尽仍在调工具） | −0.5 | 惩罚死循环 |
| 重复惩罚 | −(0~0.5) | |
| 总分 | clip 到 ±3 | |

**设计原则：主奖励 >> shaping 项。** 所有格式分加起来不超过 ±1，而 GT 分是 +2.5。
否则模型会学会「刷格式」而不是「答对题」——这就是 reward hacking 的典型形态。

`validate_gt_in_text` 用两路匹配（字符串包含 + 数值 `1e-6` 容差），
并把千分位逗号去掉，避免 `1,234` 和 `1234` 对不上。

---

## 4. 打包：把一条「多轮轨迹」变成一个训练样本

`rl_train_epoch` 里的这段：

```python
ids      = prompt_ids + response_ids            # 整条轨迹
mask     = [0]*len(prompt_ids) + response_mask  # prompt 全 0，生成段按 0/1
old_logps = [0.0]*max(len(prompt_ids)-1, 0) + old_lp   # 长度是 len(ids)-1（错一位）
```

三个约定：

- `old_logps` 的长度是 `len(ids) - 1`：因为位置 `t` 的 logits 预测 `t+1`，
  最后一个 token 没有对应的「预测位置」。
- 超长时**从左侧截断**（`ids[-max_total_len:]`），`old_logps` 也要跟着取最后 `L-1` 个，保证右对齐。
- `prompt_len = 第一个 mask==1 的位置`（截断后 prompt 可能被砍掉一部分）。

然后 `logp_pos = prompt_lens - 1 + arange(R)`，和单轮 GRPO 的公式完全一致——
**多轮的所有复杂度，都在 rollout 阶段被压平成了「一条普通序列 + 一个 0/1 mask」**。
这是这个实现最漂亮的地方。

---

## 5. rollout 引擎：训练 / 推理解耦

`trainer/rollout_engine.py` 定义了一个抽象层：

```python
class RolloutEngine(ABC):
    def rollout(...) -> RolloutResult      # 采样 + 返回 per-token logprob
    def update_policy(model)               # 把训练侧的新权重同步给推理侧
```

两个实现：

| | TorchRolloutEngine | SGLangRolloutEngine |
|---|---|---|
| 后端 | 直接用训练中的模型 `.generate()` | 独立进程的 SGLang server，HTTP |
| 优点 | 零依赖、结果确定 | 快（continuous batching、prefix cache） |
| 缺点 | 慢，且和训练抢同一张卡 | 需要同步权重、两套算子可能数值不一致 |
| 权重同步 | 同一个 `Parameter` 对象，换引用即可 | `save_pretrained` 落盘 → `/update_weights_from_disk` → 所有 rank broadcast 成功标志 |

**为什么要抽象这一层**：RL 的训练和推理是两套完全不同的最优实现
（训练要反向传播、推理要 KV cache 和批处理）。把它们解耦，才能在不改训练逻辑的前提下
换更快的推理后端——这也是工业界 vLLM / SGLang 在 RLHF 里的标准用法。

> ⚠️ `update_policy` 在 sglang 分支里会做 `dist.broadcast`，
> 所以**必须在所有 rank 上调用**，否则死锁。

---

## 6. 推理侧：模型怎么变成「服务」

```
eval_llm.py         CLI 单文件推理（加载 .pth 或 HF 目录）
  ├── 原生权重：init_model()  → .pth state_dict
  └── HF 权重：  AutoModelForCausalLM + tokenizer

scripts/convert_model.py    .pth ↔ HF 目录（映射成 Qwen3 键名）/ LoRA 合并
scripts/serve_openai_api.py FastAPI，/v1/chat/completions，SSE 流式 + tools
scripts/web_demo.py         Streamlit WebUI
scripts/eval_toolcall.py    工具调用评测（本地 / API 两种后端）
```

### 6.1 工具调用的协议

模型输出的是**纯文本**：

```
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "北京"}}
</tool_call>
```

框架负责：正则解析 → 执行函数 → 把结果以 `role="tool"` 回填 → 重新渲染模板 → 再生成。
**模型永远不执行任何代码**，它只是「按格式写一段 JSON」。这一层安全性由 `execute_tool` 的白名单 +
`safe_math_eval`（AST 白名单，禁止 `eval`）保证。

### 6.2 流式输出为什么难

服务端要一个字一个字吐（SSE），但 `<think>` / `<tool_call>` 这类标签可能被切在两个 chunk 之间。
`serve_openai_api.py` 用了一个**状态机 + pending 缓冲**：先攒着，确认不是标签前缀再吐出去。
这是所有流式服务的通用难题。

### 6.3 三个容易踩的坑

1. **`max_length` vs `max_new_tokens`**：原生 `transformers.generate` 用前者，自研 `generate` 用后者。
   写错会直接报错或截断。
2. **`open_thinking`**：开 / 关会在 assistant 前缀里插不插 `<think>`，
   **训练和推理必须一致**，否则模型看到的分布变了。
3. **`role="tool"` 的模板**：chat template 把它渲染成 `<tool_response>`；
   而走 OpenAI API 时必须带 `tool_call_id`。两套协议不能混。

---

## 7. 一句话总结

> Agent RL = **多轮 rollout（marker 对齐 token）+ 规则奖励（主奖励 + shaping）
> + 单轮 GRPO 的组内优势**。
> 所有多轮带来的复杂性，都被压缩进「一条拼接好的序列 + 一个 0/1 mask」，
> 剩下的训练代码和 `train_grpo.py` 一模一样。

---

← [03 · 对齐](03-对齐-DPO-PPO-GRPO.md) ｜ 下一篇 → [99 · 自查清单](99-自查清单与常见坑.md)

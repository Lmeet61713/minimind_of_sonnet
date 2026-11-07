# 00 · 总览：从 nanoGPT 到 MiniMind

> 目标读者：刚把 nanoGPT 跑通、知道 `Transformer = 一堆 Block + lm_head`、知道交叉熵是
> `-log p(正确 token)` 的人。这份笔记负责回答一个问题：
> **「一个只会补全文本的模型，是怎么一步步变成会聊天、会听指令、会调用工具的小助手的？」**

---

## 1. 一句话说清这个仓库

MiniMind 用 **一套结构（约 64M 参数的 decoder-only Transformer）+ 六段训练流水线**，
把「随机初始化」推到「能多轮调用工具的 Agent」。每段训练只改三样东西：

1. **数据长什么样**
2. **loss 怎么算**（哪些 token 计分、目标是什么）
3. **从谁身上继续训**（`--from_weight`）

模型结构从头到尾没变过（MoE 变体除外）。**所有魔法都在数据和 loss 里**——这是最重要的心智模型。

```
                       ┌──────────────────────────────────────────────┐
                       │   model/model_minimind.py  （结构，始终不变）  │
                       └──────────────────────────────────────────────┘
                                          ▲ 同一份结构，被 6 个 trainer 复用
                                          │
  语料 ──► train_tokenizer.py ──► vocab=6400
                                          │
   pretrain_t2t_mini.jsonl ──► ① 预训练   pretrain_768.pth      「会说话」
                                          │
   sft_t2t_mini.jsonl     ──► ② 全参SFT  full_sft_768.pth       「会听指令」
                                          │
              ┌───────────────┬───────────┴────────────┬──────────────────┐
              ▼               ▼                        ▼                  ▼
      ③ LoRA 微调      ④ 知识蒸馏                ⑤ DPO 偏好对齐     ⑥ PPO / GRPO
      lora_*.pth       full_dist_768.pth         dpo_768.pth        ppo_actor / grpo
      （垂直能力）      （MoE→Dense）              （人类偏好）        （AI 反馈 + 规则）
                                                                        │
                                                              ⑦ Agent RL（多轮工具）
                                                              agent_768.pth
                                                                        │
                                          推理：eval_llm.py / serve_openai_api.py / web_demo.py
```

编号 ③④⑤⑥⑦ 是**并列分支**，都从 ② 的 `full_sft` 出发，不是必须串行（⑤⑥⑦ 也可以从 ① 出发，
但实践中「先 SFT 再对齐」收敛快得多）。

---

## 2. 六个阶段各自改了什么

| 阶段 | 数据 | 监督信号 | loss | 从谁开始 | 产出 |
|---|---|---|---|---|---|
| ① 预训练 | 纯文本 `{"text": ...}` | 下一个 token（全序列都算） | 移位交叉熵 | 随机初始化 | 基座 |
| ② SFT | 多轮对话 `conversations` | **只有 assistant 的回复**算 | 移位交叉熵 + mask | pretrain | 指令模型 |
| ③ LoRA | 同上 | 同上 | 同上，但只更新低秩矩阵 | full_sft | 适配器（0.39M） |
| ④ 蒸馏 | 同上 | teacher 的完整分布 | `α·CE + (1-α)·T²·KL` | teacher+student 各自 | 小模型 |
| ⑤ DPO | chosen / rejected 成对 | 偏好排序 | `-log σ(β·Δmargin)` | full_sft | 对齐模型 |
| ⑥ PPO / GRPO | 只有 prompt | 标量 reward（模型或规则给） | 策略梯度 + clip + KL | full_sft | 对齐模型 |
| ⑦ Agent RL | prompt + tools + gt | 规则 reward（工具是否调对） | 同 GRPO，多轮拼接 | full_sft | 工具模型 |

**看懂这张表，就懂了整个仓库的骨架。**

---

## 3. 模型解剖：64M 是怎么算出来的

`hidden=768, layers=8, heads=8, kv_heads=4, head_dim=96, vocab=6400, intermediate=2432`

| 部件 | 公式 | 参数量 |
|---|---|---|
| embedding（与 lm_head 共享） | `V*D = 6400*768` | 4,915,200 |
| 每层 attention（q,k,v,o） | `768*768 + 768*384 + 768*384 + 768*768` | 1,769,472 |
| 每层 QK-Norm | 两个 96 维 gain | 192 |
| 每层 MLP（SwiGLU 三个投影） | `3*768*2432` | 5,603,328 |
| 单层小计 | | 7,372,992 |
| 8 层 | | 58,983,936 |
| final RMSNorm | | 768 |
| **合计** | | **63,899,904 ≈ 64M** |

三个「省参数」的设计值得记住：

- **GQA**：K/V 只有 4 个头（`768→384`），Q 有 8 个。省 1/3 的 attention 参数，
  更重要的是**推理时 KV cache 直接减半**。
- **tie_word_embeddings**：输出层和 embedding 是同一个 `Parameter` 对象，省 4.9M（约 7.7%）。
  代价是「输入表征」和「输出 logits」被绑定，表达能力受限，但小模型上净收益为正。
- **intermediate = ceil(D·π/64)·64 = 2432**：LLaMA 常用 `8/3·D = 2048`（只有 56.8M），
  这里用 π 是为了**凑到 64M 这个整数指标**；`ceil(...64)*64` 是为了让矩阵宽度对齐 GPU tile。

结构上等价于一个缩小版的 Qwen3 / LLaMA：
`RMSNorm + RoPE + GQA + SwiGLU + pre-norm 残差`，`scripts/convert_model.py` 就是靠这个等价性
把权重映射成 Qwen3 的键名，从而白嫖 transformers / vllm / sglang 的推理生态。

---

## 4. 与 nanoGPT 的逐项对照

| 维度 | nanoGPT | MiniMind |
|---|---|---|
| 规模 | 124M（GPT-2 small） | 64M，8 层 |
| 归一化 | LayerNorm | **RMSNorm**（不减均值、无 bias） |
| 位置编码 | 可学习绝对位置 `wpe` | **RoPE**（旋转位置编码，可外推） |
| 注意力 | MHA（Q=K=V 头数相同） | **GQA**（8 Q 头 / 4 KV 头） |
| FFN | `4d` 两层 GELU | **SwiGLU** `down(silu(gate(x))*up(x))` |
| logits 裁剪 | 无 | `logits_to_keep`（推理只算最后一个 token） |
| 生成 | 手写循环 | 手写 `generate` + KV cache + top_k/top_p/重复惩罚 |
| 训练阶段 | 只有 pretrain | pretrain → SFT → LoRA/蒸馏 → DPO → PPO/GRPO → Agent RL |
| 推理接口 | 自己写脚本 | HF `PreTrainedModel` + `GenerationMixin` + OpenAI 兼容 API |

**从 nanoGPT 过来，模型结构只需要新学 4 个东西**：RMSNorm、RoPE、GQA、SwiGLU。
它们各自解决的问题是：

- RMSNorm：除以 RMS 而不是标准差，少一次均值计算，数值更稳（尤其 bf16）。
- RoPE：把位置信息编码成「旋转角度」，`q·k` 自然只依赖相对距离 → `train_short / test_long` 更容易外推。
- GQA：在 MHA（质量）和 MQA（省显存）之间折中，现在几乎所有开源模型都用。
- SwiGLU：门控 FFN，同样参数下比 GELU 效果好，代价是多一个投影。

---

## 5. 一次前向的数据流（务必背下来）

```
input_ids            [B, T]        int64, 取值 [0, 6400)
   │ embed_tokens            (tie: 与 lm_head 同一个矩阵)
   ▼
hidden_states        [B, T, 768]
   │ × 8 × Block
   │     ├─ RMSNorm → Attention(q[B,8,T,96], k/v[B,4,T,96]) → RoPE → SDPA → o_proj
   │     └─ 残差 += ；RMSNorm → SwiGLU MLP → 残差 +=
   ▼
hidden_states        [B, T, 768]
   │ final RMSNorm → lm_head
   ▼
logits               [B, T, 6400]     未归一化打分
   │ 训练：logits[:, :-1, :]  对齐  labels[:, 1:]
   ▼
loss = cross_entropy(logits[..., :-1, :].reshape(-1, V), labels[..., 1:].reshape(-1),
                     ignore_index=-100)
```

两个必须记牢的点：

1. **错一位**：位置 `t` 的输出预测的是 `t+1`。所以 `logits` 丢掉最后一个位置、`labels` 丢掉第一个位置。
   整条流水线里所有「对齐」问题（DPO/PPO/GRPO/Agent）都是这一条的延伸。
2. **`ignore_index=-100`**：这些位置既不进分子也不进分母。SFT 里 prompt 全部是 -100，
   只有 assistant 的回复和结尾 `<eos>` 参与 loss。

---

## 6. 贯穿全仓库的 5 个约定

### 6.1 两种 mask 语义，别混

| 场景 | 手段 | 谁在用 |
|---|---|---|
| 训练算 loss | `labels = -100`，交给 `cross_entropy(ignore_index=-100)` | pretrain / SFT / 蒸馏 |
| RL 算策略梯度 | `mask ∈ {0,1}` 浮点张量，手动 `(loss * mask).sum() / mask.sum()` | DPO / PPO / GRPO / Agent |

为什么 RL 不用 -100？因为 RL 的 loss 不是交叉熵，是自己乘 mask 的，用 0/1 更直观
（也不会被 `ignore_index` 意外吞掉）。

### 6.2 chat template 是唯一的「协议」

模型本身只认 token。`user` / `assistant` / `tool` 这些角色，全靠 `tokenizer_config.json` 里的
jinja 模板渲染成 `<|im_start|>user\n...<|im_end|>\n` 这样的纯文本。
**改模板 = 改 token 分布 = 必须重训**。本仓库所有数据集都走
`tokenizer.apply_chat_template(...)`，没有一处手拼字符串，这是它能同时支持
`tools` / `open_thinking` / 多轮的原因。

### 6.3 SFT 的 loss mask 怎么定位「哪里是回答」

`dataset/lm_dataset.py::SFTDataset.generate_labels` 不按 message 拼接，而是：

1. 先把整段对话渲染成**一个字符串**，tokenize 成 `input_ids`；
2. 再在 token 流里找 `bos_id`（`<|im_start|>assistant\n` 的 token 序列）；
3. 从它后面开始，一路标到 `eos_id`（`<|im_end|>\n`），这些位置才算 loss。

**为什么这么绕？** 因为模板是黑盒（可能插 `<think>`、处理 `tool_calls`、插工具描述），
在 Python 侧重写一遍「哪段是回答」迟早会漂移；而在最终 token 流里找标记，
模板一改都不会错。

### 6.4 权重命名与目录

- 训练产物：`out/<name>_<hidden>[_moe].pth`，纯 `state_dict`，**半精度**。
- 续训产物：`checkpoints/<name>_<hidden>[_moe]_resume.pth`，含 optimizer/scaler/scheduler/epoch/step。
- `--from_weight` 的值就是文件名前缀（`pretrain` / `full_sft` / `dpo` …），**拼错不会报错**，
  因为 `load_state_dict(strict=False)` 会静默跳过——这是本仓库第一大坑。

### 6.5 loss 的口径不能跨阶段比

- pretrain 的 loss ≈ 每个 token 的平均 NLL，语料杂，绝对值高（典型区间 1.2 ~ 1.8）。
- SFT 的 loss 只算回复部分，语料干净，绝对值更低（典型区间 0.6 ~ 1.0）。
- DPO 的 loss 开局应该 ≈ `ln2 = 0.693`（policy 与 ref 同权重时 margin=0）。
- PPO/GRPO 的 `policy_loss` 是带负号的策略梯度，**可以是负数**，它的大小没有「越低越好」的绝对含义。

> 看到「SFT loss 比 pretrain 低，所以 SFT 更好」这种说法，可以直接判定为误解。

---

## 7. 建议的读码顺序

按这个顺序读，每一步都只依赖前面读过的内容：

```
① model/model_minimind.py       结构（Config → RMSNorm/RoPE → Attention → MLP → Block → ForCausalLM）
② dataset/lm_dataset.py         数据怎么变成 (input_ids, labels)，重点 SFTDataset.generate_labels
③ trainer/trainer_utils.py      公共件（lr 调度、checkpoint、init_model）
④ trainer/train_pretrain.py     最小训练循环（到这里 pretrain 就闭环了）
⑤ trainer/train_full_sft.py     只改了数据类和超参，对照 ④ 看差异
⑥ model/model_lora.py + train_lora.py   LoRA 插在哪、冻了什么
⑦ trainer/train_distillation.py 教师/学生、KL、T²
⑧ trainer/train_dpo.py         偏好对、β、-logsigmoid
⑨ trainer/rollout_engine.py    rollout 抽象（后面三个 RL 都靠它）
⑩ trainer/train_grpo.py        组内相对优势（比 PPO 简单，建议先读这个）
⑪ trainer/train_ppo.py         critic + GAE + clip（最复杂）
⑫ trainer/train_agent.py       多轮 + 工具 + 规则奖励
⑬ eval_llm.py / scripts/*      推理侧
```

> 想快速理解 RL 部分，先读 ⑩ GRPO 再读 ⑪ PPO：GRPO = PPO 去掉 critic、用组内均值当 baseline。

---

## 8. 最小可跑通路径

```bash
# 0) 环境
pip install -r requirements.txt          # torch / transformers / datasets

# 1) 分词器（仓库已带 model/tokenizer.json，一般不用重训）
python trainer/train_tokenizer.py

# 2) 预训练（1.2GB 数据，单卡 8G 显存大约几小时）
python trainer/train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl

# 3) 全参 SFT（从 pretrain 权重继续）
python trainer/train_full_sft.py --from_weight pretrain

# 4) 试试效果
python eval_llm.py --weight full_sft
```

其余阶段都是在第 3 步的产物上继续，跑不跑取决于你想验证哪一段。

---

## 9. 十个常见误解

1. **「SFT 是教模型新知识」** —— 不是。SFT 主要是**教格式和服从性**；知识在预训练里。
   SFT 数据量小（1.6GB 对话）且 lr 只有 1e-5，本质是「在已有分布上重新加权」。
2. **「loss 越低模型越好」** —— 只在同一阶段、同一数据、同一口径下成立。
3. **「DPO 是 RLHF」** —— DPO 是**离线**偏好优化，没有采样、没有 reward model、没有 on-policy。
   它是 RLHF 目标的一个闭式近似。
4. **「GRPO 不需要 baseline」** —— 需要，只不过 baseline 是**同组其他样本的平均分**。
5. **「LoRA 能加速训练」** —— 不能。省的是**优化器状态和权重体积**，前向反向照样要过全网。
6. **「RL 一定比 SFT 好」** —— RL 优化的是 reward，reward 有偏就会 reward hacking（复读、变长、套话）。
7. **「temperature 越高越有创造力」** —— 高温度是「更接近原始分布」，低温度是「更贪心」。
   创造性来自分布本身，不是采样参数。
8. **「多轮 Agent 就是把历史拼起来」** —— 拼的是 **token**，而且工具返回的 token **不能算 loss**。
   怎么精确对齐是 `train_agent.py` 里最难的一段。
9. **「MoE 一定比 Dense 好」** —— MoE 参数多但激活少，训练更不稳（需要 aux loss 防塌缩），
   小模型上收益不一定为正。
10. **「模型结构决定上限」** —— 在这个规模上，**数据质量和 loss mask 的正确性**远比结构重要。

---

## 10. 文件地图

| 路径 | 作用 |
|---|---|
| `model/model_minimind.py` | 模型结构（Config / RMSNorm / RoPE / Attention / MLP / MoE / Block / ForCausalLM / generate） |
| `model/model_lora.py` | LoRA 的插入、保存、加载、合并 |
| `dataset/lm_dataset.py` | 5 个 Dataset：Pretrain / SFT / DPO / RLAIF / AgentRL |
| `trainer/trainer_utils.py` | lr 调度、日志、checkpoint、分布式、reward model 封装、安全求值 |
| `trainer/train_pretrain.py` | ① 预训练 |
| `trainer/train_full_sft.py` | ② 全参 SFT |
| `trainer/train_lora.py` | ③ LoRA |
| `trainer/train_distillation.py` | ④ 知识蒸馏 |
| `trainer/train_dpo.py` | ⑤ DPO |
| `trainer/train_ppo.py` | ⑥ PPO |
| `trainer/train_grpo.py` | ⑥ GRPO |
| `trainer/train_agent.py` | ⑦ Agent RL |
| `trainer/rollout_engine.py` | rollout 抽象（torch / sglang 两种后端） |
| `trainer/train_tokenizer.py` | BPE 分词器训练 |
| `eval_llm.py` | CLI 推理 |
| `scripts/convert_model.py` | 权重格式转换 / LoRA 合并 |
| `scripts/serve_openai_api.py` | OpenAI 兼容 API |
| `scripts/web_demo.py` | Streamlit WebUI |
| `scripts/eval_toolcall.py` | 工具调用评测 |

---

← [返回索引](README.md) ｜ 下一篇 → [01 · 预训练与 SFT](01-预训练与SFT.md)

# 01 · 预训练与 SFT

> 这两个阶段共享同一套训练循环（`train_epoch` 几乎逐行相同），区别只在**数据**和**哪些 token 计分**。
> 把这两段吃透，后面 DPO / PPO / GRPO 的代码你会觉得「只是换了个 loss」。

---

## 1. 先有词表，才有模型

`trainer/train_tokenizer.py` 训一个 BPE 分词器，`vocab_size = 6400`。

- 词表大小直接决定 `embed_tokens` 的形状：`[6400, 768]` = 4.9M 参数，
  **占整个 64M 的 7.7%**。词表再大一点，小模型就只剩骨头了。
- 特殊 token 在训练前就要登记（`<s>` `<pad>` `</s>` `<unk>` `<|im_start|>` `<|im_end|>`
  `<think>` `</think>` `<tool_call>` `</tool_call>` …）。它们的 id 是稳定的，
  后面所有 mask 逻辑都依赖这几个 id。
- `tokenizer_config.json` 里的 `chat_template` 是一个 jinja 模板，
  **它是全仓库唯一的「角色协议」**：`user` / `assistant` / `tool` 怎么排列、`tools` 怎么注入、
  `<think>` 插不插，全由它决定。

> ⚠️ 换了分词器 = 换了 `input_ids` 的语义 = 所有权重作废。`MiniMindConfig.vocab_size` 必须同步改。

---

## 2. 预训练：让模型学会「下一个词」

### 2.1 数据

```jsonl
{"text": "……一段连续的自然语言……"}
```

`PretrainDataset` 做四件事：

```python
tokens = tokenizer(text, add_special_tokens=False,
                   max_length=max_length - 2, truncation=True).input_ids
tokens = [bos] + tokens + [eos]          # 首尾各加一个特殊 token
input_ids = tokens + [pad] * (max_length - len(tokens))   # 右侧补齐到定长
labels = input_ids.clone()
labels[input_ids == pad] = -100          # padding 不计 loss
```

两个细节：

- `max_length - 2`：给 `bos`/`eos` 留位置，否则截断后加不下。
- **为什么不把整篇文档打包（packing）？** 这个实现是「一篇一条」，短文本会浪费大量 pad。
  生产级做法是把多篇拼进一个定长序列并用 mask 隔开，能显著提升吞吐——但实现复杂，
  教学版选择了简单。

### 2.2 loss：什么都会，什么都不精

```python
res  = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss        # 非 MoE 时 aux_loss 恒为 0
loss = loss / accumulation_steps
scaler.scale(loss).backward()
```

模型内部：

```python
x, y = logits[..., :-1, :], labels[..., 1:]      # 错一位
loss = F.cross_entropy(x.view(-1, V), y.view(-1), ignore_index=-100)
```

于是模型学的就是

```
P(next_token | 之前所有 token)  ——  一个被 6400 类 softmax 参数化的条件分布
```

**它学会了语法、事实、推理的毛坯；但它不知道「用户提问 → 我该回答」这件事**。
给它一句「中国的首都是」，它会续写成「北京。美国的首都是华盛顿。……」——
因为它见过的语料就是这样连续排版的。这就是为什么需要 SFT。

### 2.3 超参与训练现象

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_seq_len` | 340 | README 建议 mini 数据用 ≈768，显存够就调大 |
| `batch_size` | 32 | 单卡 |
| `accumulation_steps` | 8 | **有效 batch = 32×8 = 256 条序列** |
| `learning_rate` | 5e-4 | 是 SFT 的 50 倍——因为从随机初始化开始 |
| `epochs` | 2 | 1.2GB 数据跑 2 遍 |

现象：

- loss 从 `ln 6400 ≈ 8.76` 附近（随机猜）快速降到 1.x。
- 生成能力：能续写、能补全短句；**不能对话**（问它问题，它会继续编问题）。
- 这个阶段最容易翻车的不是 loss，而是**数据重复 / 脏数据**：小模型会逐字背下来。

---

## 3. SFT：教模型「你是助手，该这样回答」

### 3.1 数据长什么样

```jsonl
{"conversations": [
   {"role": "user", "content": "帮我写个快排"},
   {"role": "assistant", "content": "好的，思路是……"}
]}
```

`SFTDataset` 的流程：

```
conversations
  → pre_processing_chat()      20% 概率注入一条随机 system prompt（让模型见过有/无 system 两种分布）
  → apply_chat_template()      渲染成整段纯文本（可能带 <think>、可能带 tools 描述）
  → post_processing_chat()     80% 概率删掉空的 <think></think>（防止模型学会输出空思考）
  → tokenizer()                变成 input_ids
  → generate_labels()          扫 token 找 assistant 段，标成 labels，其余 -100
```

### 3.2 loss mask 到底遮住了什么（重点）

假设渲染后是：

```
<|im_start|>system
你是 minimind，一个小巧但有用的语言模型。<|im_end|>
<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
等于 2。<|im_end|>
```

转成 token 后：

| 位置 | 内容 | label |
|---|---|---|
| 0 .. 12 | `<im_start>system …… <im_end>` | **-100** |
| 13 .. 20 | `<im_start>user 1+1=? <im_end>` | **-100** |
| 21 .. 24 | `<im_start>assistant\n`（角色头） | **-100** |
| 25 .. 30 | `等于 2。<im_end>\n`（**回复正文 + 结束符**） | 正常 id |
| 31 .. | `<pad>` | -100 |

三条规则，缺一不可：

1. **问题部分必须屏蔽**。否则模型在学「如何提问」，而不是「如何回答」——
   推理时它就会自问自答，或者把用户的下一句也写出来。
2. **角色头 `assistant\n` 必须屏蔽**。否则模型会自己生成 `<|im_start|>assistant`，
   多轮时角色就乱了。
3. **`<|im_end|>` / `<eos>` 必须参与 loss**。这是模型学会「什么时候停」的唯一途径。

> 🔍 对照 nanoGPT：预训练时 `labels = input_ids`，每个位置都算。
> SFT 只是把一部分位置改成 -100，**训练循环一行都没变**。

### 3.3 为什么用「token 扫段」而不是「按 message 拼接」

`generate_labels` 的实现：

```python
for i in range(len(input_ids)):
    if input_ids[i:i+len(bos_id)] == bos_id:      # 找到 <|im_start|>assistant\n
        start = i + len(bos_id)
        end   = start
        while end < len(input_ids) and input_ids[end:end+len(eos_id)] != eos_id:
            end += 1                              # 一路扫到 <|im_end|>\n
        labels[start : end+len(eos_id)] = input_ids[start : end+len(eos_id)]
```

优点：

- **对模板免疫**：模板明天加了 `<think>`、加了工具描述、改了分隔符，这段代码不用动。
- **天然支持多轮**：模板里每个 assistant 段都能被扫到，全部参与 loss。
- **不依赖 Python 侧的分词边界**：BPE 是上下文相关的，`"assistant\n"` 单独 tokenize
  和放在长文本里 tokenize 的结果**不一定相同**。在最终 token 流里找标记，才是唯一可靠的做法。

代价：`bos_id` / `eos_id` 必须是多 token 序列（`<|im_start|>assistant\n` 里的 `assistant`
会被 BPE 拆成几段，所以 `bos_id` 是一个长度 >1 的 id 列表），
因此只能用滑窗比较，不能 `input_ids.index(bos_id)`。这也是为什么代码里到处是
`input_ids[i:i+len(bos_id)] == bos_id` 这种写法。

### 3.4 与预训练的三个差异（可以直接对照两个文件）

| 项 | train_pretrain.py | train_full_sft.py |
|---|---|---|
| 数据类 | `PretrainDataset` | `SFTDataset` |
| `learning_rate` | 5e-4 | **1e-5**（50 倍差距，防止遗忘预训练知识） |
| `batch_size` | 32 | 16 |
| `accumulation_steps` | 8 | 1 |
| `max_seq_len` | 340 | 768（对话更长） |
| `from_weight` | none | **pretrain** |

其余 90% 的代码逐行相同。这就是「工程化配置」的意义：改数据，不改流程。

### 3.5 训练现象与排错

| 现象 | 可能原因 |
|---|---|
| loss 掉到 0.0x 并且不再动 | **mask 写错**，几乎所有位置都是 -100，或 labels 全是 pad |
| 模型不停输出、复读 | 结尾 `<eos>` 被 mask 掉了；或 `max_new_tokens` 太大而模型没学会停 |
| 模型自问自答 | 问题部分没 mask，它在续写「下一轮对话」 |
| 输出格式乱七八糟 | chat template 与训练时不一致（推理用了另一份 tokenizer_config） |
| loss 正常但答非所问 | 数据质量 / 数量不足；SFT 不能补知识，只能调整行为 |

> ⚠️ SFT 的 loss 绝对值不能和预训练比。口径不同（只算回复 + 语料干净），
> 1.0 的 SFT loss 和 1.0 的 pretrain loss 完全不是一回事。

---

## 4. 这两个阶段的本质总结

| | 预训练 | SFT |
|---|---|---|
| 学什么 | 世界的统计规律（语言、事实、推理毛坯） | 助手的「行为规范」（格式、语气、服从性） |
| 数据量 | 1.2GB 起步 | 1.6GB，通常更少也行 |
| 学习率 | 5e-4 | 1e-5 |
| 目标函数 | 全序列 CE | 只在回复上算 CE |
| 失败模式 | 数据脏 / 重复 → 背诵 | mask 错 → 学不会停、自问自答 |
| 一句话 | **会说话** | **会听话** |

---

← [00 · 总览](00-总览-从nanoGPT到MiniMind.md) ｜ 下一篇 → [02 · LoRA 与知识蒸馏](02-LoRA与知识蒸馏.md)

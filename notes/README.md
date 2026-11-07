# MiniMind 学习笔记

这套笔记配合仓库里的**全量中文注释**使用。

代码本身（`model/`、`dataset/`、`trainer/`、`scripts/`）已经被逐行注释过：
每个文件顶部有「作用 / 在流水线中的位置 / 核心思想 / 关键张量 / 阅读建议」五段式说明，
每个函数有 docstring（含形状），关键行有 `# 💡`（巧思）、`# ⚠️`（坑）、`# 🔍`（可与 nanoGPT 对照）。

笔记负责**串起来讲**：为什么要有这个阶段、公式怎么来的、代码里那几行在防什么。

---

## 阅读顺序

| # | 文件 | 你会得到 |
|---|---|---|
| 00 | [总览：从 nanoGPT 到 MiniMind](00-总览-从nanoGPT到MiniMind.md) | 全局地图、64M 参数账本、与 nanoGPT 的逐项对照、5 个贯穿全仓库的约定 |
| 01 | [预训练与 SFT](01-预训练与SFT.md) | 数据怎么变成 `(input_ids, labels)`、loss mask 到底遮住了什么 |
| 02 | [LoRA 与知识蒸馏](02-LoRA与知识蒸馏.md) | 少训 99% 参数的原理、T² 与 batchmean 的来历 |
| 03 | [对齐：DPO / PPO / GRPO](03-对齐-DPO-PPO-GRPO.md) | 三种对齐算法的完整推导与代码对照 |
| 04 | [Agent RL 与推理部署](04-AgentRL与推理部署.md) | 多轮 token 对齐（marker 技巧）、奖励设计、服务化 |
| 99 | [自查清单与常见坑](99-自查清单与常见坑.md) | 30 个问题自测；按症状查病因 |

---

## 三条使用建议

**① 先读代码，再读笔记。**
注释就写在代码旁边，先顺着注释读一遍源码（尤其 `model/model_minimind.py`、`dataset/lm_dataset.py`），
遇到「为什么」再来笔记里找整段解释。

**② 一定要自己算一遍形状。**
`[B, T, D]` → `[B, H, T, Dh]` → `[B*G, P+R]` 这些转换，看注释是「看懂」，
自己在纸上画一遍才是「会」。全仓库 80% 的 bug 都源于形状/对齐错了一位。

**③ 跑通 pretrain → SFT 就够建立直觉了。**
后面的 DPO/PPO/GRPO 可以只读代码 + 笔记，不一定非要跑（它们依赖 reward model 和大量算力）。

---

## 一句话版本

```
预训练：学会说话（next token）
SFT   ：学会听话（只在回复上算 loss）
LoRA  ：便宜地改行为（ΔW = BA）
蒸馏  ：把大模型搬小（软标签 + T²）
DPO   ：把偏好解成分类问题（−log σ(β·margin)）
PPO   ：critic + GAE + clip
GRPO  ：把 critic 换成同组平均值
Agent ：多轮 + 工具 + 规则奖励，剩余部分与 GRPO 相同
```

---

## 参考

- 原仓库：<https://github.com/jingyaogong/minimind>
- nanoGPT：<https://github.com/karpathy/nanoGPT>
- DPO 论文：*Direct Preference Optimization* (Rafailov et al., 2023)
- GRPO：*DeepSeekMath* (Shao et al., 2024)

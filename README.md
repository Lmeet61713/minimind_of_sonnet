# MiniMind 学习副本（非官方）

> **这不是 MiniMind 的官方仓库，也不是原创项目。**
>
> 代码、模型与原始文档均来自 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)，
> 由原项目作者及贡献者创作，遵循 Apache-2.0 协议。
> 本仓库只是学习过程中留下的一份**带注释的本地副本**：
> 把整条训练链路逐行读懂、写成中文注释，并整理成笔记，方便反复查阅。
>
> 如果你想了解项目原貌、下载权重或跟进更新，请以**原仓库**为准。

---

## 这个副本加了什么

| 内容 | 说明 |
|---|---|
| 全量中文注释 | 20 个 Python 文件，约 3,900 行 → 9,600 行；文件级 banner + 函数 docstring（含张量形状）+ 关键行讲解 |
| 学习笔记 | [`notes/`](notes/README.md) 共 6 篇，讲「为什么这么设计」与公式推导 |
| 代码零改动 | 仅新增注释与 docstring，可执行结构与原版逐字节一致（AST 校验） |

注释里用了三个标记：`# 💡` 作者巧思、`# ⚠️` 常见坑、`# 🔍` 可与 nanoGPT 对照的地方。

---

## 目录结构

```
minimind/
├── model/
│   ├── model_minimind.py     模型结构：RMSNorm / RoPE / GQA / SwiGLU / MoE / generate
│   ├── model_lora.py         LoRA 的插入、保存、加载、合并
│   └── tokenizer.json        BPE 分词器（vocab 6400）
├── dataset/
│   └── lm_dataset.py         5 个 Dataset：Pretrain / SFT / DPO / RLAIF / AgentRL
├── trainer/
│   ├── train_tokenizer.py    ① 训练分词器
│   ├── train_pretrain.py     ② 预训练
│   ├── train_full_sft.py     ③ 全参指令微调
│   ├── train_lora.py         ④ LoRA 微调
│   ├── train_distillation.py ⑤ 知识蒸馏
│   ├── train_dpo.py          ⑥ DPO 偏好对齐
│   ├── train_ppo.py          ⑦ PPO
│   ├── train_grpo.py         ⑧ GRPO / CISPO
│   ├── train_agent.py        ⑨ Agent RL（多轮工具调用）
│   ├── rollout_engine.py     rollout 抽象层（torch / sglang）
│   └── trainer_utils.py      公共工具：lr 调度、checkpoint、分布式、reward model
├── scripts/
│   ├── convert_model.py      权重格式转换 / LoRA 合并
│   ├── serve_openai_api.py   OpenAI 兼容 API（FastAPI）
│   ├── web_demo.py           Streamlit 对话界面
│   ├── chat_api.py           最小 API 调用示例
│   └── eval_toolcall.py      工具调用评测
├── eval_llm.py               CLI 推理入口
├── notes/                    学习笔记（本副本新增）
└── requirements.txt
```

---

## 环境

```bash
pip install -r requirements.txt
```

- Python ≥ 3.10，PyTorch 2.x（CUDA 建议 11.8 / 12.x）
- 64M 稠密模型单卡 8G 显存即可跑；MoE 版本显存需求更高
- 训练数据需自行下载后放进 `dataset/`（获取方式见原仓库的数据说明）

---

## 最小闭环：预训练 → SFT

```bash
# 1) 预训练（从随机初始化开始）
python trainer/train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl --max_seq_len 768

# 2) 全参 SFT（基于上一步的 pretrain 权重）
python trainer/train_full_sft.py --from_weight pretrain --data_path ../dataset/sft_t2t_mini.jsonl

# 3) 对话推理
python eval_llm.py --weight full_sft --load_from model
```

> `--from_weight` 取的是**权重文件前缀**（`pretrain` / `full_sft` / `dpo` …），
> 对应 `out/<前缀>_768.pth`。前缀拼错**不会报错**（`load_state_dict(strict=False)` 会静默跳过），
> 这是本仓库最容易踩的坑之一。

---

## 各阶段命令

| 阶段 | 脚本 | 关键参数 |
|---|---|---|
| 分词器 | `trainer/train_tokenizer.py` | 词表大小 6400，一般无需重训 |
| 预训练 | `trainer/train_pretrain.py` | `--data_path` `--max_seq_len` `--from_weight none` |
| 全参 SFT | `trainer/train_full_sft.py` | `--from_weight pretrain` |
| LoRA | `trainer/train_lora.py` | `--lora_rank 16` `--lora_name lora_identity` `--from_weight full_sft` |
| 蒸馏 | `trainer/train_distillation.py` | `--alpha` `--temperature` `--teacher_*` `--student_*` |
| DPO | `trainer/train_dpo.py` | `--beta 0.15` `--from_weight full_sft` |
| PPO | `trainer/train_ppo.py` | `--reward_model_path` `--rollout_engine torch\|sglang` |
| GRPO | `trainer/train_grpo.py` | `--num_generations` `--loss_type grpo\|cispo` |
| Agent RL | `trainer/train_agent.py` | `--num_generations` `--max_total_len` |

完整参数见各脚本的 `argparse`（`python trainer/xxx.py --help`）。

**训练产物路径**：

- `out/<前缀>_<hidden>[_moe].pth` — 半精度 `state_dict`，给推理用
- `checkpoints/<前缀>_<hidden>[_moe]_resume.pth` — 含优化器/调度器/step，给续训用

---

## 数据格式速查

```jsonc
// 预训练  dataset/pretrain_t2t_mini.jsonl
{"text": "一段连续的自然语言……"}

// SFT / RLAIF  dataset/sft_t2t_mini.jsonl
{"conversations": [
  {"role": "system", "content": "你是……"},      // 可选
  {"role": "user", "content": "问题"},
  {"role": "assistant", "content": "回答"}
]}

// DPO  dataset/dpo.jsonl
{"chosen":   [{"role": "user", "content": "…"}, {"role": "assistant", "content": "更好的回答"}],
 "rejected": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "较差的回答"}]}

// Agent RL  dataset/agent_rl.jsonl
{"conversations": [
  {"role": "system", "content": "……", "tools": "[{...OpenAI function schema...}]"},
  {"role": "user", "content": "北京天气怎么样"}
],
 "gt": ["28°C", "晴"]}
```

---

## 模型规格

| | Dense | MoE |
|---|---|---|
| 参数量 | ~64M | ~198M（激活约 64M） |
| hidden / layers | 768 / 8 | 768 / 8 |
| heads / kv heads | 8 / 4（GQA） | 8 / 4（GQA） |
| head_dim | 96 | 96 |
| intermediate | 2432 | 每专家 2432，4 专家 top-1 |
| vocab | 6400 | 6400 |
| 位置编码 | RoPE（`theta=1e6`，可选 YaRN 外推） | 同左 |
| 归一化 / 激活 | RMSNorm / SwiGLU | 同左 |

结构与 Qwen3 / LLaMA 一系对齐，因此可经由 `scripts/convert_model.py` 转成
transformers 格式，接入 vllm / sglang / llama.cpp 等生态。

---

## 推理与服务

```bash
# CLI 对话
python eval_llm.py --weight full_sft --load_from model

# 挂 LoRA 权重
python eval_llm.py --weight full_sft --lora_weight lora_identity

# OpenAI 兼容 API（默认 0.0.0.0:8998）
python scripts/serve_openai_api.py --weight full_sft --load_from ../model

# WebUI
streamlit run scripts/web_demo.py
```

> `serve_openai_api.py` 的 `model/tokenizer/device` 是**模块级全局变量**，只在 `__main__` 里赋值，
> 所以必须 `python scripts/serve_openai_api.py` 直接运行；用 `uvicorn module:app` 启动会 NameError。

---

## 学习路线

如果目的是**读懂整条流水线**，按这个顺序最省力：

1. [`model/model_minimind.py`](model/model_minimind.py) — 结构
2. [`dataset/lm_dataset.py`](dataset/lm_dataset.py) — 数据与 mask
3. [`trainer/train_pretrain.py`](trainer/train_pretrain.py) → [`trainer/train_full_sft.py`](trainer/train_full_sft.py) — 训练循环
4. [`trainer/train_grpo.py`](trainer/train_grpo.py) → [`trainer/train_ppo.py`](trainer/train_ppo.py) — 先易后难
5. [`trainer/train_agent.py`](trainer/train_agent.py) — 多轮与工具

配合笔记一起看：[notes/README.md](notes/README.md)

---

## 来源、许可与致谢

- **原始项目**：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- **协议**：Apache License 2.0，见 [LICENSE](LICENSE)。本副本的修改（注释与笔记）同样以 Apache-2.0 发布。
- **署名**：MiniMind 的项目名称、模型、代码与原始文档版权归原作者及贡献者所有。
  本仓库不主张任何原创性，也不代表原项目立场。
- 原 README 中提及的模型权重、数据集、在线体验、社区与赞助信息均属于原项目，
  可从上游获取；本副本未转载。

> 原始 README 已从本仓库移除，需要查阅时可从提交历史取回：
> `git show 4e8c4d0:README.md`

---

## 免责声明

本仓库仅用于个人学习与技术研究。代码按「原样」提供，不附带任何担保；
使用训练产物（模型权重）时请自行评估合规性与安全性。

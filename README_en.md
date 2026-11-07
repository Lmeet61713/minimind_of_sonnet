# MiniMind — Study Copy (unofficial)

> **This is NOT the official MiniMind repository, and it is not an original project.**
>
> All code, models and original documentation come from
> [jingyaogong/minimind](https://github.com/jingyaogong/minimind),
> created by its author and contributors under the Apache License 2.0.
> This repository is only a **locally annotated study copy**: the whole training
> pipeline was read through, commented in Chinese, and summarized into notes for
> personal reference.
>
> For the project itself, model weights, datasets or updates, please refer to the
> **upstream repository**.

---

## What this copy adds

| Item | Description |
|---|---|
| Full Chinese annotations | 20 Python files, ~3,900 → ~9,600 lines; file banner, function docstrings with tensor shapes, line-level explanations |
| Study notes | [`notes/`](notes/README.md) — 6 documents on *why* things are designed this way, with derivations |
| No code changes | Only comments/docstrings were added; the executable structure is byte-identical to upstream (verified via AST comparison) |

Annotation markers: `# 💡` design insight, `# ⚠️` common pitfall, `# 🔍` comparison with nanoGPT.

---

## Pipeline

```
train_tokenizer.py  →  train_pretrain.py  →  train_full_sft.py
                                              ├── train_lora.py          (LoRA)
                                              ├── train_distillation.py  (KD)
                                              ├── train_dpo.py           (DPO)
                                              ├── train_ppo.py / train_grpo.py
                                              └── train_agent.py         (Agent RL)
```

Quick start (see [README.md](README.md) for details, in Chinese):

```bash
pip install -r requirements.txt

python trainer/train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl
python trainer/train_full_sft.py --from_weight pretrain
python eval_llm.py --weight full_sft --load_from model
```

Model: a ~64M-parameter decoder-only Transformer
(RMSNorm + RoPE + GQA + SwiGLU, vocab 6400, hidden 768, 8 layers),
structurally aligned with the Qwen3 / LLaMA family.

---

## License & attribution

- Upstream: <https://github.com/jingyaogong/minimind>
- License: Apache License 2.0 — see [LICENSE](LICENSE).
  Modifications in this copy (annotations and notes) are released under the same license.
- The MiniMind name, models, code and original documentation remain the property of
  the original author and contributors. This repository claims no originality and
  does not represent the upstream project.

The original README (including its promotional content, badges and community links)
has been removed from this copy. It can be recovered from history:
`git show 4e8c4d0:README.md`

---

## Disclaimer

For personal study and research only. Code is provided "as is", without warranty of
any kind. Evaluate compliance and safety yourself before using any trained weights.

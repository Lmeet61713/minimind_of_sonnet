# 02 · LoRA 与知识蒸馏

> 这两件事经常被混为一谈，但它们解决的问题完全不同：
> **LoRA 解决「怎么便宜地改模型行为」，蒸馏解决「怎么把大模型的能力搬到小模型」**。
> 前者是参数高效微调（PEFT），后者是模型压缩。

---

## 1. LoRA：只训练 0.6% 的参数

### 1.1 一句话原理

冻结原权重 `W ∈ R^{d×d}`，在旁边挂一个低秩分支：

```
h = W x + (B A) x ,   A ∈ R^{r×d},  B ∈ R^{d×r},  r ≪ d
```

`r = 16` 时，一个 768×768 的矩阵被拆成 `16×768` 和 `768×16`，参数从 589,824 降到 24,576（1/24）。

### 1.2 两个初始化，一个都不能反

```python
self.A.weight.data.normal_(mean=0.0, std=0.02)   # 高斯
self.B.weight.data.zero_()                        # 全零
```

- **B = 0 ⇒ ΔW = BA = 0**：挂上去的瞬间，模型输出**逐位等于**原模型。
  「加载即无损」是 LoRA 能安全地接着 SFT 模型继续训的根本。
- **A 的梯度 ∝ Bᵀ = 0**，所以第一步 A 不动、只有 B 在学；第二步起两者一起动。
  这不是 bug，是标准做法。

> 如果两个都随机初始化，训练第一步就会把模型打出一个随机扰动，
> 小模型 + 小数据下这个扰动可能再也恢复不回来。

### 1.3 挂在哪：`apply_lora` 的一个隐蔽筛选

```python
if isinstance(module, nn.Linear) and module.in_features == module.out_features:
```

只看「方阵」。在本模型里：

| 层 | 形状 | 是否加 LoRA |
|---|---|---|
| `q_proj` | 768 → 768 | ✅ |
| `o_proj` | 768 → 768 | ✅ |
| `k_proj` | 768 → 384（GQA 4 头） | ❌ 维度不等 |
| `v_proj` | 768 → 384 | ❌ |
| `gate_proj` / `up_proj` | 768 → 2432 | ❌ |
| `down_proj` | 2432 → 768 | ❌ |
| `lm_head` | 768 → 6400 | ❌ |

所以真正被训练的是 **8 层 × 2 个模块 = 16 个 LoRA**，共
`16 × 2 × 768 × 16 = 393,216 ≈ 0.39M`，占全模型 **0.61%**。

> ⚠️ 这个「方阵」条件是作者偷懒的启发式，不是 LoRA 的数学要求。
> 标准做法会给 k/v/gate/up/down 也加上（分别用 `r×d_in` 和 `d_out×r`），
> 效果通常更好。看懂这一点，你就比「照抄脚本」的人多懂一层。

### 1.4 训练时到底省了什么

```python
for name, param in model.named_parameters():
    if 'lora' in name: param.requires_grad = True;  lora_params.append(param)
    else:              param.requires_grad = False

optimizer = optim.AdamW(lora_params, lr=1e-4)     # ← 只把 lora 参数交给优化器
clip_grad_norm_(lora_params, args.grad_clip)       # ← 只裁剪 lora 参数
save_lora(model, path)                             # ← 只存 lora 参数
```

省的是：

- **优化器状态**：AdamW 每个参数存 2 份动量。全参是 `64M × 8B ≈ 512MB`，LoRA 是 `0.39M × 8B ≈ 3MB`。
- **权重文件**：0.8MB vs 128MB（fp16）。
- **DDP 通信量**：只同步 `requires_grad=True` 的参数，几乎归零。

**不省的是**：前向和反向仍然要穿过整个网络（冻结参数只是不更新，梯度还是要算到 lora 层）。
所以「LoRA 训练更快」是误解——它省显存、省存储、省通信，不太省时间。

### 1.5 保存、加载、合并

```
训练：  full_sft 权重 + lora_xxx.pth（只有 A/B）
推理：  init_model(full_sft) → apply_lora() → load_lora()        # 动态挂载
发布：  merge_lora()：W_merged = W_base + B @ A                    # 合成一个普通权重
```

`merge_lora` 的关键是线性可加：

```
h = Wx + BAx = (W + BA)x
```

合并后**推理零额外开销**，可以当作普通 `MiniMindForCausalLM` 权重加载。

> ⚠️ 本仓库的 LoRA **没有 `alpha/r` 缩放**（标准 LoRA 是 `ΔW = (alpha/r)·BA`）。
> 所以强度只能靠 `rank` 和 `lr` 调；如果你引入 `alpha`，训练和合并两边必须一致，
> 否则合并出来的模型整体偏强/偏弱。

---

## 2. 知识蒸馏：把 MoE 的脑子塞进 Dense

### 2.1 这个仓库的蒸馏在做什么

```python
--teacher_hidden_size 768 --teacher_num_layers 8 --teacher_use_moe 1   # 教师：minimind-3-moe（198M-A64M）
--student_hidden_size 768 --student_num_layers 8 --student_use_moe 0   # 学生：minimind-3 dense（64M）
--alpha 0.5 --temperature 1.5
```

**老师是一个 MoE，学生是一个 Dense。** 这正是「用大模型的分布教小模型」的标准场景：
MoE 参数多、推理贵；Dense 便宜，但直接从数据学不到 MoE 的泛化。

### 2.2 损失函数

```
L = α · CE(y, p_student)  +  (1-α) · T² · KL(q_teacher ‖ p_student)

其中  q = softmax(z_teacher / T),  p = softmax(z_student / T)
```

三个容易看漏的细节：

**① 为什么除以 T，还要乘 T²**

KL 对 `z_student` 的梯度是 `(1/T)(p - q)`。温度一调高，蒸馏项的梯度就被缩小 1/T，
相当于**偷偷改了蒸馏项的学习率**，`alpha` 的语义就乱了。乘 `T²` 正好抵消这个 `1/T`
（净剩一个 `T` 的放大），于是「调温度」只改变软标签的形状，不改变两项的相对权重。

**② 为什么 `reduction='batchmean'`**

代码是先把 `[B, T, V]` 用 mask 压成 `[N_valid, V]` 再算的。`batchmean` = `sum / N`，
正好是「每个有效 token 的平均 KL」，量纲和 CE 可比。若用 `mean`，会再除以一个 `V = 6400`，
蒸馏项直接小三个数量级，`alpha` 就没法调了。

**③ 为什么是 KL(teacher‖student) 而不是反过来**

因为 `KL(q‖p) = CE(q,p) - H(q)`，`H(q)` 与参数无关，所以它**等价于用软标签做交叉熵**
（Hinton 的原始形式）。梯度里带 `q` 作为权重：教师认为不可能的类几乎不产生梯度，
算力集中在教师认为可能的候选上。

### 2.3 训练时的四个「冻结」

```python
teacher_model.eval()                 # 关 dropout：让软标签是确定的
teacher_model.requires_grad_(False)  # 不更新
with torch.no_grad(): ...            # 不建计算图（省显存）
# teacher 不进 optimizer
```

四道保险确保「软标签是常量」。学生侧的梯度只从自己的 `log_softmax` 那条路流回。

### 2.4 与 SFT 的差异

| 项 | SFT | 蒸馏 |
|---|---|---|
| 数据 | `sft_t2t_mini.jsonl` | 同一份 |
| 额外模型 | 无 | 一个 teacher（同结构或更大） |
| loss | CE | `α·CE + (1-α)·T²·KL` |
| lr | 1e-5 | **5e-6**（更小，分布匹配比标签匹配更敏感） |
| epochs | 2 | 6（软标签的信息量更大，可以多训） |
| 显存 | 1 份模型 | 2 份（教师 fp32 + 学生 bf16） |

> 💡 蒸馏真正值钱的地方：**软标签携带了「类间相似度」**。
> 「北京的省会是石家庄」这种错误答案，hard label 只说「错」，软标签会说
> 「错，但比『巴黎』接近得多」。这部分信息在 one-hot 里是不存在的。

---

## 3. 小结

| | LoRA | 蒸馏 |
|---|---|---|
| 目标 | 便宜地**改行为** | 便宜地**搬能力** |
| 省什么 | 优化器状态 / 权重体积 | 推理成本（大→小） |
| 关键公式 | `W + BA` | `α·CE + (1-α)·T²·KL` |
| 关键技巧 | B=0 初始化 | T² 补偿 + batchmean |
| 产出 | 0.39M 适配器 | 一个完整的 64M 模型 |

---

← [01 · 预训练与 SFT](01-预训练与SFT.md) ｜ 下一篇 → [03 · 对齐：DPO / PPO / GRPO](03-对齐-DPO-PPO-GRPO.md)

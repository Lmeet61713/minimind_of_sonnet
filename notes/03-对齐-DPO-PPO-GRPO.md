# 03 · 对齐：DPO / PPO / GRPO

> SFT 教模型「照着人类示范说话」，但它没有「哪个回答更好」的概念。
> 对齐阶段（alignment）做的就是**把「偏好」变成可优化的目标**。
> 三条路线：DPO（离线、无采样）、PPO（在线、有 critic）、GRPO（在线、无 critic）。

---

## 1. 为什么 SFT 不够

SFT 的目标函数是「最大化人类示范的似然」。它只告诉你**正确答案长什么样**，
不告诉你**错误答案有多错**。这带来三个问题：

1. 数据里出现过的错误示范也会被学会（模仿学习没有「否定」信号）。
2. 同一个问题有多种好答案，SFT 会把概率平摊，输出变得温吞。
3. 无法优化「有用 / 无害 / 诚实」这类**难以写成标准答案**的目标。

对齐的统一框架是：

```
max_θ  E_{y ~ π_θ(·|x)} [ r(x, y) ]  −  β · KL( π_θ(·|x) ‖ π_ref(·|x) )
        └── 变好 ──┘                  └── 别跑太远 ──┘
```

- `r(x,y)`：奖励（人类偏好 / AI 打分 / 规则校验）
- `π_ref`：参考模型（就是 SFT 后的权重），冻结
- `β`：KL 惩罚强度。没有它，模型会为了刷分把语言本身忘掉（reward hacking）

**三种算法的区别，只在于「怎么估这个期望、怎么算 r」。**

---

## 2. DPO：把 RL 目标解成一个分类 loss

### 2.1 推导（四步）

**① 上面那个目标的最优解有闭式形式：**

```
π*(y|x) = π_ref(y|x) · exp( r(x,y) / β ) / Z(x)
```

**② 反解出隐式奖励：**

```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

**③ 用 Bradley-Terry 建模「人类更偏好哪个」：**

```
P(y_w ≻ y_l | x) = σ( r(x,y_w) − r(x,y_l) )
```

**④ 代入 ②，`β·log Z(x)` 在相减时**约掉了**（因为两个答案共享同一个 prompt）：**

```
P(y_w ≻ y_l) = σ( β · [ log(π(y_w|x)/π_ref(y_w|x)) − log(π(y_l|x)/π_ref(y_l|x)) ] )
```

于是损失就是负对数：

```
L_DPO = − log σ( β · [ (logπ_θ(y_w) − logπ_θ(y_l)) − (logπ_ref(y_w) − logπ_ref(y_l)) ] )
```

**奖励模型消失了，采样也消失了。** 你只需要一份「chosen / rejected」的静态数据，
用两次前向（policy + ref）就能算 loss。这就是 DPO 的全部魔法。

### 2.2 代码对应（`trainer/train_dpo.py`）

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs    = (ref_log_probs    * mask).sum(dim=1)   # 整条回复的 log 概率和
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    ...
    pi_logratios  = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs    - reject_ref_log_probs
    logits = pi_logratios - ref_logratios                      # 就是方括号里那坨
    loss   = -F.logsigmoid(beta * logits)
```

三个实现细节：

- **求和而不是平均**：`(logp * mask).sum(dim=1)` 得到的是整条回复的联合对数概率。
  这让长回答的权重天然更大（有长度偏置，DPO 的标准做法）。
- **拼成一个 batch**：`x = cat([x_chosen, x_rejected])`，一次前向算完 2B 条，
  policy 和 ref 各一次，总共 2 次前向。
- **batch_size 必须是偶数**，且 `cat` 顺序不能乱：前 B 条是 chosen、后 B 条是 rejected，
  `dpo_loss` 里靠 `[:B//2]` / `[B//2:]` 切开。顺序错了，`Z(x)` 就约不掉。

### 2.3 超参的来历

| 参数 | 默认 | 为什么 |
|---|---|---|
| `learning_rate` | **4e-8** | Adam 的每步更新量级 ≈ lr，几千步累计位移 ~1e-4，属于「微调」；再大就灾难性遗忘 |
| `beta` | 0.15 | 双重身份：KL 约束强度 + sigmoid 斜率（越大越保守，且已分对的样本梯度消失越快） |
| `epochs` | 1 | 离线偏好数据训多轮必然 reward hacking |
| `batch_size` | 4 | 2B = 8 条序列 × 1024（`max_seq_len`），显存友好 |

### 2.4 开局 loss 应该 ≈ 0.693

policy 和 ref 一开始**同权重** ⇒ margin = 0 ⇒ `−log σ(0) = −log 0.5 = ln2 ≈ 0.693`。
如果你看到开局 loss 是 0.3 或 1.2，说明：权重没对齐、mask 错了、或者 batch 顺序乱了。
**这是最好的自检点。**

---

## 3. PPO：最经典、也最复杂

### 3.1 四个模型

```
actor   π_θ     要训练的「做事的人」
critic  V_φ     学出来的「裁判」，估计「这个状态值多少分」
ref     π_ref   冻结的 SFT 模型，用来算 KL
reward  r(x,y)  外部打分器（这里是 internlm2-1.8b-reward）
```

为什么需要 critic：因为 reward 只在**整条回复结束时**给一个标量（稀疏奖励），
而策略梯度需要**每个 token 的优势**。critic 提供 `V(s_t)` 作为 baseline，
把「整条序列的分数」摊到每个 token 上（信用分配）。

### 3.2 GAE：把稀疏奖励摊到每个 token

```python
token_rewards[最后一个有效 token] += reward    # 外部奖励只在结尾注入
for t in reversed(range(R)):
    delta       = r_t + γ·V(t+1) − V(t)        # TD 误差
    lastgaelam  = delta + γ·λ·lastgaelam       # 指数加权累积
    A_t         = lastgaelam
returns = A + V                                 # 价值网络的回归目标
```

- `γ = 1.0`：本任务没有「折扣」需求（回复长度有限）。
- `λ = 0.95`：`λ=0` 只看一步（方差小偏差大），`λ=1` 用完整回报（偏差小方差大）。
- `returns = A + V` 是恒等式：`A = R − V`，所以 `R = A + V`。

### 3.3 PPO 的裁剪目标

```python
ratio = exp(logπ_θ − logπ_old)                  # 重要性采样比

L = − min( ratio·A,  clip(ratio, 1−ε, 1+ε)·A )  # ε = 0.2
```

含义：

- `A > 0`（这个 token 比平均好）：鼓励提高它的概率，但最多提到 `1+ε` 倍就不给梯度了。
- `A < 0`：压低它的概率，同理最多压到 `1−ε`。
- 取 `min` 是**悲观界**：只在「确实变好」的方向上更新。

配套还有：

- **value clip**：`max(MSE, clipped MSE)`，防止 critic 单步跳太远。
- **KL 惩罚** `β·(exp(d) − d − 1)`，`d = logπ_ref − logπ_θ`。这个估计恒非负，
  且方差比直接用 `d` 小得多。
- **approx_kl 早停**：`0.5·mean(d²) > 0.25` 就停止这批数据的后续更新
  （因为它已经离 `π_old` 太远，重要性采样失效）。

### 3.4 一个 DDP 的坑（代码里专门处理了）

```python
if approx_kl_val > args.early_stop_kl:
    stop_ppo = True
...
if stop_ppo:
    loss = (policy_loss + vf_coef*value_loss + aux_loss) * 0.0
loss.backward()
```

早停不能直接 `break`：有的卡停了、有的卡还在跑，**DDP 的 allreduce 会死锁**。
正确做法是把 loss 乘 0、照样 backward，只是不产生梯度。同时 `approx_kl` 要先
`all_reduce(AVG)`，保证所有 rank 同时做决定。

---

## 4. GRPO：把 critic 换成一组成员

### 4.1 核心一句话

> **同一个问题，采样 G 个回答，用「这组回答的平均分」当 baseline。**

```python
grouped_rewards = rewards.view(-1, num_generations)          # [B, G]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(G)    # [B*G]
std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(G)
advantages = (rewards - mean_r) / (std_r + 1e-4)             # 组内标准化
```

好处：

- **省掉一个和 policy 一样大的 critic**（PPO 的四模型变三模型），显存和稳定性都改善。
- 不需要训练价值函数，就没有「critic 学偏 → 策略被带歪」这条故障路径。
- 天然适配「同一 prompt 可重复采样、结果可判定」的任务（数学、代码、工具调用）。

代价：

- 每条 prompt 要采样 G 次（本仓库 G=6 / Agent 里 G=4），采样成本上升。
- 如果一组里所有样本分数相同（全对或全错），`std ≈ 0` ⇒ `A ≈ 0` ⇒ **这一组没有学习信号**。
  这是 GRPO 的固有浪费，也是为什么它依赖「难度适中的题目」。

### 4.2 与 PPO 的代码级对照

| | PPO | GRPO |
|---|---|---|
| 模型 | actor + critic + ref + reward | actor + ref + reward |
| 优势 | GAE（逐 token） | 组内标准化（整条序列共享） |
| 采样 | 每 prompt 1 条 | 每 prompt G 条 |
| 更新 | `ppo_update_iters` × minibatch 多轮 | 一个 batch 更新一次 |
| loss 形式 | `−min(rA, clip(r,1±ε)A)` | 默认 `cispo`（只截上界 + `detach`） |

`train_grpo.py` 里还给了 `cispo` 这个变体：

```python
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()   # 只截上界，且 detach
    per_token_loss = -(clamped_ratio * advantages * per_token_logps - beta * per_token_kl)
```

- 标准 GRPO：`min(rA, clip(r)A)`，**两侧都截**，用 `min` 取悲观界。
- CISPO：只截上界 `C=5.0`，且截断后的 `ratio` 被 `detach()` 当作**权重**乘到 `logπ` 上，
  即 `∇ = −w·A·∇logπ`。截断只削梯度幅度、不改方向，比 `min/max` 更平滑。

### 4.3 KL 估计：为什么用 `exp(d) − d − 1`

```python
kl_div     = ref_logp - policy_logp          # d = log π_ref − log π_θ
per_token_kl = torch.exp(kl_div) - kl_div - 1
```

因为 `E_{π_θ}[exp(d)] = Σ π_θ · (π_ref/π_θ) = Σ π_ref = 1`，所以

```
E[exp(d) − d − 1] = 1 − E[d] − 1 = −E[d] = KL(π_θ ‖ π_ref)
```

它是 KL 的**单样本无偏估计**，且 `e^x ≥ x+1` 保证逐样本非负（方差远小于直接取 `−d`）。

> 日志里监控的 `KL_ref` 反而用的是 `d` 的均值（便宜、符号直观）。
> **loss 用低方差估计、监控用一阶估计**，是刻意的分工。

---

## 5. 三者对比与选型

| | DPO | PPO | GRPO |
|---|---|---|---|
| 数据 | 静态偏好对 | 在线采样 | 在线采样 |
| 需要 reward model | 否 | 是 | 是（或规则） |
| 需要 critic | 否 | **是** | 否 |
| 成本 | 最低（2 次前向） | 最高（4 模型） | 中（G 倍采样） |
| 稳定性 | 高 | 低（超参敏感） | 中 |
| 适合 | 有高质量偏好数据 | 有可靠 RM、追求极限 | 可验证任务（数学/代码/工具） |

**实践建议**：先 DPO（便宜、稳），再考虑 GRPO（需要「可判定」的 reward），
PPO 主要作为理解 baseline 的教学样本。

---

## 6. 常见坑清单

1. **`old_per_token_logps` 的口径**：它必须是「采样那份策略、在 T=1 的干净前向」下算出来的。
   如果拿 `generate` 里被 `temperature/top_p` 改过的 scores，`ratio` 一开始就不等于 1，
   整个重要性采样就错了。
2. **对齐错一位**：`logp_pos = prompt_len − 1 + t`。位置 `t` 的 logits 预测的是 `t+1` 的 token。
   这一个 `−1` 错了，loss 不会报错，只会学不动。
3. **mask 不匹配**：`completion_mask` 要截到**第一个 eos**。否则模型会被鼓励「不要停」，
   或者长度归一化被 padding 稀释。
4. **KL 方向**：`d = log π_ref − log π_θ` 还是反过来？PPO/GRPO 里
   `exp(d) − d − 1` 用的是 `π_ref − π_θ`，符号错了会变成「鼓励跑远」。
5. **reward 尺度**：规则分（长度、格式）的量级必须**远小于**主奖励（GT/RM），
   否则模型去刷格式分而不是答题。本仓库把所有 shaping 项压在 ±1 内、主奖励 +2.5、总分 clip 到 ±3。
6. **DPO 的 `beta` 太小** ⇒ 模型跑飞；**太大** ⇒ 学不动。0.1~0.2 是安全区。
7. **别用训练 loss 判断 RL 好坏**。看 `reward`、`response_len`、`kl_ref`、`clipfrac`。

---

← [02 · LoRA 与知识蒸馏](02-LoRA与知识蒸馏.md) ｜ 下一篇 → [04 · Agent RL](04-AgentRL与推理部署.md)

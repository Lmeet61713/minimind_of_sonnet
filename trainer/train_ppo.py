# ===========================================================================================
# train_ppo.py —— MiniMind 的 PPO（Proximal Policy Optimization）偏好对齐训练脚本
# ===========================================================================================
# 作用：用 PPO 把 SFT 之后的模型（policy / actor）往「reward 更高」的方向再推一步，
#       同时用 KL 惩罚把它拴在 ref_model 周围，避免训崩、避免 reward hacking。
#
# 在流水线中的位置：
#   上游 ← out/full_sft_768.pth，同时作为 actor 与 ref 的初始权重（两者必须同源，否则 KL 起点不是 0）
#          ← dataset/rlaif.jsonl（只有 prompt，没有 chosen/rejected 标签，所以不能做 DPO）
#          ← 外部 reward model（默认 internlm2-1.8b-reward），给 (prompt, response) 打一个分数
#   下游 → out/ppo_actor_768.pth（可直接用 eval_llm.py 推理，也可作为 DPO / GRPO 的起点）
#          → ../checkpoints/ 里带 optimizer / scheduler / critic 的续训存档
#
# 核心思想：
#   1. RLHF 的 PPO 是「四个模型同台」：actor（要训）、critic（要训，估计 V(s)）、
#      ref（冻结，KL 的锚点）、reward（冻结，只打分）。critic 的价值在于把「只有最后一步
#      才有 reward」的稀疏信号，用 GAE 摊薄成每个 token 的 advantage，从而降低方差。
#   2. on-policy：数据必须由「当前策略」采样得来。所以每一步先 rollout 一次，
#      再用同一批数据做 ppo_update_iters 轮 minibatch 更新（这就是 PPO 的数据复用）。
#   3. 信任域：ratio = π_new / π_old 被 clip 在 [1-ε, 1+ε]；同时另加一项对 ref 的 KL 惩罚，
#      双保险防止策略一步迈太远、跑到 reward model 的盲区里去。
#
# 关键张量/数据（B=batch，P=prompt 序列长度，R=生成长度，D=hidden_size=768，V=vocab_size=6400）：
#   enc.input_ids         [B, P]      prompt 的 token id（左 padding）
#   gen_out (output_ids)  [B, P+R]    prompt 与 response 拼好的整条序列
#   completion_ids        [B, R]      只含生成段（右端用 pad 补齐）
#   logits                [B, P+R, V] 每个位置对 V 个 token 的未归一化打分
#   labels = gen_out[:,1:] [B, P+R-1] 右移一位的真实「下一个 token」
#   logp_pos              [B, R]      「第 t 个生成 token」对应的 logits 下标 = prompt_len-1+t
#   per_token_logps       [B, R]      生成段每个 token 的对数概率（old / ref / new 三份）
#   values_seq            [B, P+R]    critic 对每个位置的状态价值 V 估计
#   old_resp_values       [B, R]      生成段每个 token 的 V(s_t)
#   advantages / returns  [B, R]      GAE 算出的优势、以及 critic 的回归目标
#   full_mask             [B, P+R]    attention_mask：1=可见，0=左 padding 或 eos 之后的补位
#   resp_policy_mask      [B, R]      completion / loss mask：只对「真正生成到 eos 为止」的 token 置 1
#
# 阅读建议：
#   1) 先读下面那段「本文件与 train_grpo.py 共用的 RL 循环骨架」，建立全局图景；
#   2) 再看 CriticModel（只有十几行，但 value_head 与二次 norm 两处最值得琢磨）；
#   3) 然后看 ppo_train_epoch 里「对齐与切窗」的那六行（full_mask → logp_pos → log_ratio），
#      这是全文件最容易读错的地方，看懂 next-token 错一位就通了；
#   4) 接着按 GAE 段 → 内层 minibatch 段 → 梯度累积收尾段的顺序读；
#   5) __main__ 里 argparse 一大段可以扫一眼就过，重点看「5. 初始化模型和数据」与「7. 编译和分布式包装」。
#   可以跳过：debug 打印、wandb 记录、ckpt 保存细节、sglang 相关参数（默认 torch 引擎）。
# ===========================================================================================
import os
import sys

# ⚠️ 这个 magic 赋值让本文件即使被当作脚本直接运行（__name__ == "__main__"），
#    里面的相对导入也能按包 `trainer.xxx` 解析；下面的 sys.path 追加则保证
#    model / dataset / trainer 这些顶层包能被 import 到（脚本从 trainer/ 目录里跑时要回到仓库根）。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ⚠️ 这行 import 必须排在 torch 之前：Windows 上 pyarrow 与 torch 的 DLL 加载顺序会互相踩，
#    先 import datasets(pyarrow) 再 import torch 才能避开（issue #771）。不要重排 import。
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


# ===========================================================================================
# 【本文件与 train_grpo.py 共用的 RL 循环骨架】—— 两个文件读之前都请先看这一块
# ===========================================================================================
# 一个训练 step 永远是这五拍：
#   第 0 拍 rollout 采样：把 prompt 喂给 rollout_engine，用「当前策略」采样出 response。
#                         同时把采样时的 per_token_logps 存成 old_*（它就是 importance
#                         sampling 的分母，也是「行为策略」的证据）。
#   第 1 拍 打分 reward  ：规则奖励（长度 / 格式 / 重复）+ reward model 分数，合成一个标量序列。
#   第 2 拍 算 advantage ：PPO 用 critic 估 V(s) 再走 GAE；GRPO 用「同一 prompt 的一组样本
#                         互相比较」得到组内相对优势（不需要 critic）。
#   第 3 拍 更新策略     ：PPO 目标 = clipped surrogate + KL_ref 惩罚（+ value loss）；
#                         GRPO 目标 = clipped surrogate + KL_ref 惩罚（没有 value loss）。
#   第 4 拍 同步 rollout 引擎：update_policy(actor)，把新权重推给采样端，保证下一拍仍近似 on-policy。
#
# 【PPO 与 GRPO 的对照表】
#   ┌────────────┬──────────────────────────────────┬────────────────────────────────────┐
#   │ 维度       │ train_ppo.py                     │ train_grpo.py                      │
#   ├────────────┼──────────────────────────────────┼────────────────────────────────────┤
#   │ 模型数量   │ 4 个：actor/critic/ref/reward    │ 3 个：actor/ref/reward（无 critic）│
#   │ 优势估计   │ critic 估 V(s) + GAE（gamma、lam）│ 组内标准化：(r-mean)/std           │
#   │ 采样条数   │ num_generations = 1              │ num_generations = 6（一组）        │
#   │ 信用分配   │ 逐 token（GAE 反传终局 reward）  │ 整条序列共享一个 advantage         │
#   │ 数据复用   │ ppo_update_iters 轮 minibatch    │ 一个 rollout batch 只更新一次      │
#   │ 额外显存   │ critic 一套前向 + 优化器         │ 省掉 critic，显存友好              │
#   │ 擅长什么   │ 密集/可学习价值信号              │ 结果可验证（数学、格式、Agent 工具）│
#   └────────────┴──────────────────────────────────┴────────────────────────────────────┘
#
# 💡 一句话记住区别：PPO 靠「一个学出来的裁判（critic）告诉我这一步值多少」，
#    GRPO 靠「同题多答，比同伴好就算好」——后者天然免掉了价值网络难训、难对齐尺度的问题。
# 🔍 和 nanoGPT 对照：nanoGPT 的 loss = 交叉熵(下一个 token)，梯度只来自「文本本身」；
#    这里的 loss 仍然是交叉熵的变体（per_token_logps 就是对数概率），但每个 token 的梯度
#    被乘上了一个 advantage 权重——从「模仿数据」变成「按回报加权地模仿自己」。
# ===========================================================================================


def rep_penalty(text, n=3, cap=0.5):
    """n-gram 重复惩罚：给「复读机式」输出扣分。

    作用：统计 text 里 n-gram 的重复比例，返回一个 [0, cap] 区间的惩罚值。
    参数：
        text : str，待评估的文本（PPO 里传的是去掉 think 块之后的 answer）。
        n    : int，n-gram 的阶数，默认 3（trigram）。
        cap  : float，惩罚上限，默认 0.5，防止单个退化样本把整批 reward 拉爆。
    返回：
        float，范围 [0, cap]；文本太短（凑不出一个 n-gram）时返回 0.0。

    设计理由：
        1. 为什么用 trigram：模型退化时最典型的表现就是「短语级」复读（同一句话翻来覆去），
           3-gram 恰好能抓住短语级重复，又不会被正常的中文措辞误伤。
        2. 为什么是「重复比例」而不是「最大重复次数」：比例天然归一化到 [0,1)，
           长回复和短回复可比，不会系统性惩罚长文本。
        3. 为什么要 cap：reward model 的分数本身在 [-3, 3]，规则项必须量级可控，
           否则一条胡言乱语的长回复会因为重复惩罚过大而让整批 advantage 失真。
    """
    # 粗分词：\w+ 抓连续的字母/数字/下划线（中文在这里按字切），[^\w\s] 抓单个标点符号。
    # 💡 不用 tokenizer 而用正则：这里只要「近似词」的概念，跑在 CPU 上、零显存、零依赖，
    #    而且对 reward 这种粗糙信号来说，字符级 n-gram 已经足够灵敏。
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    # 滑窗切出所有 n-gram，用 tuple 做 key 才能进 set 去重。
    # 长度关系：len(grams) = max(0, len(toks) - n + 1)。
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    # 重复比例 = (总 n-gram 数 - 去重后 n-gram 数) / 总 n-gram 数，再乘 cap*2 放大后截断到 cap。
    # 也就是说：当重复率达到 50% 时就已经吃满 cap，再高也不再加罚（饱和设计，抗离群值）。
    # ⚠️ grams 为空时 len(grams) 会是 0，除法会 ZeroDivisionError，所以用 if grams 兜底。
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# 自定义的Critic模型，继承自MiniMindLM
# -------------------------------------------------------------------------------------------
# 为什么 critic 要「继承 MiniMindForCausalLM 再把 lm_head 换成 value_head」？
#   1) critic 的输入和 actor 完全一样（同一批 token、同一个 tokenizer、同样的 RoPE），
#      需要的特征提取器也完全一样，继承可以白拿 embed_tokens + 8 层 Transformer + RoPE buffer。
#   2) 更关键的是：可以直接 load_state_dict(SFT 权重, strict=False) 初始化。一个「读过语料」的
#      底座 + 一个随机初始化的线性头，比全随机初始化的 critic 收敛快得多；PPO 初期 V(s) 不至于
#      全是噪声，advantage 才有意义。
#   3) 只换最后一层：从「V 维词表打分」换成「1 维标量打分」，其余结构一个参数都不动。
# 🔍 对照 nanoGPT：nanoGPT 的 head 是 lm_head: D → V；这里把同一位置换成 value_head: D → 1，
#    其余 8 层 block 原封不动——这也是 actor 和 critic 能共享同一个底座的根本原因。
# -------------------------------------------------------------------------------------------
class CriticModel(MiniMindForCausalLM):
    """价值网络：输入整条序列，输出每个位置的标量状态价值 V(s_t)。

    作用：给 PPO 提供 V(s)，用于算 GAE 的 delta 与 returns。
    参数：
        params : MiniMindConfig（注意变量名叫 params，实际传的是 config 对象）。
    输出形状：
        input_ids [B, T] → values [B, T]（每个位置一个标量，T = P+R）。
    设计理由：
        critic 不预测下一个 token，只预测「在这个位置往后，最终能拿到多少 reward」，
        所以输出维度是 1 而不是 V=6400。这个头的参数量只有 D*1+1 = 769，几乎不占显存。
    """

    def __init__(self, params):
        """构造 critic：先按 MiniMind 的完整结构初始化，再把输出头换成 D → 1 的线性层。

        参数：
            params : MiniMindConfig，必须和 actor 用同一份配置（层数、hidden 都要一致，
                     否则 load_state_dict 对不上，底座初始化就白做了）。
        """
        super().__init__(params)
        # lm_head 不参与 forward，仅靠 tie_word_embeddings 与 embed_tokens 共享权重，解绑后 DDP 会报未使用参数
        # 💡 展开说：默认 tie_word_embeddings=True，所以 lm_head.weight 和 embed_tokens.weight
        #    是同一个 Parameter 对象。虽然 self.lm_head 不再被调用，但这个参数仍然通过 embedding
        #    参与了 forward、拿得到梯度，DDP 的「所有参数都必须参与归约」检查才过得去。
        # ⚠️ 如果哪天把 tie_word_embeddings 改成 False，lm_head 就成了纯粹的「死参数」：
        #    DDP 会在 backward 时报 Expected to have finished reduction in the prior iteration，
        #    症状是训练卡死或直接抛错。届时应该 del self.lm_head 或设 requires_grad=False。
        # nn.Linear 默认带 bias，所以映射是 v = w·h + b；h 是 [B, T, D]，输出 [B, T, 1]。
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """前向：底座编码 → 取最后 hidden → value_head 压成标量。

        参数：
            input_ids      : [B, T] LongTensor，整条序列（prompt + response）。
            attention_mask : [B, T]，1=可见，0=padding（模型内部会变成 -1e9 的加性 mask）。
            **kwargs       : 透传给 MiniMindModel（如 past_key_values / use_cache）。
        返回：
            values : [B, T] float，每个位置的状态价值估计。
        形状链路：
            [B, T] → embedding [B, T, D] → 8 层 block [B, T, D] → norm [B, T, D]
                   → value_head [B, T, 1] → squeeze(-1) [B, T]
        """
        # 使用基础模型获取隐藏状态
        # 注意 self.model(...) 返回的是元组 (hidden_states, presents, aux_loss)，
        # 不是 HuggingFace 那种带 .logits 的输出对象，所以下面用 outputs[0] 取 hidden_states。
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 为什么这里又做一次 self.model.norm？
        #   ① 意图：保证进入 value_head 的特征就是「最终归一化后」的表示，与 lm_head 的输入同源。
        #   ② 事实：MiniMindModel.forward 末尾已经 self.norm 过一次（见 model_minimind.py），
        #      所以这里是「二次 RMSNorm」。RMSNorm 是 y = gamma * x / rms(x)，做两次不会丢信息，
        #      只是把向量先按 gamma 逐元素缩放、再按新尺度归一化一次，本质上是换了个可学习的缩放，
        #      value_head 自己完全能吸收这个尺度差异，所以数值上无害、也训得动。
        # ③ ⚠️ 但要意识到它是冗余的：自己复现时去掉这一行同样能训；保留它的好处是「语义清晰」——
        #      任何接在底座之后的头，输入都应该是 norm 之后的 hidden。
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        # [B, T, D] → [B, T, 1] → squeeze 掉最后一维 → [B, T]。
        # 后面会用 gather(1, logp_pos) 从这里取出「生成段」的 R 个位置。
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model):
    """把「规则奖励」与「reward model 分数」相加，得到每条 response 的标量回报。

    作用：PPO / GRPO 的 reward 函数（这里是 hybrid：可验证规则 + 学出来的偏好模型）。
    参数：
        prompts      : list[str]，长度 B，带 chat template 标记的 prompt 原文。
        responses    : list[str]，长度 B，rollout 采样出来的文本。
        reward_model : LMForRewardModel 实例（内部已 eval + 冻结），提供 get_score(messages, answer)。
    返回：
        rewards : [B] float Tensor，范围大致 [-4.25, 3.75]。
    设计理由：
        1. 为什么规则项和模型项要相加：reward model 很容易被「长度」「格式」这类表面特征 hack，
           用可验证的规则把明显的作弊行为（过长、过短、格式崩坏、复读）直接扣分，
           相当于给 reward model 加了一道廉价的护栏。
        2. 为什么用字符数而不是 token 数：中文的 token/字符比不固定，字符数更接近人的直觉。
        3. 为什么要单独给 thinking 段落打分、并且只把 answer 交给 reward model：
           reward model 是在「答案」上训练的，把思维链一起喂进去会干扰它的判断；
           而 thinking 的长度和格式本身是可以用规则检查的（这正是 R1 类模型的训练套路）。
    """
    # 先开一个 [B] 的零张量，后面用 += 逐项累加各规则项，最后再加上 reward model 的分数。
    rewards = torch.zeros(len(responses), device=args.device)

    # reward model 是冻结的，不需要任何梯度；no_grad 同时省掉激活显存（它的参数量比 actor 大得多）。
    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 从 chat template 里把角色和内容抠出来，还原成 messages 列表。
            # 为什么不用 tokenizer 解析：这里只有字符串、也不关心 token，正则更轻；
            # DOTALL 让 . 能跨越换行（回复里必然有换行）。
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            # answer 默认就是整条 response；若存在 think 块，后面会被替换成 think 之后的部分。
            answer = response
            # 规则项 1：长度。太短（<20 字符）多半是空回复/敷衍，太长（>800）多半是复读或跑题。
            # 💡 用「区间」而不是「越长越好」，是为了对冲 reward model 普遍存在的 verbosity bias。
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            # 若模型输出了思维链（以 </think> 为结束标记）
            if '</think>' in response:
                # 只切第一刀：split(..., 1) 保证后面即使还有 </think> 也整段归到 answer 里（后面会因格式错被罚）。
                thinking_content, answer_content = response.split('</think>', 1)
                # 规则项 2：thinking 长度奖励。太短说明没真思考，太长说明在凑字数（都会拖慢推理）。
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                # 规则项 3：格式奖励。恰好一个 </think> 才 +0.25；多出来的说明格式已经崩了。
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                # 交给 reward model 的只有「最终答案」，不含思维链。
                answer = answer_content.strip()
            # 规则项 4：复读惩罚（见 rep_penalty 的注释）。注意是 -=，所以是纯扣分项。
            rewards[i] -= rep_penalty(answer)

            # 模型项：reward model 打分（内部会 clamp 到 [-3, 3]，见 trainer_utils.LMForRewardModel）。
            # 注意它拿到的是还原出来的 messages（对话历史）与 answer，输出一个 python float。
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        # python list[float] → [B] Tensor，一次性搬到 GPU，避免逐条 .item() 造成的频繁同步。
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        # 规则项 + 模型项：这就是这一批样本的终局回报（sequence-level reward）。
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None):
    """跑一个 epoch 的 PPO 训练（本文件的核心，约 200 行）。

    参数：
        epoch           : int，当前 epoch 下标（从 0 开始，仅用于日志打印 epoch+1）。
        loader          : DataLoader，产出 batch["prompt"]（list[str]，长度 B）。
        iters           : int，本 epoch 的总 step 数（用于判断「是不是最后一步」以触发保存）。
        rollout_engine  : RolloutEngine，负责采样，并提供 update_policy 同步权重。
        ref_model       : 冻结的参考模型（SFT 权重），只做前向算 ref_logp。
        actor_scheduler : actor 的余弦退火调度器（每个 optimizer.step 走一次）。
        critic_scheduler: critic 的调度器。
        reward_model    : LMForRewardModel，只读打分。
        start_step      : int，续训时已完成的 step 数；DataLoader 已经用 SkipBatchSampler 跳过。
        wandb           : swanlab/wandb 句柄或 None。
    返回：
        None（所有副作用都发生在全局的 actor_model / critic_model / optimizer 上）。
    关于全局变量：
        为了少传参数，本函数直接引用 __main__ 里创建的 actor_model、critic_model、
        actor_optimizer、critic_optimizer、tokenizer、lm_config、autocast_ctx、args。
        读代码时心里要记着：这些名字不是本函数的局部变量。

    一个 step 的五个阶段（与文件头注释的「五拍」一一对应）：
        A. 采样与打分：tokenize prompt → rollout → 规则奖励 + reward model 分数
        B. 对齐与切窗：构造 full_mask / logp_pos / 各类 mask
        C. 常量与优势：no_grad 下算 old_values / ref_logp，跑 GAE 得 advantage
        D. 策略与价值更新：ppo_update_iters 轮 minibatch，更新 actor 与 critic
        E. 收尾：记录日志 / 保存 / 同步 rollout 引擎 / 释放显存
    """
    # actor 与 critic 都切到 train 模式（影响 dropout 与 MoE 的 aux_loss）。
    # ⚠️ 如果忘了这一步，dropout 关闭会让 on-policy 的 logp 与采样端不一致，ratio 系统性偏离 1。
    actor_model.train()
    critic_model.train()
    # 梯度累积用的「绝对计数」：不随 epoch/step 归零，保证无论 minibatch 怎么切分都按固定节奏 step。
    grad_accum_step = 0

    # start_step + 1 只是为了让续训时日志里的 step 编号接着上次往下数。
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        # RLAIFDataset 只吐字符串（prompt 里已经是完整 chat template + generation prompt），
        # 所以这里要现 tokenize。padding_side="left" 是关键：
        #   - 左 padding 保证「每行最后一个 token 一定是 prompt 的结尾」，生成才从正确位置开始；
        #   - 右 padding 会让 generate 从 pad 之后开始，位置全乱。
        # P 是本 batch 内最长的 prompt 长度（被 max_seq_len 截断）。
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]

        # 第 0 拍：rollout 采样。引擎是抽象层——torch 引擎直接调 model.generate，
        # sglang 引擎走 HTTP 请求到独立部署的推理服务。上层代码完全无感。
        # num_generations=1：PPO 有 critic 估 advantage，不需要「一题多答」；
        # temperature=0.8：保留探索（太高变胡话，太低等于贪心、没有样本多样性）。
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids  # [B, P+R]，prompt 与 response 拼好的整条序列
        completion_ids = rollout_result.completion_ids  # [B, R]，只含生成段（右端补 pad）
        # prompt_lens：每行 prompt 的宽度。torch 引擎里因为左 padding，所有行都等于 P；
        # sglang 引擎里则是「去掉 padding 后的真实长度」，逐行可以不同。
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # [B]
        responses_text = rollout_result.completions  # list[str]，解码后的文本，只用来打分
        # ⭐ old_resp_logp 是全流程的「行为策略证据」：采样那一刻策略对每个生成 token 的 logp。
        # 它就是 importance sampling 的分母，也是 ratio=1 的基准线。[B, R]，与 completion_ids 逐位对齐。
        old_resp_logp = rollout_result.per_token_logps.to(args.device)  # [B, R]
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            # 纯人工可读性调试：把 prompt / response / reward 打出来，看 reward 是不是在奖励你想要的东西。
            # 💡 调 RL 时最先怀疑的永远是 reward 而不是网络，这个开关能省掉很多猜。
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        # ----------------------------------------------------------------------------------
        # 第 B 阶段：把 [B, P+R] 的整条序列「对齐」到 [B, R] 的生成段上。
        # 这一段是全文件最容易读错的地方，核心只有一句话：
        #   logits 在位置 j 预测的是 token j+1，所以要往后错一位；而第 t 个生成 token 位于
        #   prompt_len+t，因此「预测它的 logits」位于 prompt_len+t-1。
        # ----------------------------------------------------------------------------------
        # attention_mask 的语义：1=可见，0=不可见（模型内部会算 scores += (1-mask)*-1e9，见 MiniMindAttention）。
        # 这里先用「非 pad」临时构造 [B, P+R]，随后用 scatter 精确修正生成段。
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        # labels 是「右移一位」的真实下一个 token：[B, P+R] → [B, P+R-1]。
        # 这样 labels[j] 与 logits[j] 天然对齐，后面 gather 时不用再切片。
        # clone() 只是防御性写法（gather 本身不改原张量），也让这块内存独立出来便于 del。
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        B = len(prompts)
        # resp_labels 只是 completion_ids 的别名：[B, R]，含 eos 及其后的 padding。
        resp_labels = completion_ids
        # 列索引 0..R-1，形状 [1, R]，靠广播铺到 [B, R]。
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        # ⭐⭐⭐ 全文件最关键的一行：把「第 t 个生成 token」映射回「预测它的 logits 下标」。
        #   [B, 1] - 1 + [1, R]  →  [B, R]
        #   第 t 个生成 token 的物理位置      = prompt_len + t
        #   预测它的 logits 的位置（错一位）   = prompt_len + t - 1 = logp_pos
        # ⚠️ 少了这个 -1 是最常见的实现 bug：整段 logp 会整体错位一格，训练看起来「能跑」但学不到东西。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        # completion_mask [B, R]：1=真正生成出来的 token（含 eos），0=右侧补位。
        # torch 引擎返回全 1（generate 的产物都是有效 token）；sglang 引擎按每条实际长度给 0/1。
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # ⭐ 为什么是 logp_pos + 1：logp_pos 是「预测位置」，+1 才是「token 自己的位置」。
        # scatter_ 把 completion_mask 写到 full_mask 的第 prompt_len+t 列：
        #   → 真实生成到的 token 在 attention 里可见；
        #   → eos 之后被引擎补出来的 pad 位置重新被屏蔽成 0。
        # 原地操作（下划线结尾），不改变 [B, P+R] 的形状。
        full_mask.scatter_(1, logp_pos + 1, resp_pad_mask.to(full_mask.dtype))
        # 三件事挤在一行，拆开看：
        #   resp_lengths [B]  = 每条实际生成了多少 token（含 eos）
        #   valid_resp  [B]   = 这条采样是否为非空（后面按行索引，空行会导致越界/写错位置）
        #   eos_mask [B, R]   = 「是 eos 且在有效区内」的位置
        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        # argmax 在 bool→int 上找「第一个 1」的下标，也就是第一个 eos 的位置。
        # ⚠️ 没有 eos 时 argmax 会返回 0（不是 -1），所以必须配合 has_eos 使用。
        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
        # ⭐ 截断到第一个 eos：eos 之后模型再吐出来的东西是「多余生成」，既不该进策略梯度，
        # 也不该进 value 回归。has_eos 的行取 eos_pos+1（含 eos 本身），否则保持原长度。
        # clamp(min=1) 防止长度为 0 导致后面的索引/除法出问题。
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        # resp_policy_mask [B, R] float 0/1 —— 真正的 loss mask：
        #   t < 有效长度（含 eos）且该位置本来就有 token。
        # 为什么用 float 而不是 bool：后面要直接乘在 loss/advantage 上做加性与归一化。
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        # value 用同一套 mask。分开起名字是为了将来允许 actor/critic 的有效范围不同；
        # 这里两者恰好一致，所以是 clone。
        resp_value_mask = resp_policy_mask.clone()

        # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
        # 💡 这一整段是「给更新准备常量」：old_values 供 GAE 用，ref_logp 供 KL 锚点用。
        #    它们都不参与反传，所以放在 no_grad 里；PPO 同时驻留 4 个模型，显存非常紧张。
        # ⚠️ 注意「陈旧度」：old_resp_logp 来自 rollout 那一刻的策略，old_resp_values 与 ref_logp
        #    却来自当前权重。严格 on-policy 时两者应当一致；若 ratio 长期偏离 1 太多，
        #    说明数据太旧，PPO 的 clip 正是在兜这个底。
        with torch.no_grad():  # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
            # DDP 的 forward 挂了梯度同步钩子，这里只是推理，直接取 .module 更轻也更直接。
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            # [B, P+R] → 每个位置一个标量 V，形状 [B, P+R]。
            # ⚠️ 因为 prompt 是左 padding，full_mask 里一定有 0，模型内部就无法走
            #    fused SDPA 快路径（见 MiniMindAttention 里的 all(mask==1) 判断），这是左 padding 的隐性代价。
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            # gather 沿序列维按 logp_pos [B, R] 取数 → [B, R]。
            # 位置语义：V(s_t) 是「看到第 t 个生成 token 之前」的状态价值（因为 logp_pos 是预测位置）。
            # 乘 mask 让无效位置归 0，避免 GAE 递推时把 padding 的乱值带进来。
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask

            # 参考模型的 per-token logp，分四步读（和后面 actor 的算法一模一样）：
            #   1) ref_model(...).logits          [B, P+R, V]
            #   2) [:, :-1]                       [B, P+R-1, V]  最后一个位置没有「下一个 token」可预测
            #   3) log_softmax(dim=-1)            [B, P+R-1, V]  💡 用 log_softmax 而非 log(softmax)：
            #      内部走 logsumexp，指数不会溢出、概率不会下溢成 0 再 log 成 -inf，数值稳定且梯度更好
            #   4) gather(2, labels[..., None])    [B, P+R-1, 1] → squeeze(-1) [B, P+R-1]：整条序列的 logp
            #      再 gather(1, logp_pos)          [B, R]          只留生成段
            # 一句话：per_token_logps 从 [B, T, V] 收缩成 [B, R]。
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            # token_rewards [B, R]：每个 token 的「即时外部奖励」，这里全 0。
            # 因为 RLHF 只有一个 sequence-level 的终局分数，没有 per-token 的即时奖励。
            token_rewards = torch.zeros_like(old_resp_logp)
            # last_idx [B]：每条序列最后一个有效 token 的下标。
            last_idx = resp_lengths - 1  # [B]
            # ⭐ 只在「最后一个有效 token」注入外部 reward：
            #   token_rewards[i, 末端] += reward_i，其余位置保持 0。
            #   GAE 会天然把这个终局分数沿着时间反传成每个 token 的 advantage——
            #   这就是「稀疏终局奖励」变成「稠密 token 信号」的那一步。
            # 用 += 而不是 =：留出将来叠加逐 token 形状奖励（如工具调用正确性）的余地；
            # valid_resp 过滤掉空采样行，否则 last_idx 可能越界或写错位置。
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励

            # ----------------------------------------------------------------------------------
            # GAE（Generalized Advantage Estimation）逐行拆解。
            # 记号：r_t = token_rewards[t]，V_t = old_resp_values[t]，gamma=1.0，lam=0.95。
            #   delta_t = r_t + gamma * V_{t+1} - V_t          （TD 误差，一步优势）
            #   A_t     = delta_t + gamma * lam * A_{t+1}      （GAE 递推，从后往前）
            # 直觉：lam 控制「看多远」——
            #   lam=0 → A_t 就是单步 TD：方差小、偏差大（过度相信 critic）
            #   lam=1 → A_t 就是蒙特卡洛回报减 baseline：无偏、方差大
            #   0.95 是常用折中。gamma=1.0 是因为语言模型整段就是一个 episode，没有「迟早会结束」的折扣需求。
            # ----------------------------------------------------------------------------------
            gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B, device=args.device); advs_rev = []
            # 倒着扫：只有从后往前，A_{t+1} 才是已知的。
            for t in reversed(range(gen_len)):
                # V_{t+1}：最后一步没有下一个状态，bootstrap 值取 0.0
                # （等价于 episode 在这里终止；这也是「reward 只放末端」能成立的前提）。
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                # TD 误差 = 即时奖励 + 折扣后的下一状态价值 - 当前状态价值。
                # 形状都是 [B]，逐时间步算。
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                # GAE 递推：把未来各步的 delta 按 (gamma*lam)^k 指数加权累加。
                # 注意 lastgaelam 是跨循环轮次携带的状态，这就是「递推」二字的由来。
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                # 收集时是「从后往前」的顺序，所以下面要反转回来。
                advs_rev.append(lastgaelam)
            # stack 后是 [B, R]，但时间维是倒序的，[::-1] 反转成正常顺序。
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            # ⭐ returns = advantage + V = V 的回归目标。
            #   因为 A = GAE 对 (回报 - V) 的估计，所以 回报 ≈ A + V。
            #   💡 这一步的巧思：critic 与 actor 共享同一份 V 与 A 的计算，
            #      不用再单独算一遍蒙特卡洛回报，省一次前向、也保证两者目标一致。
            returns = advantages + old_resp_values  # [B, R]

            # ----------------------------------------------------------------------------------
            # advantage 标准化：只在有效 token 上统计 mean / var，再标准化到 ~N(0,1)。
            # 为什么必须只在有效 token 上统计：padding 位置被 mask 成 0，如果把它们也算进均值，
            # 均值会被 0 拉偏（序列越短、padding 越多，偏得越厉害），advantage 的零点就错了。
            # 实现手法：先乘 mask 求和，再除以 mask 之和 —— 就是「掩码版 mean」。
            # ----------------------------------------------------------------------------------
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            # rsqrt(x) = 1/sqrt(x)，比 1/torch.sqrt(x) 少一次除法、也更稳。
            # +1e-8：当整批 advantage 全相等时方差为 0，防除零。
            # 最后再乘一次 mask，把 padding 位置彻底清零。
            # （advantage 来自 no_grad 区，天然无梯度，不需要 detach。）
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # --------------------------------------------------------------------------------------
        # 第 D 阶段：PPO 内层循环。同一批 rollout 数据被复用 ppo_update_iters 轮（默认 2 轮），
        # 每轮再切成若干 minibatch。这就是 PPO 的「数据复用」——也是它比 REINFORCE 样本效率高的原因。
        # ⚠️ 复用次数不能太大：越往后 ratio 离 1 越远，clip 一直生效，等于白算。
        # --------------------------------------------------------------------------------------
        # mini_batch_size 有可能大于 B（两者默认都是 2），min 一下避免空 batch。
        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False
        # 下面这堆都是「本轮要打印的统计量」，用 python float 累加。
        # .item() 会触发一次 GPU→CPU 同步，但每个 minibatch 才一次，代价可以接受。
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0
        for ppo_epoch in range(args.ppo_update_iters):
            # ⚠️ 这个 break 之所以安全，是因为 approx_kl 已经 all_reduce 过——
            #    所有 rank 看到同一个数值、在同一个 ppo_epoch 一起退出。
            #    若不做 all_reduce，就可能某张卡 break、另一张继续，下一次 DDP 集合通信数量不匹配 → 死锁。
            if stop_ppo:
                break
            # 每个 ppo_epoch 重新打乱一次，让 minibatch 的组合不同（提高数据复用效果）。
            b_inds = torch.randperm(B, device=args.device)
            for i in range(0, B, mb_size):
                # 最后一个 minibatch 可能不满 mb_size，这是正常的（切片天然支持）。
                inds = b_inds[i:i + mb_size]

                # critic 前向：[mb, P+R] → gather → [mb, R]。
                # ⚠️ 这里没乘 mask，保留原始 V 值，因为 value_loss 里要用 clamp(V, old±ε) 比较；
                #    乘 mask 放在 loss 汇总时做。
                mb_values_seq = critic_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                with autocast_ctx:
                    # autocast_ctx 在 CPU 上是 nullcontext()，GPU 上是 bf16 autocast。
                    # 💡 用 bf16 而不是 fp16：bf16 指数位和 fp32 一样宽，不会 underflow，
                    #    所以全程不需要 GradScaler（省掉一套 scale/unscale 逻辑）。
                    res = actor_model(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    # MoE 的负载均衡辅助损失（router importance loss）；非 MoE 用 0 标量占位，
                    # 这样下面的 loss 公式可以统一写，不用分支。
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                    # 构造方式与上面 ref 完全一致，只是这次是「新策略」的 logp，带梯度 → [mb, R]。
                    mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])

                # d = log π_new - log π_old（对数比率）。ratio = exp(d) 就是「新策略把这个 token
                # 变得更可能了多少倍」。形状 [mb, R]。
                # 💡 在第一个 ppo_epoch 的第一次迭代，若 actor 未被更新过且与采样端一致，d ≈ 0、ratio ≈ 1。
                log_ratio = mb_resp_logp - old_resp_logp[inds]

                # 可开关的诊断：观察首轮首个 minibatch 的 mb 与 old logp 差异。
                # 若 ratio_max/ratio_min 离 1 很远（比如 >2 或 <0.5），说明 rollout 引擎给的 logp
                # 与本地前向对不上：常见原因是引擎与训练端 dtype/算子不同、KV cache 实现差异，
                # 或者 update_policy 之后没真正生效。此时 PPO 的 clip 就在「兜着训练」。
                if args.debug_log_ratio and ppo_epoch == 0 and i == 0 and is_main_process():
                    _lr = log_ratio.detach()
                    _m = resp_policy_mask[inds].bool()
                    if _m.any():
                        _lrv = _lr[_m]
                        Logger(f"[DBG log_ratio] step={step} max|lr|={_lrv.abs().max().item():.6e} "
                               f"mean|lr|={_lrv.abs().mean().item():.6e} "
                               f"ratio_max={torch.exp(_lrv).max().item():.6f} "
                               f"ratio_min={torch.exp(_lrv).min().item():.6f} "
                               f"dropout={getattr(lm_config, 'dropout', None)} "
                               f"training={actor_model.training}")
                # 📐 approx_kl 用的是「二阶近似」的 KL：
                #     KL(π_old || π_new) = E[-log r + r - 1]，其中 d = log r
                #     由泰勒展开 e^d - d - 1 = d^2/2 + O(d^3)，当 d 很小时
                #     KL ≈ 0.5 * E[d^2]  ← 本行做的就是这个
                # 为什么不用精确的 k3 估计：d^2 同样恒非负，d≈0 时几乎等价，但少一次 exp，更便宜。
                # ⚠️ 0.5 这个系数不能丢：丢了之后 early_stop_kl 阈值的含义就翻倍了。
                # 分母是 mask 之和，即「按有效 token 平均」。
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                # 💡 展开说三件事：
                #   ① 为什么各卡不同：每张卡各自 rollout 各自的数据，approx_kl 自然不一样；
                #   ② 不同步会怎样：rank0 认为该早停 → break，rank1 认为不用停 → 继续进 minibatch
                #      循环 → 两卡后面的 all_reduce 次数错位 → 训练挂死（最难调的一类 bug）；
                #   ③ 为什么用 AVG 而不是 SUM：平均后每张卡拿到的都是同一个「全局平均 KL」，
                #      语义也正好是我们想要的量。
                # detach().clone()：既切断梯度，又得到一个可原地修改的独立张量
                # （all_reduce 是原地操作，不能直接作用在带计算图的张量上）。
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)

                # 早停条件：KL 涨到阈值（默认 0.25，约等于平均 |log_ratio| ≈ 0.7）说明已经跑偏，
                # 继续更新只会把策略推得更远，索性放弃本轮剩余的 minibatch。
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True

                # ratio = π_new / π_old，[mb, R]。我们手里只有 log 空间的值，所以 exp 一下。
                ratio = torch.exp(log_ratio)
                # clipfrac = 「被 clip 掉的 token 占比」，一个纯诊断量，不参与求导。
                # 健康区间大约 0.1~0.3；接近 1 说明每步都在被截断（步子太大，该降 lr 或减少复用轮数）。
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))
                # 📐 KL_ref 惩罚项，逐项拆：
                #     记 d_ref = log π_ref - log π_θ   （注意顺序：ref 在前，相反于 log_ratio）
                #     penalty = exp(d_ref) - d_ref - 1
                # 为什么它等于 KL(π_θ || π_ref) 的一个「非负单样本估计」：
                #     E_{x~π_θ}[exp(d_ref)] = Σ_x π_θ(x) * π_ref(x)/π_θ(x) = 1
                #     ⇒ E[exp(d)-d-1] = 1 - E[d] - 1 = -E[log π_ref - log π_θ] = KL(π_θ||π_ref) ≥ 0
                # 💡 为什么不直接用 -d_ref 估 KL：单个样本的 -d_ref 可正可负，方差极大，
                #    甚至能让总 loss 变成负数（日志看起来像 bug）；而 exp(x)-x-1 ≥ 0 恒成立
                #    （因为 e^x ≥ x+1），每个样本都给出非负贡献，均值方差小得多——这就是 PPO 里的标准做法。
                # 它跟 clip 的分工：clip 约束「相对旧策略」的步子；KL_ref 约束「别离开 SFT 权重太远」，
                #    后者才是防 reward hacking 的那道闸（reward model 的盲区往往离 SFT 很远）。
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                # 📐 PPO 的 clipped surrogate，写成「要最小化的 loss」：
                #     原始目标（最大化）: L = E[ min( r*A, clip(r,1-ε,1+ε)*A ) ]
                #     取负号变最小化，并把负号推进 min 里（min(-x,-y) = -max(x,y)）：
                #     policy_loss = E[ max( -A*r, -A*clip(r) ) ] + kl_coef * KL_ref
                # 逐项理解：
                #   · A > 0（好样本）：-A*r 随 r 增大而减小 → 鼓励把 r 推大；
                #     但 clip 后的 -A*(1+ε) 是下界 → r 一旦超过 1+ε，梯度就是 0，不再奖励「更更更大」。
                #   · A < 0（坏样本）：-A*r 随 r 增大而增大 → 鼓励把 r 压小；
                #     但 clip 到 1-ε 后梯度为 0，防止把某个 token 的概率一巴掌拍死（那会毁掉语言能力）。
                #   · 外层 max：两个候选里取「更悲观」的那个，保证目标只会比未截断版本更保守。
                # ⚠️ 乘 mask 再除以 mask.sum() 得到的是「按 token 平均」的 loss——
                #    否则长回复的 loss 会被序列长度放大，等价于给长样本更高权重。
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)
                # 📐 value loss：critic 的 MSE 也做成 clipped 版本，同样取「悲观界」：
                #     L_V = 0.5 * max( (V - R)^2, (clip(V, V_old-ε_v, V_old+ε_v) - R)^2 )
                #   · 0.5 是 MSE 的常规系数（对 V 求导正好消掉平方带来的 2）。
                #   · clip 的对象是「相对 old value 的变化量」：critic 抖 → advantage 抖 → actor 抖，
                #     所以价值网络同样要被限制在信任域内。
                #   · 为什么取 max：无论往哪个方向，只要更新后误差更大，就按大的那个罚，
                #     逼迫 critic 只在「确实变好」时才移动。
                #   · returns 来自上面的 no_grad 段，是固定常数（不回传梯度）。
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                # 日志用别名（detach 只是习惯，避免统计量无意间把计算图延长一级）。
                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                # 💡 为什么乘 0 而不是 continue/break：
                #    DDP 的梯度同步是挂在 backward 的 hook 上的。若某张卡跳过 backward，而其它卡做了，
                #    集合通信次数立刻对不上 → 卡死。乘 0 之后所有卡都老老实实跑完 forward+backward，
                #    通信对齐，但这次贡献的梯度恰好是 0，等价于「本 minibatch 不更新」。
                # ⚠️ 别改成 loss = torch.tensor(0.)：那样会脱离计算图，backward 拿不到梯度图，
                #    虽然不报错，但 DDP 的 hook 行为会依赖是否有 grad_fn，容易埋坑。
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    # 为什么要除以 accumulation_steps：梯度会自然累加，先除以累积步数
                    # 等价于对窗口内的 loss 取平均，这样 lr 不用随 accumulation 调整。
                    # vf_coef=0.5：policy loss 与 value loss 的量级平衡系数
                    # （value loss 是 MSE，尺度比 policy loss 大，通常取 0.5 左右）。
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps

                loss.backward()

                # 累计统计量（用于本 step 末尾打印 / 上报）。
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                # 累积计数器 +1（注意它是「函数级」变量，跨 step、跨 ppo_epoch 一直累加）。
                grad_accum_step += 1

                if grad_accum_step % args.accumulation_steps == 0:
                    # clip_grad_norm_ 把「所有参数梯度的全局 L2 范数」裁到 grad_clip（默认 1.0）。
                    # 💡 PPO 里几乎必开：advantage 标准化后仍偶尔有离群样本，
                    #    一次坏更新就能把策略打崩，裁剪是最便宜的保险。
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    # 两套参数、两个优化器，各走各的，互不干扰。
                    actor_optimizer.step()
                    critic_optimizer.step()
                    # scheduler 跟着 optimizer.step 走（每个优化步一次，而不是每个 minibatch 一次）。
                    actor_scheduler.step()
                    critic_scheduler.step()
                    # ⚠️ zero_grad 必须在 step 之后：PyTorch 的梯度是累加的，先清零等于白算。
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        # --------------------------------------------------------------------------------------
        # 梯度累积的收尾：循环结束后若计数没对齐，必须补一次 step。
        # 💡 为什么会有余数：一个 step 内的 minibatch 数 = ceil(B/mb_size) * ppo_update_iters，
        #    它不一定能被 accumulation_steps 整除。若不补这一步，这几个 minibatch 的梯度就一直挂在
        #    .grad 里：既浪费了计算，又会污染下一个 step 的第一次更新（新旧梯度混在一起）。
        # ⚠️ 注意这跟普通 SFT 的写法不同：SFT 一般是 (global_step+1) % accum == 0，
        #    这里是按「绝对累积计数」判断，因为 minibatch 数不固定。
        # --------------------------------------------------------------------------------------
        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # 把最新权重同步给 rollout 引擎（sglang 引擎会写盘 + 调接口热更新；torch 引擎只是换引用）。
        # 💡 为什么不是每步都同步：同步很贵（要传几十 MB 权重）。允许几步的轻微 off-policy，
        #    正是 PPO 的 clip / KL 能容忍的；同步太稀疏才会出问题。
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(actor_model)

        if is_main_process():
            # 只有 rank0 打印/上报，避免 N 张卡刷出 N 份日志。
            # 用累加和除以次数得到本轮均值；max(log_count, 1) 防 0 除（例如 ppo_update_iters=0 时）。
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            # resp_lengths 已经被截断到第一个 eos，所以这里统计的是「真实生成长度」。
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                # 这几个指标就是判断 PPO 是否健康的仪表盘：
                #   reward ↑（在涨）、kl_ref 平稳、approx_kl 远小于 early_stop_kl、clipfrac 在 0.1~0.3。
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # eval() 只是为了让保存时的行为稳定（关 dropout），存完立刻切回 train()。
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 两层解包，缺一不可：
            #   ① DDP 的 .module —— 否则 state_dict 的 key 会带 "module." 前缀，下次 load 对不上；
            #   ② torch.compile 的 _orig_mod —— compile 会把模型包成 OptimizedModule，
            #      key 会变成 "_orig_mod.xxx"，直接存同样会导致加载失败。
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            # half() 存 fp16：体积减半，且与 eval_llm.py / 其它脚本的权重格式保持一致。
            # cpu() 是为了不占显存（4 个模型已经在抢显存了）。
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)

            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            # 💡 只存 actor 权重是不够的：续训需要 optimizer 的动量、scheduler 的步数、
            #    以及 critic 的权重。这个函数把这些打包成一个 ckpt 目录/文件。
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        # --------------------------------------------------------------------------------------
        # 显式释放本 step 的中间张量。
        # 💡 为什么要手动 del：Python 的引用计数要等到「变量被重新赋值」或函数返回才释放，
        #    而循环里同名变量会在下一次迭代才被覆盖；显式 del 能让显存立刻回收。
        #    PPO 同时驻留 actor/critic/ref/reward 四个模型，显存是这一课最稀缺的资源。
        # ⚠️ 隐含前提：这些名字在本轮一定被赋值过。若 ppo_update_iters=0，
        #    policy_loss / loss 等根本不存在，这里的 del 会抛 NameError。
        # --------------------------------------------------------------------------------------
        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    # ⚠️ PPO 的 lr 比 SFT 小 2~3 个数量级（SFT 典型 5e-5，这里 3e-7）：
    #    策略更新的目标从「拟合数据」变成「提升 reward」，步子稍大就会把语言能力推崩。
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    # critic 的 lr 通常略高于 actor：价值函数是从随机头开始学的，需要更快追上。
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    # 💡 save_interval 同时兼任 rollout 引擎的同步周期（见训练循环里的 update_policy）。
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    # ⚠️ max_seq_len + max_gen_len 必须 ≤ 模型的 max_position_embeddings（默认 32768），
    #    否则 RoPE 的 buffer 切片会越界；本仓库里 768+1024=1792 很安全。
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    # ε=0.2 是 PPO 论文的默认值：ratio 落在 [0.8, 1.2] 之外就不再给梯度。
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    # kl_coef 太小防不住 reward hacking，太大则模型学不动；0.02 是常见起点。
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    # gamma=1.0：语言模型整段生成就是一个 episode，不需要对未来回报打折。
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    # lam 控制偏差-方差折中：0→单步 TD（方差小偏差大），1→蒙特卡洛（无偏方差大），0.95 折中。
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    # 每个 rollout batch 被复用 2 轮 minibatch；越大样本效率越高，但 off-policy 程度也越大。
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--debug_log_ratio", action="store_true", help="打印首轮首个minibatch的log_ratio差异量级，用于核查ratio≈1是否成立")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    # init_distributed_mode 会根据 torchrun 注入的环境变量（WORLD_SIZE/RANK/LOCAL_RANK）建进程组；
    # 单卡跑时它返回 0 且不初始化，所以后面的 dist.is_initialized() 判断都能安全短路。
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 每张卡用不同种子（42+rank）：若所有卡同种子，rollout 出来的样本会高度相似，等于浪费算力。
    # 注意 DataLoader 的 shuffle 由 sampler 负责，这里主要影响采样温度与 dropout。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 只有显式开启续训时才会去读 ckpt；read 失败/不存在时 lm_checkpoint 返回 None。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # CPU 上不能开 cuda.amp，用 nullcontext() 顶替，这样 with autocast_ctx 的写法在两处通用。
    # 💡 用 bf16 就不需要 GradScaler：bf16 的指数范围与 fp32 相同，不会出现 fp16 那种 underflow。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 这里其实用的是 swanlab，但它兼容 wandb 的 API，所以直接别名成 wandb。
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # resume='must'：续训时要求接着同一条曲线，避免日志被拆成两条。
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Actor模型
    # init_model 内部会 load {save_dir}/{from_weight}_{hidden}{moe}.pth，并用 strict=False 载入。
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    # 💡 ref 必须和 actor 同源（都在 SFT 权重上初始化）：KL(π_θ || π_ref) 的起点才是 0，
    #    否则一开始就有个巨大的 KL 惩罚把训练带偏。
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    # eval() 关掉 dropout，保证 ref 的 logp 稳定可复现；requires_grad_(False) 让它彻底退出反传。
    # ⚠️ ref 没有包 DDP：它不做 backward，也就不需要梯度同步；每张卡各自持有一份只读副本。
    ref_model = ref_model.eval().requires_grad_(False)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    # ⭐ strict=False 是这里的关键：SFT 权重里只有 model.* 和 lm_head.*，没有 value_head.*。
    #   strict=True 会因 missing keys 直接报错；strict=False 则「能对上的都载入」，
    #   于是 8 层 Transformer + embedding 全部继承 SFT 权重，只有 value_head 是随机初始化的。
    # 💡 这样 critic 一开始就是个「懂语言」的模型，只是还不会估价值——比全随机初始化收敛快得多。
    # 🔍 对照 nanoGPT：相当于把训练好的 GPT 复制一份，只把最后的 lm_head 换成 1 维输出再微调。
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    # reward model 是外部的（默认 internlm2-1.8b-reward），fp16 加载 + 内部 eval + 冻结。
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎
    # 工厂函数按 engine_type 返回不同实现：torch（进程内 generate）或 sglang（HTTP 到独立服务）。
    # 两者的 RolloutResult 接口完全一致，所以训练循环不需要知道用的是哪个。
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # RLAIFDataset 每条样本只返回 {'prompt': ..., 'answer': ""}：
    # 它按 thinking_ratio 的概率决定 prompt 里是否带 thinking 开启标记（模拟两种推理模式）。
    # max_length 给的是 max_seq_len + max_gen_len，因为 prompt 与生成要共享上下文预算。
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    # 只为数一下 iters（每个 epoch 有多少个 batch），不参与训练。
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 估算「一个 epoch 总共会调用多少次 optimizer.step()」，作为余弦退火的 T_max：
    #   iters 个 rollout batch × ppo_update_iters 轮 × 每轮的 minibatch 数 ceil(B/mb_size)
    #   最后再除以梯度累积步数。
    # ⚠️ 估多了/估少了都只是 lr 曲线提前或延后到底，不影响正确性。
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 注意 critic 也要恢复：它的 value 尺度如果和 optimizer 动量对不上，续训初期 advantage 会抖。
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # torch.compile 用 inductor 后端做图捕获与 kernel 融合，通常能提速 20%~100%（首次编译很慢）。
        # ⚠️ 顺序很重要：必须先 compile 再包 DDP，保存时也要先 .module 再 _orig_mod 解包。
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # freqs_cos/freqs_sin 各 rank 由 config 确定性算出，默认每步广播一次纯属浪费
        # 💡 broadcast_buffers=False 就是关掉「每个 forward 前把 buffer 从 rank0 广播一遍」的行为；
        #    RoPE 的 cos/sin 表本来就完全一致，广播它纯属浪费带宽。
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank], broadcast_buffers=False)
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank], broadcast_buffers=False)
    # 包装完之后必须重新同步一次，否则 rollout 引擎拿到的还是包装前的旧引用。
    rollout_engine.update_policy(actor_model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 用 and 短路代替 if：非分布式时 train_sampler 是 None，自动跳过。
        # set_epoch 必须每轮调一次，否则 DistributedSampler 的 shuffle 顺序每轮都一样。
        train_sampler and train_sampler.set_epoch(epoch)
        # setup_seed(42+epoch) 让每个 epoch 的随机顺序可复现；randperm 手动构造单卡时的索引序列，
        # 这样 SkipBatchSampler 无论有没有 sampler 都能工作。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 续训：只有「正在恢复的那一轮」才需要跳过已完成的前 start_step 个 batch。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # ⚠️ iters 传的是「完整长度」（len(loader)+skip），因为 batch_sampler 已经吃掉了一部分，
        #    len(loader) 只剩剩余 batch 数；只有用完整长度，step == iters 的保存条件才判得准。
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    # barrier 保证所有 rank 都跑完了再一起退出，避免 rank0 先退出导致其它卡等通信等成僵尸进程。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
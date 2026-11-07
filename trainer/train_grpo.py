# ===========================================================================================
# train_grpo.py —— MiniMind 的 GRPO（Group Relative Policy Optimization）强化学习脚本
# ===========================================================================================
# 作用：不做 critic，直接「同一个 prompt 采一组回答，按组内相对好坏给每个 token 加权」，
#       把 SFT 模型往 reward 更高的方向推，同时用 KL 惩罚拴住 ref_model。
#
# 在流水线中的位置：
#   上游 ← out/full_sft_768.pth（actor 与 ref 同源初始化）
#          ← dataset/rlaif.jsonl（只有 prompt）
#          ← 外部 reward model（默认 internlm2-1.8b-reward）
#   下游 → out/grpo_768.pth（可直接推理；DeepSeek-R1 时代最常见的「小模型对齐」做法）
#          → ./sglang_ckpt_grpo（用 sglang 引擎时，热更新的权重就落在这里）
#
# 核心思想：
#   1. 用「组内相对排名」替代 critic。同一 prompt 采 G=6 条回答，把 reward 在这一组内做
#      (r - mean)/std 标准化，得到的数就直接当 advantage 用。好处是彻底绕开「价值网络难训、
#      尺度难校准」的问题——我们只关心「这 6 条里谁更好」，不关心绝对分是多少。
#   2. 只用 3 个模型（actor / ref / reward），比 PPO 少一整套 critic 的前向 + 优化器状态，
#      显存和工程复杂度都明显下降，所以它在「结果可验证」的任务（数学、代码、Agent 工具调用）上
#      几乎取代了 PPO。
#   3. 目标函数仍是 PPO 的 clipped surrogate + KL_ref 惩罚，只是 advantage 的来源换了；
#      本文件还提供 cispo 变体（只做上界截断 + detach 的 ratio 权重），比标准 clip 更省。
#
# 关键张量/数据（B=batch，G=num_generations，P=prompt 长度，R=生成长度，D=768，V=6400）：
#   prompt_inputs.input_ids  [B, P]       左 padding 的 prompt
#   outputs                  [B*G, P+R]   rollout 展开后的整条序列（同一 prompt 的 G 条相邻）
#   completion_ids           [B*G, R]     只含生成段
#   old_per_token_logps      [B*G, R]     采样那一刻策略的 logp（ratio 的分母）
#   per_token_logps          [B*G, R]     当前策略的 logp（带梯度）
#   ref_per_token_logps      [B*G, R]     冻结 ref 的 logp
#   logp_pos                 [B*G, R]     「第 t 个生成 token」对应的 logits 下标 = prompt_len-1+t
#   rewards                  [B*G]        规则奖励 + reward model 分数
#   grouped_rewards          [B, G]       同一 prompt 的 G 条排成一行，供组内标准化
#   advantages               [B*G]        组内标准化后的优势（整条序列共享一个数）
#   completion_mask          [B*G, R]     截断到第一个 eos 的 completion mask
#
# 阅读建议：
#   1) 先读下面「共用的 RL 循环骨架」与 PPO/GRPO 对照表，再看 grpo_train_epoch 开头的 docstring；
#   2) 重点逐行读「logp_pos / scatter_」那 3 行（和 train_ppo.py 完全同构，只是 batch 变成 B*G）；
#   3) 然后是 grouped_rewards → advantages 的组内标准化，这 4 行就是 GRPO 的全部灵魂；
#   4) 接着看 completion_mask 截断到 eos 的 4 行，再读 cispo / grpo 两种 loss 的分支；
#   5) __main__ 里可以跳过 argparse，只看「5. 初始化模型和数据」。
#   可以跳过：debug 打印、wandb 记录、ckpt 保存细节、sglang 参数。
# ===========================================================================================
import os
import sys

# ⚠️ 这两行是「脚本也能当包内模块跑」的经典写法：
#    强制 __package__ 让相对导入（from model.xxx）按 trainer 包解析，
#    sys.path 追加仓库根目录，保证 model / dataset / trainer 这些顶层包能被找到。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ⚠️ 这行 import 必须排在 torch 之前：Windows 上 pyarrow 与 torch 的 DLL 加载顺序会互相踩，
#    先 import datasets(pyarrow) 再 import torch 才能避开（issue #771）。不要重排 import。
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

# 💡 gc / AutoTokenizer / AutoModel 在本文件里没有被直接引用（tokenizer 与 reward model 都由
#    trainer_utils 内部加载）；保留它们不影响运行，读到可以直接忽略。
warnings.filterwarnings('ignore')


# ===========================================================================================
# 【本文件与 train_ppo.py 共用的 RL 循环骨架】—— 两个文件读之前都请先看这一块
# ===========================================================================================
# 一个训练 step 永远是这五拍：
#   第 0 拍 rollout 采样：把 prompt 喂给 rollout_engine，用「当前策略」采样出 response，
#                         同时存下采样时的 per_token_logps 作为 old_*（importance sampling 的分母）。
#                         GRPO 在这里一次采样 G 条（PPO 是 1 条）。
#   第 1 拍 打分 reward  ：规则奖励（长度 / 格式 / 重复）+ reward model 分数，合成一条标量回报。
#   第 2 拍 算 advantage ：PPO 用 critic 估 V(s) 再走 GAE；GRPO 直接在同一 prompt 的 G 条样本
#                         内部做 (r-mean)/std —— 这就是「组内相对优势」，完全不需要 critic。
#   第 3 拍 更新策略     ：PPO 目标 = clipped surrogate + KL_ref（+ value loss）；
#                         GRPO 目标 = clipped surrogate + KL_ref（没有 value loss）。
#   第 4 拍 同步 rollout 引擎：update_policy(actor)，让下一次采样仍近似 on-policy。
#
# 【PPO 与 GRPO 的对照表】
#   ┌────────────┬──────────────────────────────────┬────────────────────────────────────┐
#   │ 维度       │ train_ppo.py                     │ train_grpo.py                      │
#   ├────────────┼──────────────────────────────────┼────────────────────────────────────┤
#   │ 模型数量   │ 4：actor / critic / ref / reward │ 3：actor / ref / reward            │
#   │ 优势估计   │ GAE（gamma、lam 两个旋钮）       │ 组内标准化（没有旋钮，只有 G）     │
#   │ 采样条数   │ num_generations = 1              │ num_generations = 6                │
#   │ 信用分配   │ 逐 token（终局 reward 反传）     │ 整条序列共用一个 advantage         │
#   │ 数据复用   │ ppo_update_iters 轮 minibatch    │ 一个 rollout batch 只更新一次      │
#   │ 归一化粒度 │ 整个 minibatch 的 token          │ 先按序列、再按样本                 │
#   │ 典型场景   │ 有可学习的密集价值信号           │ 结果可验证（数学/格式/工具调用）   │
#   └────────────┴──────────────────────────────────┴────────────────────────────────────┘
#
# 💡 一句话理解 GRPO：PPO 需要「一个学出来的裁判告诉我这一步值多少分」，
#    GRPO 只需要「同一道题做六遍，比平均好的就加分」——把「价值估计」降级成「组内排名」，
#    用采样换掉了 critic。
# 🔍 和 nanoGPT 对照：loss 仍然建立在「下一个 token 的对数概率」上（和我们熟悉的交叉熵同源），
#    区别是每个 token 的梯度被乘上了一个 advantage 权重：从「模仿文本」变成「按相对好坏加权地模仿自己」。
# ===========================================================================================


def rep_penalty(text, n=3, cap=0.5):
    """n-gram 重复惩罚：给「复读机式」输出扣分。

    作用：统计 text 里 n-gram 的重复比例，返回 [0, cap] 区间的惩罚值。
    参数：
        text : str，待评估文本（GRPO 里传去掉 think 块之后的 answer）。
        n    : int，n-gram 阶数，默认 3。
        cap  : float，惩罚上限，默认 0.5。
    返回：
        float，[0, cap]；文本凑不出一个 n-gram 时返回 0.0。
    设计理由：
        退化输出的典型症状是「短语级复读」，trigram 恰好能抓住它，又不会误伤正常的中文措辞。
        用「重复比例」而不是「最大重复次数」，是为了让长短回复可比；
        用 cap 封顶，是为了让规则项的量级始终小于 reward model 的 ±3，不至于一条坏样本毁掉整批。
    """
    # 粗分词：\w+ 抓字母/数字/下划线（中文按字切），[^\w\s] 抓单个标点。
    # 不用 tokenizer：这里只要「近似词」的概念，跑在 CPU、零显存、零依赖。
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    # 滑窗切 n-gram，用 tuple 做 key 才能进 set 去重；len(grams) = max(0, len(toks)-n+1)。
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    # 重复比例 = (总数 - 去重数)/总数，再乘 cap*2 放大后截断到 cap：
    # 重复率 50% 就吃满惩罚，再高也不加罚（饱和设计，抗离群值）。
    # ⚠️ grams 为空时 len(grams)=0 会除零，所以用 if grams 兜底。
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    """GRPO 版的奖励函数：给展开后的 B*G 条回复逐条打分。

    作用：规则奖励 + reward model 分数，得到每条样本的标量回报。
    参数：
        prompts      : list[str]，长度 B，带 chat template 的 prompt 原文。
        responses    : list[str]，长度 B*G —— rollout 已把每个 prompt 展开成 G 条，顺序是
                       [p0g0, p0g1, ..., p0g(G-1), p1g0, ...]，与 repeat_interleave 一致。
        reward_model : LMForRewardModel（已 eval + 冻结）。
    返回：
        rewards : [B*G] float Tensor。
    设计理由：
        为什么这里写成 i/j 双层循环而不是 PPO 里的 zip：关键在于「下标的换算」。
        rollout 引擎对 prompt 做了 repeat_interleave(G)，所以第 i 个 prompt 的 G 条样本
        连续占据 [i*G, i*G+G) 这一段；写成 i*G+j 既明确了这个排布，也方便后面
        rewards.view(B, G) 直接 reshape 成「一行一组」。
    """
    # 一次性开 [B*G] 零张量，后面用 += 累加规则项，最后加上 reward model 的分数。
    rewards = torch.zeros(len(responses), device=args.device)

    # reward model 冻结，不需要梯度；no_grad 同时省掉激活显存。
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                # ⭐ 下标换算：第 i 个 prompt 的第 j 条回答。这个顺序必须和引擎的
                #   repeat_interleave(G) 完全一致，否则后面 rewards.view(-1, G) 就会串组，
                #   advantage 会变成「拿别人的分数做比较」——训练照样跑，但学不到东西。
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 从 chat template 还原 messages 列表（reward model 的接口要这个格式）。
                # DOTALL 让 . 能跨换行——回复里必然有换行。
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                # answer 默认是整条回复；若有 think 块，会被替换成 think 之后的部分。
                answer = response
                # 规则项 1：长度。太短多为空回复/敷衍，太长多为复读或跑题。
                # 用区间而非「越长越好」，是对冲 reward model 普遍存在的 verbosity bias。
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                # 若模型输出了思维链（以 </think> 结束）
                if '</think>' in response:
                    # 只切第一刀，后面即使还有 </think> 也整段留在 answer 里（会因格式错被罚）。
                    thinking_content, answer_content = response.split('</think>', 1)
                    # 规则项 2：thinking 长度（太短=没真思考，太长=凑字数）。
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    # 规则项 3：格式。恰好一个 </think> 才加分，多个说明格式已崩。
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    # 交给 reward model 的只有最终答案，不含思维链（它是在答案上训练的）。
                    answer = answer_content.strip()
                # 规则项 4：复读惩罚，纯扣分项。
                rewards[response_idx] -= rep_penalty(answer)

                # 模型项：reward model 打分（内部把分数 clamp 到 [-3, 3]）。
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        # python list[float] → [B*G] Tensor，一次搬上 GPU，避免逐条 .item() 频繁同步。
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        # 规则项 + 模型项：这就是每条样本的终局回报（sequence-level reward）。
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    """跑一个 epoch 的 GRPO 训练（本文件核心，约 120 行）。

    参数：
        epoch          : int，当前 epoch 下标（打印时 +1）。
        loader         : DataLoader，产出 batch['prompt']（list[str]，长度 B）。
        iters          : int，本 epoch 的总 step 数（用来判断「是不是最后一步」）。
        rollout_engine : RolloutEngine，负责一次采 G 条，并提供 update_policy 同步。
        ref_model      : 冻结的参考模型（SFT 权重）。
        reward_model   : LMForRewardModel。
        start_step     : int，续训时已完成的 step 数（DataLoader 已跳过前面的 batch）。
        wandb          : swanlab/wandb 句柄或 None。
        use_sglang     : bool，标记当前是否用 sglang 引擎；⚠️ 当前函数体没有用到它，
                         保留是为了兼容旧的调用签名（采样差异已在 rollout_engine 内部消化）。
    返回：
        None（副作用都发生在全局的 model / optimizer / scheduler 上）。

    与 ppo_train_epoch 的结构差异（一定要对照着看）：
        · 没有 critic、没有 GAE、没有 ppo_update_iters 内层循环、没有 early stop；
        · 整个 B*G 一次前向、一次 backward、一次 optimizer.step（大 batch 换样本效率）；
        · advantage 不是逐 token 的，而是「一条序列一个标量」，直接广播乘到每个 token 上。
    关于全局变量：
        这里的 model、ref_model、tokenizer、optimizer、scheduler、lm_config、autocast_ctx、args
        都来自 __main__，不是本函数的参数。
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        # 与 PPO 的差别：这里关掉 add_special_tokens（prompt 里已经带 chat template 的特殊标记，
        # 再自动加 BOS 会重复）；也关掉 token_type_ids（模型没有 segment embedding，传了纯属噪音）。
        # padding_side='left' 仍然是必须的：保证生成从每行最后一个 token 之后开始。
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        # 从左侧截断到 max_seq_len：左 padding 下，左侧先被切掉的是 pad，再往里才是最早的历史，
        # 所以这个写法等价于「保留最近的上下文」。
        # ⚠️ 截断之后 prompt_lens 会由引擎按「实际传入的宽度」重新计算，后面所有位置索引
        #    都从 prompt_lens 推出来，因此仍然自洽；但如果外面自己缓存了 prompt 长度就会错位。
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # 第 0 拍：一次采样 G 条（引擎内部 repeat_interleave(G)），得到 [B*G, P+R]。
        # G=6 是 GRPO 的关键超参：太小（如 2）组内均值/方差噪声大，太大则显存和采样时间都吃不消。
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids  # [B*G, P+R]
        completion_ids = rollout_result.completion_ids  # [B*G, R]
        completions = rollout_result.completions  # list[str]，长度为 B*G
        # ⭐ old_per_token_logps 是「采样那一刻的策略」给出的 logp，[B*G, R]。
        # 它就是 ratio 的分母：ratio = exp(logp_new - logp_old)，衡量当前权重离采样端有多远。
        # detach()：这是常量，绝不能带梯度（有些引擎可能返回带图的张量，这里统一掐断）。
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # [B*G]
        # attention_mask：1=可见，0=不可见（模型内部会算 scores += (1-mask)*-1e9）。
        # 先用「非 pad」临时构造，随后用 scatter 精确修正生成段。
        full_mask = (outputs != tokenizer.pad_token_id).long()
        # ⭐⭐⭐ 定位「第 t 个生成 token 对应的 logits 位置」：
        #   第 t 个生成 token 在整条序列里的物理位置 = prompt_len + t
        #   而 logits 在位置 j 预测的是 token j+1（next-token 错一位）
        #   ⇒ 预测该 token 的 logits 位置 = prompt_len + t - 1
        # [B*G, 1] - 1 + [1, R]  →  [B*G, R]
        # ⚠️ 少了这个 -1 是最常见的实现 bug：整段 logp 整体错位一格，训练能跑但学不到东西。
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)
        # ⭐ 为什么偏置 +1：logp_pos 是「预测位置」，+1 才是「token 自己的位置」。
        # scatter_ 把 completion_mask 写到 full_mask 的第 prompt_len+t 列：
        # 让真正生成出来的 token 在 attention 里可见，同时把引擎补出来的 pad 屏蔽成 0。
        # 原地操作，形状仍是 [B*G, P+R]。
        full_mask.scatter_(1, logp_pos + 1, rollout_result.completion_mask.to(args.device, dtype=full_mask.dtype))

        # 第 1 拍：打分。[B*G]
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        with autocast_ctx:
            # 一次前向吃下整批 B*G（不切 minibatch）：GRPO 用「更大的 batch」换「更少的更新次数」。
            # autocast_ctx 在 CPU 上是 nullcontext()，GPU 上是 bf16 autocast；
            # 💡 bf16 指数范围与 fp32 相同，不会 underflow，所以全程不需要 GradScaler。
            res = model(outputs, attention_mask=full_mask)
            # MoE 的负载均衡辅助损失；非 MoE 用 0 标量占位，保持后面的 loss 公式统一。
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 当前策略的 per-token logp，四步读（与 PPO 完全同构）：
            #   1) res.logits            [B*G, P+R, V]
            #   2) [:, :-1, :]           [B*G, P+R-1, V]  最后一个位置没有下一个 token 可预测
            #   3) log_softmax(dim=-1)   [B*G, P+R-1, V]  💡 用 log_softmax 而非 log(softmax)：
            #      内部走 logsumexp，避免 exp 溢出 / 概率下溢成 0 再 log 成 -inf，数值稳定、梯度更好
            #   4) gather(2, outputs[:,1:])  [B*G, P+R-1, 1] → squeeze(-1) [B*G, P+R-1]：整段 logp
            #      再 gather(1, logp_pos)     [B*G, R]         只留生成段
            # 一句话：从 [B*G, T, V] 收缩成 [B*G, R]。
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # ref 的 logp 只当常数用：no_grad 省掉整张图的激活显存（ref 与 actor 同规模，很吃内存）。
        # 注意 ref_model 已经在外面 .eval().requires_grad_(False) 过。
        with torch.no_grad():
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            # 调 RL 时最先怀疑的永远是 reward，而不是网络：把同一组的 G 条回答并排打出来，
            # 一眼就能看出 reward 到底在奖励什么、组内有没有区分度。
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        # --------------------------------------------------------------------------------------
        # 第 2 拍：GRPO 的灵魂——组内相对优势。
        # --------------------------------------------------------------------------------------
        # ⭐ view(-1, G) 能成立，靠的是 rollout 用 repeat_interleave(G) 展开：
        #   [p0g0, p0g1, ..., p0g5, p1g0, ...] → 每 G 个连续元素恰好是同一个 prompt 的一组。
        #   如果哪天换成 repeat(G)（先铺满 g0 再铺 g1），这个 reshape 就会串组，务必对齐。
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        # 组内均值/标准差，再 repeat_interleave 回 [B*G]，让每条样本拿到「自己那组的统计量」。
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 💡 unbiased=False（总体标准差）：这一组的 G 条就是「全部」，不需要贝塞尔校正；
        #    更重要的是 unbiased=True 在 G=1 时会出现 0/0 = nan，而这里不会。
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 📐 GRPO 的 advantage：
        #     A_i = (r_i - mean(r_group)) / (std(r_group) + 1e-4)
        # 为什么不除以某个学出来的 baseline：同一组的 reward 共享「同一个 prompt、同一个 reward model」，
        # 系统性偏差在相减时被消掉，只剩「这组里谁更好」的相对信号 —— 这正是 critics-free 的关键。
        # ⚠️ 当一组内 6 条 reward 完全相同时 std=0、A≈0：没有区分度就没有梯度，这是预期行为
        #    （全对/全错说明这道题已经没有学习信号，该换数据或调 reward）。
        # +1e-4 就是防这个除法炸掉。
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        # --------------------------------------------------------------------------------------
        # completion_mask：截断到「第一个 eos」为止。
        # --------------------------------------------------------------------------------------
        # completion_mask 来自引擎，[B*G, R]，1=该位置是真实生成出来的 token。
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # is_eos：既是 eos、又在有效区内（右端补位的 pad 里可能恰好也是 eos_token_id，要排除）。
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        # 默认值：没有 eos 的样本就当成「生成长度 = R」（整段都算有效）。
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        # 只在「至少有一个 eos」的行上覆盖成第一个 eos 的位置。
        # argmax 在 bool→int 上返回第一个 1 的下标；没有 eos 时它返回 0（不是 -1），所以必须用掩码保护。
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # ⭐ 生成 mask：(列号 <= eos_idx) 且 该位置本来有 token。
        # 为什么一定要截断到第一个 eos：
        #   (a) eos 之后的 token 是「模型已经正确停止之后又被引擎补出来的东西」，优化它们等于教模型别停；
        #   (b) 不截断的话，一条提前收尾的样本会带着大段 pad 参与长度归一化，把每条样本的权重稀释掉；
        #   (c) 这些位置在 old_per_token_logps 里也没有可靠的行为策略概率，ratio 无意义。
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        # --------------------------------------------------------------------------------------
        # 第 3 拍：目标函数（KL 估计 → ratio → loss）。
        # --------------------------------------------------------------------------------------
        # 📐 记 d = log π_ref - log π_θ（注意顺序：ref 在前，和下面 ratio 的 d 相反）。
        #    逐 token KL 惩罚 = exp(d) - d - 1
        # 为什么它是 KL(π_θ || π_ref) 的「非负单样本估计」：
        #     E_{x~π_θ}[exp(d)] = Σ_x π_θ(x) * π_ref(x)/π_θ(x) = 1
        #     ⇒ E[exp(d)-d-1] = 1 - E[d] - 1 = -E[log π_ref - log π_θ] = KL(π_θ||π_ref) ≥ 0
        # 💡 为什么不直接用 d（即 -KL 的一阶估计）：单个样本的 d 可正可负、方差极大，
        #    甚至能让总 loss 变负（日志看起来像 bug）；而 exp(x)-x-1 ≥ 0 恒成立（e^x ≥ x+1），
        #    每个 token 都给出非负贡献，均值方差小得多 —— 这是 PPO 系实现的标准写法。
        # 为什么需要它：clip 只约束「相对 old 策略」的步子，管不住模型一路跑到 reward model 的盲区；
        #    KL 锚在 ref（SFT 权重）上，才是防 reward hacking 的那道闸。
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        # importance sampling 比率：r = π_θ / π_old = exp(logp_new - logp_old)，[B*G, R]。
        # 为什么 ratio 可能明显偏离 1：old_per_token_logps 来自 rollout 引擎（可能是 SGLang 的另一套
        # 算子，也可能因为 update_policy 只在 save_interval 调用而慢了几步权重）。
        # 环境完全 on-policy 时 r≈1，此时下面两种 loss 都会退化成普通策略梯度。
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
        if args.loss_type == "cispo":
            # 📐 CISPO：只做「上方截断」的 clipped importance sampling。
            #     loss_token = -[ stop_grad(clamp(r, max=C)) * A * log π_θ ] + beta * KL
            # 与 grpo 分支的三点区别：
            #   ① 只有上界 C=epsilon_high（默认 5.0），没有 1-ε 的下界：允许 token 概率大幅下降，
            #      但仍给一个上限，防止某个 off-policy 样本的权重爆炸。
            #   ② 权重是「直接乘到 log π_θ 上」的常数，而不是乘到 ratio 上：
            #      梯度 = -w*A*∇log π_θ，就是最朴素的（加权）策略梯度，数值上非常稳，
            #      不会像 PPO 的 r*A 那样出现「分子分母同时指数变化」的抵消问题。
            #   ③ .detach()：ratio 只是「权重系数」，不对它求导。这正是 clipped importance sampling
            #      的定义——重要性权重被当成常数，截断只影响梯度的大小，不改变梯度方向。
            # 💡 这也是为什么 CISPO 在高 off-policy 程度下比 PPO clip 更稳：它没有 min/max 的硬切换，
            #    token 被截断时梯度只是被「削平」，不会突然变成 0（PPO 那样会浪费掉一半样本）。
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # 📐 标准 GRPO（= PPO 的 clipped surrogate）：
            #     loss_token = -[ min( r*A, clip(r,1-ε,1+ε)*A ) - beta * KL ]
            # 逐项直觉（A 是整条序列共享的标量，advantage.unsqueeze(1) 从 [B*G,1] 广播到 [B*G,R]）：
            #   · A > 0（这条回答比组内平均好）：鼓励把每个 token 的 r 推大；一旦 r > 1+ε，
            #     clip 后的乘积不再变化，梯度归零，避免「好样本被无限放大」。
            #   · A < 0（比组内平均差）：鼓励压低 r；但 r < 1-ε 后梯度归零，
            #     防止把某些 token 的概率一巴掌拍死（那会毁掉语言能力）。
            #   · min：取两个候选里更保守的那个（注意这里是 loss 里的 min，等价于对目标取 min）。
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            # 注意这里用的是 min（而不是 max）：per_token_loss 已经带负号，
            # 外层 -(...) 再把符号翻回来，所以「越小越保守」对应「目标取更小的那个」。
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # 📐 两级归一化，这一步是 GRPO 与 PPO 在工程上的又一个差异：
        #    ① 先按「每条序列的有效 token 数」归一化：sum / mask.sum()  → [B*G]
        #       —— 每条序列等权，长回复不会因为 token 多而主导梯度；
        #    ② 再对 B*G 条序列取 .mean()：一批里每条样本等权。
        #   PPO 那边是「整个 minibatch 按 token 总数平均」，长序列天然权重大；
        #   两种都常见，但逐序列归一化对「长度方差大」的生成任务更公平。
        # clamp(min=1)：全空回复（mask 全 0）的行不会除零。
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        # 除以 accumulation_steps：梯度会累加，先除等价于对窗口内 loss 取平均，lr 不用随 accumulation 改。
        # aux_loss 只有 MoE 才非零。
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        # ⚠️ 注意这里的累积判断用的是「1-based 的 step 取模」，和 PPO 里那个跨 step 的绝对计数器不同：
        #    意思是「每 accumulation_steps 个 step 更新一次」；而 step == iters 保证最后一步一定更新，
        #    不会有梯度残留到下一个 epoch。唯一的副作用是最后一个窗口可能不足 accumulation_steps 个
        #    batch，等效于那一步用了更小的有效 batch size。
        if step % args.accumulation_steps == 0 or step == iters:
            if args.grad_clip > 0:
                # 把全部参数梯度的 L2 范数裁到 grad_clip（默认 1.0）。
                # PPO 系训练里几乎必开：advantage 标准化后仍偶有离群样本，一次坏更新就能打崩策略。
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 只有一个优化器、一套参数（没有 critic），比 PPO 少一半的优化器状态显存。
            optimizer.step()
            scheduler.step()
            # ⚠️ zero_grad 必须在 step 之后：PyTorch 的梯度是累加的。
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            # loss 被除过 accumulation_steps，日志里乘回来才是真实量级。
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            # 注意：这里是「截断到 eos 之后」的真实生成长度，不是 R。
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            # 🔍 有意思的细节：日志里的 KL_ref 用的是 d 的加权均值（一阶估计），
            #    而 loss 里用的是 exp(d)-d-1（非负、低方差）。
            #    前者便宜、符号直观，适合当监控指标；后者数值稳，适合进 loss。
            #    两者都在估同一个 KL，数值上会有轻微差异，这是刻意的选择。
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            # advantage 的均值/标准差：这是判断 GRPO 是否健康的仪表盘。
            # std 接近 0 说明组内没有区分度（reward 饱和或全同分），此时训练基本停滞。
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # eval() 只为保存时行为稳定（关 dropout），存完立刻切回 train()。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 两层解包，缺一不可：
            #   ① DDP 的 .module：否则 key 会带 module. 前缀，下次 load 直接对不上；
            #   ② torch.compile 的 _orig_mod：compile 后 key 会变成 _orig_mod.xxx，同样会导致加载失败。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # half() 存 fp16（体积减半，与 eval_llm.py 等脚本的权重格式一致）；cpu() 不占显存。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 完整续训存档：模型 + optimizer 动量 + scheduler 步数。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 第 4 拍：把新权重同步给 rollout 引擎（torch 引擎只换引用；sglang 引擎要写盘 + 调接口）。
        # 💡 同步频率 = save_interval。中间几步允许轻微 off-policy，这正是 ratio + clip 存在的意义。
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # 显式释放中间张量：Python 的引用计数要等变量被重新赋值才回收，显式 del 让显存立刻释放。
        # GRPO 虽然省掉了 critic，但 B*G 条序列同时前向（actor + ref）依然很吃显存。
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    # ⚠️ 注意 batch_size 是「prompt 条数」，真正前向的序列数是 batch_size * num_generations，
    #    默认 2*6=12 条，显存按这个算。
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    # RL 阶段的 lr 比 SFT（典型 5e-5）小两个数量级：目标是「提升 reward」而不是「拟合数据」，
    # 步子稍大就会把语言能力推崩。
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
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    # G：每个 prompt 采几条。这是 GRPO 唯一「决定优势估计质量」的超参——
    # 它同时决定显存（正比）和组内统计量的噪声（反比）。
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    # beta：KL 惩罚系数，跟 PPO 的 kl_coef 是同一个角色（0.1 比 PPO 的 0.02 大，
    # 因为 GRPO 的 advantage 量级更小，需要相对更强的锚定）。
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    # loss_type：cispo（默认，只截上界 + detach 权重）或 grpo（标准 PPO clip）。
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    # epsilon_high：cispo 的上界（默认 5.0，很宽）。它不是「差一点就截断」，
    # 而是「只在比率离谱到 5 倍以上时才削平」，平时几乎不生效。
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    # 单卡跑时它返回 0 且不建进程组，后面的 dist.is_initialized() 判断都能安全短路。
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 每张卡不同种子：同种子的 rollout 会高度相似，等于浪费采样。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # ⚠️ 这里的 max_seq_len 写成了 prompt + 生成的总预算（768+1024=1792），
    #    它会被存进 config，并在下面的 RLAIFDataset 里作为 max_length 使用；
    #    模型内部真正决定 RoPE 表长度的是 max_position_embeddings（默认 32768），两者别搞混。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    # 只有显式开启续训时才读 ckpt；不存在时返回 None。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # CPU 上不能开 cuda.amp，用 nullcontext() 顶替，这样 with autocast_ctx 的写法在两处通用。
    # 💡 bf16 不需要 GradScaler：它的指数范围与 fp32 相同，不会出现 fp16 的 underflow。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 实际用的是 swanlab，但 API 兼容 wandb，所以直接别名。
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # resume='must'：续训时要求接到同一条曲线，避免日志被拆成两条。
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    # 💡 ref 必须与 actor 同源（都从 SFT 权重初始化）：KL(π_θ||π_ref) 起点才是 0，
    #    否则一开始就有一大笔 KL 惩罚把训练带偏。
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    # eval() 关 dropout 保证 ref 的 logp 稳定；requires_grad_(False) 让它彻底退出反传。
    # ⚠️ ref 没有包 DDP：不做 backward 就不需要梯度同步，每张卡各持一份只读副本。
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎（可插拔替换，只负责 policy 推理）
    # 工厂按 engine_type 返回 Torch 或 SGLang 实现，二者 RolloutResult 接口完全一致，
    # 所以上面的训练循环完全不需要知道底下用的是哪个。
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据和优化器
    # RLAIFDataset 每条只返回 {'prompt': ...}；thinking_ratio 控制 prompt 里是否带「开启思考」标记。
    # max_length 用 lm_config.max_seq_len = prompt + 生成的预算。
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 只为数一下每个 epoch 有多少个 batch。
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 余弦退火的 T_max = 一个 epoch 里 optimizer.step() 的总次数。
    # ⚠️ 这里只算了 iters/accumulation，没乘 epochs，所以多 epoch 时 lr 会提前触底后维持 eta_min。
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # torch.compile 做图捕获 + kernel 融合，通常提速 20%~100%（首次编译很慢）。
        # ⚠️ 顺序：先 compile 再包 DDP；保存时要先 .module 再 _orig_mod 解包。
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # 同 train_ppo：RoPE buffer 各 rank 一致，每步广播纯属浪费
        # 💡 broadcast_buffers=False 关掉「每次 forward 前把 buffer 从 rank0 广播一遍」，
        #    freqs_cos/freqs_sin 由 config 确定性算出，各 rank 本来就完全相同。
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    # 包装之后必须重新同步一次，否则引擎拿到的还是包装前的旧引用。
    rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 用 and 短路代替 if：非分布式时 train_sampler 是 None，自动跳过。
        # set_epoch 必须每轮调一次，否则 DistributedSampler 每轮的 shuffle 顺序都一样。
        train_sampler and train_sampler.set_epoch(epoch)
        # 每个 epoch 用不同种子，randperm 手动造出单卡时的索引序列，让 SkipBatchSampler 统一处理。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 续训：只有「正在恢复的那一轮」才需要跳过已完成的前 start_step 个 batch。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # ⚠️ iters 传「完整长度」（len(loader)+skip），因为 batch_sampler 已经吃掉一部分，
        #    只用 len(loader) 的话 step == iters 的保存/更新条件会判错。
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    # barrier 保证所有 rank 都跑完再一起退出，避免 rank0 先退出让其它卡等通信等到僵尸进程。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
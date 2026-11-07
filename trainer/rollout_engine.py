"""
# 如果使用sglang加速，需通过以下命令首先启动（transformers格式）模型：
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
"""

# ==============================================================================================
# 作用：把 RL 训练里「用当前策略采样生成（rollout）」抽象成一个可替换的引擎，并给出两个实现：
#       TorchRolloutEngine（HF/MiniMind 自带 generate，简单、单机、慢）
#       SGLangRolloutEngine（HTTP 调用外部推理服务，服务端做连续批处理，快）
# 在流水线中的位置：
#   上游 ← trainer/train_agent.py、train_grpo.py 等 RL 脚本，传入「当前 policy 模型 + tokenizer + device」；
#   下游 → 产出 RolloutResult（token ids + 每 token logprob + mask），供 RL 侧算 advantage / ratio / KL。
#   SGLang 模式下还负责反向同步：训练一步后把新权重推给推理服务（update_policy）。
# 核心思想：
#   1) 训练与推理解耦。RL 要求「生成数据的策略」和「被更新的策略」是同一份权重（on-policy），
#      但训练侧要 autograd + DDP，推理侧要 paged attention + 连续批处理，两者最优实现完全不同。
#      所以用 ABC 把「怎么生成」隔离出来，训练循环只依赖 rollout() 和 update_policy() 两个接口。
#   2) old_logps 必须由「生成这份数据的那份权重」给出。PPO/GRPO 的 ratio = exp(logp_new - logp_old)，
#      如果 old_logps 来自另一套实现（甚至另一个 dtype），ratio 一开始就不等于 1，训练会莫名地漂。
#      所以 RolloutResult 里必须带 per_token_logps，SGLang 那边也必须开 return_logprob。
#   3) 推理引擎手里是一份「权重的拷贝」，需要显式同步：Torch 引擎共享同一份参数对象（天然同步），
#      SGLang 引擎在另一个进程里，只能把权重落盘再 HTTP 通知服务端重新加载。
# 关键张量/数据（B=batch, G=num_generations, P=prompt 长度, R=completion 长度）：
#   prompt_ids      [B, P]        训练侧拼好的 prompt token
#   attention_mask  [B, P]        1=有效 token，0=padding
#   output_ids      [B*G, P+R]    prompt + completion 的完整 id
#   completion_ids  [B*G, R]      只有模型生成的部分
#   per_token_logps [B*G, R]      生成每个 token 时策略给出的对数概率（= on-policy 的 old_logps）
#   completion_mask [B*G, R]      1=真实生成，0=padding
#   prompt_lens     [B*G]         每条样本的 prompt 长度
# 阅读建议：
#   1) 先看 RolloutResult 的字段含义，再看 RolloutEngine 的两个接口（只有 10 行，却决定了整个设计）；
#   2) 然后看 compute_per_token_logps —— 它是「teacher forcing 重算 logprob」的通用工具，两个引擎都用它；
#   3) 再看 TorchRolloutEngine.rollout（最短，能把整条链路跑通）；
#   4) 最后看 SGLangRolloutEngine（HTTP payload / logprob 对齐 / 权重同步）。不用 SGLang 的话可以先跳过。
# ==============================================================================================
import os
import sys

# 与 train_*.py 同样的套路：允许 `python trainer/rollout_engine.py` 直接跑时也能做仓库根目录的绝对导入。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


# ===== 计算每个 token 的 logprob =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """算出「给定 token 序列」在模型下的逐 token 对数概率。

    作用：
        对 input_ids 做一次 teacher forcing 前向，取出最后 n_keep 个 token 各自的 log pi(token | 前文)。
        这是 RL 里最重要的一个量：old_logps / ref_logps 都靠它得到，ratio 和 KL 都从它派生。

    参数：
        model:          MiniMindForCausalLM，可以是 DDP 包装过的；必须处于 eval 语义（本函数不自带 no_grad）
        input_ids:      [B, T] 完整 token 序列（prompt + completion 拼在一起）
        n_keep:         只算「最后 n_keep 个 token」的 logprob。n_keep<=0 时直接返回空张量
        attention_mask: [B, T] 可选，1=有效 token，0=padding

    返回：
        [B, n_keep] float32 的逐 token 对数概率（一定是普通 tensor，不是 inference tensor）

    设计理由 1：为什么要 logits_to_keep=n_keep+1
        lm_head 的投影是 [B,T,D] @ [D,V]，V=6400，全量 logits 是 [B,T,V]，显存和算力都随 T 线性涨。
        而 RL 只关心「新生成的那 R 个 token」的概率，前面 P 个 prompt 位置的 logits 全是浪费。
        传 logits_to_keep=n_keep+1 后，模型只对最后 n_keep+1 个 hidden state 做 lm_head
        → 中间张量从 [B,T,V] 降到 [B,R+1,V]，长 prompt 场景能省好几倍显存。
        🔍 对照 nanoGPT：nanoGPT 的 estimate_loss 是「整段 logits 拉平后 cross_entropy」，
           这里做的是同一件事的逐 token 版 —— 只把「需要的那几行」算出来、只 gather 出目标列。

    设计理由 2：为什么是 n_keep+1 再切掉最后一列
        「位置 t 的 logits」预测的是「位置 t+1 的 token」。要覆盖最后 n_keep 个 token 的 logprob，
        需要位置 T-n_keep-1 ... T-1 共 n_keep+1 个位置的 logits；
        多出来的最后一个位置预测的是「序列之外的下一个 token」，没有标签，所以 [:, :-1] 丢掉。

    设计理由 3：为什么用 log_softmax 而不是 log(softmax(...))
        softmax 先 exp 再 log，在 bf16 下很容易把极小概率压成 0 再取 log 得到 -inf；
        log_softmax 内部先减 max 再算，数值稳定，且梯度形式更干净。

    ⚠️ 常见坑：
        n_keep 必须和「completion 长度」严格一致（调用方自己保证）。
        如果 n_keep 比实际 completion 长，就会把 prompt 尾部也算进来；ret=0 的位置对不上，ratio 会错。
        n_keep 超过有效长度（左侧 padding 很短的样本）时，尾部会吃到 padding 位置的 logits。
    """
    if n_keep <= 0:
        # 没有 token 要算（例如 completion 为空）→ 返回 [B, 0]。
        # 显式指定 float32：空张量后面会参与 stack/乘法，dtype 必须和正常路径一致，否则 downstream 报 dtype mismatch。
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    # DDP 的 forward 会触发一次 allreduce 且要求梯度可用，纯推理没必要；取出原始模块直接跑。
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    # ⚠️ inference tensor 的坑：MiniMindForCausalLM.generate 被 @torch.inference_mode() 装饰，
    #    它返回的 output_ids 是 "inference tensor"。inference tensor 不能参与 autograd
    #    （后面 old_logps 要拿去做 ratio 乘法、要 saved for backward），所以先 detach().clone() 转成普通 tensor。
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    # 只保留最后 n_keep+1 个位置的 logits：[B, n_keep+1, V]；[:, :-1] 后是 [B, n_keep, V]
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    # 逐行 gather：第 i 行要取的列是 ids_row 里的 token id，每行都不同，所以 gather 的 index 必须逐行给。
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        # 逐行再过一次 inference 检查：切片出来的视图可能仍带着 inference 标记。
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            # logits_row.log_softmax(-1): [n_keep, V] → gather(1, ids_row.unsqueeze(1)): [n_keep, 1] → squeeze: [n_keep]
            # 等价于 「-cross_entropy(logits_row, ids_row)」，但不需要构造 one-hot，也不会触发 reduction。
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    # [B 个 [n_keep]] → [B, n_keep]。stack 而不是 cat，是因为每一行长度相同。
    # 💡 这里用 Python 循环而不是一次性 gather：一次性写法是 log_softmax(logits).gather(2, ids.unsqueeze(-1))，
    #    但那样会同时把 [B, n_keep, V] 的 log_softmax 结果整个驻留显存；逐行处理能少一份峰值显存，
    #    也方便对每一行单独做 dtype/detach 处理。train_agent.py 在训练前向里用的是那个一次性写法。
    return torch.stack(per_token_logps)


# ===== Rollout 结果 =====
@dataclass
class RolloutResult:
    """一次 rollout 的产物，两个引擎都返回这个结构。

    作用：
        把「采样出来的东西」打包成训练循环能直接吃的形式，避免到处传 6 个裸张量。

    字段（B=batch, G=num_generations, P=prompt_len, R=completion_len）：
        output_ids:      [B*G, P+R]  prompt+completion 的完整 id（sglang 路径的 prompt 已去掉左 padding，P 因样本而异）
        completion_ids:  [B*G, R]    只有生成的部分；sglang 路径右侧用 pad_token_id 补齐
        per_token_logps: [B*G, R]    生成每个 token 时策略的 log 概率；sglang 路径补 0.0 的位置对应 mask=0
        completions:     List[str]   解码后的文本，规则奖励（数 <tool_call>、匹配 GT）都靠它
        prompt_lens:     [B*G]       每条样本的 prompt 长度
        completion_mask: [B*G, R]    1=真实生成 token，0=padding

    设计理由：
        用 dataclass 而不是 tuple / dict：
          - tuple 有 6 个字段，调用方解包顺序一错就是静默 bug（返回值里全是同形状张量，错了也不报错）；
          - dict 没有字段名提示；
          - dataclass 支持属性访问 + 类型标注，后续加字段（比如 finish_reason）不会破坏已有调用。
    """
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    """rollout 引擎的抽象基类：只暴露两个方法，训练循环只依赖这两个方法。

    为什么值得单独抽一层（而不是在训练脚本里 if/else）：
        1) 训练脚本（train_agent.py / train_grpo.py）不需要知道「生成是靠 HF generate 还是 HTTP 服务」，
           也不用关心 sglang 的 payload 长什么样；换引擎只改命令行参数。
        2) 把「on-policy 数据怎么来」这件事显式化：任何实现都必须同时提供 rollout（怎么生成）
           和 update_policy（怎么把新权重同步过去），二者缺一不可，缺了就不是 on-policy。
        3) 🔍 和 nanoGPT 对照：nanoGPT 的采样写在 generate() 里、和训练脚本混在一起；
           这里把它抽成接口，是因为 RL 需要「同一份权重在一次训练循环里被反复用于生成」，
           而生成的成本远高于一步训练，必须能换更快的推理后端。

    tokenizer 是类属性而不是实例属性的默认值：
        子类各自会在 __init__ 里覆盖它；这里给 None 只是为了让「没初始化就访问」时报 AttributeError
        而不是 NameError，读起来更清晰。
    """
    tokenizer = None
    
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """给定一批 prompt，为每条 prompt 生成 num_generations 条 completion。

        参数：
            prompt_ids:      [B, P] prompt token（可能带左侧 padding）
            attention_mask:  [B, P] 同上，1=有效
            num_generations: G，每条 prompt 采几条。GRPO 需要同组多条才能算组内相对优势
            max_new_tokens:  单条 completion 的最大长度 R
            temperature:     采样温度。只影响「采样的随机性」，不影响返回的 logprob（logprob 是 T=1 的分布下算的）

        返回：
            RolloutResult，其中所有张量的第 0 维都是 B*G。

        约定（子类必须遵守）：
            第 0 维的排列必须是「同一 prompt 的 G 条样本连续相邻」，
            即样本 idx = sample_idx * G + j。因为训练侧用 rewards.view(-1, G) 做组内分组。
        """
        pass
    
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        """把训练侧的当前权重同步给 rollout 引擎。

        参数：
            model: 可能是裸模型、DDP 包装后的模型，或 torch.compile 包装后的模型；实现方负责逐层解包。

        返回：
            无（SGLang 实现额外返回 bool，但接口上不强制）。

        为什么必须显式同步：
            Torch 引擎和训练共享同一份 nn.Parameter（换引用即可，等于天然同步）；
            而 SGLang 引擎在另一个进程里跑着另一份权重，不推送就会一直用旧权重采样
            → ratio 越来越偏离 1，训练直接崩（而且日志上看不出来，只看到 loss 乱跳）。
        """
        pass


# ===== PyTorch 原生推理引擎 =====
class TorchRolloutEngine(RolloutEngine):
    """用 MiniMind 自带的 generate 直接采样：最朴素、最容易读的实现。

    优点：零外部依赖、权重天然同步、调试方便（可以直接 pdb 进 generate）。
    缺点：batch 内不同长度的序列要一起跑到最长的那条结束（等长 padding），
          多轮场景下 batch=1，GPU 利用率很低 —— 所以有了下面的 SGLang 版本。
    """
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        """保存策略模型与解码所需的状态。

        参数：
            policy_model: 训练中的模型（裸模型 / DDP / torch.compile 都行，rollout 里会解包）
            tokenizer:    用于 batch_decode 与取 pad/eos id
            device:       只用于记录，真正的 device 由张量自身决定
            autocast_ctx: 训练侧的 autocast 上下文（bf16/fp16），传进来是为了让
                          「采样的前向」和「训练的前向」用同一个 dtype。
                          ⚠️ dtype 不一致会让 old_logps 和 new_logps 有系统偏差，ratio 一开始就偏离 1。
                          传 None 时用 nullcontext()（CPU/fp32 场景）。
        """
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """用 generate 采样，然后「用 teacher forcing 重算一遍 logprob」。"""
        # 解包 DDP：DDP.forward 会做梯度相关的同步，采样完全是浪费；而且 batch=1 时也不该走 DDP。
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            # ⚠️ 关键：repeat_interleave 而不是 repeat / num_return_sequences
            #   repeat_interleave(G, dim=0): [p0,p1] → [p0,p0,p1,p1]   ← 同 prompt 连续相邻（我们要的）
            #   repeat(G, 1)               : [p0,p1] → [p0,p1,p0,p1]   ← 交错，view(-1,G) 会分错组
            #   MiniMind 的 generate 内部 num_return_sequences 用的是 .repeat()（见 model_minimind.py），
            #   所以这里必须自己 interleave 好再 num_return_sequences=1，否则组内标准化会静默算错。
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),          # [B,P] → [B*G,P]
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),  # [B,P] → [B*G,P]
                max_new_tokens=max_new_tokens,
                do_sample=True,          # 必须采样：argmax 会让同组 G 条完全一样，组内 std=0 → 优势恒为 0
                temperature=temperature,
                num_return_sequences=1,  # 见上：多样性由 interleave 提供，不交给 generate
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]
            # 说明上面那个 .clone()：
            #   generate 被 @torch.inference_mode() 装饰，返回的是 inference tensor；
            #   出了 generate 之后 clone 一次就得到普通 tensor，后面才能 stack / 存进 dataloader / 参与 autograd。
            # ⚠️ 没有传 top_p / top_k，于是 MiniMind generate 的函数默认值 top_p=0.85、top_k=50 生效。
            #    而 SGLang 那边默认是 top_p=1.0、top_k=-1（纯 temperature）。两个引擎的采样分布其实并不相同，
            #    对比实验时要注意这一点。
            prompt_len = prompt_ids.size(1)
            # 右侧切出 completion：[B*G, P+R] → [B*G, R]。
            # 注意用的是「物理长度」prompt_len，不是 attention_mask.sum()：
            # 左 padding 的情况下有效 token 靠右，物理位置上的 [P:] 才是生成部分。
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
            # full_mask: prompt 部分沿用原 attention_mask，completion 部分全 1（生成的 token 一定有效）
            #   cat 后 [B*G, P] + [B*G, R] = [B*G, P+R]，正好对应 output_ids。
            #   传给 compute_per_token_logps 是为了在重算 logprob 时屏蔽 pad —— 否则 pad 位置会参与注意力
            #   （MiniMind 用 attention_mask 造 causal mask），上下文就串了，logprob 全错。
            full_mask = torch.cat([attention_mask.repeat_interleave(num_generations, dim=0), attention_mask.new_ones(output_ids.size(0), completion_ids.size(1))], dim=1)
            # 💡 为什么不用 generate 内部的 scores 而重新前向一次：
            #   generate 里的 logits 被 temperature / top_k / top_p 改过，不是模型原始分布；
            #   这里用 T=1 的干净前向重算，才能和训练时的 per_token_logps 放在同一个尺度上做 ratio。
            #   🔍 对照 nanoGPT：nanoGPT 的 generate 只返回 token，不返回概率；
            #     这里是「生成完再 teacher forcing 补一遍概率」，是 RL 场景特有的需求。
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        # skip_special_tokens=True：把 <|im_end|> 等控制 token 从文本里去掉，喂给「规则奖励」的文本才干净。
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             # new_full 继承 prompt_ids 的 device 与 int64 dtype，省掉两次 .to()
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             # ⚠️ completion_mask 在这里是「全 1」——这是有意的简化，但要理解它的代价：
                             #   MiniMind 的 generate 是「整批都结束才 break」，先结束的行会被反复填 eos（id=2）
                             #   直到所有行结束，这些补位的 eos 在这里也被标成了 1。
                             #   train_agent.py 用 batch=1 + 「eos 位置 mask 置 0」规避了这个问题；
                             #   如果你把 num_generations 直接交给 generate（而不是 interleave），就必须自己按 eos 截断。
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))
    
    def update_policy(self, model: torch.nn.Module):
        """Torch 引擎的同步是「假同步」：只是把引用换成训练侧的新模块。

        为什么这样就够了：
            训练循环里 optimizer.step() 是原地更新 nn.Parameter 的 data，
            引擎手里持有的 self.policy_model 指向的还是同一批 Parameter 对象
            → 下一轮 rollout 自动就是新权重，什么都不用做。

        那为什么还要换引用：
            训练开始时 model 会被 DDP / torch.compile 重新包装，引擎必须拿到包装后的对象
            （rollout 里会 .module 解包），否则要么用的是未包装的旧对象，要么漏掉 compile 的加速。
        """
        self.policy_model = model


# ===== SGLang HTTP API 推理引擎 =====
class SGLangRolloutEngine(RolloutEngine):
    """把生成委托给一个独立进程里的 SGLang 服务（见文件顶部启动命令）。

    为什么值得这么做：
        多轮 Agent RL 的瓶颈几乎 100% 在 rollout：一条轨迹要跑 3 轮、每轮都要一次完整前向，
        而 batch=1 的 HF generate 在 4090 上 GPU 利用率极低。SGLang 在服务端做连续批处理
        （很多请求拼一个 batch）和 paged attention 的 KV 复用，同样的卡能快好几倍。

    代价（也是这个类里全部复杂度的来源）：
        1) 权重在另一个进程里 → 必须落盘 + HTTP 通知（update_policy）；
        2) 采样发生在服务端 → logprob 必须由服务端一起返回（return_logprob），而且格式要对齐；
        3) 服务端不接受 attention_mask → 左侧 padding 必须自己先摘掉；
        4) 服务端返回的是变长 list → 要自己 pad 成张量、自己造 completion_mask。
    """
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        """初始化 HTTP 客户端与 tokenizer。

        参数：
            base_url:        SGLang 服务地址，例如 http://localhost:8998
            model_path:      只用来加载 tokenizer。⚠️ 必须和 server 端加载的模型是同一份词表，
                             否则 token id 对不上，生成的文本会变成乱码（而且不报错，极难排查）。
            shared_ckpt_path: 训练进程和服务端都能访问的目录，update_policy 把权重写在这里，
                             tokenizer 也一起存（服务端 reload 时需要）。
            timeout:         HTTP 超时（秒）。单条最长 max_new_tokens=768，慢的时候要几十秒，120 是留足余量。
        """
        self.base_url = base_url.rstrip('/')       # 去掉结尾斜杠，避免拼出 //generate
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        # trust_remote_code=True：MiniMind 的 tokenizer 带自定义 chat template 与特殊 token，需要远程代码支持。
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 存成属性而不是直接 requests.post，是为了单测时能注入 mock，不用真的起一个 SGLang。
        self.http = requests
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """一次 HTTP 调用完成「B*G 条生成 + 拿回 logprob」，再把变长结果拼成张量。

        整体流程（和下面的行内注释一一对应）：
            1) 按 attention_mask 摘掉左侧 padding，得到「有效 token 列表」
            2) 每条 prompt 复制 G 份（保持同 prompt 相邻）
            3) POST /generate，带 return_logprob=True
            4) 解析 meta_info 里的 output_ids 与 output_token_logprobs，并做长度对齐
            5) 手工 pad 成张量，手工造 completion_mask
        """
        # 去除左侧 padding tokens，只保留有效 token
        # 为什么必须做：/generate 只接受 input_ids 列表，没有 attention_mask 参数。
        # 如果直接把 [P] 里带 pad 的 ids 发过去，pad 会被当成真实上下文参与注意力，
        # 而且 prompt_lens 会算成 P（含 pad），后面所有位置对齐全错。
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            # mask.bool() 作索引：[P] → [P_valid]。这是「左 padding 且有效 token 靠右」场景下的通用写法，
            # 不依赖「padding 一定在左边」这个假设（虽然实际上一定在左边）。
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        # 展开成 B*G：外层遍历 prompt、内层重复 G 次 → [p0,p0,p0,p1,p1,p1]，同 prompt 连续相邻。
        # 效果等价于 torch 侧的 repeat_interleave，两边顺序必须一致，否则训练的组内分组会和采样错位。
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        
        payload = {
            "input_ids": all_input_ids,          # list[list[int]]，每条长度可以不同（服务端自己处理）
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                # stop_token_ids：遇到 eos 就停。⚠️ 服务端在 stop 处通常也会把这个 token 本身放进 output_ids，
                # 具体行为随 sglang 版本略有差异，所以下面的长度对齐逻辑必须容错。
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            # 必须为 True：否则服务端不返回 output_token_logprobs，我们就拿不到 old_logps，
            # ratio 没有分母，整个 on-policy 校正就没了。
            "return_logprob": True,
        }
        
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        # 4xx/5xx 直接抛异常：SGLang 出错时返回的 json 结构完全不同，
        # 静默往下解析只会得到一堆空字符串，然后在训练 10 分钟后才莫名其妙地崩。
        resp.raise_for_status()
        
        results = resp.json()
        # 单请求时 sglang 可能直接返回 dict 而不是 list，统一成 list 才不会在下面 for 循环里炸。
        if not isinstance(results, list):
            results = [results]
        
        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            # 生成结果放在 meta_info 里；result 顶层也可能有（不同版本），所以两级 .get 兜底。
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            logprobs = []
            for item in raw_logprobs:
                # sglang 的返回格式随版本变化：可能是 [logprob, token_id, text] 的三元组，
                # 也可能直接就是一个 float。两种都兼容，取第 0 个元素。
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            
            if len(logprobs) < len(completion_ids):
                # ⚠️ 对齐补全：logprob 比 token 少，缺的一定是「最早的几个 token」，
                # 所以补在前面（[0.0] * gap + logprobs），不能补在后面 —— 补在后面等于把整条 logprob 序列错位。
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                # 多了就砍左边：同样因为尾部才是「最近生成的」，砍头保住尾部对齐。
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []
            prompt = all_input_ids[i]
            # 注意这里的 prompt 是「去掉 padding 后」的那条，长度因样本而异 → 变长，后面要 pad。
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            # 文本也由本地 tokenizer 解码（不用服务端返回的 text 字段），保证和训练侧同一套解码规则。
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        
        device = prompt_ids.device
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))   # max(1, ...)：全部为空时也留一列
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len  # prompt 最长 + completion 最长（上界，宽松但正确）
        
        def pad_to_tensor(seqs, max_len, pad_val=0):
            # 纯 Python 补位再一次性建张量：这里的数据本来就在 CPU 的 list 里，逐个 scatter 反而更慢。
            # 每行 s + [pad_val]*(max_len-len(s)) → 等长 list → torch.tensor → [N, max_len]
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            # prompt 用 pad_token_id 补齐（右边补，因为每个样本的有效内容在左边，左 padding 已经在前面摘掉了）
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            # 💡 per_token_logps 补 0.0 而不是 -inf：
            #   0 = log(1)，表示「概率为 1」，看起来很不合理，但真正的保护是下面的 completion_mask=0 ——
            #   这些位置根本不参与 loss。补 -inf 反而危险：一旦哪里漏乘 mask，就会得到 inf * 0 = nan，
            #   而 nan 会顺着梯度污染整个模型（比一个 0 难查一万倍）。
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            # prompt_lens 是「去掉 padding 后的有效长度」，不是 pad 之后的行宽。
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            # completion_mask：前 len 个 1，后面全是补位的 0。torch 引擎那边是全 1，这两个引擎的 mask 语义不同，
            # 但都满足「1=真实生成、0=无效」，训练侧只依赖这一点。
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )
    
    def update_policy(self, model: torch.nn.Module):
        """把当前权重推给 SGLang 服务：落盘 → HTTP 通知 → 全 rank 同步成败。

        返回：
            bool，True 表示同步成功（失败会把所有 rank 一起抛 RuntimeError）。

        为什么必须「所有 rank 一起失败」：
            这是 DDP 里最容易写出死锁的地方。假设 rank0 落盘失败（磁盘满 / 服务没起）：
              - 如果只有 rank0 抛异常、其它 rank 继续往前跑；
              - 下一步其它 rank 会进 forward/backward 的 allreduce，而 rank0 已经退出 → 永久 hang。
            所以用一个 int 张量把 ok 广播给所有 rank，全体看到 False 后再统一抛错：
            进程是「整齐地一起死」，而不是「一个死七个等」。

        ⚠️ broadcast 的 src 张量必须在 GPU 上：NCCL 后端不支持 CPU tensor。
           所以下面用 next(model.parameters()).device 而不是 "cpu"。
        """
        ok = True
        # 只有 rank0 真的干活：8 个进程同时写同一个目录会把文件写坏，8 次 HTTP 请求也是纯浪费。
        # 未初始化分布式（单卡直接跑）时，当前进程就是唯一的 rank，也要干活。
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                # 脱 torch.compile：编译后的模块叫 OptimizedModule，真身挂在 _orig_mod 上。
                # 不脱这层，state_dict 的 key 会带 "_orig_mod." 前缀，服务端加载时会说「一个参数都没匹配上」。
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)
                abs_path = os.path.abspath(self.shared_ckpt_path)
                # half() + cpu()：存 fp16 省一半磁盘和 IO，cpu 是因为 save_pretrained 会做序列化，没必要占显存。
                # ⚠️ bf16 → fp16 是有风险的：bf16 指数位宽、范围大，fp16 范围小。
                #    如果模型里出现了超出 6e4 的权重（训练崩过就容易有），转 fp16 会变成 inf，
                #    服务端加载后直接输出 nan。真遇到就把 .half() 改成 .bfloat16()。
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                # safe_serialization=False：存 pytorch_model.bin，避开 safetensors 对共享权重（tie_word_embeddings）的限制，
                # sglang 的 from_disk 加载也能读。
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                # tokenizer 一起存：服务端 reload 时可能重建 tokenizer，目录里没有就会报错。
                self.tokenizer.save_pretrained(abs_path)
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                # 任何异常都收敛成 ok=False，交给下面的广播统一处理：
                # 不能让 rank0 抛一个「别人不知道」的异常，否则就是上面说的死锁。
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False
        if dist.is_initialized():
            # int(ok) → 0/1 张量，放在当前 rank 的 GPU 上（NCCL 要求）。
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            # broadcast：rank0 的值覆盖所有 rank。barrier 再等一拍，确保大家都拿到了才继续。
            dist.broadcast(ok_t, src=0); dist.barrier(); ok = bool(ok_t.item())
        # 所有 rank 一起在这里抛：整齐失败，不会 hang。
        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok
    
    def flush_cache(self) -> bool:
        """清空服务端的 KV cache / prefix cache。

        为什么需要：SGLang 默认开启 prefix caching（相同前缀复用 KV）。
        权重更新后，cache 里还留着「旧权重算出来的 K/V」，一旦被复用就会污染新权重的输出，
        表现为「明明换了权重，输出还是老样子」。所以换权重后调用一次最稳。

        返回：True=清理成功。
        """
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    def health(self) -> bool:
        """探活：服务没起来时提前返回 False，比在训练中途卡 120 秒超时要好。

        返回：True=服务健康。异常全部吞掉（连接被拒、超时、DNS 失败……此时「不健康」就够了，不需要区分原因）。
        """
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            # 裸 except 在这里是合理的：探活只关心「能不能用」，任何异常都等价于 False。
            return False


# ===== 工厂函数 =====
def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    """按名字构造一个 rollout 引擎，训练脚本只需要一个 --rollout_engine 开关。

    参数：
        engine_type:       "torch" 或 "sglang"
        policy_model:      仅 torch 引擎需要（sglang 引擎的模型在服务端）
        tokenizer:         仅 torch 引擎需要（sglang 引擎自己从 model_path 加载，保证和服务端词表一致）
        device / autocast_ctx: 仅 torch 引擎需要
        sglang_base_url:   SGLang 服务地址
        sglang_model_path: SGLang 的 tokenizer 路径（必须是服务端加载的同一份模型目录）
        sglang_shared_path: 权重落盘的共享目录

    返回：
        RolloutEngine 实例。

    设计理由：
        把「参数名 → 构造函数参数」的映射集中在一处。新增引擎（vllm / TensorRT-LLM）时只改这个函数，
        训练脚本一行都不用动；反过来，如果直接在 train_*.py 里 if/else，每加一个引擎要改遍所有训练脚本。
        ⚠️ 注意 sglang 分支忽略 tokenizer / device 参数：传了也不报错，但不会生效，容易误以为「用的是我传的 tokenizer」。
    """
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        # 早点报错，别让训练跑到一半才发现引擎名打错了。
        raise ValueError(f"不支持的引擎类型: {engine_type}")

# =============================================================================
# trainer/trainer_utils.py —— 全流水线共用的训练工具函数
#
# 作用：把「每个 trainer 都要重写一遍」的脏活集中到一处：学习率调度、DDP 初始化、
#       断点续训、模型构建、跳过已训练的 batch、奖励模型封装、数学表达式安全求值。
#
# 在流水线中的位置：
#   上游 ← model/model_minimind.py（MiniMindConfig + MiniMindForCausalLM）、model/ 下的
#          tokenizer，以及各阶段存进 ../out 与 ../checkpoints 的 .pth 权重
#   下游 → 8 个 trainer 全部 import 本文件的方法；
#          train_ppo.py / train_grpo.py 用 LMForRewardModel 给 rollout 打分，
#          train_agent.py 额外用 safe_math_eval 实现 calculate_math 工具，
#          scripts/eval_toolcall.py 与 scripts/web_demo.py 也复用 safe_math_eval。
#
# 核心思想：
#   1) 断点续训必须能「一模一样地接着跑」：除了权重，还要存 optimizer 状态、GradScaler、
#      scheduler 和第几个 step，并且用「先写临时文件再原子替换」保证崩溃时不会留下半个文件。
#   2) 训练要两份权重：xxx.pth 半精度给推理（小、快、加载便宜），
#      xxx_resume.pth 全精度 + 优化器给续训（大、准、能精确恢复动量）。
#   3) 模型输出的一切都不可信：agent 的工具参数是模型生成的文本，拿它算数学必须走 AST
#      白名单，绝不能直接 eval —— 否则一句 __import__('os').system(...) 就能接管进程。
#
# 关键张量/数据：
#   state_dict : dict[str, Tensor]，形状由 config 决定（D=768、V=6400、H=8、D_h=96、KV 头数 4）
#   ckp_data   : dict(model, optimizer, epoch, step, world_size, wandb_id,
#                     scaler, scheduler, critic_model, critic_optimizer, critic_scheduler)
#   score      : 标量 float，LMForRewardModel 输出后裁剪到 [-3, 3]
#   safe_math_eval 返回 int 或 float（不是 Tensor），调用方用 str(...) 转成文本喂回模型
#
# 阅读建议：
#   先读 get_lr（10 行，把余弦退火讲透）和 init_distributed_mode（DDP 其实就这么几行）；
#   lm_checkpoint 最长，按「保存」和「加载」两半读，重点看 _orig_mod 与 os.replace；
#   SkipBatchSampler 只看 __iter__ 就够；LMForRewardModel 与 safe_math_eval 是独立的，
#   不影响主线，可以放到最后读。
# =============================================================================
"""
训练工具函数集合
"""
import os
import sys
# 💡 让「cd trainer && python train_pretrain.py」这种从子目录直接运行的方式也能 import 到
# 仓库根目录下的 model / dataset 包：把仓库根（本文件的上一级）追加进 sys.path。
# 注意是 append 而不是 insert(0)，所以当前目录仍然优先。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ast
import operator
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
# AutoModelForSequenceClassification 本文件里没有直接用到（奖励模型走 AutoModel + 自定义
# get_score），保留在这里是为了切换到「分类头式」奖励模型时不用改 import 行。
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM


def get_model_params(model, config):
    """打印模型的总参数量，MoE 模型还会额外打印「激活参数量」。

    参数：
        model : nn.Module，MiniMindForCausalLM 实例（dense 或 MoE 都能处理）。
        config: MiniMindConfig，用来读出专家数等超参。

    返回：
        None（结果直接 print 出来）。

    为什么会有两套数字：MoE 把全部专家都放在显存里（总量大），但每个 token 只经过
    top-k 个专家（算力/激活量小）。所以「显存看总数、速度看激活数」，两个都得报。
    """
    # 除以 1e6 把「个」换算成「百万（M）」，方便和 64M 这种说法对齐。
    total = sum(p.numel() for p in model.parameters()) / 1e6
    # 三层 getattr：不同版本的 config 字段名不一样（n_routed_experts / num_experts），
    # 都没有就取 0。这样 dense 模型传进来也不会 AttributeError。
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    # 只统计编号 0 的那一个专家 = 单个专家的参数量；后面乘专家数就是全体专家的量。
    # 用参数名匹配而不是去数 module，是因为 named_parameters 的 key 路径稳定、跨实现一致。
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    # 总参数里减掉「所有路由专家 + 所有共享专家」，剩下的就是 embedding + attention + norm + router，
    # 也就是「无论走哪个专家都一定会算」的基座部分。
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    # 激活 = 基座 + 每个 token 实际经过的 top-k 个路由专家 + 全部共享专家（共享专家永远激活）。
    active = base + (expert * n_active) + (shared_expert * n_shared)
    # dense 模型时 n_routed = 0，于是 active == total，只打印一个数，不会出现 "64M-A64M" 的废话。
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """当前进程是不是 0 号进程。

    返回：
        bool。没初始化 DDP 时恒为 True（单卡场景下「我就是主进程」）。

    用途：print / wandb.init / 存盘这些「只该做一次」的副作用，都要先过这道闸，
    否则 8 个 rank 会同时打印同一行日志、同时往同一个文件写。
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """只在主进程打印，避免 DDP 下日志被 rank 数量刷屏。

    参数：
        content: 任意可 print 的对象（这里是 str）。
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """余弦退火学习率：lr_t = lr * (0.1 + 0.45 * (1 + cos(pi * t / T)))

    参数：
        current_step: int，全局第几步（0 <= current_step <= total_steps）。
        total_steps : int，整个训练的总步数 T。
        lr          : float，峰值学习率（即 args.learning_rate）。

    返回：
        float，当前步应使用的学习率。

    取值范围（把余弦从 1 代到 -1）：
        t = 0  ->  0.1 + 0.45 * (1 + 1)  = 1.00  ->  1.00 * lr   （峰值）
        t = T  ->  0.1 + 0.45 * (1 - 1)  = 0.10  ->  0.10 * lr   （谷值）
    所以系数严格落在 [0.1, 1.0] 之间，最低只掉到峰值的 10%，不会退到 0 ——
    留一点点尾巴是为了最后阶段参数还能继续微调，不至于完全冻住。

    💡 作者巧思：为什么不做 warmup？
        warmup 存在的理由是：训练早期 Adam 的二阶矩估计还很不可靠，大 lr 会把刚随机
        初始化的权重一步打飞。这里敢省掉，靠的是三个前提：
          1) 模型只有 64M（D=768），梯度尺度本身温和；
          2) 全程 clip_grad_norm_ 限幅，单步更新量有上界；
          3) pretrain 之后的每个阶段都是「在已有权重上微调」，起点已经是合理模型。
        代价也要说清：真从随机初始化起跑且 lr 调得偏大时，前几十步 loss 会明显抖，
        这是已知取舍，不是 bug。
    """
    # 🔍 可以和 nanoGPT 对照：nanoGPT 的 get_lr 是「线性 warmup + 余弦衰减到 min_lr」，
    # 需要 warmup_iters / lr_decay_iters / min_lr 三个超参；这里把 min_lr 固定成 0.1*lr，
    # 并且省掉了 warmup，于是只剩「当前步 / 总步数 / 峰值 lr」三个入参。
    # 注意 current_step / total_steps 是「全局步数 / 总步数」，不是 epoch，
    # 所以每个 epoch 结尾 lr 不会跳回去，整条曲线是一条连续下降的余弦。
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """根据环境变量判断要不要走 DDP；要走就初始化进程组并返回 local_rank。

    返回：
        int，本进程使用的 GPU 序号（非 DDP 模式返回 0）。

    设计理由：
        torchrun 会给每个子进程注入 RANK / LOCAL_RANK / WORLD_SIZE 等环境变量，
        而单卡直接 python train_xxx.py 时这些变量根本不存在。所以「RANK 存不存在」
        本身就等价于「是不是 torchrun 启动的」，不需要再加一个 --distributed 开关，
        也不会出现「忘了传开关就静默退化成单卡」的坑。
    """
    # 取不到 RANK 时给 -1（而不是 0），这样「没设置」和「rank 0」能区分开。
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    # backend="nccl"：NVIDIA GPU 之间通信的标准后端，DDP 每步的梯度 all-reduce 就靠它。
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    # ⚠️ 必须在建模型 / 分配任何 CUDA 张量之前绑定卡号。否则所有进程都默认用 cuda:0，
    # 8 个进程抢同一张卡，直接把显存打爆（而且报错信息通常很难懂）。
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """固定所有随机源，保证同一 seed 下初始化、数据顺序、dropout 都可复现。

    参数：
        seed: int，随机种子。调用方通常传 args.seed + rank（DDP 下让各卡数据不同）
              或 args.seed + epoch（每个 epoch 换一种 shuffle 顺序）。

    ⚠️ cudnn.benchmark = False 会让卷积慢一点，但本仓库几乎全是 attention/linear，
    影响很小；换来的是「同一输入必得同一输出」，复现偶发 nan 时这点非常值钱。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # manual_seed_all 覆盖所有卡：DDP 下每个进程只能看到自己的卡，但保险起见全设一遍。
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """保存 / 加载检查点，一次写两份：推理用的半精度权重 + 续训用的全精度完整状态。

    参数：
        lm_config : MiniMindConfig，只用到 use_moe 与 hidden_size，用来拼文件名。
        weight    : str，权重前缀，如 'pretrain' / 'full_sft' / 'dpo' / 'ppo_actor'。
        model     : nn.Module | None。传了 = 保存模式；None = 加载模式。
        optimizer : torch.optim.Optimizer，保存模式必传。
        epoch, step: int，当前训练进度，写进 resume 文件。
        wandb     : swanlab / wandb 句柄，用来取 run id，续训时接回同一条曲线。
        save_dir  : str，ckpt 目录。
        **kwargs  : 额外要一起存的对象，如 scaler / scheduler / critic_model。
                    有 state_dict() 的存 state_dict，其余直接存原对象。

    返回：
        加载模式且 resume 文件存在时返回 dict（含 model / optimizer / epoch / step / ...）；
        否则返回 None。

    文件命名规则（dense 与 MoE 用 _moe 后缀区分，避免互相覆盖）：
        {save_dir}/{weight}_{hidden_size}{_moe}.pth          <- 半精度，给推理
        {save_dir}/{weight}_{hidden_size}{_moe}_resume.pth   <- 全精度 + 优化器，给续训
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ⚠️ 常见坑之一：DDP 会把模型包一层，state_dict 的 key 全部带 'module.' 前缀。
        # 不拆包直接存，下次用裸模型 load 时一个 key 都对不上 —— 再配合下面的 strict=False，
        # 就是「静默什么也没加载」，你会在毫无提示的情况下从随机初始化开始训练。
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # ⚠️ 常见坑之二：torch.compile 会再包一层，key 前缀变成 '_orig_mod.'。
        # getattr 取不到就返回原对象，所以这一行同时兼容「编译过」和「没编译」两种情况。
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        # 逐个转成 CPU 张量再存：state_dict() 返回的只是参数的「引用」而不是拷贝，
        # 转成 CPU 张量后才是一份独立快照，后面 .half() 与写 resume 两次读写也因此拿到一致的数据。
        state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
        # 💡 原子写：先写 .tmp 再 os.replace 覆盖正式文件。torch.save 到一半被 kill
        # 会留下一个损坏的 .pth；而同一分区内的 os.replace 是原子操作，
        # 于是外部永远只能看到「旧的完整文件」或「新的完整文件」，不存在中间态。
        ckp_tmp = ckp_path + '.tmp'
        # 推理权重转 half：体积减半、加载更快，推理精度损失可以忽略。
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        # 取 wandb run id，续训时能接回同一条曲线（swanlab 与 wandb 的 API 形态略有差异，两种都试）。
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            # 注意复用的是上面那份 CPU 副本，此时还是 fp32 全精度（.half() 只作用于临时 dict）。
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            # 记下当时的卡数：换卡续训时要靠它换算 step（见下面的加载分支）。
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                # hasattr(value, 'state_dict')：GradScaler、scheduler、critic_model 都满足，
                # 于是 scaler 的 loss_scale、scheduler 的 last_epoch 都能原样恢复。
                if hasattr(value, 'state_dict'):
                    # critic_model 也是 DDP 包装的，同样要解包 + 解 _orig_mod，否则 key 带前缀。
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        # 及时把大字典和显存还给系统，避免存盘刚结束就和下一步前向抢显存。
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            # map_location='cpu'：先落到内存再由调用方 load_state_dict 到目标设备，
            # 这样「在 GPU 上存、在别的机器上读」也不会因为设备号不同而报错。
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                # ⚠️ 换卡数必须换算 step：step 是「优化器更新次数」，一次更新吃掉的是
                # world_size * batch_size 条数据（前提是每卡 batch 与梯度累积不变）。
                # 8 卡换 4 卡，跑完同样多的数据需要 2 倍 step，所以
                #   step_new = step_old * saved_ws / current_ws
                # 这里用 // 整除，宁可少跳一个 batch，也不要多跳（多跳会丢数据）。
                # 反过来，如果改了 batch_size 或梯度累积，这个换算就不准了，需要手动改。
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        # 没有 resume 文件：返回 None，调用方会当成「从零开始 / 从头微调」处理。
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """按 config 造模型，并按需加载某个阶段的权重，返回 (model, tokenizer)。

    参数：
        lm_config   : MiniMindConfig（hidden_size / num_hidden_layers / use_moe ...）。
        from_weight : str，权重前缀，如 'pretrain' / 'full_sft' / 'dpo' / 'ppo_actor'；
                      传 'none' 表示不加载、纯随机初始化（从零训练时用）。
        tokenizer_path: str，tokenizer 目录。
        save_dir    : str，权重所在目录。
        device      : str，'cuda' / 'cuda:0' / 'cpu'。

    返回：
        (model, tokenizer)。model 已 .to(device)，且处于默认的 train 模式。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        # 文件名必须和 lm_checkpoint 里的 ckp_path 完全一致，否则会 FileNotFoundError。
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        # ⚠️ strict=False 是一把双刃剑：
        #   好处 —— 架构小改（新增字段、LoRA 注入后 key 名不同、词表微调）时也能直接启动，
        #           不用为了跑通去改 checkpoint。
        #   坏处 —— 路径写错、key 前缀不对（如存的时候没解 _orig_mod）时它不会报错，
        #           而是安安静静地「一个权重都没加载」。你会以为在微调，其实在从随机
        #           初始化训练，loss 曲线还「看起来正常」。
        #   排查方法：临时改成 strict=True 看它报哪些 missing / unexpected keys，
        #            或打印 set(model.state_dict()) - set(weights) 的差集。
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    # 单独统计「可训练参数」：LoRA 微调时这里会远小于总参数（通常 1% 量级），
    # 这是验证「LoRA 是否真的只训了旁路、主干是否被冻住」最快的一眼。
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """包一层已有的 sampler，按 batch 重组后「丢掉最前面的 skip_batches 个 batch」。

    续训场景：从第 k 步恢复时，我们希望第 k+1 步看到的数据和「没中断时第 k+1 步看到的」
    完全一样。最省事又最不容易算错的做法不是「重新推偏移量」，而是把整个 epoch 的采样
    顺序原样再走一遍，前 k 个 batch 直接扔掉、不 yield。代价只是启动时多扫
    O(skip_batches * batch_size) 个索引，相比一次前向可以忽略。
    """

    def __init__(self, sampler, batch_size, skip_batches=0):
        """参数：
            sampler     : 可迭代的索引来源。DDP 下是 DistributedSampler，
                          单卡下是 torch.randperm(len(ds)).tolist()。
            batch_size  : int，每个 batch 的样本数。
            skip_batches: int，要跳过的 batch 数，也就是续训起点 step。
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """逐个 yield 索引 batch；前 skip_batches 个凑满就丢，不产出。

        注意这里不 shuffle 也不重排：顺序 100% 由传入的 sampler 决定。
        ⚠️ 所以调用方必须先用同一个 seed 重新生成同一份 indices，否则跳过的是另一批数据。
        """
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    # 已恢复过的 batch：直接扔掉，相当于「快进」过这段数据。
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 尾部不足一个 batch 的残余：只有「已经跳完」才 yield。
        # 若 skipped < skip_batches，说明整个 epoch 都在跳过区间里（skip 给大了），
        # 这时残余也应当一起丢掉。
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """返回实际会产出的 batch 数：ceil(样本数 / batch_size) - skip_batches，下限 0。

        ⚠️ 调用方用 len(loader) 算总步数去喂 get_lr，所以这个值必须准确；
        同时 DataLoader 一旦传了 batch_sampler，就会忽略 batch_size / shuffle / drop_last，
        也就是说「最后一个不满的 batch 要不要」完全由这里决定。
        """
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """把「奖励模型 / LLM-as-judge」封装成一个只暴露 get_score 的小接口。

    PPO / GRPO / Agent RL 都靠它给模型生成的回答打标量分，这个分直接当 reward 用：
    reward 越高，说明该回答越像「人类会喜欢的那个」。
    """

    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        """参数：
            model_path: str，奖励模型目录（通常自带 get_score 方法与自定义 modeling 代码）。
            device    : str，放到哪张卡。
            dtype     : torch.dtype，默认 float16 —— 打分只跑前向，半精度足够且省一半显存。

        设计理由：
            trust_remote_code=True 是因为打分逻辑写在奖励模型自己的 modeling 文件里，
            不是标准的 AutoModelForSequenceClassification 前向。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        # .eval()：关掉 dropout。否则同一个回答打两次分会得到不同的数，
        # advantage 里就混进了纯噪声，训练会莫名不稳定。这一行比看起来重要得多。
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """给「多轮历史 + 当前问题 + 待评回答」打一个标量分。

        参数：
            messages: list[dict]，一条对话历史，最后一条是当前 user 的提问。
            response: str，待评价的模型回答（调用方通常已经剥掉 <think> 段）。

        返回：
            float，裁剪到 [-3, 3]。

        设计理由：
            奖励模型是个「单轮 (user, assistant) 打分器」，它读不懂多轮 chat 结构。
            所以这里把历史压平成一行行 'role: content' 塞进 user 内容里，再配一条
            assistant 回答，把多轮问题伪装成单轮 —— 既不用改奖励模型，
            又能让它看到完整上下文。
        """
        # messages[:-1] 是「除最后一条提问以外的历史」，逐行拼成 'user: ...' / 'assistant: ...'。
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        # 最后一条才是「本轮要回答的问题」；messages 为空时给空串兜底，避免 IndexError。
        last_query = messages[-1]['content'] if messages else ""
        # 有历史时加一句分隔语，让奖励模型知道「前面是背景，下面才是要回答的新问题」；
        # 没有历史（单轮）就直接把问题当上下文。
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        # 拼成标准的单轮对话：user 侧放上下文，assistant 侧放待评回答。
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        # 💡 裁剪到 [-3, 3] 的三个理由：
        #   1) 奖励模型训练时的输出范围大概就在这里，超出的部分属于外推，本来就不可信；
        #   2) reward 会直接变成 advantage，一个异常大的分会让整组样本的梯度被它独占；
        #   3) 这也是防 reward hacking 的一道闸 —— 别让策略发现「狂刷某个句式就能爆分」。
        return max(min(score, 3.0), -3.0)


# ===== 数学表达式安全求值：替代 eval，只放行算术运算与 math 白名单（长度上限 512） =====
def safe_math_eval(expression):
    """对模型生成的数学表达式求值：支持 + - * / // % **、math 白名单函数与 pi/e/tau 常量。

    参数：
        expression: str，模型输出的表达式，如 '2^10 + sqrt(16)'、'3 × 4 ÷ 2'。

    返回：
        int 或 float（不是 Tensor）。

    设计理由：
        agent 的 calculate_math 工具会把模型生成的字符串直接拿来算。若用 eval，
        模型（或一次提示注入）只要输出 __import__('os').system('...') 就是远程代码执行。
        这里改成「先 ast.parse 成语法树、再逐节点白名单求值」：树上可能出现的节点类型
        被显式列举，遇到没见过的类型直接 ValueError。

    ⚠️ 这是白名单而不是黑名单：宁可算不出来，也绝不放过去。
    """
    def pow_guard(base, exp):  # 幂运算统一走这里，拦截 9**9**9 / 10**99999 这类把进程算死的输入
        """带结果上限的幂运算。

        ⚠️ 为什么单独防 **：大整数幂是最廉价的 DoS —— '9**9**9' 一共只有 7 个字符，
        但 Python 要算的是一个约 3.7 亿位十进制的大整数，直接把进程卡死好几分钟。
        两道闸：
            abs(exp) > 1e4                     -> 指数本身已经大得离谱；
            abs(exp) * log10(|base|) > 100     -> 结果超过约 1e100（用对数估算，避免真的算出来）。
        注意第二个条件要求 |base| > 1：底数是 0/1 时结果不会爆炸，log10 也不该参与判断。
        """
        if abs(exp) > 1e4 or (abs(base) > 1 and abs(exp) * math.log10(abs(base)) > 100): raise ValueError('幂运算结果过大')
        return base ** exp
    def resolve(node):  # 同时兼容 sqrt(4) 与 math.sqrt(4) 两种写法
        """把 Name / Attribute 节点统一成一个「名字字符串」。

        - Name('sqrt')            -> 'sqrt'
        - Attribute(math.sqrt)    -> 'sqrt'
        - 其它                    -> None（调用方查表时会落进 ValueError）
        """
        if isinstance(node, ast.Name): return node.id
        # 只认「math.xxx」这种属性访问；node.value 不是 Name（比如 foo().bar）时 getattr 返回 ''。
        if isinstance(node, ast.Attribute) and getattr(node.value, 'id', '') == 'math': return node.attr
    def walk(node):
        """递归求值：只认常数、常量名、一元/二元运算、白名单函数调用，其它一律报错。"""
        # 💡 用 type(node.value) in (int, float) 而不是 isinstance(...)：
        # 因为 bool 是 int 的子类（isinstance(True, int) 为 True），用 isinstance 会把
        # True 当成 1 放进来；complex 也会被误放。白名单必须用「精确类型」。
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        name = resolve(node)
        if name in consts: return consts[name]
        if isinstance(node, ast.UnaryOp): return unary_ops[type(node.op)](walk(node.operand))
        # 先递归求左右子树再套算子；字典里没有的算子（如 <<、&）会抛 KeyError。
        if isinstance(node, ast.BinOp): return bin_ops[type(node.op)](walk(node.left), walk(node.right))
        # not node.keywords：只允许位置参数。带关键字参数的调用直接拒掉 ——
        # 白名单里的函数全都能用位置参数表达，没必要再给关键字留一个可能绕过检查的口子。
        if isinstance(node, ast.Call) and not node.keywords: return funcs[resolve(node.func)](*map(walk, node.args))
        # 走到这里说明遇到了不在白名单里的语法（Subscript、Lambda、ListComp、Attribute...）。
        raise ValueError(f'不支持的表达式语法: {type(node).__name__}')
    # 💡 闭包晚绑定：walk 里引用的 consts / funcs / bin_ops / unary_ops 是在下面才定义的，
    # 但函数体要到真正调用时才求值，那时这些表都已就位，所以可以「先用后定义」。
    consts = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
    # 'pow' 被覆盖成带守卫的版本；其余函数用 getattr(math, 同名) 直接拿，
    # 这份字符串白名单里没有 abs / min / max / round，更没有 __builtins__。
    funcs = {'pow': pow_guard, **{n: getattr(math, n) for n in 'sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil trunc fabs fmod hypot gcd degrees radians'.split()}}
    # 算子表：只映射到这几个安全的算术算子。ast.Pow 也指向 pow_guard，所以 ** 走同一道闸。
    # 注意 ast.Div -> truediv（1/2 = 0.5），ast.FloorDiv -> floordiv（5//2 = 2）。
    bin_ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: pow_guard}
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    # 模型很爱输出这些「人味」符号：^ 表示幂、× ÷ − 是排版符号（− 是 U+2212 真减号，
    # 不是连字符）、² ³ 是上标、括号可能是全角。先统一翻译成 ASCII，否则 ast.parse 直接 SyntaxError。
    chars = str.maketrans({'^': '**', '×': '*', '÷': '/', '−': '-', '²': '**2', '³': '**3', '（': '(', '）': ')'})
    # str(...) 容忍非字符串入参（模型的参数有可能是数字）；先翻译再 strip。
    expr = str(expression).translate(chars).strip()
    # 长度闸：既防「一亿层括号」把递归打爆（512 字符最多 256 层括号，远低于默认递归上限 1000），
    # 也顺手挡掉空表达式。
    if not expr or len(expr) > 512: raise ValueError('表达式为空或过长')
    try:
        # mode='eval' 只允许「一个表达式」：赋值、import、for、多语句全部在解析阶段就被
        # SyntaxError 挡住 —— 这是第一道、也是最强的闸，比任何黑名单都可靠。
        return walk(ast.parse(expr, mode='eval').body)
    except (KeyError, SyntaxError, TypeError):
        # KeyError   : 函数名 / 算子不在白名单（funcs[name] / bin_ops[op] 取不到）
        # SyntaxError: 语法不合法，或含语句而非表达式
        # TypeError  : 参数个数不对（如 sqrt(1, 2)）
        # 统一转成 ValueError，给上层一个可读的错误；from None 抹掉内部 traceback，
        # 不把实现细节漏给调用方。
        # ⚠️ ValueError 本身（包括 pow_guard 抛的「幂运算结果过大」）不会被这里捕获，
        #    会原样抛给上层 —— 这是有意为之，那条信息对调用方更有用。
        raise ValueError('不支持的表达式语法') from None

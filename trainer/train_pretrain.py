# ==============================================================================================
# MiniMind · 预训练（Pretraining）
# ==============================================================================================
# 作用：在纯文本语料上做 next-token prediction 的自回归语言建模，把随机初始化的
#       MiniMind（D=768 / 8 层 / V=6400 / 约 64M 参数）训练成一个「会续写」的基座模型。
#
# 在流水线中的位置：
#   上游 ← BPE 分词器（../model，V=6400，由 train_tokenizer.py 训练得到）
#          + dataset/pretrain_t2t_mini.jsonl 纯文本语料
#   下游 → ../out/pretrain_{D}[_moe].pth，供 train_full_sft.py 用 --from_weight pretrain 载入
#
# 核心思想：
#   1. 自监督：文本自己就是标签（labels 复制 input_ids，只把 pad 涂成 -100），
#      模型只学 P(next token | prefix)，零人工标注，语料越多越好。
#   2. 预训练只学「语言」，不学「听话」：产物更像一个超强输入法，必须再接 SFT
#      才变成 chat 模型——这也解释了为什么 SFT 的学习率要小一个多数量级。
#   3. 工程上用 bf16 autocast + 梯度累积，把「32×8 = 256 条序列」的有效 batch
#      塞进单张消费级显卡，再用 clip + 余弦退火撑住 5e-4 这种偏大的学习率。
#
# 关键张量/数据：
#   input_ids : [B, T] long，B=batch_size=32（单卡 micro-batch），T=max_seq_len=340
#   labels    : [B, T] long，与 input_ids 同值；pad 位置被置成 ignore_index=-100
#   hidden    : [B, T, D]，D=hidden_size=768（模型内部）
#   logits    : [B, T, V]，V=vocab_size=6400；内部切 [B, T-1, V] 去对齐 labels[:, 1:]
#   loss      : 标量，-100 位置被 cross_entropy 跳过；日志里要乘回 accumulation_steps 才是真实值
#   aux_loss  : 标量，仅 --use_moe 1 时的路由负载均衡损失，dense 模型恒为 0
#
# 阅读建议：
#   先顺着文件底部 __main__ 的 9 个步骤读一遍（跑起来的主干全在那里），
#   再回头看 train_epoch 里「前向 → 反传 → 累积 → clip → scaler.step」这十几行；
#   wandb(swanlab)、续训恢复、torch.compile 三块可以先跳过，不影响理解训练本身。
# ==============================================================================================

import os
import sys

# 这两行是「以脚本方式运行」的补丁：此时 sys.path[0] 是 trainer/ 而不是仓库根目录，
# 直接 import model.xxx / dataset.xxx 会 ModuleNotFoundError。
# 手动把上一级目录（仓库根）追加进 sys.path，同时把 __package__ 设成 "trainer" 让相对包语义成立。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ⚠️ 常见坑：这一行必须排在 import torch 之前。Windows 下 pyarrow 与 torch 会争抢同名 DLL，
# 先把 datasets/pyarrow 的 DLL 加载好，可以规避后续 import torch 时崩溃（详见 issue #771）。
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
# torch.distributed 只有在 torchrun 启动、且设置了 RANK/LOCAL_RANK 环境变量时才真正初始化
import torch.distributed as dist
# nullcontext 是一个「什么都不做」的上下文管理器，用来在 CPU 上顶替 autocast，保持代码结构统一
from contextlib import nullcontext
from torch import optim, nn  # nn 在本文件里没有直接用到，保留是为了与同目录其它 trainer 脚本的 import 结构一致
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# ==============================================================================================
# 预训练到底在学什么？（一个必读的小节）
# ==============================================================================================
# 一句话：学 P(next token | prefix)，即「给定前 t 个 token，第 t+1 个 token 的概率分布」。
# 具体到这套代码，一条样本是这样被造出来的：
#   1. PretrainDataset 把一段文本编码成  [BOS] t1 t2 ... tn [EOS] [PAD] ... 并补齐到 max_seq_len；
#   2. labels 直接复制 input_ids，只有 PAD 位置被改成 -100（不参与 loss）；
#   3. 模型 forward 内部做右移：用 logits[..., :-1, :] 去预测 labels[..., 1:]，
#      于是「第 i 个位置的输出」负责预测「第 i+1 个 token」，这就是 teacher forcing；
#   4. 交叉熵在 [B*(T-1), V] 上求平均，-100 的位置被跳过，
#      模型因此被逼着把「下一个词该是什么」的概率越调越准。
# 所以预训练的目标函数就是最朴素的下一 token 交叉熵，没有任何花活；它的全部魔力来自数据规模。
# 💡 作者巧思：labels 不做 shift，右移放在模型里做。好处是 dataset 只需要返回两个【形状相同】的
#    张量，SFT/DPO/PPO 等下游脚本可以复用完全一样的 (input_ids, labels) 接口，
#    只是把「不想让模型学的 token」涂成 -100 而已。这比 nanoGPT 里在 get_batch 中
#    现场做 x[:, :-1] / y[:, 1:] 更通用。
# ==============================================================================================


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """跑完一个 epoch（或从 start_step 续跑到 epoch 末尾）。

    参数:
        epoch: 当前 epoch 下标，从 0 开始。只用于日志显示 [epoch+1/总epochs] 和换算全局 step。
        loader: DataLoader，每次产出一个 (input_ids, labels) 二元组，两者形状都是 [B, T]，
                dtype=torch.long。B 是单卡 micro-batch，不是有效 batch。
        iters: 本 epoch 的「总 step 数」，即 micro-batch 的个数（不是 optimizer step 数）。
               ⚠️ 梯度累积下，一个 optimizer step 会吃掉 accumulation_steps 个 batch，
               但这里仍然按 batch 计数，因为 step 这个变量语义就是「第几个 micro-batch」。
        start_step: 断点续训时已经跑完的 step 数。enumerate 从 start_step+1 开始，
               于是日志里的 step 与断点前严格连续，余弦退火的进度也接得上。
        wandb: swanlab 模块句柄（接口与 wandb 兼容）；非主进程或未开启时为 None。

    返回:
        None。副作用有二：把梯度写回 model 参数；把日志打印到 stdout（只有 rank0 会打）。

    形状流转:
        input_ids [B, T] --embedding--> [B, T, D] --N 层 Transformer--> [B, T, D]
        --lm_head--> logits [B, T, V]；模型内部把它和 labels 各切一刀，得到 [B, T-1, V] vs [B, T-1]，
        再 view 成 [B*(T-1), V] vs [B*(T-1)]，最后 cross_entropy 得到一个标量 loss。
    """
    start_time = time.time()  # 只用于估算剩余时间 ETA，和训练正确性无关
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 搬到 GPU。因为 DataLoader 用了 pin_memory=True（页锁定内存），这次 H2D 拷贝通常能与计算重叠。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # lr 按【全局 step】算，而不是 epoch 内的 step：epoch*iters + step 才是「一共走了多少 batch」。
        # get_lr 是带下限的余弦退火：lr_t = lr * (0.1 + 0.45 * (1 + cos(pi * t / T)))，
        # t=0 时系数为 1.0（满 lr），t=T 时系数为 0.1（保留 10% 的 lr，不彻底退到 0）。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 手动改 param_group 里的 lr，等价于 scheduler.step()，但省掉了一份 scheduler 状态，
        # 续训时也就不用恢复 scheduler（lr 完全由 step 决定，是纯函数）。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # autocast：只把 matmul / softmax 这类算子降为 bf16 计算，
        # 参数本身和梯度累加仍保持 fp32，兼顾速度与数值稳定。
        # 模型返回的 loss 已经是 batch 内、T-1 个位置上的平均值（标量）。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            # 💡 作者巧思：先除以 accumulation_steps 再 backward。
            # 因为 backward 默认是「梯度累加」而不是「梯度平均」，
            # 若每个 micro-batch 都用完整 loss，累积 K 次后梯度会被放大 K 倍（等效 lr × K）。
            # 提前除以 K，则 K 次累加后的梯度恰好等于「K 个 batch 的平均梯度」，与真实大 batch 等价。
            loss = loss / args.accumulation_steps

        # scaler.scale：先把 loss 乘一个很大的放大系数（初始 65536），再反传。
        # 目的：fp16 的尾数只有 10 位，小梯度容易直接下溢成 0；乘大数后落到可表示区间，反传完再除回来。
        # bf16 的指数位和 fp32 一样宽（8 位），没有下溢问题，所以 dtype=bfloat16 时
        # GradScaler 是 enabled=False 的空操作，这行等价于 loss.backward()。
        scaler.scale(loss).backward()

        # 何时真正更新参数：每累积 accumulation_steps 个 micro-batch 一次；
        # 或者已经到本 epoch 的最后一个 step（step == iters），强制把尾巴上不足 K 个的梯度也消化掉，
        # 否则最后几个 batch 白跑。
        if step % args.accumulation_steps == 0 or step == iters:
            # ⚠️ 顺序不能错：clip 必须在 optimizer.step() 之前，而且必须先 unscale_。
            # 此时 .grad 里还带着 scaler 的放大系数，不先还原，算出来的梯度范数会大 65536 倍，
            # 每个 step 都会被裁到 grad_clip，训练直接崩。
            scaler.unscale_(optimizer)
            # 按【全局 L2 范数】裁剪：如果所有参数梯度拼成一根向量的模长超过 grad_clip，
            # 就整体等比缩小到 grad_clip。防的是个别异常 batch 把参数打飞（预训练时尤其常见）。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # scaler.step 内部会先检查梯度里有没有 inf/nan：有就跳过这次更新，并降低 scale；
            # 没有才真正调用 optimizer.step()。
            scaler.step(optimizer)
            # ⚠️ update 必须在 step 之后：它要根据「这次有没有溢出」来动态调整 scale 系数。
            scaler.update()

            # set_to_none=True：把 .grad 置为 None 而不是写 0。
            # 省一次全参数写带宽 + 省下梯度显存（对 64M 参数约 256MB），
            # 代价是下一步的 .grad 由第一次 backward 直接创建，首步略慢一点点。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 因为前面手动除过 accumulation_steps，这里要乘回来才是「真实 loss」，
            # 否则日志里的 loss 会莫名只有 1/8，容易让人误判训练崩了。
            # ⚠️ 注意它只反映【最后一个 micro-batch】的 loss，不是整个累积窗口的平均值。
            current_loss = loss.item() * args.accumulation_steps
            # MoE 模型额外有一项路由负载均衡损失，要从总 loss 里拆出来单独看：
            # aux_loss 一直很小（系数 5e-4 量级），如果它突然变大，通常意味着专家路由塌缩。
            # dense 模型下模型返回的是 0 维零张量，不是 None，所以这个分支实际不会走到 0.0。
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # ETA 估算：已用时间 / 已跑步数 = 每步平均耗时，再乘剩余步数，整除 60 换成分钟。
            # max(..., 1) 是防除零；用 // 而不是 / 只是为了让打印更干净，不是精密计时。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 保存前切 eval、保存后切回 train：这是导出权重的仪式性写法。
            # 本脚本保存路径里不做前向，其实不影响数值，但保持习惯可以防止以后在此处插入推理代码时被 dropout 坑。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # DDP 包装后真实模型在 .module 里；torch.compile 包装后真实模型在 ._orig_mod 里。
            # 两层都要剥掉，否则 state_dict 的 key 会带上 module. / _orig_mod. 前缀，
            # 之后 init_model 里 load_state_dict 会全部对不上（严格来说 strict=False 会静默跳过，更可怕）。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 每个张量先 half() 再 cpu()：权重文件体积减半（fp32 → fp16），
            # 且 .cpu() 之后这块显存可以立刻释放。推理时再 cast 回需要的 dtype。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另存一份「续训专用」的 ckp 到 ../checkpoints：除了权重，还带 optimizer/scaler/epoch/step，
            # 注意它是全精度 state_dict + fp32 优化器状态，体积比上面那份大得多，只用于续训、不要拿来发布。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            # ⚠️ state_dict 里挂着全部参数引用（虽然已 .cpu()，仍是几百 MB 内存），
            # 显式 del 让 Python 立刻回收，避免保存后内存曲线一路爬升（CPU 端的 OOM 往往就来自这里）。
            del state_dict

        # 手动释放这一轮的大张量。真正占显存的是 res：它把完整 logits [B, T, V] 也挂在里面，
        # 32×340×6400×2 字节（bf16）≈ 139MB，不删的话它会一直活到这个 step 的局部变量被下一轮覆盖。
        # 🔍 对照 nanoGPT：nanoGPT 每个 step 只保留 loss，不保留 logits，所以从不需要这行。
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # 下面这些默认值就是「能在一张 8G 卡上跑通 64M 模型」的量级：
    # batch_size=32 / accumulation_steps=8 → 有效 batch 256 条 × 340 token ≈ 87k token 每次更新。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    # from_weight='none' 表示「不加载任何上游权重，从随机初始化开始」——这正是预训练脚本的默认行为；
    # 下游 SFT/LoRA 脚本会把默认值改成 'pretrain' / 'full_sft'，形成一条权重接力链。
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # init_distributed_mode()：如果环境里没有 RANK（不是 torchrun 启动的），直接返回 0，走单卡；
    # 否则 init_process_group(nccl) 并 torch.cuda.set_device(local_rank)，返回本进程的 local_rank。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # ⚠️ 这个 +rank 的偏移很关键：若所有卡用同一个种子，模型初始化、dropout、数据打乱会完全一致，
    #    多卡就等于在重复同一份随机流，白白浪费了「多份独立随机性」；
    #    用 seed+rank 让每张卡有各自的随机流，同时保证「同一组超参 + 同样的 world_size」可完全复现。
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 只传 hidden_size / num_hidden_layers / use_moe，其它超参（头数、head_dim、RoPE theta 等）
    # 都用 MiniMindConfig 里的默认值——这样 out 里的权重文件名只用 {hidden}{_moe} 就能唯一标识结构。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # lm_checkpoint 在不传 model 时是【读取模式】：去 ../checkpoints 找 {weight}_{D}{_moe}_resume.pth，
    # 找不到返回 None。这里只是探测「有没有断点」，真正 load_state_dict 在第 6 步。
    # 💡 它还会比对 ckp 里记录的 world_size 与当前 world_size，若卡数变了会按比例换算 step，
    #    这样 8 卡训练中断、改用 4 卡续训也能接上（见 trainer_utils.lm_checkpoint）。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 上 torch.cuda.amp.autocast 无意义（会报警告），用 nullcontext 顶替，
    # 好处是下面 with autocast_ctx 的代码在 CPU/GPU 上完全不用分叉。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # swanlab 的 API 与 wandb 兼容，所以直接 import 成名字 wandb 来用（国内网络更友好）。
    # ⚠️ 只有 rank0 会初始化并上传，其它 rank 的 wandb 变量保持 None，
    #    日志因此不会重复上报（Logger 与 wandb.log 都只在主进程生效）。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # resume='must' 表示「必须接着这个 run id 写」，若 id 已失效会直接报错而不是悄悄新建一条曲线。
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # init_model：建 MiniMindForCausalLM + 从 ../out/{from_weight}_{D}{_moe}.pth 载入权重（none 则跳过），
    # 最后打印总参数/可训练参数。返回值是 (model.to(device), tokenizer)。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 无 DDP 时给 None，后面每轮自己 randperm 打乱；
    # 有 DDP 时 DistributedSampler 负责把数据集切成 world_size 份，各卡样本互不重叠，
    # 配合每轮 set_epoch(epoch) 更换切分种子，否则每轮每卡看到的样本集合会一模一样。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # ⚠️ enabled 只在 fp16 时开：bf16 不需要 loss scaling（指数位足够宽），
    #    如果 bf16 也开 scaler，反而会引入无意义的乘除与溢出判断。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW：把 weight decay 从梯度里解耦出来（真·衰减，而非 L2 正则），是大模型训练的默认选择。
    # 这里只显式给了 lr，betas 用默认 (0.9, 0.999)；没有 warmup，靠余弦退火 + clip 兜底。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # ⚠️ 顺序坑：必须先把 model/optimizer/scaler 都建好才能 load；
        #    而且 load 必须在第 7 步 DDP 包装【之前】——DDP 会给参数名加上 module. 前缀，之后就对不上了。
        # load_state_dict 按「参数自身的 dtype」拷贝，所以不管 ckp 里是 fp16（out/*.pth）还是
        # fp32（这里的 resume 文件），都不需要手动转换，会自动对齐。
        model.load_state_dict(ckp_data['model'])
        # 优化器状态也要恢复：AdamW 的 exp_avg / exp_avg_sq 决定了每个参数的「动量」，
        # 只恢复权重不恢复优化器，等于让模型带着惯性重新起步，续训初期 loss 会明显抖一下。
        optimizer.load_state_dict(ckp_data['optimizer'])
        # GradScaler 的 scale 系数是动态调整出来的，也要一起恢复，否则 fp16 下可能要重新爬升。
        scaler.load_state_dict(ckp_data['scaler'])
        # 💡 epoch/step 存的是「最后完成的那个 step」的位置，所以下面 range(start_epoch, ...) 会
        #    重新进入同一个 epoch，并用 skip 跳过已经跑过的 batch，而不是从下一个 epoch 开始。
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    # 顺序：先 compile 再 DDP。compile 返回的是 OptimizedModule，把原模型藏在 _orig_mod 里；
    # DDP 只关心 backward 时 all-reduce 梯度，对 forward 有没有被编译无感。
    # ⚠️ 一旦 compile，state_dict 的 key 会变成 _orig_mod.xxx，所以保存时必须先拆包（见 train_epoch）。
    # 🔍 对照 nanoGPT：那里没有 compile、没有 DDP，只有一个裸 model，所以保存时直接 model.state_dict()。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # DDP：每张卡各自 forward/backward 一份 micro-batch，反传时自动 all-reduce 平均梯度，
    # 因此「有效 batch = batch_size × world_size × accumulation_steps」。
    # device_ids=[local_rank] 限定 NCCL 只用本卡通信，多机多卡时也不会拿错设备。
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 每轮都要调用：DistributedSampler 用 epoch 当种子重新 shuffle + 重新切分，
        # 忘了这行的话，每轮每张卡看到的样本顺序完全相同（等于数据没打乱）。
        train_sampler and train_sampler.set_epoch(epoch)
        # 重新播种再 randperm：保证「同样的 seed + 同样的 epoch」得到同样的打乱顺序，可复现。
        # 💡 注意 DDP 分支下 indices 其实用不上（下面 train_sampler or indices 会短路取 sampler），
        #    这段 randperm 是为非 DDP 的单卡路径服务的。
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 只有「恢复后的第一个 epoch」才需要跳过已跑的 batch；之后每个 epoch 都从头开始。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler 把一个「索引来源」（DistributedSampler 或纯 list）按 batch_size 切块，
        # 并丢掉前 skip 个 batch。于是断点续训时，损失曲线的横轴 step 能无缝接上。
        # ⚠️ 它按【batch】跳过，所以续训时 accumulation_steps / world_size 必须与中断时一致，
        #    否则 skip 个 batch 对应的 optimizer step 数会变（lm_checkpoint 只帮我们换算 step 的卡数影响）。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # num_workers=8 起多进程读数据；pin_memory=True 用页锁定内存，H2D 拷贝会更快。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 💡 因为 loader 已经砍掉了 skip 个 batch，总 step 数要加回来（len(loader) + skip），
            #    否则 iters 变小，余弦退火的分母和日志里的 (step/iters) 都会算错。
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    # barrier：等所有卡都跑完最后一次 all-reduce / 保存，再一起退出。
    # 少了它，rank0 可能先退出并把进程组销毁，其它卡还在等通信，会看到 NCCL 超时或直接卡死。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ==============================================================================================
# MiniMind · 全参指令微调（Full-parameter SFT, Supervised Fine-Tuning）
# ==============================================================================================
# 作用：拿预训练好的基座模型，在「多轮对话」数据上继续做 next-token 训练，
#       但只对 assistant 自己说的话算 loss，把「会续写的语言模型」改造成「会听话的助手」。
#
# 在流水线中的位置：
#   上游 ← ../out/pretrain_{D}[_moe].pth（由 train_pretrain.py 产出）
#          + dataset/sft_t2t_mini.jsonl（多轮对话，经 chat template 渲染成一段文本）
#   下游 → ../out/full_sft_{D}[_moe].pth，供 train_dpo.py / train_lora.py / eval_llm.py 载入
#
# 核心思想：
#   1. 目标没变，还是 next-token 交叉熵；变的是「标签怎么来」：预训练里所有文本都是标签，
#      这里只把 assistant 的回复当标签，system/user 以及 "assistant\n" 这个头部全部涂成 -100。
#      于是模型从「学语言」转向「学 P(response | prompt)」——同样的算力，只优化真正想让它说的那部分。
#   2. 为什么必须 mask：如果不 mask，模型会花大量容量去「复述用户的问题」，
#      而推理时 prompt 本来就是已知输入，预测它没有意义；更糟的是会让回复被 prompt 的措辞带跑。
#   3. 为什么 lr 只有 1e-5（预训练是 5e-4，差 50 倍）：SFT 是在一个已经收敛的流形上做「小幅偏移」，
#      lr 一大就会把预训练学到的知识结构打散（catastrophic forgetting），模型很快过拟合到对话风格上。
#
# 与 train_pretrain.py 的差异（逐条对照，三个脚本骨架几乎一模一样，差别全在这些常数上）：
#   数据      : PretrainDataset → SFTDataset
#               前者：纯文本 → [BOS] ... [EOS] [PAD]，labels=input_ids，只把 PAD 涂 -100；
#               后者：jsonl 的 conversations → apply_chat_template 渲染成一段文本，
#                    再用 generate_labels 扫描出「每段 assistant 回复」并只为它填标签，其余保持 -100。
#   loss mask : 预训练只屏蔽 pad；SFT 还屏蔽 system / user / 工具调用声明 / "assistant\n" 头部，
#               只学回复正文（含结尾的 <eos>）——这就是术语表里的 completion_mask / response_mask。
#   lr        : 5e-4 → 1e-5（见上面核心思想第 3 条）
#   batch     : 32 → 16（序列从 340 拉长到 768，显存换长度）
#   accumulation: 8 → 1（micro-batch 变小、步数变多，不再需要用累积凑大 batch；
#               代价是每个 optimizer step 只看 16 条样本，SFT 数据本来就只有几万条，够用）
#   epochs    : 2 → 2（两者相同；真正变多的是 train_lora.py 的 10——只训 0.4M 参数、数据又少，
#               需要更多轮次才能把 LoRA 分支的 B 矩阵从 0 拉起来）
#   max_seq_len: 340 → 768（对话上下文更长；预训练语料是短句，340 足够）
#   from_weight: none → pretrain（只有一个字符串的差别，却决定了整条流水线的接力关系）
#   权重保存  : 全量 state_dict（与预训练相同）；只有 train_lora.py 才只存 LoRA 权重
#
# 关键张量/数据：
#   input_ids : [B, T] long，B=batch_size=16，T=max_seq_len=768，PAD 补齐
#   labels    : [B, T] long，只有 assistant 回复段是真 token id，其余全是 -100（约 90% 的位置被屏蔽）
#   logits    : [B, T, V]，V=6400；内部切 [B, T-1, V] 去对齐 labels[:, 1:]
#   loss      : 标量，只对 mask 为 1 的位置求平均（分母是「有效 token 数」，不是 B*T）
#   aux_loss  : 标量，仅 --use_moe 1 时的路由负载均衡损失
#
# 阅读建议：
#   这个文件与 train_pretrain.py 的代码差异只有 3 行（数据类、超参默认值、lm_checkpoint 的换行），
#   建议先看「与 train_pretrain.py 的差异」这一节 + 第 5 步的 SFTDataset，训练循环本身可以跳过；
#   真正值得逐行读的是 train_epoch 里的 mask 语义与 scaler 顺序，以及第 6 步续训、第 7 步包装。
# ==============================================================================================

import os
import sys

# 以脚本方式运行时，sys.path[0] 是 trainer/ 而非仓库根目录，手动把上一级加进来，
# 否则 import model.xxx / dataset.xxx / trainer.xxx 都会失败。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ⚠️ 常见坑：必须排在 import torch 之前。Windows 下 pyarrow 与 torch 会争抢同名 DLL，
# 先让 datasets/pyarrow 把 DLL 加载好，可以规避后续 import torch 时崩溃（详见 issue #771）。
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
# nullcontext 是「什么都不做」的上下文管理器，用来在 CPU 上顶替 autocast，保持代码结构统一
from contextlib import nullcontext
from torch import optim, nn  # nn 本文件未直接使用，保留以与同目录其它 trainer 脚本保持一致
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# ==============================================================================================
# SFT 的 loss mask 到底屏蔽了什么？（读代码前先建立这张图）
# ==============================================================================================
# 一条 SFT 样本经过 chat template 之后，长这样（示意，实际由 tokenizer 的模板决定）：
#   [BOS] system 你是一个AI助手 <eos> [BOS] user 帮我写个快排 <eos>
#   [BOS] assistant <回复正文> <eos> [PAD] [PAD] ...
# SFTDataset.generate_labels 会扫描 "BOS + assistant\n" 这个分界串，从它【之后】一直填到 <eos>，
# 其它位置全部保持 -100。所以：
#   - system / user 段：-100，不算 loss（推理时它们是输入，不需要被预测）
#   - "assistant" 这个角色头：-100，否则模型会学会「自己喊角色名」
#   - 回复正文 + 结尾 <eos>：真实 token id，算 loss（<eos> 必须学，否则模型不会停）
#   - PAD：-100（labels 初始化就是 -100，天然被屏蔽）
# 交叉熵的 ignore_index=-100 会把这些位置从分子分母里一起剔除，
# 所以 loss 的实际含义是「平均每个【回复 token】的负对数似然」，而不是整条序列的平均。
# 💡 作者巧思：mask 放在 dataset 里做，训练循环一行都不用改——预训练脚本和 SFT 脚本的
#    train_epoch 才能保持完全一致，区别只在数据。这也是 HF transformers 的通行做法。
# ⚠️ 常见坑：如果哪天 mask 写错（比如把 user 段也放出来），训练 loss 会明显更低、收敛更快，
#    但模型会开始复读用户的问题——看到这种「反常的好 loss」，先怀疑 mask。
# ==============================================================================================


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """跑完一个 epoch（或从 start_step 续跑到 epoch 末尾）。

    和 train_pretrain.py 的同名函数逐行一致，唯一区别在数据侧：loader 吐出来的 labels
    已经被 SFTDataset 涂过 -100，所以这里的 loss 天然只统计 assistant 回复。

    参数:
        epoch: 当前 epoch 下标（从 0 开始），用于日志显示与全局 step 换算。
        loader: DataLoader，每次产出 (input_ids, labels)，形状都是 [B, T]，dtype=long。
                labels 里约 90% 的位置是 -100，真正参与 loss 的只有 assistant 回复那几个 token。
        iters: 本 epoch 的 micro-batch 总数（注意不是 optimizer step 数）。
        start_step: 断点续训时已跑完的 step 数；enumerate 从 start_step+1 开始接上。
        wandb: swanlab 模块句柄（wandb 兼容接口），非主进程/未开启时为 None。

    返回:
        None。副作用是把梯度写回 model、把日志打到 stdout（仅 rank0）。

    形状流转:
        input_ids [B, T] --embedding--> [B, T, D] --8 层 Transformer--> [B, T, D]
        --lm_head--> logits [B, T, V]；模型内部切出 [B, T-1, V] 与 labels[:, 1:] 求 CE，
        其中 labels 为 -100 的位置被 ignore_index 跳过，最后得到标量 loss。
    """
    start_time = time.time()  # 只用于估算剩余时间 ETA
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # [B, T] 的两块 long 张量上卡；真正的 [B, T, D] 隐藏态在模型内部才展开
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # lr 按全局 step 算：epoch*iters + step。T=args.epochs*iters 是总步数。
        # get_lr 是带下限的余弦退火：lr_t = lr * (0.1 + 0.45 * (1 + cos(pi * t / T)))，
        # 从满 lr 平滑降到 10% lr。对 SFT 来说 1e-5 → 1e-6，属于温和收尾。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 直接改 param_group，省掉 scheduler 对象及其状态序列化；lr 是 step 的纯函数，续训也不怕。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # autocast 只把 matmul/softmax 降到 bf16 计算，参数与梯度累加仍是 fp32。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # 总 loss = 语言建模 loss + MoE 路由负载均衡 loss（dense 模型后者恒为 0）
            loss = res.loss + res.aux_loss
            # 💡 先除再反传：backward 是「累加」梯度，不除的话 K 次累积会让梯度放大 K 倍（等效 lr×K）；
            # 除完 K 次累加正好等于 K 个 batch 的平均梯度，与真的用了 K 倍大 batch 等价。
            # SFT 默认 accumulation_steps=1，这行退化成除以 1，但仍保留是为了超参可调。
            loss = loss / args.accumulation_steps

        # fp16 下先把 loss 放大再反传，防止小梯度下溢成 0；bf16 的指数位够宽，
        # GradScaler 是 enabled=False 的空操作，这行等价于 loss.backward()。
        scaler.scale(loss).backward()

        # 每累积 accumulation_steps 个 micro-batch 更新一次；step==iters 时强制收尾，
        # 把最后一个不满窗口的梯度也用掉。
        if step % args.accumulation_steps == 0 or step == iters:
            # ⚠️ 必须 unscale 后才能 clip：此时梯度还带着放大系数，
            # 直接算范数会大出几个数量级，导致每个 step 都被裁到 grad_clip。
            scaler.unscale_(optimizer)
            # 全局 L2 范数裁剪，防单个异常 batch 把参数打飞。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # scaler.step 内部先检查 inf/nan，溢出就跳过本次更新并降低 scale；
            # 然后 update 再根据「是否溢出」调整 scale。两步顺序不能反。
            scaler.step(optimizer)
            scaler.update()

            # set_to_none=True 用「置 None」代替「写 0」：省一次全参数写带宽 + 省梯度显存。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 乘回 accumulation_steps 才是真实 loss（否则日志里只有 1/K）；
            # ⚠️ 它只代表最后一个 micro-batch，不是整个累积窗口的均值。
            current_loss = loss.item() * args.accumulation_steps
            # 把 MoE 的 aux_loss 单独拆出来看：它突然变大通常意味着专家路由塌缩。
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # ETA：已用时间/已跑步数 × 剩余步数 ÷ 60。max(...,1) 防除零；// 只是让打印干净。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            # ⚠️ 提醒：SFT 的 loss 数值只对「回复 token」求平均，不能拿来和预训练的 loss 绝对值比较。
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 切 eval 再保存、保存完切回 train，是导出权重的标准仪式（防止保存路径里插入前向时被 dropout 影响）。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 剥掉两层包装：DDP 的 .module 与 torch.compile 的 ._orig_mod。
            # 不剥的话 key 会带 module. / _orig_mod. 前缀，下游 load_state_dict 全部对不上。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 全量保存（SFT 会更新所有参数，不能像 LoRA 那样只存一小撮）。
            # 每个张量 half().cpu()：文件体积减半，且把显存立刻还回去。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 续训专用 ckp（在 ../checkpoints）：额外带 optimizer/scaler/epoch/step/wandb_id，
            # 体积远大于上面那份 fp16 权重，只用于续训，不要拿去发布。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            # 显式删引用，让几百 MB 的 CPU 侧 state_dict 立刻被回收（避免内存曲线缓慢爬升）。
            del state_dict

        # 释放本轮大张量：真正占显存的是 res 里挂着的完整 logits [B, T, V]，
        # 16×768×6400×2 字节（bf16）≈ 157MB，不删就要等到下一轮赋值才释放。
        # 🔍 对照 nanoGPT：那个实现每个 step 只有 loss，没有保留 logits，所以不需要这行。
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    # 与预训练相比：batch 减半、lr 降 50 倍、序列拉长到 768、不再需要梯度累积。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    # 💡 这就是整条流水线的「接力棒」：从 pretrain 权重起步，而不是从随机初始化。
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 没有 RANK 环境变量（非 torchrun 启动）时返回 0，走单卡；否则 init_process_group(nccl) 并 set_device。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # ⚠️ seed+rank：让每张卡有独立的随机流（模型初始化/数据打乱不会四张卡完全一样），
    #    同时保证「同超参 + 同 world_size」时可复现。
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 只暴露 hidden_size / num_hidden_layers / use_moe，其余结构超参用 MiniMindConfig 默认值，
    # 这样权重文件名只用 {D}{_moe} 就能唯一标识一个模型结构。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 不传 model 时是读取模式：找 ../checkpoints/{weight}_{D}{_moe}_resume.pth，没有就返回 None。
    # 💡 若 ckp 里记录的 world_size 与当前不同，lm_checkpoint 会按比例换算 step，支持换卡数续训。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 上 cuda.amp.autocast 无意义，用 nullcontext 占位，让下面的 with 语句在 CPU/GPU 上同构。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # swanlab 兼容 wandb 的 API，直接 import as wandb。只有 rank0 初始化，避免多卡重复上报。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # resume='must'：必须接在该 run id 后面写，id 失效会直接报错（而不是悄悄开一条新曲线）。
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # init_model：建模型并从 ../out/{from_weight}_{D}{_moe}.pth 载入权重（本例即 pretrain 权重），
    # 权重里的未登录词/结构差异会用 strict=False 宽宽松松地跳过；返回 (model, tokenizer)。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 🔑 三者中唯一发生实质变化的就是这一行：换成 SFTDataset。
    # 它内部用 tokenizer.apply_chat_template 把多轮对话拼成一段文本，再 generate_labels 只给
    # assistant 回复打标签，其余位置保持 -100 —— 训练的「魔法」全在这里，不在训练循环里。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 下把数据切成 world_size 份；单卡时为 None，后面每轮用 randperm 自己打乱。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # ⚠️ 只有 fp16 才需要 loss scaling；bf16 传 False，scaler 变成空操作。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 全参微调：优化器收下所有参数（对比 train_lora.py 只收 lora_params）。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # ⚠️ 必须在 DDP 包装之前、且在 optimizer 建好之后 load（否则 key 对不上 / 无对象可 load）。
        # load_state_dict 按参数自身 dtype 拷贝：out/*.pth 是 fp16、这里 resume 文件是 fp32，
        # 两种情况都会自动对齐到模型参数的 fp32，不需要手动转换。
        model.load_state_dict(ckp_data['model'])
        # AdamW 的 exp_avg/exp_avg_sq 是「动量」，不恢复的话续训初期 loss 会明显抖一下。
        optimizer.load_state_dict(ckp_data['optimizer'])
        # GradScaler 的动态 scale 系数也一并恢复，fp16 下不用重新爬升。
        scaler.load_state_dict(ckp_data['scaler'])
        # 💡 存的是「最后完成的那个 step」，所以下面会重新进入同一个 epoch，靠 skip 跳过已跑过的 batch。
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    # 先 compile 再 DDP。⚠️ compile 后 state_dict 的 key 带 _orig_mod. 前缀，保存时必须拆包。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # DDP 在 backward 时 all-reduce 平均梯度，所以有效 batch = 16 × world_size × accumulation_steps。
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 每轮必须重设 epoch，否则 DistributedSampler 的 shuffle/切分种子不变，数据等于没打乱。
        train_sampler and train_sampler.set_epoch(epoch)
        # 重新播种 + randperm，保证打乱顺序可复现；DDP 下 indices 用不上（被 train_sampler 短路）。
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 只有「续训后的第一个 epoch」需要跳过已跑的 batch，之后每个 epoch 从头开始。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler 把索引流按 batch_size 切块并丢掉前 skip 块，让断点续训的 step 曲线无缝衔接。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # num_workers 多进程读数据；pin_memory 用页锁定内存加速 H2D 拷贝。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 💡 loader 已经砍掉 skip 个 batch，所以总 step 数要加回来，否则余弦退火的分母会算错。
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    # barrier 保证所有卡都跑完最后一次通信再退出，否则先退出的进程会销毁进程组，让其它卡 NCCL 超时。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

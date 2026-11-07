# ==============================================================================================
# MiniMind · LoRA 低秩微调（Low-Rank Adaptation）
# ==============================================================================================
# 作用：把基座权重【全部冻结】，只在每个方阵 Linear 旁边挂一对小矩阵 A、B（W'x = Wx + BAx），
#       只训练这 0.4M 参数，就完成一次领域/风格定制（如 lora_medical 医疗问答、lora_identity 自我认知）。
#
# 在流水线中的位置：
#   上游 ← ../out/full_sft_{D}[_moe].pth（SFT 之后的权重）+ dataset/lora_medical.jsonl
#   下游 → ../out/{lora_name}_{D}[_moe].pth（只有 LoRA 权重，约 0.8MB 级，可随手分享）
#          使用方：eval_llm.py --lora_weight、scripts/serve_openai_api.py --lora_weight
#          （推理时动态加载），或 scripts/convert_model.py 的 merge_lora 把 BA 合并回基座
#
# 核心思想：
#   1. 低秩假设：微调带来的权重变化 ΔW 是「低秩」的，用 rank=16 的 BA 去近似就够了。
#      参数量从 768×768=590k 降到 768×16+16×768=24.6k，省 24 倍。
#   2. 零损起步：A 用 N(0, 0.02) 随机初始化、B 初始化为全 0，所以训练开始时 BA=0，
#      模型行为与基座【完全一致】——不会像随机初始化那样一上来就把基座打坏。
#   3. 省的不是算力而是「优化器状态」：基座仍要参与前向/反向（冻结参数没有梯度，但仍要算激活），
#      真正省下的是 AdamW 给每个可训练参数存的 2 份动量（fp32 下每参数 8 字节）：
#      全参 64M → 约 512MB，LoRA 0.39M → 约 3MB。
#
# 与 train_pretrain.py 的差异（逐条对照）：
#   数据       : PretrainDataset → SFTDataset（LoRA 用的也是对话数据，loss mask 同 SFT）
#   lr         : 5e-4 → 1e-4（比全参 SFT 的 1e-5 大 10 倍：只训 0.4M 新参数、B 从 0 起步，
#                梯度信号很弱，需要更大的步长才推得动；同时因为不碰基座，大 lr 也不会毁掉知识）
#   batch      : 32 → 32（与预训练相同；数据是短对话，显存吃得下）
#   accumulation: 8 → 1
#   epochs     : 2 → 10（LoRA 数据只有几千条、可训练参数又只有 0.4M，必须多轮才能把 B 拉起来；
#                轮数多也不容易过拟合基座，因为改动被低秩约束住了）
#   max_seq_len: 340 → 340（与预训练相同）
#   from_weight: none → full_sft（必须站在 SFT 权重上做定制，否则连对话格式都不会）
#   保存       : 只 save_lora 存 LoRA 权重；train_pretrain / train_full_sft 存全量 state_dict。
#                ⚠️ 但 ../checkpoints 里的续训文件仍是【全量】state_dict（含冻结基座），别拿它去发布。
#   冻结/优化器: 新增「冻结非 lora 参数 + optimizer 只收 lora_params + clip 只裁 lora_params」
#   torch.compile: 新增「自动关闭」——apply_lora 用 monkey-patch 改了 module.forward，与编译不兼容。
#
# 关键张量/数据：
#   input_ids : [B, T] long，B=32，T=max_seq_len=340
#   labels    : [B, T] long，只有 assistant 回复段是真 id，其余 -100（同 SFT）
#   x         : [B, T, D]，D=768 —— LoRA 分支看到的输入，A: [D, r]、B: [r, D]，r=lora_rank=16
#   ΔW = B @ A : [D, D]，但只有两个 768×16 / 16×768 的因子是可学的
#   loss      : 标量，只对回复 token 求平均；梯度只流向名字里带 'lora' 的参数
#
# 阅读建议：
#   1. 先看第 5 步（apply_lora → 统计参数 → 冻结）—— LoRA 的全部魔法在这个 20 行里；
#   2. 再看 train_epoch 的签名：多了一个 lora_params 参数，clip 只裁它；
#   3. 最后看第 8 步为什么把 use_compile 强制改回 0。
#   scaler / 梯度累积 / 续训这些与前面两个脚本完全一样，可以快速扫过。
# ==============================================================================================

import os
import sys

# 脚本方式运行时 sys.path[0] 是 trainer/，手动补上仓库根目录，否则 import model.* 会失败。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ⚠️ 必须排在 import torch 之前：Windows 下 pyarrow 与 torch 争抢同名 DLL，先加载可规避崩溃（issue #771）。
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn  # nn 本文件未直接使用，保留以与其它 trainer 脚本一致
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# ==============================================================================================
# 先搞清楚 apply_lora 到底挂在哪些层上（model/model_lora.py 只有 20 行，但很关键）
# ==============================================================================================
# apply_lora 的筛选条件是 isinstance(module, nn.Linear) and module.in_features == module.out_features，
# 也就是「方阵 Linear」。对着 MiniMind 的结构数一遍：
#   q_proj : 768 → 768   ✅ 方阵，挂 LoRA
#   k_proj : 768 → 384   ❌ GQA 只有 4 个 KV 头（4×96=384），不是方阵
#   v_proj : 768 → 384   ❌ 同上
#   o_proj : 768 → 768   ✅ 方阵，挂 LoRA
#   MLP 的 gate/up/down : 768 ↔ 3072   ❌ 不是方阵
#   lm_head : 768 → 6400  ❌ 不是方阵
# 于是 8 层 × 2 个方阵 = 16 个 LoRA 模块，参数量 = 16 × (768×16 + 16×768) = 393,216 ≈ 0.39M，
# 约占 64M 总参数的 0.61% —— 和启动时打印的「LoRA 参数占比」应该对得上。
# 挂载方式是 monkey-patch：setattr(module, "lora", lora) 注册成子模块（所以参数名里带 'lora'），
# 再把 module.forward 换成一个闭包，输出 = 原层输出 + lora(x)。
# ⚠️ 也正因为是「实例属性上的闭包」，torch.compile 无法对它做 graph capture（见第 8 步）。
# 💡 这份实现没有 alpha/r 缩放（标准 LoRA 是 ΔW = (alpha/r)·BA），B 直接加到原输出上，
#    所以没有 lora_alpha 这个「强度旋钮」，强度只能靠 rank 和学习率调。
# ==============================================================================================


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """跑完一个 epoch（或从 start_step 续跑到末尾）。

    与 train_pretrain.py 的同名函数相比只多了一个参数 lora_params，其余逻辑一致：
    冻结的基座参数没有梯度，所以「前向 → 反传 → 累积 → clip → scaler.step」这条链
    自动只会更新 LoRA 参数。

    参数:
        epoch: 当前 epoch 下标（从 0 开始），用于日志与全局 step 换算。
        loader: DataLoader，产出 (input_ids, labels)，形状均 [B, T]，labels 已被 SFTDataset 涂过 -100。
        iters: 本 epoch 的 micro-batch 总数（不是 optimizer step 数）。
        lora_params: 只包含名字带 'lora' 的参数列表。用来做 clip —— 显式限定范围，
                避免把（理论上不存在的）基座梯度也算进范数里。
        start_step: 断点续训时已跑完的 step 数；enumerate 从 start_step+1 接上。
        wandb: swanlab 句柄（wandb 兼容接口），非主进程/未开启时为 None。

    返回:
        None。副作用是把梯度写回 LoRA 参数、把日志打到 stdout（仅 rank0）。

    形状流转:
        input_ids [B, T] --embedding--> [B, T, D] --8 层--> [B, T, D] --lm_head--> [B, T, V]，
        模型内部切 [B, T-1, V] 与 labels[:, 1:] 求 CE；LoRA 只在这些层内部把 [B, T, 768]
        先降到 [B, T, 16] 再升回 [B, T, 768] 加回去。
    """
    start_time = time.time()  # 只用于估算 ETA
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # [B, T] long 上卡；LoRA 不改变数据形状，只改变「哪些参数可学」
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # lr 按全局 step 走余弦退火（带 10% 下限）：lr_t = 1e-4 * (0.1 + 0.45*(1+cos(pi*t/T)))。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 直接写进 param_group，省掉 scheduler；optimizer 里只有 lora 参数，所以只有它们被改 lr。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # autocast 降精度只影响算子内部计算，参数（含 LoRA 的 A/B）仍是 fp32。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            # 💡 先除再反传：backward 是累加梯度，不除会让 K 次累积把梯度放大 K 倍（等效 lr×K）。
            # 本脚本默认 accumulation_steps=1，这行等于除以 1，保留是为了超参可调。
            loss = loss / args.accumulation_steps

        # fp16 下放大 loss 防止梯度下溢；bf16 时 GradScaler 是空操作，等价于 loss.backward()。
        # ⚠️ 冻结参数 requires_grad=False，autograd 不会为它们累积 .grad（省下约 256MB），
        #    但反向传播仍要穿过整个网络——梯度得一路传回靠前层的 LoRA 分支。
        #    所以「LoRA 省显存」主要省在梯度与优化器状态上，不是省在反向计算量上。
        scaler.scale(loss).backward()

        # 每累积 accumulation_steps 个 micro-batch 更新一次；step==iters 时强制收尾。
        if step % args.accumulation_steps == 0 or step == iters:
            # ⚠️ 顺序：先 unscale（还原放大系数）→ 再 clip（按真实范数裁剪）→ 再 step。
            # 少一步 unscale，梯度范数会大几万倍，每步都被裁到阈值，LoRA 直接学不动。
            scaler.unscale_(optimizer)
            # 🔑 只裁 lora_params（对比 SFT 的 model.parameters()）：语义上「只关心我训的这部分」，
            # 工程上也省掉了对 64M 冻结参数的一次遍历筛选（它们 .grad 为 None 会被跳过，但没必要传进来）。
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            # scaler.step 内部先查 inf/nan，溢出就跳过本次更新并降低 scale；update 必须在 step 之后。
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 乘回 accumulation_steps 才是真实 loss；⚠️ 只代表最后一个 micro-batch，不是窗口均值。
            current_loss = loss.item() * args.accumulation_steps
            # MoE 的 aux_loss 拆出来单独看（dense 模型恒为 0）。
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # ETA：平均每步耗时 × 剩余步数 ÷ 60；max(...,1) 防除零。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 切 eval 再保存、保存后切回 train：导出权重的标准仪式。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # LoRA只保存LoRA权重
            # 具体做法（见 model_lora.save_lora）：遍历所有带 .lora 子模块的层，只收集 lora.A/lora.B，
            # key 形如 layers.0.self_attn.q_proj.lora.A.weight。体积约 0.8MB，可以随手分享。
            # 💡 save_lora 内部自己会处理 DDP 的 module. 前缀与 compile 的 _orig_mod，所以这里不用手动拆包。
            # ⚠️ 对外发布只能用这个文件；不能把下一行生成的 resume 文件（含全量基座）当 LoRA 权重发出去。
            save_lora(model, lora_save_path)
            # 续训专用 ckp 存的是【全量】state_dict（基座 + LoRA + 优化器状态），
            # 文件名用 lora_name 区分不同任务的断点，否则多个 LoRA 任务会互相覆盖。
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        # 释放本轮大张量：res 里挂着完整 logits [B, T, V]，32×340×6400×2B（bf16）≈ 139MB。
        # 🔍 对照 nanoGPT：那边每个 step 只有 loss 一个标量，不需要手动清引用。
        del input_ids, labels, res, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    # ⚠️ 注意：本脚本没有 --seed 参数，种子在下面硬编码为 42（见第 1、9 步）。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--lora_rank", default=16, type=int, help="LoRA rank")
    # lora_name 同时决定「输出的 LoRA 文件名」和「续训 ckp 的文件名」，换个任务就要换个名字。
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    # 💡 必须是 full_sft：LoRA 是「在已经会对话的模型上做领域定制」，
    #    若从 pretrain 起步，LoRA 这点参数量根本学不会对话格式。
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # ⚠️ 这里写死 42 而不是 args.seed——LoRA 脚本没有暴露 --seed，可复现性靠这个常量。
    #    +rank 的偏移仍然保留，保证多卡各有独立随机流。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 读取模式的续训探测，key 用 lora_name（不是 save_weight，本脚本没有这个参数）。
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 上 cuda.amp.autocast 无意义，用 nullcontext 占位，让 with 语句在两种设备上同构。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # swanlab 兼容 wandb 接口；只有 rank0 初始化，避免多卡重复上报。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # resume='must'：必须接在该 run id 后续写，id 失效会直接报错。
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    # 先加载 full_sft 全量权重，再「挂」LoRA——顺序不能反：
    # 若先挂 LoRA 再 load_state_dict，基座权重里没有 lora.* 这些 key，strict=False 也只是静默跳过，
    # 虽然能跑，但会让人误以为 ckp 里有 LoRA 权重。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 🔑 apply_lora：给每个「方阵 Linear」注册 lora 子模块并 monkey-patch forward，
    #    新输出 = 原层输出 + lora(x)。此时 B 是全 0，所以模型输出与 full_sft 逐位相等（零损起步）。
    apply_lora(model, rank=args.lora_rank)
    
    # 统计参数
    # total_params：全模型（基座 + LoRA），约 64M；lora_params_count：名字里含 'lora' 的那些，约 0.39M。
    # ⚠️ 一定要在 apply_lora 之后统计，否则数不到 LoRA 参数（它们此刻才被注册成子模块）。
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非LoRA参数，收集LoRA参数
    # requires_grad=False 之后：autograd 不再为这些参数累积 .grad（省约 256MB），optimizer 也不会更新它们，
    # DDP 也不会为它们 all-reduce 梯度（通信量因此极小）。
    # ⚠️ 用名字里是否含 'lora' 来判断，是因为 apply_lora 把子模块命名为 "lora"；
    #    这是「约定优于配置」的写法，改 model_lora.py 里的属性名时这里必须同步改。
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    # LoRA 也用 SFTDataset：同样是对话数据、同样只对 assistant 回复算 loss。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # ⚠️ fp16 才需要 loss scaling，bf16 传入 False 让 scaler 变成空操作。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 🔑 优化器只收 lora_params（对比 SFT 的 model.parameters()）。
    #    这才是 LoRA 省显存的关键：AdamW 不会为 64M 个冻结参数分配 exp_avg / exp_avg_sq。
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # strict=False：容忍键不完全匹配（例如改了 lora_rank 后新增/缺失的 lora.* 键）——
        # 缺失的键保持当前值（基座来自 full_sft、LoRA 是刚初始化的），多余的键直接忽略。
        # ⚠️ 但「形状不一致」仍然会报错，改了 rank 就别指望旧 ckp 能接上，不如直接从头训。
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        # 同前：存的是「最后完成的 step」，所以会重进同一个 epoch，用 skip 跳过已跑的 batch。
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. 编译和分布式包装 ==========
    # ⚠️ torch.compile 与 apply_lora 的 monkey-patch 不兼容：
    #    apply_lora 把 module.forward 换成了实例属性上的闭包（forward_with_lora），
    #    Dynamo 追踪时看到的是「被替换过的 Python 函数对象」而不是静态的 forward 定义，
    #    既可能追踪失败（graph break 甚至报错），也可能悄悄绕过 LoRA 分支。
    #    稳妥做法就是直接关掉它——这里把 args.use_compile 改写为 0，属于纠正参数而不是改训练逻辑。
    if args.use_compile == 1:
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
    # DDP 只 all-reduce requires_grad=True 的参数梯度，所以 LoRA 训练的通信量几乎可以忽略。
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 每轮重设 epoch，让 DistributedSampler 换切分种子，否则每轮数据顺序完全一样。
        train_sampler and train_sampler.set_epoch(epoch)
        # 硬编码 42 + epoch（与第 1 步呼应）；DDP 下 indices 用不上，被 train_sampler 短路。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 只有续训后的第一个 epoch 需要跳过已跑的 batch。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler 按 batch 跳过前 skip 块，让续训的 step 曲线无缝衔接。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # num_workers 多进程读数据；pin_memory 用页锁定内存加速 H2D 拷贝。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 💡 loader 已砍掉 skip 个 batch，总 step 数要加回来，否则余弦退火的分母与日志分母都会错。
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 清理分布进程 ==========
    # barrier：等所有卡跑完最后一次通信再退出，否则先退出的进程会销毁进程组，让其它卡 NCCL 超时。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

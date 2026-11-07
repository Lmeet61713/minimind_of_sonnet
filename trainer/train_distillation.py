# ==========================================================================================
# 作用：知识蒸馏（Knowledge Distillation）训练脚本 —— 让学生在 SFT 数据上同时学习
#       「真实答案的 one-hot 标签」和「教师模型输出的软标签分布」。
# 在流水线中的位置：
#     上游 ← init_model 载入 args.from_student_weight / args.from_teacher_weight（默认都 full_sft）
#             + SFT 数据 ../dataset/sft_t2t_mini.jsonl
#     下游 → ../out/full_dist_{hidden}[_moe].pth：一个更小的学生模型，
#             可被 eval_llm.py 直接加载，也可作为 DPO / LoRA / RL 的新起点
# 核心思想：
#   1. 软标签的信息量远大于 one-hot：teacher 在同一位置给出的整个 [V] 分布，
#      编码了「哪些错答案勉强能接受、哪些错得离谱」，这是 hard label 完全没有的监督。
#   2. temperature 把分布「加热」变平，让原本 1e-8 量级的小概率类别浮出来，
#      否则蒸馏几乎没有梯度；再乘 T^2 把被温度缩小掉的梯度量级补回来。
#   3. 支持异构蒸馏：teacher 可以更深 / 更宽 / 带 MoE，只要词表 V 一致即可 ——
#      学生只对齐「输出分布」，不需要对齐中间层形状，所以蒸馏天然跨架构。
# 关键张量/数据：
#   input_ids: [B, T]（T=max_seq_len=340）；labels: [B, T]，只有 assistant 回复位有值，其余 = -100
#   student_logits / teacher_logits: [B, T-1, V]，V=6400
#   loss_mask: [B, T-1] 取 0/1，1 = 「这个位置该由模型生成」，是 CE 与 KL 共用的 mask
# 阅读建议：
#   先精读 distillation_loss（本文件唯一真正新的东西）→ 再读 train_epoch 里标了
#   「1) CE」「2) Distill」「3) 加权」的三段 → __main__ 与 SFT 脚本几乎一致，
#   重点只看第 5 节「定义学生和教师模型」，其余扫一眼小标题即可。
# ==========================================================================================
import os
import sys

# 把脚本所在目录（trainer/）的上一级（仓库根）塞进 sys.path，这样下面才能
# from model... / from dataset... / from trainer... 三层都 import 得到。
# __package__ 手动设成 "trainer" 是为了让 `python trainer/train_distillation.py`
# 这种直接跑法也拥有正确的包名（否则相对导入和 checkpoint 里的模块名会错）。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
# nullcontext：CPU 上不做 autocast，用「什么都不做的上下文」让代码只有一条路径，
# 省掉 if device_type == 'cuda' 的分支。
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
# 蒸馏复用 SFT 数据集：同样的 chat template、同样的 -100 loss mask，
# 所以这里的「hard label」部分事实上就是一次 SFT。
from dataset.lm_dataset import SFTDataset
# get_lr: 余弦退火（lr → 0.1*lr）；Logger: 只在 rank0 打印；
# lm_checkpoint: 存「权重+优化器+scaler+epoch/step」以便续训；
# init_model: 建模型 + 载权重（strict=False）；SkipBatchSampler: 续训时跳过前 N 个 batch。
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 中文语料 + tokenizer 的 padding 告警会把控制台刷爆，且不影响训练，直接屏蔽。
warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """KL 蒸馏损失（Hinton et al. 2015 的经典形式，只是写成了 KL 而非 CE 的形式）。

    参数:
        student_logits: [N, V] 学生未归一化打分。N 是「有效 token 数」——
                        调用方已经用布尔索引按 loss_mask 筛过，不是 B*T
        teacher_logits: [N, V] 教师未归一化打分，N 与上面逐行对应（同一批位置）
        temperature:    蒸馏温度 T。先 z/T 再 softmax，T>1 让分布变平
        reduction:      'batchmean' / 'sum' / 'mean' / 'none'，透传给 F.kl_div

    返回:
        scalar 张量，已经乘过 T^2 的 KL 散度

    形状:
        [N, V] × [N, V] → scalar

    设计理由:
        1) teacher 只提供「目标分布」不参与反传，所以 softmax 外面套 no_grad：
           不建计算图既省显存，也杜绝了梯度误传到教师参数的任何可能。
        2) 先 log_softmax 再 kl_div，而不是 log(softmax(...))。log_softmax 内部
           用「先减最大值」的实现，不会 exp 溢出，也不会 log(0) 得到 -inf。
           ⚠️ 常见坑：把未归一化的 logits 直接喂给 F.kl_div 会算出无意义的数，
           因为 kl_div 的 input 必须已经在 log 空间。
        3) KL 的方向是 KL(teacher || student)（见下面公式），这正是 Hinton 蒸馏要的：
           梯度里带 teacher_probs 作为权重，教师认为不可能的类别几乎不产生梯度，
           模型把算力集中在「教师认为可能」的那一小撮候选上。
    """
    # 公式（q = teacher 分布，p = student 分布，z 为 logits，T 为温度）：
    #   q_v = softmax(z_t / T)_v ,  p_v = softmax(z_s / T)_v
    #   kl = T^2 * KL(q || p) = T^2 * sum_v q_v * ( log q_v - log p_v )
    # F.kl_div(input, target) 逐元素算的是 target * (log(target) - input)，
    # 所以 kl_div(p_logprobs, q_probs) 得到的正是 KL(q||p)。
    # 又因为 KL(q||p) = CE(q, p) - H(q)，而 H(q) 与参数无关（常数），
    # 所以这等价于「用软标签做交叉熵」——差别只是常数项，梯度一模一样。
    # 目标分布里 q_v = 0 的位置 PyTorch 内部用 xlogy 处理，不会算出 NaN。
    with torch.no_grad():
        # 已经是 no_grad 了，末尾的 .detach() 是双保险：
        # 万一将来这段被挪到 no_grad 外面，它仍然能挡住梯度回流到 teacher。
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生侧必须保留计算图（这里要对它求导），所以只能 log_softmax 不能 no_grad。
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 💡 作者巧思 / 为什么要乘 T^2：
    #   设 p = softmax(z_s/T)，KL 对 z_s 的梯度是 (1/T)*(p - q)。
    #   温度越大，KL 分支的梯度被缩得越厉害；如果直接和 CE 相加，
    #   那「调温度」就等于偷偷改了 KL 项的学习率，alpha 的语义就乱了。
    #   乘上 T^2 恰好把这个 1/T 抵消（净剩下一个 T 的放大），
    #   于是「换温度」只改变软标签的形状，不改变两项之间的相对权重。
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    """跑完一个 epoch：每个 step 前向一次 student、前向一次 teacher，算 CE + KL 的加权和。

    参数:
        epoch:              当前 epoch 下标（从 0 开始），只用于日志和学习率
        loader:             DataLoader，产出 (input_ids [B,T], labels [B,T])
        iters:              本 epoch 的总 step 数；续训时传 len(loader)+start_step，
                            这样 get_lr 里的 total_steps 和「全局 step」才对得上
        teacher_model:      冻结的教师模型；为 None 时退化成纯 CE 的 SFT
        lm_config_student:  学生配置，这里只用来判断 use_moe（决定要不要加 aux_loss）
        start_step:         续训时已经跑过的 step，用于跳过和日志对齐
        wandb:              swanlab/wandb 句柄，None 表示不上报
        alpha:              hard label CE 的权重，总损失 = alpha*CE + (1-alpha)*KL
        temperature:        蒸馏温度 T

    返回:
        None（副作用：更新 model 参数、周期性地存盘）

    设计理由:
        把「一个 step 的全部逻辑」写成一个扁平 for 循环，而不是拆成一堆小函数，
        是为了让刚学完 nanoGPT 的人从上往下顺着读就能看懂一条样本的完整生命周期；
        代价是 args / model / optimizer / scaler 都依赖模块级全局变量（在 __main__ 里定义）。
    """
    start_time = time.time()

    # 教师的 eval + 冻结放在 epoch 入口再确认一次（__main__ 里已经设过）：
    # eval() 关掉 dropout 等随机性，保证同一句话每次蒸馏出的软标签都一样，
    # 否则 target 自己带噪声，KL 就变成了「追一个抖动的水面」。
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 💡 为什么 mask 从 labels 上算，而不是从 input_ids 上算：
        #   数据集已经把「用户提问 + 系统提示」位置的 labels 写成 -100，
        #   所以 (labels != -100) 天然就是「模型该自己生成的 token」。
        #   labels[..., 1:] 与下面 logits[..., :-1, :] 严格对齐
        #   （位置 t 的 logits 预测的正是 labels 里的第 t+1 个 token）。
        #   形状：[B, T] → [B, T-1]
        loss_mask = (labels[..., 1:] != -100).float()
        # 余弦退火：get_lr 返回 lr*(0.1 + 0.45*(1+cos(pi*progress)))，即 lr → 0.1*lr。
        # 这里用「全局 step」= epoch*iters + step，保证跨 epoch 连续下降而不重置。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（学生模型）
        with autocast_ctx:
            res = model(input_ids)
            # 丢掉最后一个位置：它后面没有「下一个 token」可以预测，留着是纯浪费。
            # [B,T,V] → [B,T-1,V]；contiguous() 是必须的，因为后面要 .view(-1, V)，
            # 而切片后张量在内存里不连续，非连续张量上调用 view 会直接报错
            # （换成 reshape 则会偷偷 copy 一份，反而多占显存）。
            # 🔍 对照 nanoGPT：那边是在 loss 里做 logits[:, :-1] 对 targets[1:]，
            #    这里把「切 logits」放在前向之后、「切 labels」放在算 loss 之前，等价。
            student_logits = res.logits[..., :-1, :].contiguous()

        # 教师模型前向传播（只在eval & no_grad）
        if teacher_model is not None:
            with torch.no_grad():
                # ⚠️ 注意 teacher 的前向没有套 autocast_ctx：也就是说教师始终在 fp32 下算。
                #    好处是软标签更准（bf16 只有约 3 位十进制精度，1e-4 以下的小概率会被抹平），
                #    代价是显存和耗时大约翻倍。如果在 24G 卡上塞大 teacher 塞不下，
                #    把这一行挪进 autocast 是「只改精度、不改逻辑」的安全优化。
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                vocab_size_student = student_logits.size(-1)
                # 异构蒸馏的兜底：teacher 词表可能比 student 大，这里按「前 V_student 个
                # id 语义相同」的假设截断对齐：[B,T-1,V_t] → [B,T-1,V_s]。
                # 💡 只要两边用同一个 tokenizer，这一步就是恒等操作；但留着它，
                #    脚本就不会因为 vocab 不一致而在 F.kl_div 处炸一个形状错误。
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss
        # 这里没有直接用 F.cross_entropy(..., ignore_index=-100) 的平均值，而是
        # reduction='none' 再自己加权平均，原因有两个：
        #   a) 同一份 mask 后面还要用来筛蒸馏位置，逐 token 的 ce 得先留下来；
        #   b) 顺手能单独报一个「纯 CE」指标，方便观察蒸馏到底把 loss 拉低了多少。
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)          # [B, T-1] → [B*(T-1)]
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),   # [B*(T-1), V]
            shift_labels.view(-1),                              # [B*(T-1)]
            ignore_index=-100,
            reduction='none'                                    # → [B*(T-1)]，逐 token
        )
        # 只对 mask=1 的位置求平均；+1e-8 是防除零的保险：
        # 万一某个 batch 全是 prompt（没有回复），mask 全 0，0/0 会变 NaN 并污染全局。
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # MoE 的负载均衡辅助损失：不加它，router 会把几乎所有 token 都塞给少数几个 expert，
        # 训练早期就有大批 expert「饿死」变成死参数。dense 学生的 aux_loss 恒为 0。
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        else: ce_loss = ce_loss_raw

        # 2) Distillation Loss
        # ⚠️ 关键一步：布尔索引 [loss_mask_flat == 1] 把 [B*(T-1), V] 压成 [N_valid, V]，
        #    N_valid = 本 batch 里所有回复 token 的总数。这么做有两个后果：
        #    a) prompt 和 padding 位置完全不参与蒸馏（监督范围和 CE 保持一致），
        #       否则会逼学生去模仿 teacher「读题」时的分布，纯属浪费算力；
        #    b) 摊平成「每个 token 一行」之后，KL 的 reduction='batchmean'
        #       正好等于「每个 token 的平均 KL」，量纲和 CE 完全对齐，alpha 才有可比性。
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            # 没有 teacher 时给一个 0.0 常数张量（注意要在 device 上，且是 float），
            # 这样下面 alpha 混合的公式不用写 if-else，代码路径只有一条。
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        #   alpha=1.0 → 退化成普通 SFT（纯 CE，蒸馏完全不生效）
        #   alpha=0.0 → 纯蒸馏（只有软标签，注意此时若 teacher 也没有，loss 恒为 0）
        #   默认 0.5 是折中：CE 保证「答案写对」，KL 保证「整个分布的形状像 teacher」，
        #   后者正是小模型能逼近大模型「手感」的来源。
        # 除以 accumulation_steps 是为了让累积 N 个微 batch 后的梯度和「一次大 batch」等价。
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # GradScaler 只在 fp16 下真正生效（见 __main__ 里 enabled=(dtype=='float16')）；
        # bf16 时 scale / unscale_ / step / update 全是 no-op，保留这套写法只是为了
        # --dtype float16 时脚本能原样跑。
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            # ⚠️ unscale_ 必须排在 clip 之前：GradScaler 把梯度整体放大过 S 倍，
            #    先 clip 等于按 S 倍的阈值裁剪，grad_clip 就形同虚设。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)      # 内部会检查 inf/nan，有就跳过这一步的更新
            scaler.update()             # 动态调整 scale，避免 fp16 梯度下溢成 0
            # set_to_none=True：把 .grad 置成 None 而不是全 0，
            # 省掉一次全参数 memset，顺带让「哪些参数这一步没梯度」变得可查。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # loss 在 backward 前被除过 accumulation_steps，这里乘回去，
            # 日志里报的才是「真实 loss 量级」，方便和单步训练的结果对比。
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            # 剩余时间的粗估：平均每步耗时 × 剩余步数，// 60 换成分钟（整除，忽略余数）。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            # 同时报 ce 和 distill 两个分量，是为了判断「是学生在学知识」还是
            # 「alpha 把某一项压没了」——只盯总 loss 会看不出来是哪一项在降。
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 存盘前切 eval、存完再切回 train，纯粹是为了让 state_dict 的取值确定
            # （有些实现会在 eval 分支里做 weight norm 的缓存合并，这里保持习惯性写法）。
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            # 三层剥壳，顺序不能反：
            #   DDP → .module 拿到原始模型；torch.compile → _orig_mod 拿到未被包装的模型。
            # 忘了任何一层，state_dict 的 key 都会多出 "module." / "_orig_mod." 前缀，
            # 之后 eval_llm.py 用 load_state_dict(strict=False) 才会「看起来能加载但其实全没加载」。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # .half().cpu()：存盘统一转 fp16 并把张量搬回内存，文件体积减半、
            # 也不会在续训 / 推理时把显存钉死；与 eval_llm.py 的加载格式一致。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另一份「可续训」的 checkpoint：额外带 optimizer / scaler / epoch / step，
            # 和上面的纯权重 .pth 是两个不同用途的文件（一个给推理，一个给续训）。
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            # 手动断引用，让这份 CPU 侧的 state_dict 立刻被回收，不等函数返回。
            del state_dict

        # 手动断引用释放显存。这些张量在下一轮 for 循环里会被新值覆盖，
        # 如果不 del，它们的生命周期会一直拖到「下一轮赋值那一刻」，
        # 于是「上一 step 的中间结果」和「这一 step 的新张量」在显存里同时存在，
        # 峰值凭空多出一大截（尤其 res 里还挂着各层 hidden states）。
        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss


if __name__ == "__main__":
    # 模拟用moe模型蒸馏dense模型，也可以用更大teacher_hidden_size模型蒸馏更小student_hidden_size的
    # （默认参数就是第一种玩法：teacher 8 层 + MoE，student 8 层 dense，
    #   两份权重都从 full_sft 出发，所以学生不是从零学，而是「重塑」已有的能力。）
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    # 蒸馏的 lr（5e-6）比 SFT 的 1e-5 还小：学生的起点已经是会聊天的 full_sft，
    # 大 lr 会把这份先验冲掉，只剩「像 teacher」而忘了「怎么说话」。
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    # alpha 混合的是「两个 loss 标量」，不是梯度：先各自算出标量再加权，两个分支的
    # 计算图仍然连在一起，反传时梯度会按 alpha 的权重分别回流，等价于加权求和的目标。
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    # 温度 1.5：比 1 稍高，把 teacher 分布里 1e-3 量级的次优候选抬高到 1e-2 量级，
    # 让学生真的能「看见」它们。太大（>3）会把分布压成近似均匀，蒸馏信号退化成噪声。
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # DDP 由环境变量 RANK 决定是否启用；非 DDP 时返回 0 且不 init_process_group。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 每个 rank 用不同种子，避免所有进程生成出一模一样的 dropout / 数据顺序
    # （DDP 只同步梯度，不同 rank 的随机性必须自己负责错开）。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 两个独立的 config：student 和 teacher 可以层数 / 宽度 / 是否 MoE 完全不同，
    # 这是蒸馏相对「剪枝」的最大优势——不需要两者结构兼容。
    # 注意 hidden_size 决定权重文件名（见 init_model），所以改这里必须改对应的 .pth。
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
    # 续训只在 from_resume=1 时去 ../checkpoints 找「带 optimizer 状态」的那份；
    # 找不到就返回 None，于是 start_epoch/start_step 保持 0，等价于从头训。
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    # CPU 上不支持 autocast，用 nullcontext() 让 with 语句语法保持一致，
    # 避免全文件到处写 if device_type == 'cuda'。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    # 只有 rank0 建 experiment，否则 8 张卡会开出 8 个同名 run。
    wandb = None
    if args.use_wandb and is_main_process():
        # 项目用 swanlab 顶替 wandb，接口兼容，所以这里 import 后直接改名叫 wandb。
        import swanlab as wandb
        # 从 checkpoint 里捞回 wandb_id，续训时就能接上同一条曲线而不是新开一条。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义学生和教师模型 ==========
    # init_model 内部做三件事：建 MiniMindForCausalLM(lm_config) → 从
    # ../out/{from_weight}_{hidden_size}[_moe].pth 载权重（strict=False）→ .to(device)。
    # 学生用 full_sft 初始化（不是随机！），意味着蒸馏的起点已经会对话，
    # 这一步本质是「换一个更小/更便宜的网络去复现原网络的行为」。
    # ⚠️ strict=False 只容忍「key 缺失 / 多余」，张量形状不匹配仍然会报错：
    #    所以 --student_hidden_size 必须与权重文件对应（文件名里就带着这个数，
    #    init_model 是按 {from_weight}_{hidden_size} 拼路径的）；
    #    而 --student_num_layers 可以改小 —— 多出来的层被当成 unexpected key 丢掉，
    #    留下的前 N 层照常加载，相当于一次「浅层迁移 + 新层随机初始化」。
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 教师是完整独立的一份权重，显存占用 = 学生 + 教师（这里教师通常更大）。
    # 第二个返回值 tokenizer 被丢掉：teacher 和 student 必须共用同一个 tokenizer，
    # 否则 V 的每一维不再代表同一个 token，KL 就完全没意义了。
    # 💡 teacher 的 hidden_size / num_layers 跟 student 完全无关，可以随便换：
    #    蒸馏只在「输出分布」这个界面上对齐，不像剪枝那样要求结构兼容。
    #    这也是为什么上面要写 teacher_logits[..., :vocab_size_student] 那句兜底。
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    # 再次 eval + 冻结：init_model 内部不会自动冻结，这里必须显式做，
    # 否则 teacher 的 .grad 会被分配（哪怕 no_grad 下不写入），白占显存。
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DistributedSampler 负责把数据集切成 world_size 份并保证各 rank 不重不漏；
    # 非 DDP 时返回 None，后面用 train_sampler or indices 兜底成普通随机顺序。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 只有 fp16 需要（bf16 的指数位和 fp32 一样宽，不会下溢）；
    # enabled=False 时它退化成一组 no-op，所以下面代码不用分支。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # ⚠️ 只把 model.parameters() 交给优化器：teacher 不在里面，
    # 即使某处漏了 no_grad，教师也永远不会被这一步更新。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    # 三件套要一起恢复：权重、优化器动量、scaler 的 scale 值。
    # 少恢复 optimizer 会让续训前几十步的更新方向「回到初动量」，曲线出现明显拐点。
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    # torch.compile 先包，DDP 后包：反过来 DDP 会先去 trace 一个已被包装的模块，
    # 且存盘时剥壳顺序也要跟着变，容易出错。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # set_epoch 决定 DistributedSampler 这一轮的打乱方式；
        # 不调用它，每个 epoch 的切分完全一样，等于没有 shuffle。
        train_sampler and train_sampler.set_epoch(epoch)
        # 非 DDP 路径：自己按固定种子打乱下标，保证可复现。
        # 注意这里 setup_seed(42+epoch) 会重置全局随机状态（包括 dropout 的随机性），
        # 是「同 epoch 内可复现」的保证。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 续训：只在「第一个要跑的 epoch」跳过已经训过的 batch；
        # 从第二个 epoch 起 skip=0，否则会一路跳到底、再也训不到数据。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 传 batch_sampler 而不是 batch_size，是为了让「跳过前 N 个 batch」这件事
        # 发生在采样层，既不浪费一次前向，也不用在循环里写 continue。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # iters 传 len(loader)+skip：跳过的部分也要计入总步数，
            # 否则学习率会从「中途」重新按余弦曲线起步，出现一次跳变。
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)

    # ========== 9. 清理分布进程 ==========
    # barrier 保证所有 rank 都跑完（否则 rank0 先退出会让 NCCL 通信超时），
    # 再销毁进程组，释放 NCCL 的端口与显存。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

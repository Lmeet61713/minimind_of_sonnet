# ==========================================================================================
# 作用：DPO（Direct Preference Optimization，直接偏好优化）偏好对齐训练脚本 ——
#       用「成对的 chosen / rejected 对话」把 SFT 模型往人类偏好方向推，
#       既不需要 reward model，也不需要在线采样。
# 在流水线中的位置：
#     上游 ← full_sft 权重 + ../dataset/dpo.jsonl（每条样本是同一个 prompt 的两个回答）
#     下游 → ../out/dpo_{hidden}[_moe].pth：更「懂人类偏好」的对话模型，
#            也是 LoRA / PPO / GRPO 等后续对齐实验的常见起点
# 核心思想：
#   1. 把「先训 reward model、再跑 PPO」这套昂贵的 RLHF 流程解析地化简成一个
#      二分类问题：偏好概率可以只用 policy 与 ref 的对数概率差写出来，
#      所以不需要 RM、不需要 critic、不需要 rollout，一份离线数据跑一个 epoch 就够。
#   2. 隐式奖励 r(x,y) = beta * log( pi(y|x) / pi_ref(y|x) )：模型自己就是 reward model，
#      而冻结的 ref_model 负责把策略「锚」在 SFT 的行为附近，等价于一个 KL 约束。
#   3. chosen 和 rejected 拼成一个 batch 只做一次 policy 前向 + 一次 ref 前向，
#      训练成本大约是两次 SFT 前向，是这条流水线上最便宜的对齐步骤。
# 关键张量/数据：
#   x: [2B, T]（T=max_seq_len=1024；前 B 行 chosen，后 B 行 rejected）
#   logits: [2B, T, V]（V=6400）；per_token_logps: [2B, T]；mask: [2B, T]（只对回复位=1）
#   每条回复最后被压成一个标量「log 概率和」→ [2B] → 两两相减得 [B] → mean → scalar
# 阅读建议：
#   先看 dpo_loss 的推导注释（本文件的灵魂，看懂它 DPO 就懂了）→ 再看
#   logits_to_log_probs 里 gather 的形状变化 → 最后看 train_epoch 里 cat / no_grad 两处。
#   __main__ 与 SFT 脚本几乎一样，重点只看第 5 节「定义模型和参考模型」。
# ==========================================================================================
import os
import sys

# 把脚本目录（trainer/）的上一级（仓库根）加进 sys.path，下面的 model / dataset /
# trainer 三层包才 import 得到；__package__ 手动设成 "trainer"，
# 是为了 `python trainer/train_dpo.py` 这种直接运行方式也能拿到正确的包名。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
# nullcontext：CPU 上不做 autocast 时的「空操作上下文」，
# 让 with autocast_ctx 这一行在 CPU / GPU 上写法一致，省掉 if 分支。
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
# DPODataset 每次返回一条样本的 chosen 和 rejected 两套 (x, y, mask)，
# 并且已经在数据集内部把 labels 右移了一位（x = ids[:-1], y = ids[1:]）。
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """从 logits 里取出「每个位置真实 token(labels) 的对数概率」。

    参数:
        logits: [B, T, V] 模型输出的未归一化打分
        labels: [B, T] 每个位置的目标 token id（这里是已经右移过一位的 y）

    返回:
        [B, T] 的 log pi(y_t | x_{<t})，每个位置一个标量

    形状:
        [B,T,V] --log_softmax(dim=2)--> [B,T,V]
                --gather(dim=2, index=labels.unsqueeze(2)[B,T,1])--> [B,T,1]
                --squeeze(-1)--> [B,T]

    设计理由:
        1) 必须用 log_softmax，不能写成 log(softmax(...))：log_softmax 内部先减
           最大值再取 exp/log，bf16 下也不会 exp 溢出或 log(0) = -inf。
        2) gather 是「按索引取数」，它的反向是 scatter：比 one-hot 乘再求和
           省得多 —— one-hot 要额外 O(B*T*V) 显存，gather 只要 O(B*T)。
        3) 🔍 对照 nanoGPT：那边直接 F.cross_entropy(logits, targets)，
           而 cross_entropy 就等于 -logits_to_log_probs(...).mean()。
           这里把「取对数概率」单独摘出来，是因为 DPO / PPO / GRPO 后面要对它
           做加减（算 logratio），而不是求 CE。
    """
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    # 注意 dim=2：在词表维（最后一维）上做归一化，绝不能写成 dim=-1 之外的东西。
    log_probs = F.log_softmax(logits, dim=2)
    # labels.unsqueeze(2)：[B,T] → [B,T,1]，因为 gather 要求 index 与输入同维数。
    # gather(dim=2)：对每个 (b,t) 沿词表维取出第 labels[b,t] 个元素 → [B,T,1]。
    # squeeze(-1)：去掉那个长度 1 的维度 → [B,T]。
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """DPO 的负对数似然损失（Bradley-Terry 偏好模型 + 隐式奖励）。

    参数:
        ref_log_probs:    [2B, T] 冻结参考模型给「真实 token」的逐位置对数概率
        policy_log_probs: [2B, T] 当前策略模型在同一位置的逐位置对数概率
        mask:             [2B, T] 0/1，只对 assistant 自己生成的回复位为 1
                          （prompt、系统提示、padding 位置都是 0）
        beta:             温度 / KL 正则系数，命令行默认 0.15

    返回:
        scalar：一个 batch 里 B 对偏好的平均损失

    形状:
        [2B,T] --乘 mask 再在 T 上求和--> [2B]（每条回复一个 log 概率和）
        --按前后半切分--> 各 [B] --相减--> [B]（每对一个 margin）--mean--> scalar

    设计理由与推导（这是本文件的灵魂，值得慢慢看）:

        第一步，用 mask 加权求和，把整条回复压成一个数：
            log pi(y|x) = sum_t log pi(y_t | x, y_<t)
        注意是「求和」而不是「平均」：语言模型的序列概率本来就等于各 token
        条件概率的连乘，取对数后就变成连加。⚠️ 求和带来的副作用是长回复的
        log 概率绝对值更大、天然权重更高；DPO 原论文也是这么做的，如果想减轻
        长度偏置可以改成除以 mask.sum()，但那就不再是论文里的那个目标了。

        第二步，DPO 假设存在一个「隐式奖励」：
            r(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) ) + beta * log Z(x)
        其中 pi_ref 是 SFT 之后冻结的模型，Z(x) 是只与 x 有关的配分函数。

        第三步，Bradley-Terry 模型假设人类选 chosen 的概率是
            P(y_w > y_l | x) = sigmoid( r(x, y_w) - r(x, y_l) )
        把隐式奖励代进去：beta*log Z(x) 在相减时被完全约掉
        （这正是「chosen 和 rejected 必须来自同一个 prompt x」的根本原因）；
        pi_theta 自己的配分函数也早在推导中被吸收进了 log Z(x)。

        于是：
            loss = -log sigmoid( beta * [ (log pi_theta(y_w|x) - log pi_theta(y_l|x))
                                       - (log pi_ref(y_w|x)   - log pi_ref(y_l|x)) ] )
        令 pi_logratios  = log pi_theta(y_w) - log pi_theta(y_l)
           ref_logratios = log pi_ref(y_w)   - log pi_ref(y_l)
        就是下面代码里的两行相减。

        看到这里应该明白 DPO 为什么不需要 reward model 了：
        「奖励差」被改写成了「策略与参考策略的对数概率比之差」，
        全部都是前向就能算出来的量，一个 RM 都不需要。

    beta 的作用（KL 约束强度）:
        DPO 等价于解 max_pi E_{y~pi}[ r(x,y) ] - beta * KL(pi || pi_ref)。
        beta 就是这个 KL 惩罚的系数：
            beta 大 → 策略被钉在 ref 附近，只敢小步偏移，学得保守但稳；
            beta 小 → 允许离 ref 更远，偏好拟合更激进，也更容易「训崩 / 遗忘」。
        同时它还是 sigmoid 的斜率：beta 越大，同样的 margin 给出越饱和的梯度，
        已分对的样本会更快地「不再产生梯度」。

    数值细节:
        -logsigmoid(z) 用 log(1+exp(-z)) 的稳定实现：z 很大时约为 0，z 很小时
        约为 -z，全程不会 exp 溢出，比自己写 -log(1/(1+exp(-z))) 稳得多。
        z = 0 时 loss = ln2 ≈ 0.693。而训练第一步 policy 和 ref 权重完全相同，
        所以 pi_logratios - ref_logratios 必然为 0 —— 「开局 loss 应该约等于
        0.693」。如果一上来就明显偏离这个数，说明权重没对齐或 mask 写错了。
    """
    # 第一步：把每条回复压成一个「log 概率和」
    # mask 是 0/1（dataset 里是 long，乘 float 会自动提升成 float），
    # 相乘等价于「只保留回复位置，其余位置清零」：[2B,T] * [2B,T] → [2B,T]
    # sum(dim=1)：在序列维上求和 → [2B]，每条回复得到一个 log 概率总和。
    # 💡 这里就是「求和而非平均」的实现处，量纲是 nat（自然对数单位）。
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    # ⚠️ 强假设：一次前向的 2B 行必须是 [chosen × B, rejected × B] 的排布
    #    （train_epoch 里用 torch.cat([chosen, rejected], dim=0) 保证），
    #    且第 i 行与第 i+B 行来自同一个 prompt。所以 args.batch_size 请保持偶数：
    #    一个 micro batch 恰好是 B 个完整 pair（2B 行），切分点 B 才落在 pair 边界上。
    #    一旦有人在 collate 里打乱顺序，chosen 和 rejected 就不再同源，
    #    「配分函数 Z(x) 能约掉」的前提没了，这个 loss 立刻退化成噪声。
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 策略模型「更偏好 chosen 多少」的对数比：[B] - [B] → [B]
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 参考模型原本「已经偏好 chosen 多少」：这是 SFT 模型自带的那点偏好，要减掉
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # 💡 灵魂两行：真正要优化的是「相对于 ref，策略有没有把 chosen 抬得更高」。
    #    如果 chosen 本来在 SFT 下概率就更高，ref_logratios 已经贡献过了，
    #    减掉之后模型只需学「增量」—— 这也是 DPO 不容易破坏已有先验的原因。
    # ⚠️ 命名坑：这个变量叫 logits，但它不是模型 logits，
    #    而是「隐式奖励差」（margin）。别被名字带偏。
    logits = pi_logratios - ref_logratios
    # loss = -log sigmoid( beta * ( pi_logratios - ref_logratios ) )
    # 梯度直觉：d loss / d(beta*logits) = -sigmoid(-beta*logits)
    #   margin 还很负（分错了）→ 梯度接近 -1，全力纠正；
    #   margin 已经很大（分得很对）→ 梯度趋近 0，自动「不再纠结」。
    #   这就是 DPO 不需要采样、也不需要 PPO 式 clip 的原因：
    #   错得越离谱的样本，自带越大的梯度。
    loss = -F.logsigmoid(beta * logits)
    # 每对样本一个标量，再对 batch 求平均 → 与 batch size 无关的稳定量级
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """跑完一个 epoch：每步「拼 batch → policy 前向 → ref 前向(no_grad) → DPO loss → 反传」。

    参数:
        epoch:      当前 epoch 下标，只用于日志和学习率
        loader:     DataLoader，每项是含 x_chosen / x_rejected / y_* / mask_* 的 dict
        iters:      本 epoch 总 step 数；续训时传 len(loader)+start_step 以对齐 get_lr
        ref_model:  冻结的参考模型（与 policy 同结构、同初始权重）
        lm_config:  配置，只用来判断 use_moe（决定存盘文件名的后缀）
        start_step: 续训时已经跑过的 step
        wandb:      日志句柄或 None
        beta:       DPO 的 beta。注意函数默认值 0.1 只是兜底，
                    命令行实际传的是 args.beta（默认 0.15）

    返回:
        None

    设计理由:
        结构与 SFT 的 train_epoch 完全一致（方便对照着读），唯一的新东西是
        「两次前向 + 一个自定义 loss」。注意这里没有 token 级 CE：
        loss 是「每对样本一个标量」再平均，所以数值量级天然比 CE 小得多，
        日志里看到 loss 在 0.6 附近徘徊是正常的（开局 ln2 = 0.693）。
    """
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        # 六个张量全是 [B, T]（B = args.batch_size，T = max_seq_len）：
        #   x = input_ids[:-1]（喂给模型），y = input_ids[1:]（要算概率的目标），
        #   mask = loss_mask[1:]（只保留 assistant 回复位），右移对齐由 DPODataset 完成。
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        # 💡 核心效率技巧：把 chosen 和 rejected 沿 batch 维拼起来，
        #    [B,T] + [B,T] → [2B,T]，这样 policy 和 ref 各只做「一次」前向，
        #    GPU 利用率比跑两次 B 高得多（更大的矩阵乘、更少的 kernel 启动）。
        #    ⚠️ 代价是显存翻倍：logits 会变成 [2B,T,V]，2*4*1024*6400 个 bf16 ≈ 105MB，
        #    加上 ref 那份就是两倍 —— 这也是默认 batch_size 只有 4 的原因之一。
        #    ⚠️ 同时这里也定下了 dpo_loss 里「前一半 chosen、后一半 rejected」的约定。
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 余弦退火，和 SFT 同一套：lr → 0.1*lr；用全局 step 保证跨 epoch 连续。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                # 参考模型只前向、不反传：它不提供梯度，只提供「锚点」。
                # no_grad 让整张计算图不建，[2B,T,V] 的中间激活也就不占计算图内存。
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            # ⚠️ 这一行虽然写在 no_grad 外面，但输入 ref_logits 的 requires_grad 已经是
            #    False，所以 ref_log_probs 同样不带梯度，不会有梯度从 policy 漏到 ref。
            #    （把它挪进 no_grad 里能再省一点点 autograd 元数据，属于可选优化。）
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 策略模型前向 —— 只有这一条路径需要梯度。
            # 🔍 注意 ref 和 policy 用的是同一个 x：同一 prompt 的 chosen/rejected
            #    一起算，才能真正比较「同一个条件 x 下两条回复谁更被偏好」。
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # MoE 的负载均衡损失必须加上，否则 router 会把 token 全塞给少数 expert，
            # 训着训着就塌缩了。dense 模型的 aux_loss 恒为 0（zeros 标量），加了也无害。
            loss = dpo_loss_val + outputs.aux_loss
            # 梯度累积：除以累积步数，让累积 N 个 micro batch 的梯度和「一次大 batch」等价。
            loss = loss / args.accumulation_steps

        # 与 SFT 一致：GradScaler 只在 fp16 下真生效，bf16 时全是 no-op。
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            # ⚠️ unscale_ 必须在 clip 之前，否则 clip 的阈值被放大 S 倍而形同虚设。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)      # 内部检查 inf/nan，有就跳过这一步更新
            scaler.update()             # 动态调整 scale，防 fp16 梯度下溢
            # 置 None 而不是填 0：省一次全参数 memset，也便于观察哪些参数没梯度。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # loss 在 backward 前被除过 accumulation_steps，乘回去才是真实量级。
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            # 剩余时间粗估：平均每步耗时 × 剩余步数，// 60 换算成分钟（整除）。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            # 单独把 dpo_loss 打出来：总 loss 里混了 aux_loss，
            # 只看总数会分不清「偏好没学动」还是「expert 塌缩了」。
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 只有 rank0 存盘，否则 N 个进程会同时写同一个文件，内容互相踩踏。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 三层剥壳，顺序不能反：DDP → .module，torch.compile → _orig_mod。
            # 少剥一层，state_dict 的 key 就会多出 "module." / "_orig_mod." 前缀，
            # 而 eval_llm.py 用 strict=False 加载，会「不报错但什么也没加载上」。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 统一存 fp16 + CPU：体积减半，也不会把显存钉住；与推理端加载格式一致。
            # ⚠️ 参考模型不存 —— 它和 policy 起点相同，需要时用 full_sft 重新 init 即可。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 另一份「可续训」的 checkpoint（带 optimizer / scaler / epoch / step），
            # 和上面的纯权重 .pth 用途不同：一个给推理，一个给断点续训。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 手动断引用、尽早释放显存。不 del 的话这些张量会活到下一轮循环被赋值那一刻，
        # 「上一步的 [2B,T,V] logits」和「这一步的新张量」在显存里同时存在，
        # 峰值会凭空多出一大块，很容易 OOM。
        # 🔍 想更省可以更早 del ref_outputs / logits：算完 log_probs 后它们就没用了，
        #    现在它们会被挂到循环末尾，也就是「反传期间」还占着显存。
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    # 默认只训 1 个 epoch：DPO 是「离线偏好数据 + 极小学习率」的细活，
    # 训多了不但不会更好，还会出现 reward hacking（把 chosen 的概率硬拉高，
    # 语言质量一起崩掉）。
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    # 每个 step 实际喂给模型的是 2*4 = 8 条序列（chosen + rejected），
    # 加上 ref 的额外一份前向，4 已经是 24G 卡上的常见配置。
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    # 🔍 4e-8 小得反直觉，原因有两层：
    #   1) AdamW 的每步更新量级约等于 lr 本身（与梯度大小关系不大，
    #      因为 Adam 用二阶动量把梯度归一化了），所以 4e-8 就是「每步只挪 4e-8」，
    #      几千步下来参数总位移也就 1e-4 量级 —— 刚好是「微调」而非「重训」。
    #   2) DPO 的梯度方向只关心「chosen 比 rejected 高一点」，太大会直接
    #      把 SFT 学到的语言能力冲掉（灾难性遗忘）。
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    # T=1024 比 SFT 的 340 长得多：DPO 数据里 chosen/rejected 是完整对话，
    # 而且「回复长度」本身就是偏好的重要维度，截太短会把差异切掉。
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    # ⚠️ 必须和 policy 用同一份 full_sft：下面 ref_model 也用这个参数初始化，
    #    两者开局完全一致，才能保证第一步的 margin 恰好为 0（loss = ln2）。
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    # beta=0.15 偏小：允许策略离 SFT 参考模型远一点，把偏好信号吃进去；
    # 显存/数据不够时可以适当调小，训不稳（语言质量下滑）就调大。
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 非 DDP 时返回 0 且不建进程组；DDP 时按环境变量 RANK 初始化 NCCL。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 每个 rank 用不同的种子，避免所有进程的数据顺序 / dropout 完全一样
    # （DDP 只同步梯度，随机性必须自己负责错开）。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 只有 from_resume=1 才去 ../checkpoints 找带优化器状态的那份；找不到就返回 None，
    # 于是 start_epoch / start_step 保持 0，等价于从头开始。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    # CPU 上不支持 autocast，用 nullcontext() 让 with 语句语法在两边一致。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    # 只有 rank0 建 experiment，否则 8 张卡会开出 8 个同名 run。
    wandb = None
    if args.use_wandb and is_main_process():
        # swanlab 的接口与 wandb 兼容，import 后直接改名复用，下游代码不用改。
        import swanlab as wandb
        # 从 checkpoint 里找回 wandb_id，续训时能接上同一条曲线。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型和参考模型 ==========
    # init_model = 建 MiniMindForCausalLM(config) + 从 ../out/{from_weight}_{hidden}[_moe].pth
    # 载权重（strict=False）+ .to(device)。policy 从这里开始「边学偏好边被训」。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型（ref_model冻结）
    # 💡 注意 ref 用的是「同一份权重 + 同一个 config」，而不是某个单独的 ref 文件：
    #    DPO 的参考分布就是「SFT 之后、对齐之前」的那个策略，用 full_sft 精确对应。
    #    代价是显存里要多放一整个模型（64M 参数的 fp32 约 256MB，bf16 约 128MB）。
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    # eval() 关掉 dropout 等随机性 —— 参考分布必须是确定的，否则每一步的「锚点」
    # 都在抖动，pi_logratios - ref_logratios 里会混进纯噪声。
    ref_model.eval()
    # 冻结参数：requires_grad_(False) 会递归作用到所有子模块。
    # 这一步和下面的 no_grad 是双保险，也避免优化器/梯度缓冲为它分配显存。
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 下把数据集切成 world_size 份且不重不漏；非 DDP 返回 None。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # fp16 才需要 GradScaler（bf16 指数位与 fp32 同宽，不会下溢），enabled=False 时是 no-op。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # ⚠️ 注意只传了 model.parameters()：ref_model 不在优化器里，
    #    所以哪怕某处漏了 no_grad，ref 也不会被这一步更新。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    # 权重 / 优化器动量 / scaler 三件套要一起恢复，少一样续训曲线就会拐一下。
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    # 先 compile 再 DDP：顺序反了会导致存盘时剥壳顺序也要跟着变，容易错。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 不调用 set_epoch，每个 epoch 的 DDP 切分完全一样，等于没 shuffle。
        train_sampler and train_sampler.set_epoch(epoch)
        # 非 DDP 路径：自己按固定种子打乱下标（分号写法与原脚本保持一致）。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 只在「第一个要跑的 epoch」跳过已训过的 batch；之后的 epoch skip=0，
        # 否则会一路跳到底、再也训不到数据。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 用 batch_sampler 而不是 batch_size：把「跳过前 N 个 batch」下沉到采样层，
        # 既不白跑前向，也不用在训练循环里写 continue。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # iters 传 len(loader)+skip：跳过的那部分也计入总步数，
            # 否则余弦学习率会从「中途」重新起步，出现一次跳变。
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 9. 清理分布进程 ==========
    # barrier 等所有 rank 跑完（否则 rank0 先退会让 NCCL 通信超时），再销毁进程组。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ==============================================================================
# LoRA 低秩微调（model/model_lora.py）
# ==============================================================================
# 作用：
#   给已经 SFT 好的模型“外挂”一组极小的可训练参数（ΔW = B @ A），冻结原权重，
#   只训这一小撮参数。本文件提供四件事：LoRA 模块本身、挂载、保存/加载、合并导出。
#
# 在流水线中的位置：
#   上游 ← trainer/train_full_sft.py 产出的 SFT 权重（.pth），作为冻结的 base 模型；
#           trainer/train_lora.py 调用 apply_lora() 挂载后只训练 LoRA 部分，
#           每个 epoch 用 save_lora() 存一份几 MB 的 lora 权重。
#   下游 → scripts/convert_model.py 用 apply_lora()+merge_lora() 把 LoRA 烘进主权重，
#           导出成普通模型；eval_llm.py 与 scripts/serve_openai_api.py 则用
#           apply_lora()+load_lora() 在推理时动态挂载。
#
# 核心思想：
#   1) 一个 D×D 的全参微调矩阵 ΔW，用两个瘦长矩阵 B[D,r] @ A[r,D] 近似，
#      r=16 时参数量只有原来的 2*r/D ≈ 4%（768 维下每层 24576 vs 589824）。
#   2) B 零初始化 → 训练开始时 ΔW 严格等于 0，模型行为与 SFT 权重一模一样，
#      不会用随机噪声破坏已有能力（这是 LoRA 能“无损接着训”的关键）。
#   3) 因为 y = xW^T + x(BA)^T = x(W+BA)^T，两个分支是线性可加的，
#      所以推理时可以把 BA 直接加进 W，合并后零额外延迟（merge_lora 干的就是这事）。
#
# 关键张量 / 数据：
#   D   = hidden_size = 768；r = rank，默认 16
#   A.weight : [r, D]   = [16, 768]    低秩矩阵 A（nn.Linear(D, r) 的权重）
#   B.weight : [D, r]   = [768, 16]    低秩矩阵 B（nn.Linear(r, D) 的权重）
#   x        : [B, T, 768]   （这里的 B 是 batch，别和矩阵 B 混了）
#   ΔW       : [D, D] = B @ A，形状与 nn.Linear.weight 完全一致，才能直接相加
#   单模块参数量 r*D*2 = 16*768*2 = 24,576；整个模型只挂 q_proj/o_proj
#   → 8 层 * 2 个模块 * 24,576 = 393,216 ≈ 0.39M，占总参数 64M 的约 0.6%
#
# 阅读建议：
#   1) 先看 LoRA 类的 __init__ 与 forward，两行就够，重点是“为什么 B 要是 0”；
#   2) 再看 apply_lora 的筛选条件 in_features == out_features —— 这一点最容易看漏；
#   3) load/save 是纯粹的工程细节，扫一眼即可；merge_lora 的 B@A 公式值得单独记。
# ==============================================================================
import torch
from torch import optim, nn
# 注：optim 在本文件里其实没被用到（train_lora.py 自己 import optim 建优化器），
# 保留它是为了不打断已有的 import 习惯，属于历史遗留，不影响功能。


# 定义Lora网络结构
class LoRA(nn.Module):
    """单个 LoRA 旁路：用 B @ A 这个低秩分解去近似“全参微调的增量矩阵”。

    作用：
        给某个 nn.Linear 并联一条小分支，前向时把两条分支的输出相加：
            y = W x + B A x = (W + B A) x

    参数：
        in_features   被挂载的 Linear 的输入维度（768）
        out_features  被挂载的 Linear 的输出维度（768）
        rank          低秩维度 r，默认 16；r 越小越省、越容易欠拟合

    返回：
        forward 返回 [B, T, out_features]

    形状：
        x        [B, T, 768]
        A(x)     [B, T, 16]     ← 先把 768 维压到 16 维
        B(A(x))  [B, T, 768]    ← 再升回 768 维
        注意 nn.Linear 的 weight 是 [out, in]，所以
            A.weight = [r, D] = [16, 768]
            B.weight = [D, r] = [768, 16]
            B.weight @ A.weight = [768, 16] @ [16, 768] = [768, 768] = ΔW ✓

    设计理由：
        完整的 ΔW 有 D*D = 589,824 个参数；分解成 B@A 后只有 2*r*D = 24,576 个，
        降了 24 倍。论文的假设是“微调带来的变化本身是低秩的”——任务适配只需要
        调整少数几个方向，不需要动整个矩阵。
        🔍 和全参微调（trainer/train_full_sft.py）对照：全参要更新全部 64M 参数，
        优化器还得为每个参数存两份 fp32 状态（Adam 的 m 和 v）；LoRA 只更新 0.39M，
        优化器状态同步缩小约 160 倍，所以同一张卡可以把 batch 开得更大、或者多开几路
        rollout —— 这是它在 RL 阶段（PPO/GRPO 要同时驻留 actor/ref/reward）特别香的原因。
    """
    def __init__(self, in_features, out_features, rank):
        """构造 A、B 两个无 bias 线性层，并按 LoRA 论文的方式初始化。

        参数：
            in_features   768
            out_features  768
            rank          16
        返回：
            None
        形状：
            A.weight [16, 768]，B.weight [768, 16]
        """
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A，768 → 16（降维/投影到子空间）
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B，16 → 768（从子空间升回来）
        # 矩阵A高斯初始化
        # std=0.02 是 transformer 里最常用的初始化量级；A 只要“非零且小”即可，
        # 它的大小不敏感，因为后面 B 会学出合适的缩放
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        # 💡 这是 LoRA 最关键的一行：B=0 使得 ΔW = B@A = 0 严格成立，
        #    所以挂上 LoRA 的那一刻，模型输出和 SFT 权重一模一样，训练不是
        #    “从一个随机扰动过的模型重新开始”，而是“从零增量开始”。
        #    副作用（也是设计的一部分）：第一步只有 B 有梯度（∂L/∂B ∝ A(x) ≠ 0），
        #    A 的梯度 ∝ B^T = 0 所以不动；于是训练节奏是“先让 B 定方向，A 随后跟上”。
        # ⚠️ 如果反过来（A=0、B 随机），第一步就变成只有 A 能动，结论对称，
        #    但论文选了 A 高斯 + B 零这个组合，因为这样可以复用“正态初始化 A”的
        #    现成初始化逻辑，且 B 全零后 state_dict 里能一眼看出“还没训过”。
        self.B.weight.data.zero_()

    def forward(self, x):
        """LoRA 分支前向：B(A(x))。

        参数：
            x : [B, T, in_features]
        返回：
            [B, T, out_features]，即 x @ (B@A)^T

        形状：
            [B,T,768] → A → [B,T,16] → B → [B,T,768]

        设计理由（为什么是 B(A(x)) 而不是先算 B@A 再乘 x）：
            训练时输入有很多 token，先降维再升维能把中间张量压到 [B,T,16]，
            计算量从 2*T*D*D 降到 2*T*D*r；而推理时如果想省这点开销，
            就用 merge_lora 把 B@A 提前算好加进 W（只在部署时做一次）。
        """
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """给模型里“输入输出维度相同的 Linear”挂上 LoRA 旁路，并改写它的 forward。

    作用：
        遍历所有子模块，对符合条件的 nn.Linear 做三件事：
        1) 造一个 LoRA(r) 并 setattr 成 module.lora；
        2) 保存原始 forward；
        3) 用 forward_with_lora 覆盖 module.forward。

    参数：
        model  MiniMindForCausalLM（或任何 nn.Module）
        rank   16

    返回：
        None —— 就地修改 model（原地 monkey-patch，没有返回值）

    形状：
        q_proj: 768 → 768  → 挂 LoRA，ΔW [768,768]
        o_proj: 768 → 768  → 挂 LoRA，ΔW [768,768]

    💡 关键点：为什么筛选条件是 in_features == out_features（只挑方阵）？
        在 MiniMind 里，形状是方阵的 Linear 只有两个：
            self_attn.q_proj : 768 → 8*96  = 768   ✓ 方阵
            self_attn.o_proj : 768 → 768           ✓ 方阵
        其余全都不等，所以一个都不会被挂上：
            self_attn.k_proj : 768 → 4*96 = 384    ✗（GQA，KV 头只有 4 个）
            self_attn.v_proj : 768 → 384           ✗
            mlp.gate_proj    : 768 → 2432          ✗
            mlp.up_proj      : 768 → 2432          ✗
            mlp.down_proj    : 2432 → 768          ✗
            lm_head          : 768 → 6400          ✗（而且它通常和 embedding 共享权重）
            MoE 的 router    : 768 → 4             ✗
        所以“只挑方阵”等价于“只给注意力里的 q_proj 和 o_proj 加 LoRA”。
        LoRA 原理上并不要求方阵（A/B 可以是 r×in、out×r 任意形状），
        作者这么写是一个刻意的简化：方阵时 A、B 形状最对称、参数量最好估，
        而且实测只调 q/o 就已经足够（论文也认为注意力投影是最值得调的部分）。
        ⚠️ 副作用：这是个“看形状不看语义”的启发式。如果你把 intermediate_size
        改成 768，那么 mlp 的三个投影也会变成方阵、被误挂上 LoRA。

    设计理由（为什么用 monkey-patch 而不是新写一个 LoRALinear 类）：
        模型是从 .pth 加载的，直接换类会改变 state_dict 的 key（多一层前缀），
        导致老权重加载失败；而在实例上改 forward，state_dict 里只是“多了
        xxx.lora.A.weight 这几个 key”，加载老权重时 strict=False 即可兼容。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            # model.device 取第一个参数所在设备；DDP 下每个 rank 的模型已经在自己卡上，
            # 所以这里直接跟着走就行。⚠️ 如果模型被切分到多张卡（device_map），
            # 这里会挂错设备。
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            # 💡 setattr 一个 nn.Module 会触发 nn.Module.__setattr__ 的注册逻辑：
            #    lora 被塞进 module._modules，于是它的参数会自动出现在
            #    model.parameters() / .to(device) / .half() / state_dict() 里。
            #    如果 setattr 的是普通 Tensor/对象，就不会有这个效果。
            setattr(module, "lora", lora)
            # 取出“原来的 forward”。注意这时拿到的是绑定方法（bound method），
            # self 已经绑在 module 上，所以后面调用 layer1(x) 不需要再传 module。
            original_forward = module.forward

            # 显式绑定
            # ⚠️ 这里的默认参数 layer1=..., layer2=... 不是可有可无的装饰：
            #    Python 的闭包是“晚绑定”的。如果写成
            #        def forward_with_lora(x): return original_forward(x) + lora(x)
            #    那么循环结束后，所有层的函数都会去找最后一份 original_forward/lora，
            #    结果是全模型共用同一个 LoRA —— 训练能跑但完全错误，且很难发现。
            #    用默认参数在“定义那一刻”求值并固化，是最简洁的解法。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 前向：原分支 + LoRA 分支。
                # 因为 B 初始为 0，此刻 layer2(x) 恒等于 0，输出和原模型完全一致。
                # 💡 函数只声明了一个必须参数 x：直接赋给 module.forward 后，它不会
                #    被当作方法自动绑定 self（实例属性上的普通函数不做描述符绑定），
                #    所以调用 module(x) → __call__ → self.forward(x) 正好只传一个 x。
                #    ⚠️ 如果这里写成 (self, x)，就会多传一个参数直接报 TypeError。
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
            # 把函数挂到“实例”上，覆盖类的方法查找。这样其他代码完全不用改，
            # 任何 module(x) 调用都会自动带上 LoRA。


def load_lora(model, path):
    """把磁盘上的 LoRA 权重灌回已被 apply_lora 挂载过的模型。

    作用：
        读 pth → 去掉 DDP 前缀 → 按模块名分组 → 逐模块 load_state_dict。

    参数：
        model  已经调用过 apply_lora 的模型（必须已经有 .lora 子模块）
        path   lora 权重路径

    返回：
        None（就地加载）

    设计理由：
        只加载 A/B 这几十万个参数，几 MB 的文件，不需要动 64M 的主权重；
        推理服务因此可以“一份 base 权重 + 多个任务 LoRA”热切换。
    """
    # map_location 直接落到模型所在设备，省掉“先读进 CPU 再搬到 GPU”的一次拷贝
    state_dict = torch.load(path, map_location=model.device)
    # 去掉 DDP 的 "module." 前缀（7 个字符），让保存时是否被 DDP 包过变得无所谓
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 从大字典里筛出属于本模块的 key，并把 "model.layers.0.self_attn.q_proj.lora."
            # 这个前缀也剥掉，这样剩下的就是 lora 模块自己的 "A.weight"/"B.weight"
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
            # ⚠️ load_state_dict 默认 strict=True：如果某个模块的关键字在文件里找不到，
            #    会直接抛 missing keys 错误。这其实是好事 —— 传错文件（比如传了 base
            #    权重）会立刻报错，而不是静默地“加载了 0 个参数却看起来成功”。


def save_lora(model, path):
    """只把 LoRA 的 A/B 参数存成一个小 pth。

    作用：
        遍历所有挂了 lora 的模块，收集 A.weight / B.weight，加回模块名前缀后保存。

    参数：
        model  训练中的模型（可能被 torch.compile 或 DDP 包过）
        path   输出路径

    返回：
        None

    设计理由：
        1) 只存 LoRA（约 0.39M 参数，fp16 下不到 1MB），而不是 64M 的整个模型；
        2) 存 cpu + half，避免把 GPU 显存写进文件、也避免文件过大；
        3) 还原出“带模块全名”的 key，这样 load_lora 能按名字精确匹配。
    """
    # 💡 torch.compile 会把模型包成 OptimizedModule，真正的参数挂在 _orig_mod 上；
    #    getattr(..., default) 这一个写法同时兼容“编译过”和“没编译过”两种情况。
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 去掉 DDP 加的 "module." 前缀，保证保存出来的 key 是干净的全名
            clean_name = name[7:] if name.startswith("module.") else name
            # f'{clean_name}.lora.{k}'：k 是 "A.weight"/"B.weight"，
            # 拼完形如 model.layers.0.self_attn.q_proj.lora.A.weight
            # .cpu().half()：搬回内存 + 半精度，纯粹为了省磁盘
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """把 LoRA 的 ΔW = B@A 直接加进主权重，导出一个“普通模型”。

    作用：
        先加载 LoRA，再对所有挂了 LoRA 的 Linear 做 W ← W + B@A，
        最后把整份（不含 .lora.* 的）state_dict 存盘。

    参数：
        model      已经 apply_lora 过、且结构与 lora 权重严格对应的模型
        lora_path  训练产出的 lora pth
        save_path  合并后主权重（与普通 MiniMindForCausalLM 的 state_dict 同构）

    返回：
        None

    公式：
        W_merged = W_pretrained + ΔW = W + B @ A
        W        : [out, in] = [768, 768]
        B.weight : [out, r]  = [768, 16]
        A.weight : [r, in]   = [16, 768]
        B@A      : [768,16] @ [16,768] = [768,768] ✓ 与 W 完全同形，可以直接相加

    设计理由（为什么可以合并、合并有什么好处）：
        y = x W^T + x (B A)^T = x (W + B A)^T
        两条分支之间没有非线性夹着，是纯粹的线性可加，所以数学上完全等价。
        合并后推理时不再有额外的矩阵乘，延迟和显存都和原模型一模一样 ——
        这是 LoRA 相对 Adapter 类方法的一大优势（部署零成本）。
    """
    load_lora(model, lora_path)   # 先把 A/B 灌进模型，避免调用方自己拼路径、自己排序
    raw_model = getattr(model, '_orig_mod', model)   # 兼容 torch.compile 的包装
    # 先拷贝一份“不含 LoRA 的全部参数”（embedding、各层 attn、mlp、norm、lm_head...），
    # 过滤 '.lora.' 是因为那些 key 不该出现在合并后的产物里
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # ⚠️ 这里用“名字里有没有 .lora.”来过滤，而不是 isinstance(module, LoRA)：
            #    因为 lora.A 和 lora.B 本身也是 nn.Linear，它们的名字形如
            #    model.layers.0.self_attn.q_proj.lora.A —— 用名字一次就能排掉。
            #    这也是为什么 apply_lora 要保证挂上去的子模块名叫 "lora"。
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            # 显式重写一遍权重：保证这个 key 一定存在、且一定是 fp16、一定在 cpu 上，
            # 不依赖上面那行 raw_model.state_dict() 的遍历顺序
            if hasattr(module, 'lora'):
                # 只对“挂了 LoRA 的方阵”做加法；其余 Linear 保持原样
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
                # B[768,16] @ A[16,768] → ΔW[768,768]，再转 cpu+half 后原地加到 W 上。
                # ⚠️ 注意这个 += 是在 fp16 上做的（左边已是 half），rank=16 时误差
                #    可以忽略；如果对精度极其敏感，应该先在 fp32 里相加再转 half。
    torch.save(state_dict, save_path)
    # 💡 合并后模型对象里的 lora 子模块并没有被删除，仍占着那 0.39M 参数；
    #    但它们不会进 state_dict，所以产物是一个“干净的普通模型”，
    #    可以直接用 MiniMindForCausalLM 加载（config 完全不用改）。

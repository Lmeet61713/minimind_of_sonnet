# ==============================================================================
# MiniMind 主模型定义（model/model_minimind.py）
# ==============================================================================
# 作用：
#   用纯 PyTorch 手写一个 decoder-only 的 LLaMA 风格小语言模型：
#   RMSNorm + RoPE + GQA + SwiGLU + pre-norm 残差，可选 MoE 稀疏前馈；
#   同时提供 HF 兼容的 config / PreTrainedModel / 手写采样版 generate。
#
# 在流水线中的位置：
#   上游 ← 无（本文件是被所有 trainer 依赖的“骨架”，不从任何权重加载）：
#     trainer/train_tokenizer.py 训出 vocab=6400，决定这里的 vocab_size；
#     trainer/train_pretrain.py 从随机初始化开始做 next-token 预测；
#   下游 → 8 个 trainer 全部 import 这里的 MiniMindConfig / MiniMindForCausalLM：
#     train_pretrain / train_full_sft / train_lora / train_distillation /
#     train_dpo / train_ppo / train_grpo / train_agent，
#     它们各自产出 .pth（state_dict），再交给 eval_llm.py 与 scripts/* 推理。
#
# 核心思想：
#   1) 结构完全对齐 LLaMA 一代：pre-norm 残差 + RMSNorm + RoPE + SwiGLU。
#      所以“读懂这一个文件”≈“读懂主流开源 LLM 的 90% 结构”。
#   2) 用 GQA（8 个 Q 头共用 4 组 K/V）+ tie_word_embeddings 把参数量压到 64M，
#      让单张 8G 显存的卡也能从零跑完 pretrain → sft → dpo → ppo 全链路。
#   3) 对外契约极简：MiniMindModel.forward 返回 (hidden_states, presents, aux_loss)，
#      loss 只在 MiniMindForCausalLM 里算，trainer 各自决定用不用 aux_loss。
#
# 关键张量 / 数据：
#   input_ids     : [B, T]           int64，取值 [0, V)，V = vocab_size = 6400
#   hidden_states : [B, T, D]        D = hidden_size = 768
#   q             : [B, H, T, Dh]    H = 8，Dh = head_dim = 96
#   k / v         : [B, KVH, T, Dh]  KVH = num_key_value_heads = 4（GQA）
#   attn_scores   : [B, H, T, T_kv]  未归一化点积 / sqrt(Dh)
#   logits        : [B, T, V]        未归一化打分，喂给 cross_entropy 前不再手动 softmax
#   labels        : [B, T]           int64，ignore_index = -100 的位置不计 loss
#   freqs_cos/sin : [max_pos, Dh]    = [32768, 96]，RoPE 查表，persistent=False 不进 state_dict
#
# 阅读建议：
#   1) 先看 MiniMindConfig 里每个数为什么这么取（顺带把 64M 算清楚）；
#   2) 再看 RMSNorm 与 precompute_freqs_cis 两个小函数，它们是全篇的数学地基；
#   3) 然后 Attention（GQA + QK-Norm + flash 双分支）与 FeedForward（SwiGLU）；
#   4) MiniMindBlock 只是把两者拼成 pre-norm 残差，MiniMindModel 只是 for 循环堆叠；
#   5) MiniMindForCausalLM.forward 的“logits 右移 / labels 左移”是训练的灵魂，必看；
#   6) generate 是手写采样循环，可以最后看 —— 它的价值是把 temperature / top_k /
#      top_p / repetition_penalty / KV cache 五件事一次讲清，性能不是重点；
#   7) use_moe=False 时，MOEFeedForward 那一段可以先整段跳过，不影响主线。
# ==============================================================================
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    """MiniMind 的超参容器（继承 HF 的 PretrainedConfig，所以能直接 from_pretrained）。

    作用：
        集中声明模型结构的所有可调项。注意它不只是“配置”，它同时定义了这个模型的
        “身份”（model_type），HF 的 save_pretrained / from_pretrained 靠它序列化成
        config.json 并还原。

    参数（全部有默认值，所以 MiniMindConfig() 就能跑）：
        hidden_size=768        残差流宽度 D，也是 embedding 维度
        num_hidden_layers=8    Transformer block 层数
        use_moe=False          是否把每一层的 FFN 换成 MoE 稀疏前馈
        **kwargs               其余超参都用 kwargs.get 取，好处是 HF 传进来一堆无关字段
                               （dtype、device_map、_name_or_path...）时不会 TypeError

    返回：
        MiniMindConfig 实例，可当作 dict 一样用 to_dict() / to_json_string()

    形状：
        本身不含张量；它决定的形状是 [B, T, D] 与 [B, T, V]

    设计理由（64M 是怎么来的）：
        embedding（tie 后只算一份）  V*D    = 6400*768  = 4,915,200
        每层 attn  q 768*768=589,824 + k 768*384=294,912 + v 294,912 + o 589,824
                                                                  = 1,769,472
        每层 QK-Norm（两个 head_dim 长度的 gain）                =       192
        每层 MLP  3*D*I = 3*768*2432                             = 5,603,328
        单层合计                                                  = 7,372,992
        8 层                                                     = 58,983,936
        final RMSNorm 的 gain                                    =       768
        合计 = 58,983,936 + 768 + 4,915,200 = 63,899,904 ≈ 63.9M → 对外就叫 64M。
        ⚠️ 如果把 intermediate_size 改成 LLaMA 常用的 8/3*D = 2048，总量只有
        约 56.8M，就凑不到 64M 了 —— 所以这里用 π 而不用 8/3，是有意的“凑数”。
    """
    model_type = "minimind"   # 写进 config.json；from_pretrained 靠它路由回本类
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        """构造配置对象；只做字段赋值，不创建任何张量。

        参数：
            hidden_size          D，残差流宽度
            num_hidden_layers    block 层数
            use_moe              是否启用 MoE（影响 MiniMindBlock 里选哪个 MLP）
            **kwargs             其余字段，缺省时用下面的默认值补齐
        返回：
            None（就地初始化 self）
        """
        super().__init__(**kwargs)   # 先让 PretrainedConfig 处理 dtype / device_map 等公共字段
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        # dropout 默认 0：模型只有 64M，数据量也不算大，dropout 反而拖慢收敛；
        # 真正做正则的是 weight decay 和（SFT 阶段的）小学习率
        self.dropout = kwargs.get("dropout", 0.0)
        # 6400 来自 trainer/train_tokenizer.py 训出的 BPE 词表
        # ⚠️ 换分词器必须同步改这里，否则 embedding 形状对不上，checkpoint 直接加载失败
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)   # <s>，见 tokenizer 的特殊 token 布局
        self.eos_token_id = kwargs.get("eos_token_id", 2)   # </s>，同时充当 pad 与停止符
        # 只是“允许”走 SDPA；真正是否用 flash kernel 还要看 Attention.forward 里的三个前置条件
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)      # H = 8
        # GQA：4 组 K/V 头被 8 个 Q 头共享（每 2 个 Q 头共用 1 组）→ KV cache 直接减半
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        # 96 = 768/8；它不是 64 的倍数，这在 LLaMA 系里很正常 —— head_dim 与 hidden
        # 是解耦的，只要 H*Dh 能凑回 hidden（8*96=768）即可，o_proj 于是仍是方阵
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')   # SwiGLU 里的 SiLU（swish）
        # ceil(768 * π / 64) * 64 = ceil(37.699) * 64 = 38 * 64 = 2432
        # 用 π≈3.1416 而非常见的 8/3≈2.667，是为了让 64M 这个目标参数量成立；
        # 向上对齐到 64 的倍数是老 GPU kernel 的偏好（tile 对齐，避免尾块浪费）
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        # RoPE 表预计算到 32k：训练只用 2048 左右，但表多算不花什么时间，
        # 推理时想直接上长上下文就不用重建模型
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        # 比 LayerNorm 常用的 1e-5 更小：RMSNorm 不减均值，分母天然更小更稳
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        # base 越大 → 低频维度转得越慢 → 长距离的位置差更容易分辨。
        # LLaMA-2 用 1e4，LLaMA-3 用 5e5，这里取 1e6 是给“2k 训练、32k 推理”留外推空间
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        # 输入 embedding 与输出 lm_head 共享同一个 Parameter（省 4.92M 参数）
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        # 只在“推理阶段想超长上下文”时才置 True；训练阶段必须关：
        # 训练长度 ≤ 2048 时开 YaRN 会把本来够用的位置分辨率白白插值糊掉
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        # YaRN 参数：beta_fast/beta_slow 是“波长边界”，用来把频率维度分成三段
        # （不改 / 线性过渡 / 全量缩放）；factor=16 表示把可分辨距离拉长 16 倍；
        # original_max_position_embeddings=2048 是预训练时的真实长度；
        # attention_factor 是 YaRN 论文里补偿 softmax 温度的量（论文建议 0.1*ln(s)+1，
        # 这里给 1.0 即不补偿）；type 只是个标记字段，本实现不解析它
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        # 4 个专家、每 token 只激活 1 个（top-1）：专家参数 4 份，算力仍是 1 份的量
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        # 每个专家单独的隐层宽度；默认与 dense 版一致，方便直接对比
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        # top-1 时归一化是恒等变换，保留这个开关是为了和 DeepSeek 系实现对齐
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        # 负载均衡损失的权重：5e-4 很小，因为它是“辅助目标”，绝不能压过语言建模主 loss
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    """Root-Mean-Square LayerNorm：LLaMA 系的默认归一化层。

    作用：
        对最后一维做归一化：y = x / sqrt(mean(x^2) + eps) * g，再做缩放。

    参数：
        dim  : 归一化的维度长度。MiniMindModel 里传 D=768，
               Attention 里的 q_norm/k_norm 传 Dh=96（每个头内部各归一化）
        eps  : 防止除零，默认 1e-5，实际调用时传 config.rms_norm_eps=1e-6

    返回：
        与输入同形状的归一化结果

    形状：
        x        [..., dim]  →  [..., dim]
        weight   [dim]

    设计理由（与 LayerNorm 的区别）：
        LayerNorm  : y = (x - mean) / sqrt(var + eps) * g + b   —— 要减均值、要学 bias
        RMSNorm    : y = x / sqrt(mean(x^2) + eps) * g         —— 不减均值、无 bias
        对 decoder-only 的 LLM 来说，“减去均值”这一步几乎不影响效果（它只是把
        表征平移了一下，而后面紧跟的线性层本来就能吸收平移），砍掉它可以少一次
        reduction 和一组 bias，前向更快、显存更省。这个经验被 LLaMA / PaLM 验证过。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        """初始化归一化层。

        参数：
            dim  归一化维度长度（768 或 96）
            eps  数值稳定项
        返回：
            None
        形状：
            weight 被初始化为全 1，即初始时 RMSNorm ≈ 恒等变换（乘 1）
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 只有 gain，没有 bias

    def norm(self, x):
        """核心归一化：把 x 的均方根（RMS）化为 1。

        参数：
            x : [..., dim]，应当已经是 fp32
        返回：
            [..., dim]，每一行的 RMS ≈ 1（eps 影响下略小于 1）

        形状：
            x.pow(2)         [..., dim]
            .mean(-1, keepdim=True)  [..., 1]
            torch.rsqrt(...)         [..., 1]
            x * (...)                [..., dim]   ← 广播，逐行缩放

        设计理由：
            - rsqrt = 1/sqrt：GPU 上是一条指令，比 1/torch.sqrt(...) 少一次除法；
            - eps 放在 sqrt 内部，x 全 0 时得到 1/sqrt(eps)=1000 而不是 inf/nan；
            - 只算均方根不减均值，所以这里没有 x - mean 这一步。
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """归一化并乘可学习 gain，最后转回输入精度。

        参数：
            x : [..., dim]，可能是 bf16 / fp16（autocast 下）
        返回：
            [..., dim]，dtype 与输入相同

        形状：
            norm(x.float())   [..., dim]（fp32）
            weight * (...)    [..., dim]（fp32，weight 是 fp32 参数，会把结果提回 fp32）
            .type_as(x)       [..., dim]（回到 x 的 dtype）

        设计理由（为什么先 .float() 再 type_as 回去）：
            1) bf16 只有 8 位尾数，x.pow(2) 对大数值很容易丢精度甚至溢出（bf16 上限
               虽然很大，但平方后有效位掉得更快），而“除以一个统计量”对精度极敏感；
            2) 归一化在 fp32 里算完再降回 bf16，前向数值稳定，反向也不会出现
               “梯度被 bf16 的粗粒度量化卡住”的问题；
            3) type_as(x) 让这个模块对 dtype 完全透明，autocast 混用 fp16/bf16 时
               不会因为 dtype 不一致报错，也不需要额外存一份 fp32 激活。
            ⚠️ 常见坑：如果省略 .float()，在 bf16 训练到中后期（激活尺度变大）时
            会偶发 loss 变 nan，且很难定位。
        """
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算 RoPE（旋转位置编码）的 cos / sin 查表。

    作用：
        为每个位置 p、每个频率对 i 算出旋转角 theta = p * base^(-2i/dim)，
        并把 cos/sin 存成可以直接广播相乘的表。

    参数：
        dim         每个头的维度 Dh=96（不是 hidden！RoPE 是“头内”旋转）
        end         最大位置数，取 config.max_position_embeddings=32768
        rope_base   RoPE 的 base，取 config.rope_theta=1e6
        rope_scaling YaRN 配置 dict，None 表示不做外推

    返回：
        (freqs_cos, freqs_sin)，各为 [end, dim] = [32768, 96] 的 fp32 张量

    公式：
        theta_i = 1 / base^(2i/dim),   i = 0, 1, ..., dim/2 - 1
        位置 p 在第 i 对上的旋转角 = p * theta_i
        i 越大 → theta 越小 → 周期越长：前几维区分相邻 token，后几维区分远距离

    设计理由：
        RoPE 的角度只跟“位置 × 频率”有关，与输入内容无关，所以可以在 __init__
        里一次性算好，训练/推理时只做切片（见 MiniMindModel.forward 里的
        freqs_cos[start_pos:start_pos+seq_length]）。省掉了每层每步的重复三角函数。
    """
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    # arange(0, 96, 2) → 0,2,...,94 共 48 个偶数下标；再 [:dim//2] 是冗余保护（保证只取一半）
    # 除以 dim 再取负幂 → 1/(base^(2i/96))：这就是每个“频率对”的角速度 theta_i
    # attn_factor 先给 1.0，只有启用 YaRN 时才可能被下面的配置改写
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        # 原注释保留：YaRN 的核心公式 f'(i) = f(i) * ((1-γ) + γ/s)
        # γ 是一个随维度下标线性变化的 ramp：γ=0 的维度完全不改，γ=1 的维度整体除以 s
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )   # 一次取齐五个量：原始训练长度、外推倍数、两个波长边界、注意力温度补偿
        if end / orig_max > 1.0:   # 只有“要外推”（32768/2048=16 > 1）时才启用缩放
            # inv_dim(b)：反解“哪个维度下标的波长正好等于 b 圈（b*2π 个位置）”
            # 由 base^(2i/dim) = orig_max/(b*2π) 解出 i，即把“波长阈值”翻译成维度下标
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # dim=96, orig_max=2048, base=1e6 时：beta_fast=32 → low≈8（floor），
            # beta_slow=1 → high≈21（ceil）；再夹紧到 [0, dim//2-1] 防止配置离谱时越界
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # ramp 就是公式里的 γ：下标 < low 处为 0（不动），> high 处为 1（全缩放），
            # 中间线性过渡。max(high-low, 0.001) 是防止 high==low 时除零
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # f'(i) = f(i) * (1 - γ + γ/s)：乘 1/16 等于把该维的旋转速度降到 1/16，
            # 也就是“把位置轴拉长 16 倍”，等效于能分辨 16 倍远的距离
            freqs = freqs * (1 - ramp + ramp / factor)
            # 💡 只有低频维被拉长、高频维原样保留：近距离的相对位置关系（局部语序）
            #    不会被插值糊掉，这正是 YaRN 比“整表线性插值”更强的地方
    t = torch.arange(end, device=freqs.device)                 # [32768]
    freqs = torch.outer(t, freqs).float()                      # [32768, 48]，第 p 行第 i 列 = p * theta_i
    # 为什么要把 cos/sin 各复制两份拼成 [32768, 96]：
    # rotate_half 把向量切成前后两半 x1=x[:48]、x2=x[48:]，第 i 个和第 i+48 个元素
    # 是“同一对”、共享同一个旋转角 theta_i。所以 cos 也要变成 [c0..c47, c0..c47]，
    # 这样后面一次逐元素乘就能同时完成 48 个 2x2 旋转，不需要 reshape/转置
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    # 保持 fp32：bf16 存 cos 只有 8 位尾数，旋转角会明显失真；使用方再按需 .to(dtype)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """给 q / k 施加 RoPE 旋转。

    作用：
        对每一对元素 (x_i, x_{i+Dh/2}) 按角度 theta_i 做二维旋转，使 q·k 只依赖
        “两个 token 的相对距离”。

    参数：
        q, k        查询/键。形状 [B, H, T, Dh]（本文件调用时已 transpose 到 H 在第 2 维）
                    注意 k 的头数可能是 4（GQA），cos/sin 与头数无关，可广播
        cos, sin    [T, Dh]，来自 precompute_freqs_cis 的切片
        unsqueeze_dim  cos/sin 补哪一维：默认 1（补成 [T,1,Dh] 去对齐 [B,H,T,Dh]；
                    若你保持 [B,T,H,Dh] 布局则传 2）

    返回：
        (q_embed, k_embed)，形状与 dtype 同输入

    公式（对第 i 对 (x_i, x_{i+Dh/2})）：
        y_i          = x_i      * cos_i - x_{i+Dh/2} * sin_i
        y_{i+Dh/2}   = x_{i+Dh/2} * cos_i + x_i      * sin_i
        矩阵形式： [[cos_i, -sin_i], [sin_i, cos_i]] @ [x_i, x_{i+Dh/2}]

    设计理由：
        把 Dh/2 个独立的 2x2 旋转写成“一次逐元素乘 + 一次 swap”的向量化形式，
        比 reshape 成 [..., Dh/2, 2] 再矩阵乘快得多，也不需要额外 kernel。
    """
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # rotate_half: [..., 96] → [..., 96]，输出前半 = -x2，后半 = x1（x1=x[:48], x2=x[48:]）
    # 为什么长这样：要让 x*cos + rotate_half(x)*sin 展开后正好等于
    # [x1*cos - x2*sin, x2*cos + x1*sin]，即上面那个旋转矩阵；
    # 负号在 -x2 上（而不是交换顺序），因为 sin 项本身要带符号。
    # 🔍 和 nanoGPT 对照：nanoGPT 用可学习绝对位置表 wpe 直接加到 token embedding 上；
    #    RoPE 改成“在 Q/K 上旋转”，好处是不占参数、且分数天然只依赖相对距离。
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    # cos/sin 是 fp32 表，与 bf16 的 q 相乘时 PyTorch 会提升到 fp32 计算，
    # 最后 .to(q.dtype) 转回来 —— 先算后降精度，比先把 cos 转成 bf16 再乘要准
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 GQA 的 K/V 从 KV 头数“复制”成 Q 头数，好让注意力能按头对齐。

    作用：
        8 个 Q 头、4 个 K/V 头时，每 2 个 Q 头共用同一组 K/V；本函数把这 4 组
        在“头”这一维上复制 2 份，变成 8 组。

    参数：
        x     [B, T, KVH, Dh] = [B, T, 4, 96]
        n_rep H // KVH = 8 // 4 = 2

    返回：
        [B, T, H, Dh] = [B, T, 8, 96]

    形状变化：
        [B,T,4,96] --unsqueeze(3)--> [B,T,4,1,96]
                   --expand-->      [B,T,4,2,96]   （stride=0 的“假复制”，不占显存）
                   --reshape-->     [B,T,8,96]     （真正落成连续内存）

    设计理由（为什么 K/V 只要 4 个头、为什么要 expand 而不是 repeat）：
        1) 推理时显存大头是 KV cache：每层 2 * B * T * KVH * Dh * sizeof(dtype)。
           把 KVH 从 8 砍到 4，cache 直接减半，长上下文才跑得动；而 Q 仍是 8 头，
           表达力损失很小（GQA 论文的结论：它是 MHA 与 MQA 之间的甜点）。
        2) expand 只改 stride、不分配显存，reshape 时才按需拷贝；如果直接
           x.repeat(1,1,n_rep,1)，会立刻先分配一份完整的新张量。
        ⚠️ expand 出来的张量 stride 为 0，后面必须用 reshape 不能用 view（view 会报错）。
        💡 reshape 把 (KVH, n_rep) 两维压平，所以第 h 个 Q 头用第 h//2 组 K/V
           （0,1→kv0；2,3→kv1...），而不是 h%4。两种分组数学等价，但相邻分组对
           cache 的连续性更友好。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x   # 普通 MHA 时直接返回，省一次拷贝
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    """多头自注意力（GQA + QK-Norm + RoPE + KV cache + flash/普通双分支）。

    作用：
        把 [B, T, D] 的隐状态做一次因果自注意力，再投影回 [B, T, D]。

    参数：
        x                  [B, T, D]           D=768，T 是“本次新进的 token 数”，有 cache 时通常为 1
        position_embeddings (cos, sin)，各 [T, Dh]=[T,96]，已按绝对位置切好
        past_key_value     (k, v) 或 None，k/v 各 [B, past_len, 4, 96]
        use_cache          是否返回新的 KV cache
        attention_mask     [B, T_total] 的 0/1 padding mask，或 None

    返回：
        (output [B, T, D], past_kv 或 None)

    设计理由（投影维度为什么不同）：
        q_proj: 768 → 8*96 = 768      （8 个 Q 头）
        k_proj: 768 → 4*96 = 384      （只有 4 组 K/V，这是 GQA 的全部含义）
        v_proj: 768 → 4*96 = 384
        o_proj: 768 → 768             （把 8 个头拼回来再混合）
        如果写成 k_proj: 768→768 就成了普通 MHA，KV cache 会大一倍。
    """
    def __init__(self, config: MiniMindConfig):
        """构造四个投影、两个 QK-Norm、两个 dropout，并探测 SDPA 是否可用。

        参数：
            config  MiniMindConfig
        返回：
            None
        形状：
            q_proj.weight  [768, 768]
            k_proj.weight  [384, 768]
            v_proj.weight  [384, 768]
            o_proj.weight  [768, 768]
        """
        super().__init__()
        # 允许 num_key_value_heads=None 表示“退化成普通 MHA”
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads      # 8
        self.n_local_kv_heads = self.num_key_value_heads     # 4
        self.n_rep = self.n_local_heads // self.n_local_kv_heads   # 2，每个 KV 头被几个 Q 头共享
        self.head_dim = config.head_dim                      # 96
        self.is_causal = True   # 硬编码：这是 decoder-only，没有 bidirectional 需求
        # 注意 q_proj 的输出维度写成 H*Dh 而不是 hidden_size：数值相等（8*96=768），
        # 但写成乘积才能支持 head_dim ≠ hidden/heads 的配置（Llama-3 就是这种）
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # QK-Norm：对每个头的 96 维单独做 RMSNorm（不是对整个 768 维）
        # 作用：把 q/k 的模长压到稳定范围，防止注意力 logits 随训练爆炸。
        # 💡 这其实是把“注意力温度”显式归一化了：scores = q·k/sqrt(Dh)，
        #    若 |q|、|k| 无约束地增长，softmax 会迅速饱和成 one-hot、梯度消失。
        # 🔍 nanoGPT 没有 QK-Norm，只靠 1/sqrt(d) 缩放；小模型 + 小数据够用，
        #    但一旦上 RL（PPO/GRPO 里同一批数据反复更新）就容易炸，所以这里加上。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)    # 作用在 softmax 之后
        self.resid_dropout = nn.Dropout(config.dropout)   # 作用在 o_proj 之后
        self.dropout = config.dropout
        # 运行时探测 + 配置开关：老版本 PyTorch 没有 scaled_dot_product_attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """一次因果自注意力前向。

        参数：
            x                  [B, T, D]
            position_embeddings (cos, sin)，各 [T, Dh]
            past_key_value     上一轮留下的 (k, v)，各 [B, past_len, KVH, Dh]
            use_cache          是否把本轮的 k/v 也返回
            attention_mask     [B, T_total]，1=有效，0=padding
        返回：
            (output [B, T, D], past_kv 或 None)

        形状流转：
            x            [B,T,768]
            xq           [B,T,768] → view → [B,T,8,96]
            xk / xv      [B,T,384] → view → [B,T,4,96]
            cat cache 后 xk [B,past+T,4,96]
            transpose 后 xq [B,8,T,96]，xk/xv [B,8,past+T,96]
            scores       [B,8,T,past+T]
            output       [B,8,T,96] → transpose+reshape → [B,T,768]
        """
        bsz, seq_len, _ = x.shape   # seq_len 是“本次新进的 token 数”，有 cache 时通常等于 1
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)        # [B,T,768] → [B,T,8,96]
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)     # [B,T,384] → [B,T,4,96]
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)     # [B,T,384] → [B,T,4,96]
        xq, xk = self.q_norm(xq), self.k_norm(xk)   # 只在最后一维 96 上归一化，v 不归一化
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)   # 这里 q/k 是 [B,T,H,96]，cos 补在第 1 维后为 [T,1,96] 可广播
        # 注意：RoPE 必须在“拼 cache 之前”做，因为 cos/sin 是按绝对位置切好的；
        # 如果把旧 k 也再转一遍，角度就重复叠加了（这是新手最常见的 RoPE bug）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)   # [B,past,4,96] + [B,seq,4,96] → [B,past+seq,4,96]
            xv = torch.cat([past_key_value[1], xv], dim=1)   # 同上
        # 存的是“拼好的、还没 repeat 的”4 头版本：cache 只要 4 头，省一半显存
        past_kv = (xk, xv) if use_cache else None
        # 先 repeat_kv 再 transpose：repeat_kv 要求输入是 [B,T,KVH,Dh] 布局
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        # → xq [B,8,T,96]；xk/xv [B,8,past+T,96]（头在倒数第 3 维，方便按头做矩阵乘）
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 走 flash / memory-efficient kernel 的三个前置条件：
            # 1) seq_len > 1：解码时 q_len=1，flash kernel 没有收益（瓶颈在访存），
            #    而且走普通分支才能把 mask 精确放到“最后 seq_len 列”上；
            # 2) past_key_value is None：SDPA 的 is_causal=True 只支持 q_len == kv_len 的
            #    “左上角对齐”上三角 mask；用了 cache 后 q_len=1 而 kv_len=past+1，
            #    SDPA 会误以为只有第 0 个位置可见 → 结果全错，必须自己造 mask 走 else；
            # 3) attention_mask 全 1：这个调用里根本没传 attn_mask，flash 分支无法表达
            #    padding；只要存在 padding 就必须落到普通分支。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 普通分支：显式算 scores 再 softmax。先把点积升到 fp32 再缩放，
            # 因为 bf16 下 QK^T 的累加很容易溢出/掉精度
            # [B,8,T,96] @ [B,8,96,T_kv] → [B,8,T,T_kv]，再统一除以 sqrt(Dh)
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 为什么是 scores[:, :, :, -seq_len:] 而不是 scores[:, :, :, :]：
            #   query 是“最后 seq_len 个位置”，能看的 key 是 [0, T_kv) 全部，
            #   要屏蔽的是“比自己更靠后”的那些，所以 mask 贴在最后 seq_len 列上。
            #   训练时 seq_len == T_kv，-seq_len 就是 0，退化成标准上三角；
            #   解码时 seq_len=1，mask 只覆盖最后一列（实际是空操作）。
            #   ⚠️ 如果这里写死 0，decode 时会把整行几乎全屏蔽掉，输出直接是垃圾。
            # triu(1)：主对角线以上（不含对角线）= 未来位置，恰好是“不能看”的那些
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # attention_mask [B,T_kv] → [B,1,1,T_kv] 广播到 [B,8,T,T_kv]：
            #   1 的位置加 0（保留），0 的位置（padding）加 -1e9
            # 为什么这里用 -1e9 而上面用 -inf：
            #   -inf 能保证 softmax 后严格为 0，但如果某一行全是 padding（batch 里有空样本），
            #   整行 -inf 会让 softmax 算出 nan，污染整个训练；
            #   -1e9 是有限数，最差也只是退化成近似均匀分布，不会出 nan。
            #   而上面的 causal mask 每行至少能看到自己、不可能全 -inf，所以敢用 -inf。
            # 注意 scores 已经是 fp32，-1e9 不会溢出；若在 bf16 上算，e^1e9 会直接 inf
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv
            # softmax 在 fp32 里算完再降回 xq 的 dtype（bf16），保证数值稳定；
            # dropout 放在归一化之后、加权之前（对齐标准 Transformer 的实现）
            # [B,8,T,T_kv] @ [B,8,T_kv,96] → [B,8,T,96]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # [B,8,T,96] → [B,T,8,96] → [B,T,768]：把“头”和“头内维度”合并回 hidden。
        # ⚠️ 必须用 reshape 不能用 view：transpose 之后张量不连续，view 会报错。
        # 合并顺序必须是 (H, Dh)，且要和 o_proj 权重的布局约定一致（也等价于
        # 和上面 q_proj view 的顺序一致）——训练时随机初始化，自洽即可。
        output = self.resid_dropout(self.o_proj(output))
        # o_proj：768 → 768，作用是把 8 个头的信息线性混合到一起
        return output, past_kv

class FeedForward(nn.Module):
    """SwiGLU 前馈层（LLaMA 的标准 MLP）。

    作用：
        对每个 token 独立做一次“升维 → 门控 → 降维”的非线性变换。

    形状：
        x        [B, T, 768]
        gate/up  [B, T, 2432]
        逐元素乘  [B, T, 2432]
        down     [B, T, 768]

    公式：
        y = W_down ( SiLU(W_gate x) ⊙ (W_up x) )

    参数：
        config             MiniMindConfig
        intermediate_size  隐层宽度 I；MoE 里会传 moe_intermediate_size 覆盖默认值

    设计理由（三个投影各自的作用 + 为什么用 SwiGLU）：
        gate_proj：产生“门控信号”，过 SiLU 后落在 (0, +∞)，起到阀门作用；
        up_proj  ：“内容”通道，提供被门控的原始特征；
        两者逐元素相乘 = 用 gate 去调制 up 的每一个特征通道（这就是 GLU 的“门”）；
        down_proj：把 2432 维压回 768 维。
        参数量 3*D*I = 3*768*2432 ≈ 5.60M，占单层 7.37M 的 76%，是模型里最肥的地方。
        门控结构在同等参数量下 perplexity 更好（PaLM/LLaMA 都验证过），代价是需要
        3 个矩阵而不是 2 个，所以 I 通常取 8/3*D 左右来抵消参数增长。
        🔍 和 nanoGPT 对照：nanoGPT 的 MLP 是 Linear(4d) → GELU → Linear(d)，
        只有 2 个投影、隐层 4d；这里换成 3 个投影、隐层 2432 ≈ 3.17d，
        总参数几乎一样，但表达力更强。
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        """构造三个无 bias 投影 + 激活函数。

        参数：
            config             MiniMindConfig
            intermediate_size  None 时用 config.intermediate_size（2432）
        返回：
            None
        形状：
            gate_proj.weight  [2432, 768]
            up_proj.weight    [2432, 768]
            down_proj.weight  [768, 2432]
        """
        super().__init__()
        # 用 `or` 而不是 `if x is None`：None（默认）和 0 都会回退到 config 的值，
        # 这里传 0 没有实际意义，所以两种写法的差别无害
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # down_proj 形状是“反的”（I → D）：输出必须回到 D，才能和残差流相加
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # ACT2FN[silu] → nn.SiLU()，用 HF 的注册表是为了和 config.hidden_act 字符串联动
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """SwiGLU 前向：down(act(gate(x)) * up(x))。

        参数：
            x : [B, T, 768]
        返回：
            [B, T, 768]
        形状：
            gate(x)  [B,T,2432] → SiLU → [B,T,2432]
            up(x)    [B,T,2432]
            相乘      [B,T,2432]（逐元素，不是矩阵乘）
            down     [B,T,768]
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    """MoE 稀疏前馈：4 个专家 + 一个 router，每个 token 只走 top-k 个专家。

    作用：
        在参数总量 ×E 的前提下，把每个 token 的实际计算量保持在 1 个专家的水平。
        只有 config.use_moe=True 时才会被 MiniMindBlock 创建。

    参数：
        config  MiniMindConfig（num_experts=4, num_experts_per_tok=1,
                moe_intermediate_size=2432, router_aux_loss_coef=5e-4）

    形状：
        x        [B, T, 768] → x_flat [N, 768]，N = B*T（路由是 per-token 的，所以展平）
        scores   [N, 4]      每个 token 对 4 个专家的偏好概率
        topk_idx [N, 1]      选中的专家编号
        y        [N, 768] → [B, T, 768]

    返回：
        [B, T, 768]

    关键设计：
        1) 路由：一个小 Linear(768, 4)，参数量才 3072，几乎不计入总参数；
        2) 稀疏算力：循环里只把“选中该专家的 token”送进去算，所以只算 N/4 左右；
        3) 负载均衡：一个人为加的 aux_loss 防止 router 塌缩到少数专家；
        4) 副作用通道：aux_loss 不通过返回值传，而是挂在 self.aux_loss 上，
           由 MiniMindModel 汇总后交给 trainer（所以 trainer 必须真的把它加进 loss）。
    """
    def __init__(self, config: MiniMindConfig):
        """构造 router 与 E 个独立专家。

        参数：
            config  MiniMindConfig
        返回：
            None
        形状：
            gate.weight            [4, 768]
            experts[i].gate/up/down 与 FeedForward 相同（每个都是 5.6M）
        """
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)   # router，输出 4 个分数
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]   # 这里其实没用到（专家内部各自有），留着只为接口一致

    def forward(self, x):
        """MoE 前向：路由 → 分组 → 分专家计算 → 按权重汇总。

        参数：
            x : [B, T, 768]
        返回：
            [B, T, 768]

        形状流转：
            [B,T,768] → view → [N,768]
            gate → [N,4] → softmax → topk → topk_weight/topk_idx [N,1]
            y 逐专家 index_add_ → [N,768] → view → [B,T,768]
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)   # [B,T,768] → [N,768]；路由是 per-token 的，展平后可直接 index_add
        scores = F.softmax(self.gate(x_flat), dim=-1)   # [N,4]，每行是该 token 对 4 个专家的偏好概率
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        # topk_weight/topk_idx 各 [N, k]；k=1 时就是“最偏好的那个专家”。
        # sorted=False 略快一点（k=1 时排不排序都一样）
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1: 
                # k>1 时把选中的 k 个权重重新归一化到和为 1（DeepSeek-V2 的做法），
                # 保证输出尺度不随 k 变化；+1e-20 只是防 0 除的保险
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # k=1 的理论归一化是恒等变换，但作者换成了 STE（直通估计）：
                # 先算出 top1 概率（注意输入 detach 了，梯度不回流到 token 表征）
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0 # k=1: 1.0 forward on purpose, gradient via straight-through
                # 原注释保留 + 展开：
                #   前向： top1 - top1.detach() = 0，所以整体恒等于 1.0
                #          → 输出就是“未加权的专家输出”，和不做归一化完全等价；
                #   反向： d(top1 - detach(top1))/dθ = d top1/dθ，梯度原样回传给 router。
                # 💡 为什么非要这么绕：如果直接把权重写死成常量 1.0，router 就完全不
                #    进计算图，DDP 会报 "unused parameters"，且 router 永远学不到东西。
                #    STE 让“前向不变 + 梯度可回传”两件事同时成立。
        y = torch.zeros_like(x_flat)   # [N,768] 输出容器（fp32/bf16 同 x，由 autocast 决定）
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)   # [N,k] bool：哪些 token（的哪个槽位）选中了专家 i
            if mask.any():
                # 只处理“真的选中了本专家”的那些 token —— 这就是稀疏计算的落点
                token_idx = mask.any(dim=-1).nonzero().flatten()   # [n_i]，被专家 i 接管的 token 行号（k>1 时去重）
                weight = topk_weight[mask].view(-1, 1)             # [n_i,1]，按同样顺序取权重
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
                # index_add_ 按行号累加：同一个 token 被多个专家（k>1）选中时会自动相加；
                # 用原地 index_add_ 而不是 python 循环赋值，避免建大量中间张量
                # weight [n_i,1] 广播到 [n_i,768]；.to(y.dtype) 处理 autocast 下的 dtype 不一致
            elif self.training:
                # 这个专家本轮一个 token 都没接到 → 它的参数完全没进计算图。
                # DDP 的梯度 all-reduce 只处理“本次 backward 出现过的参数”，没出现的
                # 会被当成 unused，下一轮就可能报 "Expected to have finished reduction
                # in the prior iteration"（要么开 find_unused_parameters=True，代价是
                # 每步多一遍图遍历）。
                # 0 * sum(p.sum()) 是“数值上等于 0、但依赖所有参数”的项，加上它就等于
                # 告诉 autograd 这些参数“用过了”。
                # ⚠️ 只在 training 时加；eval 没有 DDP，加了纯属浪费。
                # 💡 加在 y[0,0] 上而不是新建变量，是为了让它一定留在输出路径上，
                #    否则可能被当成死代码 / 不参与 loss 而被剪掉。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # 负载均衡损失（Switch Transformer / DeepSeek 的 aux loss）
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # [N,k] → [N,k,E] → 对 (N,k) 求均值 → [E]
            # load[e] = 分给专家 e 的 (token, 槽位) 比例，∑_e load[e] = 1
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
            # scores.mean(0) → [E]：所有 token 对每个专家的平均路由概率
            # 公式： aux = E * coef * Σ_e ( load_e * mean_score_e )
            #   完全均匀时 load_e = mean_score_e = 1/E，乘起来求和 = 1/E，再乘 E 归一化到 1；
            #   塌缩到单一专家时该值会涨到 E —— 所以最小化它就是在“逼 router 均衡”。
            #   coef=5e-4 很小：负载均衡只是辅助目标，绝不能压过语言建模主 loss。
            # ⚠️ aux_loss 只是挂在 self 上，由 MiniMindModel 汇总后通过输出对象返回；
            #    trainer 如果忘了把它加进 total loss，这一项就等于没写。
        else:
            # eval 或系数为 0 时给一个 0 维标量，保证上层 sum 不会因为 None 报错
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)   # [N,768] → [B,T,768]；y 是新建的连续张量，可以 view

class MiniMindBlock(nn.Module):
    """一个 Transformer block：pre-norm 残差 + 注意力 + 前馈。

    作用：
        y = x + Attn(Norm(x))
        z = y + MLP(Norm(y))

    参数：
        layer_id  仅为可读性/调试保留，本实现没用到（有些实现会传给 cache 做分层索引）
        config    MiniMindConfig

    形状：
        全程 [B, T, 768]，不变

    设计理由（为什么 norm 放在子层之前 = pre-norm）：
        post-norm（原始 Transformer）是 x + Norm(Attn(x))，残差主干要穿过 LayerNorm；
        深层网络的梯度要连乘很多个 LN 的雅可比，必须靠精心设计的 warmup 才训得起来。
        pre-norm 给残差留了一条“恒等高速公路”：梯度可以从最后一层直接流回 embedding，
        不算任何 LN，8 层以上优势非常明显，所以现在所有主流 LLM 都是 pre-norm。
        🔍 和 nanoGPT 对照：nanoGPT 的 Block 就是 ln_1 在 attn 前、ln_2 在 mlp 前，
        可以直接对上；区别只是这里用 RMSNorm 且没有 LN 的 bias。
        副作用：pre-norm 的残差流范数会随层数累积增长，所以 MiniMindModel 最后
        必须再补一次 self.norm（post-norm 不需要，因为它每层都在归一化）。
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        """构造注意力子层与 MLP 子层，以及两个前置 RMSNorm。

        参数：
            layer_id  层号（本实现未使用，保留是为了和上游日志/调试对齐）
            config    MiniMindConfig
        返回：
            None
        """
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)           # 注意力之前
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # MLP 之前
        # 用 use_moe 在“稠密 FFN”和“MoE FFN”之间二选一，两者对外的形状契约完全一致
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """一次 block 前向，返回 (hidden_states, present_key_value)。

        参数：
            hidden_states       [B, T, 768]
            position_embeddings (cos, sin)，各 [T, 96]
            past_key_value      本层上一轮的 (k, v)
            use_cache           是否返回新的 cache
            attention_mask      [B, T_total] padding mask
        返回：
            (hidden_states [B,T,768], present_key_value)
        形状：
            全程 [B, T, 768]
        """
        residual = hidden_states   # 先把残差存下来，因为下一行会把它覆盖
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )   # norm 直接在实参位置做，不需要额外命名；注意只有送进 attn 的那份被归一化
        hidden_states += residual
        # 原地加，省一份 [B,T,768] 的显存。
        # 安全性：hidden_states 这时是 self_attn 的返回值（全新张量），改它不会污染
        # residual；如果两者是同一个对象，这里就会算错。
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        # 第二次残差必须基于“注意力之后”的新值，所以用 + 赋给新变量，
        # 而不能 += residual（那样会把注意力之前的输入再加一遍）
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    """不带头部的模型主干：embedding + N 个 block + final norm + RoPE 表。

    作用：
        把 token id 变成上下文相关的隐状态，不负责算 logits、不负责算 loss。

    参数：
        config  MiniMindConfig

    形状：
        input_ids     [B, T]         int64
        hidden_states [B, T, 768]

    返回：
        (hidden_states [B,T,768], presents 列表, aux_loss 标量)
        presents 是每层的 (k, v)，长度 = num_hidden_layers

    设计理由：
        和 HF 的 LlamaModel 同构，但返回顺序是 (hidden, presents, aux_loss)，
        这是本仓库自定义的契约（多了 aux_loss），trainer 直接按位置解包。
    """
    def __init__(self, config: MiniMindConfig):
        """构造 embedding、N 层 block、final norm，并预计算 RoPE 表。

        参数：
            config  MiniMindConfig
        返回：
            None
        形状：
            embed_tokens.weight  [6400, 768]
            freqs_cos / freqs_sin [32768, 96]
        """
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)   # [B,T] → [B,T,768]
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)   # pre-norm 结构必须的收尾归一化
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        # 💡 为什么 persistent=False：freqs 完全由 config 决定、随时可重算，存进 state_dict
        #    只会让每个 checkpoint 白白多 2*32768*96*4B ≈ 25MB，而且换 max_position 时
        #    还会和旧权重里的表冲突。
        # 为什么用 register_buffer 而不是普通属性：这样 .to(device) / .half() 的遍历
        # 会自动带上它（虽然 persistent=False 不进 state_dict）。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """主干前向：查表 → 逐层 → final norm。

        参数：
            input_ids       [B, T]  int64；有 cache 时 T 可以只是新进的 1 个 token
            attention_mask  [B, T_total] 0/1 padding mask
            past_key_values list，长度 = 层数，每项 (k,v) 或 None
            use_cache       是否返回新的 KV cache
        返回：
            (hidden_states [B,T,768], presents list, aux_loss 0 维张量）
        形状：
            [B,T] → embed → [B,T,768] → 逐层 → [B,T,768]
        """
        batch_size, seq_length = input_ids.shape   # seq_length 是“本次新进”的长度，不是总长度
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # HF 新版本会把 cache 包成 DynamicCache 之类的对象（有 .layers 属性），
        # 本实现只认 list[tuple]，遇到就直接丢弃（等价于“忽略外部传入的 cache”）。
        # ⚠️ 这意味着如果用 HF 的 generate() 调这个模型，cache 不会被复用，会退化成
        #    每步全量重算（结果正确但慢很多）。本文件自带的 generate 没有这个问题。
        past_key_values = past_key_values or [None] * len(self.layers)   # 统一成“每层一个”，后面循环不用判空
        # 第 0 层的 k 是 [B, past_len, 4, 96]，shape[1] 就是已经算过的 token 数，
        # 也就是本次第一个 token 的绝对位置 —— RoPE 必须知道绝对位置，这是关键一行
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # [B,T] → [B,T,768]；查表就是最简单的“离散 token → 连续向量”
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        # 原注释保留：transformers>=5 用 meta device / to_empty 建模型时，buffer 会是 0。
        # 正常情况下 freqs_cos[0,0] = cos(0*θ_0) = cos(0) = 1，永远不可能是 0，
        # 所以“第一项是 0”就是一个可靠的哨兵：一旦成立，说明 RoPE 表是坏的。
        # ⚠️ 如果不兜底：cos=sin=0 → q/k 旋转后变成 0 → 注意力全废，但 loss 看起来
        #    “还有个正常数值”，非常难查。
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
            # 注意这里用“赋值”而不是 register_buffer 重注册：直接替换属性即可，
            # 因为同名的 buffer 已存在，nn.Module.__setattr__ 会认得它
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        # 只按“本次的绝对位置区间”切片 → 各 [T,96]。
        # 💡 同一份 tables 被所有层共享，因为 RoPE 的角度与层无关，不必每层存一份。
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            # zip 让“第 i 层”和“第 i 层的 cache”一一对应，避免手写下标
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)   # use_cache=False 时这里全是 None，返回值仍然对齐 HF 的接口
        hidden_states = self.norm(hidden_states)   # pre-norm 结构必须在最后补一次归一化
        # sum(list, start)：start 给一个 0 维 0，保证“没有 MoE 层”时也能返回张量而不是 int 0
        # isinstance 过滤掉稠密层，因为它们根本没有 aux_loss 属性
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """带语言模型头的完整模型：主干 + lm_head + loss + 采样生成。

    作用：
        trainer 真正实例化的就是它。forward 里算 next-token 的交叉熵，
        generate 里做自回归采样。

    参数：
        config  MiniMindConfig，None 时自动建一个默认配置

    形状：
        input_ids  [B, T]
        logits     [B, T, 6400]
        loss       0 维标量

    设计理由：
        继承 PreTrainedModel 白拿 save_pretrained / from_pretrained / device_map；
        继承 GenerationMixin 是为了让 HF 生态的 generate 接口存在（但本文件覆写了
        generate，用自己的采样实现）。
    """
    config_class = MiniMindConfig   # from_pretrained 靠它决定读哪个 config 类
    # 告诉 HF “这两个 key 是同一份权重”：save_pretrained 只存一份，load 时按映射补齐。
    # ⚠️ 没有这一行，safe_serialization=True 保存时会因为“共享张量”报错或存出两份。
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        """构造主干、lm_head，并按开关做权重绑定。

        参数：
            config  MiniMindConfig 或 None
        返回：
            None
        形状：
            lm_head.weight  [6400, 768]
        """
        self.config = config or MiniMindConfig()   # 允许 MiniMindForCausalLM() 直接可用，同时兜住 None
        super().__init__(self.config)   # PreTrainedModel 的初始化（dtype/device/post_init 钩子等）
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # [B,T,768] → [B,T,6400]；无 bias，和 embedding 的对称性一致
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        # 💡 直接把 lm_head 的 Parameter 对象赋给 embedding，两者从此是同一块内存：
        #    前向时词表和输出投影永远同步，反向时梯度会自动累加（PyTorch 对同一个
        #    Parameter 在多处使用时会 sum 梯度）。
        #    参数量 6400*768 = 4.92M 只算一份，否则总参数会变成约 68.8M。
        #    为什么能 tie：输入“查表”和输出“打分”本质都是 token ↔ 向量的映射，
        #    共享等于给 embedding 加了一个强正则，对 6400 这种小词表收益明显。
        self.post_init()   # HF 钩子：调用 _init_weights 初始化参数、处理 tie、设置 generation_config

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """前向 + 可选的 next-token 交叉熵。

        参数：
            input_ids       [B, T] int64
            attention_mask  [B, T_total] padding mask
            past_key_values KV cache（list 或 None）
            use_cache       是否返回新 cache
            logits_to_keep  0=算全部位置的 logits；N>0=只算最后 N 个（推理省显存）
            labels          [B, T] int64，-100 的位置不计 loss；None 时只返回 logits
        返回：
            MoeCausalLMOutputWithPast(loss, aux_loss, logits, past_key_values, hidden_states)
        形状：
            logits [B, T, 6400]（logits_to_keep=N 时是 [B, N, 6400]）
            loss   0 维标量
        """
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # 注意顺序：模型返回的第二个是 presents（新的 KV cache），这里直接覆盖同名变量
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits_to_keep=0（默认）→ slice(-0, None) 等价于 slice(0, None) → 全量；
        # logits_to_keep=N>0 → 只保留最后 N 个位置。
        # 💡 推理时其实只需要最后一个 token 的 logits，但默认会把整个 [B,T,768] 都投到
        #    6400 维：T=2048 时要算 2048*6400 个数，纯属浪费。只算最后 1 个能省 99%+
        #    的 logits 显存和 lm_head 计算量。
        # ⚠️ 训练时绝不能开（否则下面和 labels 的错位就对不上了），只在 generate / rollout 里用。
        logits = self.lm_head(hidden_states[:, slice_indices, :])   # [B,T,768] → [B,T,6400]
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # 错位关系（整个训练的核心）：
            #   位置 t 的 logits 预测的是 token t+1，所以
            #   logits 砍掉最后一个位置（它没有 t+1 可预测），
            #   labels 砍掉第一个位置（token 0 前面没有输入去预测它）。
            #   于是 x 的第 t 行和 y 的第 t 行都表示“用前 t+1 个 token 预测第 t+2 个”。
            # .contiguous()：切片后张量不连续，后面要 view，必须先把内存理顺。
            # ⚠️ 如果 logits 已经因为 logits_to_keep 被截短，这里再错位就全错了 ——
            #    所以训练必须传 logits_to_keep=0。
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
            # [B*(T-1), 6400] vs [B*(T-1)]，一次算完整个 batch 的 CE。
            # 内部用 logsumexp 技巧算 log_softmax，比自己写 -log(softmax) 稳得多
            # （后者在极端 logits 下会 log(0) = -inf）。
            # ignore_index=-100：这些位置既不贡献 loss 也不贡献梯度 —— SFT/DPO 的
            # loss mask 就是靠这个实现的（prompt 部分全填 -100）。
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        # 用 MoE 版的输出类（它比普通版多一个 aux_loss 字段）；多返回一个字段对 HF 生态无害

    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    # inference_mode 比 no_grad 更彻底：连 autograd 的版本计数都关掉，采样循环里
    # 几百次 forward 能省一点。
    # ⚠️ 代价是产出的 tensor 不能参与求导 —— 所以 rollout 里如果要算 logps，
    #    得在别的地方重算（或去掉这个装饰器）。
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """手写自回归采样（不是 HF 的那套 generate）。

        参数：
            inputs             [B, L] int64 的 prompt；也可用关键参数 input_ids 传
            attention_mask     [B, L] 0/1
            max_new_tokens     最多生成多少个 token
            temperature        温度：<1 更确定，>1 更随机
            top_p              nucleus 采样的累积概率阈值
            top_k              只在前 k 个候选里采样；<=0 表示关闭
            eos_token_id       停止符 id；None 表示不提前停
            streamer           HF 风格 streamer（有 put/end 方法）
            use_cache          是否复用 KV cache（关掉会变成 O(T^2) 次重算，只用于调试）
            num_return_sequences 同一条 prompt 并行采样几条
            do_sample          False 时退化成贪心解码
            repetition_penalty 重复惩罚系数，1.0 表示不惩罚
        返回：
            默认返回 input_ids [B*n, L+new]；kwargs 里 return_kv=True 时返回
            {'generated_ids': ..., 'past_kv': ...}

        形状：
            input_ids [B*n, L] → 每步 cat 一个 [B*n, 1] → [B*n, L+step]
            logits    [B*n, 1, 6400] → [B*n, 6400] → 采样出 [B*n, 1]

        设计理由：
            不追求性能，追求把 temperature / top_k / top_p / repetition_penalty /
            KV cache 这五件事用最短的代码讲清楚。代价是没有 beam search、
            没有 logits_processor、没有 stopping_criteria。
        """
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        # 兼容 HF 的两种调用习惯：generate(inputs) 或 generate(input_ids=...)
        # repeat：同一条 prompt 复制 n 份并行采样（KV cache 自然也是 [B*n, ...]）
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)   # 允许外部塞入已有的 cache（续写场景）
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)   # [B*n]，哪些行已经吐出 eos
        if streamer: streamer.put(input_ids.cpu())   # HF 的 streamer 约定：先推 prompt，再推新 token
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0   # 已有多少 token 进过 cache
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # 💡 只把“新 token”喂进去：有 cache 时前面算过的不用重算。
            #    第一次 past_len=0 → 全量 prefill；之后 past_len=当前长度 → 只喂 1 个 token。
            #    这就是 KV cache 的全部收益：把每步 O(L^2) 变成 O(L)。
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # 每步把 mask 加长 1。这里不做 padding、也不会提前置 eos，所以新列恒为 1；
            # 但长度必须和 input_ids 对齐，否则 Attention 里 [B,1,1,T_kv] 的广播会错位
            logits = outputs.logits[:, -1, :] / temperature
            # 只取最后一个位置的分布 → [B*n, 6400]。
            # 先除温度：temperature<1 让分布更尖（更确定），>1 更平（更随机）；
            # 除完再进 top_k/top_p 是标准顺序（先调形状，再截断）。
            # ⚠️ use_cache=False 时 outputs.logits 是 [B,T,6400]，[:, -1] 依然取最后一个，逻辑不变
            if repetition_penalty != 1.0:
                # 对每一行单独处理“已经出现过的 token”
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                    # 正 logits 除以 penalty → 变小；负 logits 乘以 penalty → 更负。
                    # 两种情况都在压低该 token 的概率，但因为是“按符号分别处理”，
                    # 不会破坏词表内各 token 的相对排序（若直接减一个常数，会把本来
                    # 排在前面的 token 顺序搅乱）。
                    # ⚠️ 这一步在 temperature 之后做，等价于在“缩放后的 logits”上加惩罚，
                    #    与 HF 的顺序一致。
                    # ⚠️ seen 里包含 prompt 的 token，所以 prompt 里高频出现的词也会被罚
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
                # topk(logits, top_k) 返回 [B,k]（值已降序），[..., -1, None] 取第 k 大的
                # 那个分数 → [B,1]，广播后把所有“比它小”的位置置 -inf。
                # softmax 后这些位置严格为 0，不可能被采样到。
                # ⚠️ 用 < 而不是 <=：第 k 名与第 k+1 名同分时会保留多于 k 个，这是无害的
            if top_p < 1.0:
                # nucleus / top-p 采样：只保留“累积概率刚好覆盖 top_p”的最小候选集
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)   # [B,6400] 降序 + 对应词表下标
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # cumsum 第一次超过 top_p 的位置开始为 True（即“核外”）
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # 把 mask 右移一位，并让第 0 位恒为 False：
                #   这样“刚好越过阈值”的那个 token 会被保留在核内，从它的下一个才开始丢。
                #   ⚠️ 没有这一步，“刚好让累积概率越过阈值的那个 token”会被一起删掉，
                #   而它恰恰是 nucleus 候选集的边界成员（top_p 越小越容易暴露这个问题）。
                #   clone() 是为了避免切片赋值的自我覆盖（右侧用的是左移前的值）。
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
                # scatter 把 mask 从“排序后的位置”散射回原始词表位置，再统一置 -inf
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            # do_sample=True：按概率抽样（multinomial 其实接受未归一化权重，这里仍先 softmax 保持语义清晰）
            # do_sample=False：贪心。此时 temperature/top_k/top_p 几乎失去意义，只有
            # repetition_penalty 还在起作用（它改的是 logits 本身）
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            # 已经结束的行强制填 eos：保持 batch 内形状固定、后续不再变化。
            # ⚠️ 已结束的行仍会继续参与 forward，算力是浪费的 —— 换来的是实现简单
            #    （不需要动态收缩 batch、也不用处理 cache 的变长）
            input_ids = torch.cat([input_ids, next_token], dim=-1)   # [B,L] → [B,L+1]
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)   # |= 是原地“或”，把所有已经出过 eos 的行标记上
                if finished.all(): break   # 全部结束就提前退出，省掉剩下的步数
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        # Agent RL 的多轮 rollout 需要拿到 KV 继续续写，所以留了这个可选出口
        return input_ids

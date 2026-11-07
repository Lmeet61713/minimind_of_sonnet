# ==============================================================================
# 文件：scripts/convert_model.py —— 权重格式转换 / LoRA 合并 / 聊天模板搬运
# 作用：在「MiniMind 原生 .pth（只有一个 state_dict）」与「HuggingFace 模型目录
#       （config.json + model.safetensors(or .bin) + tokenizer 文件）」之间双向转换，
#       另外提供「把 LoRA 合并回 base 权重」和「jinja 模板 ↔ JSON 字符串」两个小工具。
# 在流水线中的位置：
#   上游 ← out/*.pth（pretrain / full_sft / lora / dpo / grpo 任意阶段的权重）；
#   下游 → 一个能直接 from_pretrained 的目录，供 vLLM / llama.cpp / LM Studio /
#          Streamlit WebUI / OpenAI 兼容 API（scripts/serve_openai_api.py）使用。
# 核心思想：
#   1. .pth 里只有张量，没有「模型长什么样」；transformers 目录必须有 config.json
#      才能重建网络。所以转换 = 用同一份 MiniMindConfig 建模型 + load_state_dict + save_pretrained。
#   2. 只存权重时，键名（key 字符串）就是唯一的契约。转成 Qwen3 架构时用 strict=True，
#      任何一个键名对不上都会直接报错 —— 这既是硬约束，也是「结构确实等价」的证明。
#   3. LoRA 推理时是「base 权重 + B@A 的低秩增量」；合并 = 把这步加法提前算好写回 base，
#      之后不再需要 LoRA 模块，推理零额外开销。
# 关键张量/数据：
#   state_dict: Dict[str, Tensor]，键形如 model.layers.3.self_attn.q_proj.weight
#   单层形状：q_proj [768, 768]、k_proj/v_proj [384, 768]（GQA：4 个 KV 头 x head_dim 96）、
#             o_proj [768, 768]、gate/up [inter, 768]、down [768, inter]
#   LoRA 增量：B [out, r] @ A [r, in] → [out, in]，与它要修改的 base weight 同形状
# 阅读建议：
#   先看 convert_torch2transformers_minimind()（最直观：建模型→载权重→save_pretrained），
#   再看 convert_torch2transformers()（多一层「把键名翻译成 Qwen3」），
#   最后看 convert_merge_base_lora()。文件末尾 __main__ 里是调用示例，按需取消注释。
# ==============================================================================
import os
import sys
import json

# 把仓库根目录加入 sys.path，这样无论从哪个目录执行，都能 import 到 model/ 包。
# （__package__ 那行是给部分 IDE / 相对导入场景的占位写法，不影响本脚本运行）
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora

# 转换过程会触发大量 UserWarning（未初始化权重、tied weights、dtype 精度等），
# 属于预期噪音，直接静音，避免真正的错误被淹没。
warnings.filterwarnings('ignore', category=UserWarning)

def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    """把 MiniMind 原生 .pth 转成「自定义架构」的 transformers 目录（保留 MiniMind 类名）。

    作用：用 MiniMindConfig 重建网络 → 载入 state_dict → save_pretrained，
          最后把 ../model/ 下的 tokenizer 一起复制过去，凑成一个完整可加载的目录。
    参数：
        torch_path: str，输入 .pth（纯 state_dict，键名与 MiniMindForCausalLM 一致）。
        transformers_path: str，输出目录（会自动创建）。
        dtype: torch.dtype，保存精度，默认 fp16（体积减半，推理精度基本无损）。
    返回：None（产物全部写盘）。
    形状：不改变任何张量形状，只改变「存储格式 + 附带配置文件」。
    设计理由：
        1. register_for_auto_class() 会往 config.json 写 auto_map，
           使 AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
           能自动找到 MiniMindForCausalLM；没有它，别人只能手动 import 才能加载。
        2. load_state_dict(..., strict=False)：容忍键不完全匹配（例如 checkpoint 里混了
           optimizer 状态、或某个阶段没有 lm_head）。⚠️ 代价是「键名写错」也会静默通过，
           所以务必看返回值的 missing_keys / unexpected_keys。
        3. safe_serialization=False：写老格式 pytorch_model.bin。
           新版 transformers 默认写 model.safetensors，二者等价，只是容器不同。
    """
    MiniMindConfig.register_for_auto_class()  # 把 MiniMindConfig 注册为「可从仓库代码自动加载」的配置类
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")  # 注册模型类，key 是 AutoClass 名称
    # ⚠️ 常见坑：lm_config 是文件末尾 __main__ 里的全局变量，并不在本函数内定义。
    #    如果你想 import 本模块后直接调用这个函数，必须先自己造一个 lm_config，否则 NameError。
    lm_model = MiniMindForCausalLM(lm_config)  # 按 config 建出结构，此时权重是随机的
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 有卡就用卡，加载与转换更快
    state_dict = torch.load(torch_path, map_location=device)  # map_location：把权重直接解到目标设备
    lm_model.load_state_dict(state_dict, strict=False)  # strict=False 的作用与风险见 docstring 第 2 条
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    # 顺便核对参数量：64M 版的数字应落在 60~70（百万）这个量级；明显不对说明 config 与权重不匹配。
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)  # 写 config.json + 权重文件
    # ⚠️ 硬编码的相对路径 '../model/'：必须在本仓库的 scripts/ 目录下执行才能命中。
    #    它保证转换出来的目录自带一份与权重严格配套的 tokenizer（V 必须都是 6400）。
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)  # 复制 tokenizer.json / tokenizer_config.json 等
    # ======= transformers-5.0的兼容低版本写法 =======
    # 背景：transformers 5.x 改了若干配置字段（tokenizer 的 extra_special_tokens、
    # config 里把 RoPE 参数收进 rope_parameters 字典）。这里把产物改写成 4.x 能读的样式，
    # 于是同一个目录在 4.x / 5.x 上都能 load。
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        # tokenizer_class 显式指定为 fast 版；extra_special_tokens 置空，避免 5.x 自动推导出额外字段
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        # 5.x 写的是 rope_parameters 字典，4.x 只认扁平的 rope_theta / rope_scaling。
        # 所以：补回 rope_theta、把 rope_scaling 置 None（本模型不做长度外推），再删掉 rope_parameters。
        # ⚠️ 这段只在上面的 if 里执行：只有 5.x 才会有 rope_parameters 这个键。
        #    如果你把这段代码搬到别的版本上复用，del 一个不存在的键会直接 KeyError。
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


# QwenForCausalLM/LlamaForCausalLM结构兼容生态
def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    """把 MiniMind 权重转成「Qwen3 架构」的 transformers 目录（生态兼容版）。

    作用：MiniMind 的模块命名刻意与 Qwen3 对齐，于是可以直接建一个 Qwen3ForCausalLM /
          Qwen3MoeForCausalLM，把同一份 state_dict 灌进去，再 save_pretrained。
          产物能被 vLLM / llama.cpp / LM Studio / Ollama 等按 Qwen3 架构直接加载。
    参数：
        torch_path: str，输入 .pth。
        transformers_path: str，输出目录。
        dtype: torch.dtype，保存精度，默认 fp16。
    返回：None。
    形状：不改变任何张量形状，只是把 MiniMind 的参数名映射到 Qwen3 的同名参数上。
    设计理由：
        1. 自研结构没有推理框架支持；借用 Qwen3 的类名，就能白嫖整个生态的 kernel 与调度优化。
        2. 这条路要求「结构真的等价」：q/k/v/o 的维度、GQA 的 KV 头数、RMSNorm 的 eps、
           RoPE 的 theta、是否 tie embedding，全部必须一致，所以下面 common_config 逐项搬。
        3. MoE 版本要多做一步重排：MiniMind 是「每个专家一个 Linear」，
           Qwen3MoE 是「所有专家拼成一个大 3D 张量」，见 else 分支里的 gate_up_proj / down_proj。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    # 把 MiniMindConfig 的字段逐项翻译成 Qwen3Config 认识的字段名。
    # 少搬一项，后面 load_state_dict(strict=True) 就会失败（或推理结果悄悄变差）。
    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,  # GQA：KV 头数 4 < Q 头数 8，KV cache 省一半
        # Qwen3 里 head_dim 是显式字段（不一定是 hidden/heads），这里按 MiniMind 的定义算成 96
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,  # RMSNorm 分母里的 eps，防止全零向量除零
        "rope_theta": lm_config.rope_theta,      # RoPE 基频 1e6，决定位置编码各维的「波长」
        "tie_word_embeddings": lm_config.tie_word_embeddings  # 输入 embedding 与 lm_head 是否共享同一份权重
    }
    if not lm_config.use_moe:
        qwen_config = Qwen3Config(
            **common_config, 
            use_sliding_window=False, 
            sliding_window=None  # 不启用滑动窗口：64M 小模型直接全因果注意力，简单且够快
        )
        qwen_model = Qwen3ForCausalLM(qwen_config)  # 结构等价，只是外面套了 Qwen3 的类名
    else:
        # MoE 版额外要：专家总数 4 / 每个 token 激活几个（top-k）/ 专家内部中间维 / 是否归一化 topk 概率
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,  # top-k 路由：每个 token 只走 k 个专家
            moe_intermediate_size=lm_config.moe_intermediate_size,  # 单个专家内部的 FFN 中间维
            norm_topk_prob=lm_config.norm_topk_prob  # 对 top-k 的路由权重重新归一化到和为 1
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)
        # ======= transformers-5.0的兼容低版本写法 =======
        # 5.x 的 Qwen3MoE 把「每专家一个 Linear」换成了「按专家批量存储的 3D 张量」：
        #   MiniMind: experts.{e}.gate_proj.weight [moe_inter, hidden]
        #             experts.{e}.up_proj.weight   [moe_inter, hidden]
        #             experts.{e}.down_proj.weight [hidden,    moe_inter]  x E(=4) 个专家
        #   Qwen3MoE(5.x): experts.gate_up_proj [E, 2*moe_inter, hidden]
        #                  experts.down_proj    [E, hidden,      moe_inter]
        # 所以这里要显式 stack + cat 一次，把布局搬过去。
        if int(transformers.__version__.split('.')[0]) >= 5:
            # 先丢掉所有 'experts.' 开头的键，但保留路由门控 experts.gate.weight（router 不打包）
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                # stack 出 [E, moe_inter, hidden]，再沿 dim=1 拼成 [E, 2*moe_inter, hidden]
                # （前 moe_inter 行是 gate，后 moe_inter 行是 up）
                new_sd[f'{p}.gate_up_proj'] = torch.cat([torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]), torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])], dim=1)
                # down_proj 只需要 stack 到第 0 维：[E, hidden, moe_inter]
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
            state_dict = new_sd

    # strict=True：键名必须严格一一对应。这是「结构等价」的强制校验 ——
    # 一旦 MiniMind 改了模块名而这里没跟上，会立刻报错，而不是静默加载出半个随机模型。
    qwen_model.load_state_dict(state_dict, strict=True)
    qwen_model = qwen_model.to(dtype)  # 转换模型权重精度
    qwen_model.save_pretrained(transformers_path)  # 新版默认写 model.safetensors（比 pickle 更安全）
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained('../model/')  # 同样是从固定的 ../model/ 复制一份 tokenizer
    tokenizer.save_pretrained(transformers_path)

    # ======= transformers-5.0的兼容低版本写法 =======
    # 与 convert_torch2transformers_minimind() 末尾那段完全同理：把 5.x 的字段改写成 4.x 能读的样式。
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    """反向转换：HuggingFace 目录 → 原生 .pth（训练脚本认的那种纯 state_dict）。

    作用：把从网上下载的 transformers 格式权重，还原成 train_*.py 能直接 load 的 .pth。
    参数：
        transformers_path: str，输入目录（含 config.json 与权重文件）。
        torch_path: str，输出 .pth 路径。
    返回：None。
    形状：所有张量统一转 fp16 并存到 CPU，形状不变。
    设计理由：
        1. trust_remote_code=True 是必须的：自定义架构的 config.json 里写了 auto_map，
           模型代码就在本仓库里，需要允许 transformers 执行它。
        2. 逐个 .cpu().half()：既避免把 GPU 张量 pickle 进文件（换机器会加载失败），
           又让文件体积减半 —— 推理场景 fp16 足够。
    """
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)  # 只存权重，不存结构与优化器
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    """把 LoRA 增量合并进 base 权重，产出一个「没有 LoRA 模块」的普通 .pth。

    作用：base.pth + lora.pth → merged.pth。
    参数：
        base_torch_path: str，基模权重（LoRA 训练时被冻结的那一份）。
        lora_path: str，train_lora.py 存出的 LoRA 权重（只有 A/B 两个矩阵）。
        merged_torch_path: str，输出路径。
    返回：None。
    形状：
        LoRA 的 A: [r, in]、B: [out, r]（r 默认 16）→ B@A: [out, in]，与 base 的 weight 同形状；
        合并后每个被适配的 Linear 仍是 [out, in]，模块结构完全不变。
    设计理由：
        1. 先把 apply_lora() 挂上去：merge_lora() 是遍历 named_modules 找带 .lora 属性的层，
           不先挂模块就没有落点。rank 必须与训练时一致（这里用 apply_lora 的默认 16）。
        2. 合并公式：W_merged = W_base + B @ A。
           🔍 对照 peft：peft 里还会乘一个缩放系数 alpha / r，而本仓库的 LoRA 直接相加，
           所以这里必须与 train_lora.py 的实现保持一致，否则合并结果会整体偏大或偏小。
        3. 合并的意义：推理时不再需要「额外一次低秩前向」，也不用手写 LoRA 加载逻辑，
           导出的就是标准 base 结构，可以直接走 convert_torch2transformers()。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)  # 同样的全局 lm_config 依赖
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)  # 先恢复被冻结的 base 权重
    apply_lora(lm_model)  # 给每个 in==out 的 Linear 挂上 lora.A / lora.B
    merge_lora(lm_model, lora_path, merged_torch_path)  # 内部会 load_lora → W += B@A → 存盘
    print(f"LoRA 已合并并保存为基模结构 PyTorch 格式: {merged_torch_path}")


def convert_jinja_to_json(jinja_path):
    """把独立的 .jinja 模板文件转成可以粘进 tokenizer_config.json 的 JSON 字符串。

    作用：读模板 → json.dumps 转义 → 打印一行 "chat_template": "..."。
    参数：
        jinja_path: str，.jinja 文件路径。
    返回：None（结果打印到终端，供你手动粘贴或脚本替换）。
    设计理由：
        chat_template 在 tokenizer_config.json 里是一整条「换行符写成转义序列、双引号再转义一次」
        的 JSON 字符串，直接在 JSON 里编辑它极易出错。所以维护时改 .jinja 源文件，再用本函数生成
        目标字符串；json.dumps 负责把所有换行和双引号正确转义（中文若要保留可读，
        需另加 ensure_ascii=False，这里默认 True；模板本身是英文，所以不设也行）。
    """
    with open(jinja_path, 'r') as f: template = f.read()
    escaped = json.dumps(template)  # 这一步就是「加转义」，产出可直接放进 JSON 的字面量
    print(f'"chat_template": {escaped}')


def convert_json_to_jinja(json_file_path, output_path):
    """把 tokenizer_config.json 里的 chat_template 抽出来，存成独立的 .jinja 文件。

    作用：与 convert_jinja_to_json() 互为逆操作，方便阅读、diff 和编辑模板。
    参数：
        json_file_path: str，含 chat_template 字段的 json（如 ../model/tokenizer_config.json）。
        output_path: str，输出的 .jinja 路径。
    返回：None。
    设计理由：
        json.load 会自动把 转义换行 / 转义引号 之类的写法还原成真实字符，所以写出来就是干净的
        jinja 源码；反过来（改完想放回 JSON）用 convert_jinja_to_json() 再转义一次即可。
    """
    with open(json_file_path, 'r') as f: config = json.load(f)
    template = config['chat_template']  # 读出时已经是「真换行」，不再是字面量 \n
    with open(output_path, 'w') as f: f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")


if __name__ == '__main__':
    # 全局配置：hidden_size 与 num_hidden_layers 决定模型大小（768 x 8 层 = 64M 那一档），
    # max_seq_len=8192 是训练时用的上下文长度，use_moe=False 表示稠密版。
    # ⚠️ 这个 lm_config 被上面所有函数隐式依赖（函数内部直接引用全局名），
    #    所以千万不要随手删掉；要转 MoE 权重就把 use_moe 改成 True 并补齐 num_experts 等字段。
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)

    # convert torch to transformers
    # 权重命名约定：out/{阶段}_{hidden_size}{_moe}.pth，阶段如 pretrain / full_sft / dpo / grpo / ppo_actor
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3'
    # 默认走「Qwen3 架构兼容」这条路（推理生态支持最好）。
    # 若想保留 MiniMind 自己的类名（例如要让别人 trust_remote_code 加载本仓库代码），
    # 把下面这行换成 convert_torch2transformers_minimind(torch_path, transformers_path) 即可。
    convert_torch2transformers(torch_path, transformers_path)

    # # merge lora
    # 把 LoRA 权重合并回 base：三行分别指定 基模 / LoRA / 合并后的输出。
    # 合并完成后就可以直接像普通模型一样推理或再转 transformers 格式。
    # base_torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # lora_path = f"../out/lora_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # merged_torch_path = f"../out/merge_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # 反向：transformers 目录 → .pth（训练脚本可直接 load）
    # convert_transformers2torch(transformers_path, torch_path)
    # 模板双向转换：从 tokenizer_config.json 抽出 jinja / 把 jinja 转回 JSON 字符串
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')

# ==============================================================================
# 文件：eval_llm.py  —— MiniMind 命令行推理 / 对话入口
# ==============================================================================
# 作用：
#   加载训练好的 MiniMind 权重，把用户输入套上 chat template 后送进模型做自回归生成，
#   并以流式（逐 token）方式打印到终端。支持 pretrain / SFT / DPO / 各类 RL 权重、
#   LoRA 插件权重、thinking 开关、多轮历史。
#
# 在流水线中的位置：
#   上游 ← ./out/{weight}_{hidden_size}{_moe}.pth（train_pretrain / train_full_sft /
#          train_dpo / train_ppo / train_grpo / train_agent 等训练脚本产出的权重）
#          以及 train_tokenizer.py 产出的 tokenizer 目录（里面带着 chat_template）
#   下游 → 不产出新权重，只做推理。它是「训练完先看一眼效果」的第一站；
#          同源逻辑被 scripts/serve_openai_api.py（HTTP 服务）和 scripts/web_demo.py
#           （Streamlit WebUI）各自重写了一遍，三者骨架完全一致。
#
# 核心思想：
#   1. 推理是训练目标的反向使用：预训练学的是「给定前文预测下一个 token」，
#      推理就是不断把刚生成的 token 接回输入再预测下一个（KV cache 只是把这一步做快）。
#   2. 对话能力不是模型结构给的，而是「文本协议」给的：<|im_start|>role ... <|im_end|>
#      这套 chat template 把多轮对话压平成一条普通文本，模型依旧只做 next-token 预测。
#   3. thinking 是模板级开关：open_thinking=True 时在 assistant 前缀后插入 <think>，
#      模型先在这个「草稿区」里推理再作答；关掉则直接输出答案。
#
# 关键张量/数据：
#   input_ids:      [B, T_prompt]，本脚本 B=1（单条对话），T_prompt 是模板编码后的长度
#   attention_mask: [B, T_prompt]，1=真实 token，0=pad（单条推理时通常全 1）
#   generated_ids:  [B, T_prompt + T_new]，前 T_prompt 个是原样回填的 prompt，
#                   必须切掉才能只 decode 新生成的部分
#   速度指标：T_new / 秒，单位 tokens/s（含 prefill 时间，短回复会被明显拉低）
#
# 阅读建议：
#   1. 先读 init_model()：理解「原生 torch 权重」与「transformers 目录」两条加载分支，
#      以及 LoRA 的「先插结构、再灌权重」两步。
#   2. 重点读 main() 里的四步骨架：apply_chat_template → tokenizer → generate → decode，
#      所有推理脚本（服务端 / WebUI / 评测）都是这四步的变体。
#   3. 可以跳过：prompts 示例列表、argparse 参数声明（help 里已经写清楚了）。
# ==============================================================================
import os
import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
# 屏蔽 transformers / torch 的一堆版本提示、pad_token 未设置之类的警告，避免刷屏盖住生成结果
warnings.filterwarnings('ignore')

def init_model(args):
    """按需加载「原生 torch 权重」或「transformers 目录」，并返回 (model, tokenizer)。

    参数：
        args: argparse.Namespace，用到 load_from / hidden_size / num_hidden_layers /
              use_moe / inference_rope_scaling / save_dir / weight / lora_weight /
              lora_rank / device 这些字段。
    返回：
        model:     MiniMindForCausalLM，已 half()（fp16）+ eval() + .to(device)
        tokenizer: AutoTokenizer，内部带 chat_template，供 apply_chat_template 使用
    设计理由：
        用「目录名是不是 model」来区分两种权重格式，是仓库约定：
        --load_from model  表示 ./model 目录下放的是原生 state_dict（.pth）；
        --load_from 别的路径 表示放的是 HF 格式目录（config.json + 权重 + tokenizer）。
        这样同一份脚本既能跑自己训练出来的 ckpt，也能跑从 Hub 下载的导出模型。
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    # ⚠️ 判断依据是「路径最后一级目录名是否恰好等于 model」，不是看文件后缀；
    #    所以自定义权重目录务必命名成 model，否则会走到 transformers 分支然后报错
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        # 分支一：原生 torch 权重。模型结构必须现场 new 出来，且超参必须与训练时一致，
        # 否则下面 load_state_dict(strict=True) 会因为键名/形状对不上直接崩
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        # MoE 版权重的文件名带 _moe 后缀，稠密版没有；这里把后缀拼出来供后面复用
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        # map_location=args.device：直接反序列化到目标设备，避免「先占一份 CPU 内存再拷到 GPU」
        # strict=True：要求 state_dict 的键集合与模型完全一致，多一个少一个都报错，
        #             这是推理场景想要的严格性（训练中断时也能尽早发现权重不对）
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            # LoRA 二段式：先把 A/B 低秩分支「插」进模型结构（此时权重是随机的），
            # 再把训练好的 lora 权重「灌」进去。缺了 apply_lora 就没有 lora.* 这些参数名，
            # load_lora 会一个都匹配不上而静默跳过——表现为「加载了 LoRA 但效果没变」
            apply_lora(model, rank=args.lora_rank)
            # ⚠️ lora_rank 必须和训练时一致，否则 A/B 矩阵形状不匹配，load_state_dict 报错
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    else:
        # 分支二：transformers 格式目录，结构与超参都在 config.json 里，不用手动指定；
        # trust_remote_code=True 是因为 MiniMind 的建模代码是随目录一起分发的
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    # 打印参数量（MoE 模型会打印成 total-Aactive 的形式，方便看清稀疏激活了多少参数）
    get_model_params(model, model.config)
    # half()：转 fp16，显存占用与访存带宽都减半，这是 64M 小模型在消费级显卡上跑长上下文的关键
    # eval()：关掉 dropout 等训练期行为
    # .to(device)：搬到 GPU/CPU。链式写法保证三步都作用在同一个对象上
    return model.half().eval().to(args.device), tokenizer

def main():
    """命令行主循环：解析参数 → 加载模型 → 逐条 prompt 生成 → 流式打印。

    参数：无。所有配置都从命令行读取。
    返回：None。
    流程说明：
        1) 用 argparse 收集加载参数与采样参数；
        2) init_model() 得到 model 与 tokenizer；
        3) 选择「自动跑内置 prompts」或「手动打字输入」；
        4) 每条 prompt 都走：拼接模板文本 → 编码成张量 → generate → 切掉 prompt → decode。
    """
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA rank，需与训练时一致（默认16）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    # temperature / top_p 是采样的两个旋钮：temperature 先把 logits 除以 T 再 softmax（越小越尖），
    # top_p 再做 nucleus 截断（只保留累计概率到 p 的最小候选集合）
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    # open_thinking 只影响 chat template 里 assistant 前缀的写法，不改模型结构
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    # historys 的计数单位是「消息条数」而不是「轮数」，所以要成对（user+assistant）传入，
    # 否则历史里会出现孤立的 assistant 或 user，模板拼出来的角色交替不合法
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    # 内置的 8 条测试问题，覆盖「闲聊 / 常识 / 写代码 / 解释概念 / 推理 / 比较 / 推荐」几个典型场景，
    # 用来快速目测一个权重有没有训练崩（重复输出、乱码、答非所问）
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    # conversation 是「消息列表」，元素形如 {"role": "user"/"assistant", "content": "..."}，
    # 它不是 token，也不是张量，只是给 chat template 用的中间表示
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    # TextStreamer 是 transformers 提供的流式打印器：模型每产出一个 token 就回调它，
    # 它负责 detokenize 并 write 到 stdout。
    # skip_prompt=True：不重复打印我们刚传进去的 prompt（否则会看到问题被打印两遍）
    # skip_special_tokens=True：不打印 <|im_end|> 这类控制符
    # 💡 作者巧思：MiniMindForCausalLM.generate 是自己手写的采样循环，但它同样调用
    #    streamer.put(token_ids) / streamer.end()，所以可以直接复用 transformers 的 TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 两种输入模式共用同一个 for 循环：
    # 手动模式用 iter(callable, sentinel) 把「反复 input() 直到读到空串」包装成迭代器，
    # 于是「自动跑列表」和「手动逐条输入」在下面的循环体里完全同构，不用写两份代码
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        # 每轮都用一个新的随机种子重新播种：目的是让每次采样的随机数流不同，
        # ⚠️ 代价是结果不可复现（种子本身是随机的），要复现必须把种子固定下来
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        # 只保留最后 args.historys 条历史消息；historys=0（falsy）时直接清空 → 单轮问答。
        # 之所以按「消息条数」截断，是为了保证截断后仍以 user 开头、role 交替合法
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        # pretrain 权重只学过「续写网页/书籍文本」，从没见过 <|im_start|>user 这套对话标记，
        # 所以必须绕过 chat template，直接把 bos_token 拼在原始 prompt 前面做纯续写
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            # apply_chat_template 是「把消息列表渲染成一条字符串」的纯文本操作，不涉及张量。
            # tokenize=False 表示先拿文本，稍后再统一编码（方便打印/调试，也方便只编码一次）。
            # add_generation_prompt=True 会在结尾补上 <|im_start|>assistant\n，
            # 相当于把「该你说了」这个提示交给模型。
            # open_thinking：在这行 assistant 前缀后插入 <think>，让模型先思考再回答
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))

        # 文本 → 张量：input_ids [T_prompt] → [1, T_prompt]，attention_mask 同形状，然后搬到设备上。
        # ⚠️ truncation=True 用的是 tokenizer 默认的 truncation_side='right'，
        #    历史很长时会从「右边」截断，可能把结尾的 <|im_start|>assistant 生成提示切掉，
        #    导致模型看到一个不完整的对话前缀而胡言乱语；长对话应在 tokenizer_config 里改 left
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        # generate 只负责「往 input_ids 后面追加 token」，不会修改传入的 inputs；
        # 返回值 generated_ids 是「prompt + 新生成」的完整序列 [1, T_prompt + T_new]
        # 🔍 可以和 nanoGPT 对照的地方：nanoGPT 的 sample() 就是「logits / temperature →
        #    top-k 过滤 → softmax → multinomial → cat 回 idx」这一个循环；MiniMind 的 generate
        #    结构完全相同，只多了两件事：KV cache（每步只算最后一个 token）和 streamer 回调
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            # max_new_tokens 是「新增」上限，不是总长度上限
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            # do_sample=True 走多项式采样（temperature/top_p 才生效）；False 则退化成 argmax 贪心
            # pad_token_id：批量推理时用于补齐；单条推理其实用不到，但不传会触发警告
            # eos_token_id：命中 <|im_end|> 就认为该样本说完，停止生成
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            # repetition_penalty=1 表示关闭重复惩罚。MiniMind 训练时没引入它，
            # 推理时开大了会破坏格式（比如把 <think>/<tool_call> 拆坏），所以默认关掉
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        # 切片 [ (T_prompt + T_new) 中的后半段 ]：prompt 部分原样回填在 generated_ids 里，
        # 必须切掉，否则 decode 出来会把用户问题也当成回复打一遍。
        # skip_special_tokens=True 顺手把结束符 <|im_end|> 去掉
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        # 把回答写回历史，下一轮把它当作上下文一起喂给模型（多轮对话靠的就是这段拼接）
        conversation.append({"role": "assistant", "content": response})
        # T_new = 总长度 - prompt 长度；注意这个数里包含结束符，所以会和肉眼看到的字数略有出入
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        # decode 速度（tokens/s）。分母是完整墙钟时间，包含首 token 的 prefill，
        # 所以短回复的「测速」会明显偏低，想看真实吞吐要跑长输出
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    try:
        main()
    # Ctrl-C（KeyboardInterrupt）和 Ctrl-D / 空输入导致的 EOFError 都直接静默退出，
    # 避免在交互式对话里抛出一大段 traceback
    except (EOFError, KeyboardInterrupt):
        print()

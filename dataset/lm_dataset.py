# =============================================================================
# dataset/lm_dataset.py —— MiniMind 全流水线共用的 5 个 Dataset
#
# 作用：把原始 jsonl 语料变成训练要吃的张量（或字符串 prompt）。这里是整条流水线
#       唯一的「数据入口」，决定了模型看到什么文本、以及「哪些 token 算 loss」。
#
# 在流水线中的位置：
#   上游 ← model/ 下的 BPE tokenizer（vocab=6400）+ 各类 jsonl 语料
#         （pretrain_corpus.jsonl / sft_*.jsonl / dpo.jsonl / rl.jsonl / agent_rl.jsonl）
#   下游 → trainer/train_pretrain.py      用 PretrainDataset  （next-token 预测）
#         trainer/train_full_sft.py      用 SFTDataset       （全参指令微调）
#         trainer/train_lora.py          用 SFTDataset       （LoRA 微调）
#         trainer/train_distillation.py  用 SFTDataset       （teacher → student 蒸馏）
#         trainer/train_dpo.py           用 DPODataset       （chosen/rejected 偏好对齐）
#         trainer/train_ppo.py           用 RLAIFDataset     （PPO，只喂 prompt）
#         trainer/train_grpo.py          用 RLAIFDataset     （GRPO，只喂 prompt）
#         trainer/train_agent.py         用 AgentRLDataset   （多轮工具调用 Agent RL）
#
# 核心思想：
#   1) 同一个 chat 模板（tokenizer_config.json 里的 jinja 字符串）贯穿 SFT / DPO /
#      RL / Agent。训练和推理必须走同一条模板代码路径，否则就会出现「训练时见过
#      的格式，推理时没见过」的静默掉点。所以各 Dataset 只负责「喂什么内容」，
#      「长成什么格式」一律交给 apply_chat_template。
#   2) 预训练是「整段都算 loss」，SFT / DPO 是「只算 assistant 回答那几段」。
#      于是自然分叉出两种屏蔽手段：SFT 用 ignore_index=-100（交给 CE 忽略），
#      DPO 用 0/1 乘法 mask（因为要自己乘 per-token logp 再求和）。
#   3) 所有随机性（要不要 system prompt、要不要保留空 think、要不要开思考）都放在
#      Dataset 里做。目的是让模型在训练阶段就见过多种分布，而不是推理时第一次遇到。
#
# 关键张量/数据：
#   PretrainDataset : input_ids [T] int64、labels [T] int64（pad 位置 -100）
#   SFTDataset      : input_ids [T]、labels [T]，两者等长；错位（[:-1] vs [1:]）
#                     由 model/model_minimind.py 的 forward 内部完成，不在数据集里做
#   DPODataset      : x / y / mask 各 [T-1]，其中 x=ids[:-1]、y=ids[1:]，mask∈{0,1}
#   RLAIFDataset    : dict(prompt: str, answer: "")，此时还没有 token id
#   AgentRLDataset  : dict(messages: list[dict], tools: list | None, gt: dict)
#   其中 T = max_length（本文件所有 Dataset 都 pad 到定长），B 由 DataLoader 决定，
#   D = 768、V = 6400 由模型配置决定。
#
# 阅读建议：
#   先读 PretrainDataset（最短，把 bos / eos / pad / -100 四件事一次讲透）；
#   再读 SFTDataset，重点是 generate_labels 怎么在 token 流里「扫段」；
#   然后读 DPODataset，重点看 loss_mask 为什么返回 0/1 而不是 -100；
#   RLAIFDataset / AgentRLDataset 只产 prompt、非常短，最后扫一眼即可。
#   pre_processing_chat / post_processing_chat 建议对照 tokenizer_config.json 里的
#   chat_template 一起读，才能看懂 <think> 标签究竟是谁插进去的。
# =============================================================================
from torch.utils.data import Dataset
import torch
import json
import os
import random
# 🔍 可以和 nanoGPT 对照：nanoGPT 用 np.memmap 读 .bin、再按 block_size 手动切 (x, y)；
# 这里换成了 HuggingFace datasets 读 jsonl —— 底层同样是 Arrow 内存映射（随机访问 O(1)），
# 但支持「一行一条变长样本」，更适合长度参差的对话数据。
from datasets import load_dataset, Features, Sequence, Value
# ⚠️ 常见坑：DataLoader 的 num_workers > 0 时会 fork 子进程，而 tokenizers 内部有一套
# Rust 线程池，fork 之后极易死锁并刷一屏 warning。必须在构造任何 Dataset 之前关掉它。
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """在对话最前面「概率性」插入一条 system prompt（数据增强式的分布扩展）。

    参数：
        conversations  : list[dict]，每条形如 {'role': 'system'|'user'|'assistant'|'tool',
                         'content': str, ...}，来自 jsonl 的 conversations 字段。
        add_system_ratio: float，注入 system prompt 的概率，默认 0.2。

    返回：
        list[dict]，可能比入参多一条 system 消息；若含 tools 则原样返回同一对象。

    设计理由（为什么要随机加，而不是都加或不加）：
        1) 推理侧（OpenAI 兼容 API、WebUI、eval_llm.py）请求里可能带 system prompt，
           也可能不带。如果训练集 100% 都带，模型会把「永远有一个 system 段」当成
           固定的结构先验，一旦线上没传就分布外了；反过来 100% 不带，
           模型又学不会「跟随 system 指令」这件事。按概率混，两种分布都见过。
        2) 具体 prompt 文本也从 10 条里随机抽（中英各半），避免整份语料出现同一个
           恒定前缀 —— 恒定前缀对模型没有信息量，反而会占掉注意力的「锚点」位置。
    """
    # tool use 数据完整保留不做处理
    # 理由：含 tools 的样本，其 system 消息里已经写着工具 schema，chat_template 也会
    # 因为收到 tools 而走另一条分支，往里面塞通用 system prompt 只会污染工具描述。
    if any(conv.get('tools') for conv in conversations): return conversations

    # 💡 作者巧思：这 10 条 system prompt 只是「分布的代表」，不是知识。
    # 真正想教给模型的是：system 段是一个可控的指令区，而不是固定装饰。
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    # 只在「本来没有 system」时注入；数据集自带 system 的样本一律尊重原值，不覆盖。
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            # ⚠️ 这里用 [system] + conversations 生成新 list，不会原地修改入参，
            # 所以同一个样本被 __getitem__ 反复取到、每次注入结果不同也不会互相污染。
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2, remove_empty_think=None):
    """按概率删掉文本里「空的 think 块」，避免模型学会输出空思考。

    参数：
        prompt_content    : str，apply_chat_template(tokenize=False) 的整段结果。
        empty_think_ratio : float，「保留」空 think 的概率（默认 0.2，即 80% 删除）。
        remove_empty_think: bool | None，None 表示本函数自己掷骰子；传 True/False 则强制。

    返回：
        str，可能被删掉了若干处 '<think>\\n\\n</think>\\n\\n'。

    设计理由：
        chat_template 对每条 assistant 消息都会无条件输出
        '<think>' + reasoning_content + '</think>' 再加正文。当样本没有
        reasoning_content 时，就得到 '<think>\\n\\n</think>\\n\\n' 这一坨空标签。
        如果全保留，模型很快会学到「think 段永远是空的、直接输出这一坨然后答」，
        真正需要思考时也懒得想；如果全删掉，模型又没见过「空 think」这种形态，
        而推理时 open_thinking=0 的模板恰恰会注入它 —— 那就是训练/推理不一致。
        所以用 8:2 的比例混，两种形态都进模型。
    """
    # 以80%概率移除空思考标签
    # ⚠️ 命名有点反直觉：empty_think_ratio 是「保留概率」而非「移除概率」。
    # random.random() > 0.2 成立的概率是 0.8，也就是说 80% 的样本会被删。
    if '<think>\n\n</think>\n\n' in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            # ⚠️ 只做一次全量 replace：模板里每轮 assistant 都会带这一段，一次替换全部清掉。
            # 若将来模板改了换行数（比如少一个 \n），这里会静默匹配不上 —— 只能靠
            # 训练时打印样本肉眼发现，属于隐藏耦合，改模板时务必同步改这里的字面量。
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    """预训练数据集：把一整段纯文本变成一个定长的 next-token 预测样本。

    每条样本返回 (input_ids, labels)，形状都是 [T]（T = max_length）。
    labels 与 input_ids 内容相同，只是把 pad 位置改成 -100；真正的「错一位」
    （用第 t 个位置预测第 t+1 个 token）由模型 forward 里的 labels[..., 1:] 完成。
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """参数：
            data_path : str，jsonl 路径，每行至少含一个 'text' 字段。
            tokenizer : 已加载的 BPE tokenizer（vocab=6400，bos=<|im_start|>，eos=<|im_end|>）。
            max_length: int，最终张量的定长 T，默认 512。

        设计理由：用 datasets 的 Arrow 表做后端，语料只加载一次；多个 DataLoader worker
        共享同一份内存映射，不会每个 worker 再复制一份数据到自己的进程里。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        """返回样本条数（不是 token 数），DataLoader 靠它算 epoch 长度。"""
        return len(self.samples)

    def __getitem__(self, index):
        """取第 index 条文本，切成定长 (input_ids, labels) 张量对。

        返回：
            input_ids: LongTensor [T]，[BOS] + 正文 + [EOS] + [PAD] * 若干
            labels   : LongTensor [T]，与 input_ids 相同，但 PAD 位置为 -100
        """
        sample = self.samples[index]
        # ⚠️ max_length - 2：先给 BOS / EOS 各留一个位置，保证「加完特殊 token 之后」
        # 总长仍然 <= max_length。如果这里写 max_length，后面再包 BOS/EOS 就会超长，
        # 拼 pad 时长度变负数、得到比 max_length 更长的序列，collate 时直接崩。
        # add_special_tokens=False：特殊 token 由我们手动加，否则 tokenizer 可能再加一遍。
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 💡 作者巧思：BOS 用 <|im_start|>、EOS 用 <|im_end|>，与对话格式共用同一套标记。
        # 好处是预训练阶段模型就学会了「<|im_end|> 意味着这段结束」，SFT 阶段不用重新学。
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # pad 到定长：定长才能让默认 collate 直接 stack 成 [B, T]，也避免动态 shape
        # 让 torch.compile 反复重新编译。pad_token 是 <|endoftext|>，与 eos 不同 id。
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # 🔍 可以和 nanoGPT 对照：nanoGPT 是 y = x 整体错一位、没有 pad 概念；
        # 这里先把 labels 复制成 input_ids，再把 pad 位置打成 -100，交给
        # F.cross_entropy(..., ignore_index=-100) 忽略。为什么不填 0？因为 id=0
        # 恰好是 pad/unk 这个真实词，填 0 等于强迫模型去预测 pad，会稀释梯度。
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    """指令微调数据集：整段对话按 chat_template 拼好，只对 assistant 的回答算 loss。

    返回 (input_ids, labels)，两者都是 [T]，T = max_length。labels 在提问部分（system /
    user / tool 段）为 -100，只在 assistant 回答段（含结尾 <|im_end|>）保留真实 token id。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """参数：
            jsonl_path: str，每行形如 {'conversations': [{role, content, ...}, ...]}。
            tokenizer : BPE tokenizer（bos=<|im_start|>，eos=<|im_end|>）。
            max_length: int，定长 T，默认 1024。

        💡 为什么手写 Features：HuggingFace 推断 schema 时是看前若干行样本的，如果某个
        字段（如 reasoning_content / tools / tool_calls）在开头这些行里全是 null，
        Arrow 会把该列推断成 null 类型，后面遇到非空行就直接报类型错误。显式声明成
        Value('string') 之后，缺失字段统一变成 None，行与行之间不会互相拖累。
        另外 conversations 这里用「list-of-dict 字面量」声明即可，Sequence 用不上。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 🔍 这两个不是「单个 id」，而是一段 token 序列，这是本文件最关键的一个细节：
        #   '<|im_start|>assistant\n' -> [1, 1388, 570, 811, 234]  ('ass'+'ist'+'ant'+'\n')
        # 因为 BPE 把 assistant 拆成了 3 个 subword。所以后面 generate_labels 里
        # 只能用「切片相等」来匹配，不能写成 input_ids[i] == bos_id。
        # 由于加了 trailing '\n'，这个模式只会命中「assistant 回合的开头」，
        # 不会误命中出现在 user 正文里的普通单词 assistant。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        # '<|im_end|>\n' -> [2, 234]，模板在每条消息结尾都会输出 <|im_end|> 加换行。
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        """返回对话条数。"""
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """把一条结构化对话渲染成 chat_template 字符串（不做 tokenize）。

        参数：
            conversations: list[dict]，可能含 system / user / assistant / tool 四种 role。

        返回：
            str，形如 '<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...'。

        设计理由：
            tools 必须从 system 消息里「拆」出来，作为 apply_chat_template 的独立关键字
            参数传入。因为 chat_template 里是 {%- if tools %} 这种顶层分支：带上 tools 时
            模板会写一整段 '# Tools ... <tools>[...]</tools>' 的函数签名说明、并把 system
            正文拼在前面；不带 tools 时只输出普通 system 段。把 JSON 留在 system.content
            里是渲染不出工具块的。
        """
        messages = []
        tools = None
        for message in conversations:
            # ⚠️ 必须 dict(message)：Arrow 表返回的行是只读映射，直接改 tool_calls 会报错。
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                # jsonl 里 tools 存的是 JSON 字符串，而模板需要的是真正的 list[dict]
                # （它要遍历 tool、读 tool.type / tool.function.name）。
                # 这里兼容两种形态：字符串就 loads，已经是对象就直接用。
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # 同理，tool_calls 也要从 JSON 字符串还原成 list，模板才能拿到
            # tool_call.function / tool_call.arguments 这两个字段。
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        # tokenize=False：先拿回字符串，才能让 post_processing_chat 做字符串替换。
        # add_generation_prompt=False：SFT 要训练「完整对话包括最后的 assistant 回答」，
        # 若置 True 模板会额外补一个 '<|im_start|>assistant\n'，等于多出一个没有答案的空回合。
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """在 token 序列里「扫段」，把 assistant 回答区间标出来，其余留 -100。

        参数：
            input_ids: list[int] 或 LongTensor [T]，已经 pad 到 max_length 的完整序列。

        返回：
            list[int]，长度与 input_ids 完全一致（= max_length）。

        实现（本质是字符串匹配的 token 版）：
            1) 从前往后找 bos_id（'<|im_start|>assistant\\n'），找到就说明一个 assistant
               回合开始了；
            2) 从该回合的正文起点往后找 eos_id（'<|im_end|>\\n'），找不到就说明这句被
               截断了，一路标到序列末尾；
            3) 把 [正文起点, eos 结束位置) 这段的 token id 抄进 labels，其余保持 -100。

        💡 作者巧思：为什么不按 messages 一条条拼、非要事后再扫一遍 token？
            因为 apply_chat_template 是个黑盒 jinja 字符串模板（见 tokenizer_config.json），
            它在 assistant 段里还会插 '<think>'、处理 tool_calls、控制空行。如果在 Python
            里重新实现一遍「哪一段是回答、字符偏移是多少」，就等于把模板逻辑抄了第二份，
            模板一改就两边漂移。改成「在最终 token 流里认标记」，只要标记不变就永远对齐。
            这也顺带解决了 post_processing_chat 删字符导致偏移量变化的问题。

        为什么要 min(..., self.max_length)：__getitem__ 里已经 pad 到 max_length，
        正常情况下这里恒成立，属于防御性截断（万一将来有地方传进更长的序列）。
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 命中 '<|im_start|>assistant\n'，回答正文从这之后开始。
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 注意 end 可能停在 len(input_ids)（该回答被 max_length 截断，没等到 eos），
                # 此时 range 上界被 min 钳成 max_length，等于「标到末尾为止」——
                # 截断的回答前半段照样参与训练，这是可接受的。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # ⚠️ 跳过整段到 eos 之后（含 eos 本身），而不是 i += 1 继续扫。
                # 否则 assistant 正文里若出现类似 '<|im_start|>assistant' 的字样，
                # 会被误判成「又开了一个回答」。
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        # 多轮对话会自然命中多次，每一轮 assistant 回答都被标注 —— 所以 SFT 是多轮全监督。
        return labels

    def __getitem__(self, index):
        """取一条对话，产出 (input_ids [T], labels [T])。

        返回：
            input_ids: LongTensor [T]，[渲染出的整段对话 token] + [PAD] * 若干
            labels   : LongTensor [T]，提问段 = -100，assistant 回答段 = 真实 token id
        """
        sample = self.samples[index]
        # 1) 数据增强：按 20% 概率补一条 system prompt（或保持原样）。
        conversations = pre_processing_chat(sample['conversations'])
        # 2) 渲染模板：得到一整段 chatml 文本（含 <think> 结构）。
        prompt = self.create_chat_prompt(conversations)
        # 3) 文本级后处理：按 80% 概率删掉空的 <think></think> 空标签块。
        prompt = post_processing_chat(prompt)
        # 4) tokenize 并从右侧截断。注意 [ : max_length] 是「保头弃尾」：
        #    多轮对话被截断时，丢掉的是最后几轮（通常是最后一个回答的尾巴）。
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 5) pad 到定长。pad_token_id = <|endoftext|>（id 0），与 eos 不同。
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 6) 用 bos/eos 标记在 token 流里定位回答区间，生成 -100 掩码后的 labels。
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        # ⚠️ 这里不返回 attention_mask：调用方直接 model(input_ids, labels=labels)。
        # 因为 pad 全在右侧、因果掩码本来就不让前面的真实 token 看到后面的 pad，
        # 而 pad 位置的 label 又是 -100 不产生 loss，所以省掉这份 [B, T] 内存是安全的。
        # 但反过来，一个 batch 内所有 token 都被 -100 时 CE 会返回 nan，所以
        # max_length 不应小到把回答整段截没。
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """DPO 偏好数据集：同一问题下给出 chosen（好回答）与 rejected（差回答）两条完整对话。

    每条样本返回 6 个 [T-1] 张量：x / y / mask 各有 chosen 和 rejected 两份。
    其中 x = ids[:-1]，y = ids[1:]，mask = loss_mask[1:]，三者天然对齐到「预测目标位置」。
    """

    def __init__(self, file_path, tokenizer, max_length=4096):
        """参数：
            file_path : str，每行含 'chosen' 与 'rejected' 两个 list[dict] 字段。
            tokenizer : BPE tokenizer。
            max_length: int，定长 T，默认 4096（DPO 显存占用大，但 4096 足够覆盖对话）。

        为什么 DPODataset 要自己存 bos_id / eos_id：它需要复刻 SFT 的「扫段」逻辑，
        只把 assistant 回答部分计入 log-prob（mask），所以同样需要这两个标记。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 兜底 pad id：正常 tokenizer 一定有 pad_token_id，这里防的是自定义 tokenizer
        # 返回 None 的情况（None 传给 tokenizer(padding=...) 会报错）。本文件后续没再读它。
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        """返回偏好对数量（一对 = chosen + rejected 两条序列）。"""
        return len(self.samples)

    def __getitem__(self, index):
        """取一对偏好样本，返回 6 个等长张量。

        返回 dict：
            x_chosen / y_chosen / mask_chosen       : LongTensor [T-1]
            x_rejected / y_rejected / mask_rejected : LongTensor [T-1]
        """
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        # chosen / rejected 各自独立走一遍 chat_template（tokenize=False，先拿字符串）。
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        # 💡 作者巧思：同一对的 chosen 与 rejected 共用同一个 remove_empty_think 决定。
        # 如果两边独立掷骰子，就会出现「chosen 删了空 think、rejected 没删」，那么 DPO 学到的
        # 偏好里有相当一部分来自「格式差异」而不是「回答质量差异」—— 这是偏好对齐里很脏的
        # 一种伪信号（reward hacking 的温床）。共用一次采样就把这个变量控住了。
        remove_empty_think = random.random() > 0.2
        chosen_prompt = post_processing_chat(chosen_prompt, remove_empty_think=remove_empty_think)
        rejected_prompt = post_processing_chat(rejected_prompt, remove_empty_think=remove_empty_think)
        # padding='max_length'：DPO 的 chosen 与 rejected 长度天然不同，定长 pad 之后
        # 才能 torch.cat([chosen, rejected], dim=0) 拼成一个 batch 一起前向。
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        # 注意返回的是 1/0，不是 -100 —— 见 generate_loss_mask 的说明。
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # 统一做「错位」：x 是喂给模型的输入，y 是要预测的下一个 token。
        # 第 t 个位置的 logits 预测 y[t]，所以 mask 也要取 [1:] 与 y 对齐。
        # 形状：长度 T 的序列 -> 六个 [T-1] 张量（默认 collate 后是 [B, T-1]）。
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        # mask_chosen[t] 回答的是「y_chosen[t]（即 ids[t+1]）这个位置算不算数」，
        # 所以取 loss_mask[1:] 而不是 loss_mask[:-1]，差一位就全错。
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """扫出 assistant 回答区间，返回每位置是否计入 log-prob 的 0/1 掩码。

        参数：
            input_ids: list[int]，长度 T（已 pad 到 max_length）。

        返回：
            list[int]，长度 T，回答段（含结尾 <|im_end|>）为 1，其余（system / user /
            tool / pad）为 0。

        💡 为什么是 0/1 而不是 -100：SFT 把 labels 交给 F.cross_entropy 的
        ignore_index 去忽略，那是「框架帮我们丢」。而 DPO 不算 CE，它要的是
        「整段回答的 log-prob 之和」：
            log pi(y|x) = sum_t mask_t * log_softmax(logits_t)[y_t]
        乘法掩码天然贴合这个式子（见 train_dpo.py 的 (ref_log_probs * mask).sum(dim=1)），
        而 -100 在这种乘法里会把和直接算成 -100 倍的负数，完全错误。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """RL 数据集（PPO / GRPO 共用）：只产 prompt 字符串，不产答案。

    每条样本返回 {'prompt': str, 'answer': ""}。answer 恒为空字符串是刻意的：
    RL 是 on-policy 的，回答必须由当前策略现场采样（rollout），
    拿数据集里的参考答案去训练就变回 SFT 了。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        """参数：
            jsonl_path    : str，每行含 'conversations' 字段（最后一条是参考答案）。
            tokenizer     : BPE tokenizer。
            max_length    : int，此处不做实际截断，保留给调用方对照（真正截断在 trainer 里）。
            thinking_ratio: float，渲染 prompt 时「打开思考模式」的概率，默认 0.5。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        # 本类后续没有再用到 bos_id / eos_id（RL 侧不做 loss mask，只给 prompt 字符串），
        # 保留与 SFTDataset 一致的写法，方便两边对照阅读。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        """返回 prompt 条数。"""
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """把「提问部分」渲染成带生成前缀的 prompt 字符串。

        参数：
            conversations: list[dict]，最后一条是 assistant 的参考答案。

        返回：
            str，以 '<|im_start|>assistant\\n' 结尾（可能还带 '<think>\\n'）。

        设计理由：
            1) conversations[:-1] 丢掉最后一条 assistant 回答 —— 只保留「问题 + 历史」，
               答案交给 rollout 现场生成。
            2) add_generation_prompt=True 让模板在末尾补上 assistant 回合的开头，
               模型从这个位置往后续写就是回答。
            3) open_thinking 是 chat_template 的开关：
                 False -> 模板写入 '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
                          （一个空思考块，等于提示模型「别想了，直接答」）
                 True  -> 模板只写入 '<|im_start|>assistant\\n<think>\\n'
                          （开着思考块，让模型自己把推理写进去，再写 '</think>' 收尾）
               按 thinking_ratio 随机开，模型才能同时学会「该想时想、该直答时直答」。
        """
        # 与 SFT 共用同一条预处理：按 20% 概率注入 system prompt。
        conversations = pre_processing_chat(conversations)
        # 每次取样本都重新掷骰子 —— 同一条数据在不同 epoch 会以「思考/不思考」两种形态出现，
        # 相当于在不增加数据量的前提下把 prompt 分布扩了一倍。
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        """取一条样本，返回 {'prompt': str, 'answer': ""}。

        返回的是「字符串 dict」，默认 collate 会把同名字段聚成 list[str]，
        所以 GRPO / PPO 的 DataLoader 不需要自定义 collate_fn：
        直接拿到 batch['prompt'] 就是长度 B 的字符串列表。
        """
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }


class AgentRLDataset(Dataset):
    """Agent RL 数据集：返回「多轮工具调用历史的起点 + 工具定义 + 标准答案」。

    每条样本返回 {'messages': list[dict], 'tools': list | None, 'gt': dict}，
    其中 messages 已经砍掉最后一条 assistant 回答 —— 那是 agent 要自己 rollout 出来的轨迹。
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """参数：
            jsonl_path: str，每行含 'conversations'（带 tools 的 system 消息 + 多轮工具交互）
                        与 'gt'（标准答案 / 需要命中的关键信息）。
            tokenizer : 保留参数（本类只做结构解析，渲染交给 train_agent.py）。
            max_length: 保留参数（长度控制在 train_agent.py 里做）。

        为什么这里手写 open/json.loads 而不用 load_dataset('json', ...)：
            tools 是嵌套 list[dict]、gt 的结构在不同样本间还可能不同，
            用 Arrow 固定 schema 表达很别扭；数据量也不大，逐行读更直接。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                # ⚠️ 这里没有过滤空行：语料末尾若残留一个空行，json.loads('') 会直接抛异常。
                # 所以 agent_rl.jsonl 必须保证「每一行都是合法 json」，写数据时留意别留尾空行。
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        """返回轨迹条数。"""
        return len(self.samples)

    def parse_conversations(self, conversations):
        """把带 tools 的对话拆成「历史消息」和「工具定义」两部分。

        参数：
            conversations: list[dict]，通常第 0 条是 role=system 且带 tools 字段。

        返回：
            (messages[:-1], tools)
            messages: list[dict]，已去掉最后一条 assistant 回答；
            tools   : list[dict] | None，解析后的工具 schema。

        设计理由：
            与 SFTDataset.create_chat_prompt 一样，tools 必须从 system 消息里提出来
            单独传给 apply_chat_template 的 tools 参数，模板才会渲染 <tools> 块。
            这里不用关心 tool_calls 的类型转换 —— 渲染发生在 train_agent.py 里，
            那边会重新构造 messages，且 agent 的多轮历史本来就是运行时拼出来的。
        """
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        # messages[:-1]：最后一条是 assistant 的参考答案，必须丢掉，否则等于把答案喂给
        # 模型（teacher forcing 泄漏）。agent 要从 user 提问出发，自己决定调哪个工具。
        return messages[:-1], tools

    def __getitem__(self, index):
        """取一条 agent 轨迹，返回 {'messages', 'tools', 'gt'}（全是非张量对象）。

        ⚠️ 三个字段都不是张量，默认 collate 会因为「list[dict] 无法 stack」而报错，
        所以 train_agent.py 里必须提供自定义 collate_fn 把它们聚成 list。
        """
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    # 占位入口：方便直接 python dataset/lm_dataset.py 检查 import 是否报错。
    # 真正的数据构造逻辑全在 Dataset 类里，这里没有可执行的主流程。
    pass

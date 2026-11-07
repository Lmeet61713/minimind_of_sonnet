# ==============================================================================
# 文件：trainer/train_tokenizer.py —— 流水线第 0 步：训练 BPE 分词器（V = 6400）
# 作用：把原始 jsonl 文本喂给 HuggingFace tokenizers 的 ByteLevel BPE 训练器，
#       产出 tokenizer.json / tokenizer_config.json / vocab.json / merges.txt。
# 在流水线中的位置：
#   上游 ← dataset/*.jsonl（预训练的 {"text": ...} 或 SFT 的 {"conversations": [...]}）；
#   下游 → model/ 下的 tokenizer 文件，被 SFT/DPO/PPO/GRPO/评测全部复用，
#          并且它定死了 V = 6400，于是 model_minimind.py 的 embed_tokens 是 [V, D] = [6400, 768]。
# 核心思想：
#   1. 词表是「模型与文本之间唯一的约定」。换一份词表 = 换一套 id 编号，同一份权重里的
#      embedding 行会全部错位，所以社区必须共用同一份 tokenizer（见下面作者的原注）。
#   2. ByteLevel BPE：先把文本转成 UTF-8 字节再做 merge，任何字符都能表示 ⇒ 永不 OOV；
#      代价是中文常被拆成多个 byte 级子词，所以「压缩率 chars/token」是关键验收指标。
#   3. special token 必须登记在 BPE 训练之前：它们按给定顺序抢编号，
#      这样 <|im_start|> / <|im_end|> / <tool_call> 的 id 在所有 checkpoint 里都稳定。
# 关键张量/数据：
#   text:      str，一条训练文本（可能来自 text 字段，也可能是 conversations 拼接）
#   input_ids: List[int]，长度 T，取值 [0, 6400)；没有 batch 维度，是纯 Python list
#   V = 6400 = 36 个 special/buffer token + BPE 学出的普通子词
# 阅读建议：
#   先读 train_tokenizer() 里 BpeTrainer 的构造（special_tokens / initial_alphabet），
#   再读 get_texts() 的两种数据格式兼容；eval_tokenizer() 当「验收脚本」看即可。
#   文件中部那一整段 chat_template 是一整条 jinja 模板字符串（4000+ 字符），第一次可整段跳过。
# ==============================================================================

# 注：不建议再重复训练tokenizer（“词典”），MiniMind已自带，此脚本仅供学习和参考。基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

# 🔍 对照 nanoGPT：nanoGPT 直接复用现成的 character-level / tiktoken 词表，这里则亲手训一份。
#    注意 V 是写进模型结构的超参（embed_tokens 的行数），训练前必须定死，之后无法改。
DATA_PATH = '../dataset/sft_t2t_mini.jsonl'  # 训练语料：示例用 SFT 数据，也可指向预训练 jsonl
TOKENIZER_DIR = '../model_learn_tokenizer/'  # 输出目录：故意另起一个，避免覆盖正式使用的 ../model/
VOCAB_SIZE = 6400        # V：总词表大小。全仓库都假设 6400，改了就与 MiniMind 官方权重不兼容
SPECIAL_TOKENS_NUM = 36  # 预留 36 个「非普通子词」的位置（特殊符 + 缓冲符），见下面 num_buffer
MAX_LINES = 0  # 0 表示读取全部；设为正数则只取前 N 条（快速试跑）

def get_texts(data_path, max_lines=MAX_LINES):
    """逐行产出用于训练 BPE 的文本，兼容两种主线数据格式。

    pretrain_t2t(_mini).jsonl 是 {"text": ...}，
    sft_t2t(_mini).jsonl 是 {"conversations": [{"role":..., "content":...}, ...]}。
    原先只解析 conversations，所以把 DATA_PATH 指向预训练语料时一条文本都取不到，
    BPE 仅基于 initial_alphabet 训练，产出一个没有任何 merge 的词表，且全程不报错。

    作用：流式读取 jsonl，把每行还原成一段纯文本，交给 tokenizer.train_from_iterator 消费。
    参数：
        data_path: str，jsonl 路径，一行一个 JSON 对象。
        max_lines: int，最多产出多少条；0 表示不限（全量训练）。
    返回：
        生成器（generator），每次 yield 一条非空 str；一条都没取到则 raise ValueError。
    形状：
        str（长度不固定）→ ByteLevel 切成字节 → merge 成 List[int]，长度 T 视文本而定。
    设计理由：
        1. 用 yield 而不是 list(...)：语料可能上 GB，全读进内存会 OOM；
           train_from_iterator 本来就接受「一次性迭代器」，边读边训正好。
        2. 两种格式用一个函数吞掉，是为了让 BPE 统计到的字符分布与实际训练语料一致。
        3. 末尾的 used == 0 兜底，是给「静默失败」解药：空语料训出来的词表只剩
           initial_alphabet 的 256 个字节 + special token，模型能跑但压缩率极差，且不报错。
    """
    used = 0  # 只统计「真正产出」的条数，供最后的空语料检查使用
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        # errors='ignore'：脏数据（非法 UTF-8 字节）直接跳过，不让整个训练崩掉
        for line in f:
            if max_lines and used >= max_lines:  # max_lines=0 时短路，永远不进这个分支
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # 空行 / 被截断的行 / 非 JSON 行：跳过而不是中断整个训练
            if 'text' in data:
                text = str(data['text'])  # 预训练格式：一行就是一整篇文档
            else:
                # SFT 格式：把这条样本里所有 role 的 content 串起来。
                # 连 assistant 的回复也拿来训 BPE，是因为 BPE 只统计「字符 n-gram 频率」，
                # 不区分谁说的；多用一份数据，子词切分会更贴合真实分布。
                contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                text = "\n".join(contents) if contents else ''  # 用换行拼接，避免两句话粘成一个词
            if text.strip():
                used += 1
                yield text  # 纯空白文本对 BPE 统计没有贡献，提前过滤，省得白跑
    if used == 0:
        # ⚠️ 常见坑：训练集路径写错时 train_from_iterator 不会报错，只是「没有样本」，
        # 最终得到一个只有 256 字节的退化词表。这里主动抛错，把静默失败变成显式失败。
        raise ValueError(
            f'{data_path} 中没有可用文本：每行应为 {{"text": ...}} 或 '
            f'{{"conversations": [{{"role":..., "content":...}}, ...]}}。'
            f'继续训练只会得到一个空词表。'
        )

def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM):
    """训练一份 MiniMind 词表，并把配套的 json 全部写到 tokenizer_dir。

    作用：搭 ByteLevel BPE、登记 special token、扫语料学 merge，最后落盘成
          tokenizer.json + tokenizer_config.json（含 chat_template）+ vocab/merges。
    参数：
        data_path: str，语料 jsonl，格式兼容性见 get_texts()。
        tokenizer_dir: str，输出目录。
        vocab_size: int，目标总词表大小，本仓库固定 6400。
        special_tokens_num: int，special + buffer token 的总配额，本仓库固定 36。
    返回：None（所有产物直接写盘）。
    形状：
        产出的是「子词 ↔ id」映射表，规模 V = 6400；
        之后任意文本 → input_ids（长度 T）→ 模型侧 [B, T] → embedding 后 [B, T, D]。
    设计理由：
        1. vocab_size 是硬上限：BpeTrainer 会一直 merge 到词表装满为止，
           所以 6400 同时决定 embedding 行数（6400 x 768）和平均每个 token 装多少字。
        2. special token 用 special_tokens= 在训练时就传进去，而不是训完再追加，
           这样它们进入「不可再切分」集合，并且在词表尾部占一段连续编号。
        3. 36 个里有一部分是空位（buffer），留给未来新增功能 token：
           加功能时不必重训词表，也就不必改模型结构。
    """
    tokenizer = Tokenizer(models.BPE())  # 空模型：此时还没有任何 merge 规则
    # ByteLevel：先把文本编码成 UTF-8 字节，再在字节序列上做 BPE（GPT-2 同款方案）。
    # add_prefix_space=False：句首不补空格，中文语料上更贴近原始文本。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 第 1 组：聊天协议 + 多模态/语音占位符。MiniMind 主线是纯文本，但保留这些 token
    # 可以让 tokenizer 与 Qwen 系的 chat template、以及未来的多模态扩展保持一致。
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", 
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", 
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    
    # 第 2 组：Agent / 工具调用 / 思维链专用标记，train_agent.py 和 eval_toolcall.py 都依赖它们。
    # 💡 作者巧思：它们没有进上面的 special_tokens_list，所以 special 标记为 False ——
    #    这样 decode(skip_special_tokens=True) 不会把它们吞掉，评测脚本才能用正则抓到 <tool_call>。
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    # 用「36 - 已登记标记数」算出还差几个空位，补齐成 <|buffer{i}|>。
    # ⚠️ 常见坑：若往上面加了新标记却忘了同步 special_tokens_num，这里会算出 0 或负数，
    #    词表就悄悄少几个 special token，而且不会有任何报错。
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)] # 预留一定数量的token位置
    # special token 在 BpeTrainer 里的先后顺序 = 它们的 id 顺序（统一排在普通子词之后）。
    # 一旦重排这个列表，同一个 <|im_end|> 的 id 就变了，旧权重全部错位 ⇒ 这个列表视同冻结。
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,  # 目标 V；BPE 会持续 merge 直到装满（或语料里已无可 merge 的 pair）
        show_progress=True,     # 进度条：能直观看出是在正常 merge 还是在空转
        # 先把 256 个字节符号全部塞进词表。ByteLevel 的前提是「任何字节都可表示」，
        # 少了这一步，语料里没出现过的字节就会变成 OOV。
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens  # 这些串整体入表、永不被切碎，并优先占用编号
    )
    texts = get_texts(data_path)  # 惰性生成器：执行到这一行时还一条数据都没读
    tokenizer.train_from_iterator(texts, trainer=trainer)  # 真开始扫语料：数 pair、反复 merge
    tokenizer.decoder = decoders.ByteLevel()  # 解码器：把 token 的字节串拼回 UTF-8 文本
    # 只把第 1 组重新登记为 special（第 2 组保持普通 token，见上面 additional_tokens_list 的注释）。
    tokenizer.add_special_tokens(special_tokens_list)

    os.makedirs(tokenizer_dir, exist_ok=True)  # 目录可已存在，exist_ok 避免重复跑时报错
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))  # 完整 tokenizer 的 JSON 描述
    tokenizer.model.save(tokenizer_dir)  # 额外导出 HuggingFace 老格式的 vocab.json + merges.txt
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    # BpeTrainer 会把 special_tokens= 里的每个串都写成 special=True。
    # 这里把第 2 组（<tool_call>/<think>）以及 buffer 空位改回 False，目的是让这些
    # 结构标记在 decode(skip_special_tokens=True) 时仍被保留 —— 评测要靠正则抓 <tool_call>；
    # 而 <|im_start|>/<|im_end|> 依旧是 special，跳过它们不影响人读输出。
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in special_tokens_list:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)  # ensure_ascii=False：中文可读
    
    # added_tokens_decoder 是 tokenizer_config.json 里描述「额外 token」的字段：
    # key 是 id 的字符串形式，value 说明该 token 的行为（是否 special、要不要吞空格等）。
    # 它是 slow tokenizer（以及不少下游库）识别 <|im_start|>/<|im_end|> 的依据；
    # 缺了它，这些串可能被当成普通文本、被 ByteLevel 切成多个子词，
    # 于是同一句话编码出的 id 与训练时不一致 —— 模型效果会莫名其妙地崩。
    added_tokens_decoder = {}
    for i, token in enumerate(all_special_tokens):
        idx = tokenizer.token_to_id(token)  # 训练完成后反查 id，顺序即上面 all_special_tokens 的顺序
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,      # 编码时不要在左侧吞空格，否则文本还原会失真
            "normalized": False,  # 不做 unicode 归一化，保证 id 严格可逆
            "rstrip": False,      # 同理，右侧也不吞空格
            "single_word": False, # 不要求它必须是独立单词（中文没有空格分词）
            "special": True if token in special_tokens_list else False
        }

    # tokenizer_config.json 是「tokenizer 的人设卡」：它告诉 transformers 该用哪个类、
    # 哪些串算 special token、以及 apply_chat_template 要渲染哪段 jinja 模板。
    config = {
        # 不自动加 BOS/EOS：chat template 里已经明明白白写了 <|im_start|> 和 <|im_end|>，
        # 若再自动加一次就会出现两个 <|im_start|>，训练和推理的输入分布直接对不上。
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,  # 与上面 pre_tokenizer 的 add_prefix_space 保持一致
        "added_tokens_decoder": added_tokens_decoder,
        # <|endoftext|> 由 pad/unk 字段单独声明，这里排除掉避免重复登记。
        "additional_special_tokens": [t for t in special_tokens_list if t not in ["<|endoftext|>"]],
        "bos_token": "<|im_start|>",  # 句首标记：对话开始
        "clean_up_tokenization_spaces": False,  # 不要自作主张清理空格，会破坏字节级可逆性
        "eos_token": "<|im_end|>",    # 句尾标记：模型学到「该停了」就生成它
        "legacy": True,               # 兼容老版本 transformers 的配置格式
        "model_max_length": 131072,   # tokenizer 自身的长度上限（真正的上下文窗口由模型 max_seq_len 决定）
        "pad_token": "<|endoftext|>",  # 没有专门的 pad，借用 endoftext：batch 内补齐位置会被 attention_mask 屏蔽
        "sp_model_kwargs": {},         # 只有 SentencePiece 系 tokenizer 才用得到，这里留空占位
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",  # ByteLevel 理论上不会 OOV，兜底复用 endoftext
        # 以下 8 个字段是多模态/语音的占位声明：主线纯文本用不到，
        # 但保留它们可以让同一份 tokenizer 直接喂给 Qwen-VL 风格的处理器。
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        # ⚠️ 上一行 chat_template 是整段 jinja 模板（换行转义成了一行 JSON 字符串）。
        #   它来自：本文件里手写（照抄 Qwen 系的对话模板），最终由下面 json.dump 写进
        #   tokenizer_config.json —— 并不是 tokenizers 库自动生成的。
        #   它负责把 messages 渲染成模型认识的 <|im_start|>role 换行 content<|im_end|> 换行 结构，
        #   同时负责：有 tools 时把工具签名塞进 system prompt、把 role=tool 的结果包成
        #   <tool_response>...</tool_response> 放进 user 轮。想改对话格式就改它，
        #   但改完必须与 dataset/lm_dataset.py、eval_llm.py、train_agent.py 的假设对齐。
        #   小技巧：scripts/convert_model.py 里的 convert_json_to_jinja() 可以把它导出成
        #   单独一个 .jinja 文件，改起来比在 JSON 字符串里艰难地编辑 \n 舒服得多。
        "tokenizer_class": "PreTrainedTokenizerFast"  # 用 fast tokenizer（Rust 实现，支持批处理与偏移映射）
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)  # indent=4：人眼可读，方便手改
    print("Tokenizer training completed.")

def eval_tokenizer(tokenizer_dir):
    """加载刚训好的 tokenizer，把「能编码 / 能解码 / 压缩率 / 流式解码」四件事验一遍。

    作用：手工验收脚本。不是单元测试，但跑完能一眼看出词表有没有训坏。
    参数：
        tokenizer_dir: str，train_tokenizer() 写出的那个目录。
    返回：None（全部结果 print 到终端）。
    形状：
        messages（Python list）→ new_prompt（str，约 100 字符）
        → input_ids（List[int]，长度 T）→ decode 回 str，要求与 new_prompt 完全相同。
    设计理由：
        1. 先 apply_chat_template(tokenize=False) 拿到纯文本再 print，
           是为了肉眼检查 <|im_start|>/<|im_end|> 拼得对不对 —— 这是 SFT 最关键的一步，
           模板错了 loss 照样能降，但模型学到的对话格式就是错的。
        2. decode 一致性（response == new_prompt）检验「编码可逆」：
           只有 ByteLevel 解码器 + 正确的 special token 配置才能做到 token→文本 无损。
        3. 压缩率 chars/token 是分词器的核心 KPI：它决定同样上下文窗口能装多少字。
           这个数越低，说明子词切得越碎，模型要学的序列越长、越费算力。
    """
    from transformers import AutoTokenizer  # 局部 import：训练路径本身不依赖 transformers
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)  # 读 tokenizer_config.json + tokenizer.json
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自月球'},
        {"role": "user", "content": '你到底来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    # tokenize=False 表示「只要渲染后的字符串，不要 id」。
    # 这里故意保留多轮历史（system + 2 轮 user/assistant），用来检查每轮是否各自被
    # <|im_start|>...<|im_end|> 包住，且轮与轮之间恰好有一个换行。
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-'*100)
    print(new_prompt)
    print('-'*100)
    print('tokenizer词表长度：', len(tokenizer))  # 期望正好是 6400（V 有没有被悄悄改大/改小）
    model_inputs = tokenizer(new_prompt)  # 默认只返回 input_ids / attention_mask
    print('encoder长度：', len(model_inputs['input_ids']))  # 这句 prompt 会花掉多少 token
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)  # 不跳过 special，才能做可逆对比
    print('decoder一致性：', response == new_prompt, "\n")  # 期望 True；False 说明有字符被吃掉或空格异常
    print('-'*100)
    print('压缩率测试（Chars/Tokens）：')
    # 4 条样本分别覆盖：纯中文 / 纯英文 / 中英混排。压缩率 = 字符数 / token 数，
    # 例如 200 字只花 100 个 token ⇒ 2.0 chars/token，越大越省上下文。
    test_texts = [
        # 中文样本 (约200字)
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。",
        # 英文样本 (约200词/字符)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 混合样本
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
    ]
    
    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)  # List[int]，长度 T（这里没加 special token）
        token_count = len(encoded)
        char_count = len(text)  # Python 的 len(str) 按「字符」数，不是字节数
        compression_ratio = char_count / token_count  # 越小说明切得越碎、越费上下文
        total_compression += compression_ratio
        print(f"样本 {i+1} | 字符数: {char_count:4} | Tokens: {token_count:3} | 压缩率: {compression_ratio:.2f}")
    
    print(f"平均压缩率: {total_compression / len(test_texts):.2f}")
    print('-'*100)
    print('流式解码（字节缓冲）测试：')
    # 💡 作者巧思：为什么需要「字节缓冲」？
    # ByteLevel 的一个汉字常被拆成 3 个字节 token（UTF-8 每字 3 字节）。
    # 若每来一个 token 就 decode 一次，前两个 token 只凑出半个汉字，decode 会吐出
    # 替换字符 '\ufffd'（也就是问号菱形）。所以这里累积 token，只有当前缓冲能解出
    # 「完整且无 \ufffd 的文本」时才打印并清空 —— 这正是线上流式输出（SSE / WebUI）必须做的事。
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)  # 解码整个缓冲，而不是单个 token
        if current_decode and '\ufffd' not in current_decode:  # 只有凑齐完整字符才把缓冲吐出
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []  # 清空缓冲，下一个 token 重新开始攒

if __name__ == '__main__':
    # 两步走：先「造词典」，再「验收词典」。训练只需要跑一次，产物应当提交进仓库。
    # ⚠️ 再强调一次作者开头的原话：MiniMind 已自带 tokenizer，本文件仅供学习。
    #    如果你用这里产出的词表去训模型，得到的权重与社区发布的 checkpoint 无法互相加载。
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)  # 产出 tokenizer.json / tokenizer_config.json
    eval_tokenizer(TOKENIZER_DIR)  # 打印渲染结果、词表长度、压缩率、流式解码过程

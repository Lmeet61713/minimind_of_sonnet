# ==============================================================================
# 文件：scripts/eval_toolcall.py —— 工具调用（function calling）能力评测
# 作用：用 8 条固定用例，让模型自己决定「调哪个工具、填什么参数」，执行工具
#       （这里是 MOCK_RESULTS 假实现）后把结果回灌，再看模型能否收敛出最终答案。
# 在流水线中的位置：
#   上游 ← out/{weight}.pth（本地后端）或任意 OpenAI 兼容服务（api 后端）；
#   下游 → 终端上的多轮对话日志，用来人工判断 SFT / Agent RL 之后工具调用是否学会。
# 核心思想：
#   1. 工具调用 = 「让模型输出一段结构化文本」。模型本身并不执行函数，只是生成
#      <tool_call>{"name": ..., "arguments": ...}</tool_call>；由外部解析、真正执行、再把结果塞回去。
#      所以这里的一切都围绕「字符串协议」，而不是某种特殊算子。
#   2. 两种后端共用同一套 messages 流，差别只在「谁负责生成」和「工具结果怎么回填」：
#      本地后端自己 apply_chat_template；API 后端遵循 OpenAI 的 tool_calls 协议。
#   3. 工具用假实现：评测关心的是「有没有调对」，不是「天气准不准」——
#      这样评测可复现、零网络依赖、也不会因为第三方接口抖动而失败。
# 关键张量/数据：
#   messages:   List[dict]，多轮对话；每轮追加 assistant(含 tool_call) + tool(结果) 两条
#   input_ids:  [1, T]（batch=1）→ 模型内部 [1, T, D]，T 随轮数增长
#   tool_calls: List[dict]，元素形如 {"name": str, "arguments": str(JSON 文本)}
# 阅读建议：
#   先看 TOOLS / TEST_CASES 了解输入长什么样，再读 run_case() 的主循环 —— 这就是 Agent 的
#   ReAct 骨架；然后对照 parse_tool_calls()（本地解析）与 parse_tool_call_from_text()（API 兜底解析）。
#   generate() 与 chat_api() 是两条互斥路径，按你用的后端读一条即可。
# ==============================================================================
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params, safe_math_eval
warnings.filterwarnings('ignore')  # 屏蔽 transformers 的加载/生成警告，避免淹没评测日志

# 工具清单，格式就是 OpenAI function calling 的 JSON Schema：
#   {"type": "function", "function": {name, description, parameters}}
# 其中 parameters 是一段标准 JSON Schema（properties / required）。
# 💡 作者巧思：description 写得非常具体（连「如123+456、2**10、sqrt(144)」都给了示例），
#    因为模型完全靠这段文字来理解「这个工具能干什么、参数该长什么样」——
#    description 就是工具调用能力的「prompt 工程」，写糊了模型就会乱调。
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},
]

# 工具的「假实现」：名字 → lambda(args) → dict。
# 全部是纯本地计算或常量，保证评测结果只取决于模型行为，不受网络和第三方 API 影响。
# safe_math_eval 来自 trainer_utils，用 AST 白名单求值，避免直接 eval 执行任意代码。
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(safe_math_eval(args.get("expression", "0")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "晴"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

# 建一个「名字 → 完整工具定义」的索引，方便 TEST_CASES 只写名字。
TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    """按名字取出完整的工具定义，供 chat template / OpenAI 接口使用。

    作用：把 TEST_CASES 里简写的工具名展开成完整定义（含 description 与 parameters schema）。
    参数：
        names: List[str]，工具名，如 ["calculate_math", "get_current_time"]。
    返回：
        List[dict]，对应的完整 schema（顺序与入参一致）。
    设计理由：
        每条用例只开放 2~3 个工具。这是有意为之：候选越多，模型越容易选错；
        限定子集才能公平地考察「在这几个工具里该选哪个」。
    """
    return [TOOL_MAP[n] for n in names]

# 固定用例：每条覆盖一种「需要工具才能答对」的需求。
# tools 只是名字列表，运行时由 get_tools() 展开成完整 schema。
# 最后一条是英文 + 多工具组合（天气 + 单位换算），用来测跨语言与多步调用。
TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    """按 --load_from 的目录名决定用哪种方式加载模型（本地后端专用）。

    作用：一次性把 model 与 tokenizer 准备好，并打印参数量。
    参数：
        args: argparse.Namespace，用到 load_from / hidden_size / num_hidden_layers / use_moe / save_dir / weight / device。
    返回：
        (model, tokenizer)；model 已 .half() 并 .eval() 并搬到目标 device。
    形状：模型内部为 [B, T, D]，B 恒为 1（一次只评测一条用例）。
    设计理由：
        1. ⚠️ 这里用「目录名是不是正好叫 model」来判断权重格式，是一个约定俗成的 hack：
           以 ../model 结尾 ⇒ 原生 .pth；其他路径 ⇒ transformers 目录。
        2. .eval() 关掉 dropout；.half() 用 fp16 省显存、加速 decode（fp16 对生成质量基本无损）。
        3. 本地权重路径按 out/{weight}_{hidden}{_moe}.pth 拼，与训练脚本的命名保持一致。
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':  # 目录名恰为 model 才算原生权重
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'  # 如 ./out/full_sft_768.pth
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)  # 严格加载：键必须全对
    else:
        # transformers 目录：config.json 里靠 auto_map 找到 MiniMind 的自定义类
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)  # 打印参数量，顺便确认 config 与权重匹配
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    """从模型输出正文里提取所有 <tool_call>...</tool_call> 块（本地后端用）。

    作用：正则抓取 + JSON 解析，把「文本协议」还原成 Python 结构。
    参数：
        text: str，模型刚生成的那段文本（可能包含 1 个、多个、或 0 个工具调用）。
    返回：
        List[dict]，每项形如 {"name": ..., "arguments": {...}}；解析失败的块被丢弃。
    形状：无张量，纯字符串处理；一次生成可能带多个调用（并行调用）。
    设计理由：
        1. re.DOTALL 让 . 也能匹配换行 —— 模型常把 JSON 拆成多行写出，
           不加这个 flag 就只能匹配单行调用，会大面积漏抓。
        2. (.*?) 非贪婪：避免一次吞掉从第一个 <tool_call> 到最后一个 </tool_call> 的整段内容。
        3. ⚠️ except 里只 pass：解析失败的调用会被静默丢掉，看起来就像「模型没调用工具」。
           调 prompt / 训练时若发现模型总是不调用，先打印原始文本确认是不是格式写坏了。
    """
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))  # 去掉首尾空白，模型常多打一个换行
        except Exception:
            pass  # 这段不是合法 JSON，丢弃这一个，不影响其他调用
    return calls


def parse_tool_call_from_text(content):
    """把正文里的 <tool_call> 解析成 OpenAI 风格的 tool_calls（API 后端兜底用）。

    作用：当 API 后端没有返回原生 tool_calls 字段（很多自建服务只吐文本）时，
          从 assistant 的 content 里把调用捞出来，并包装成协议要求的形状。
    参数：
        content: str，assistant 回复的全部文本。
    返回：
        List[dict]，每项 {"id": "call_i", "function": {"name": ..., "arguments": "JSON 字符串"}}；
        一个都没有则返回 None。
    形状：注意 arguments 必须再 json.dumps 回字符串 —— OpenAI 协议里它就是字符串，
          很多服务端会严格校验类型，直接塞 dict 会 400。
    设计理由：
        本地后端输出的是 <tool_call>{"name":..., "arguments":{...}}</tool_call>，
        而 OpenAI 协议要求 arguments 是「字符串化的 JSON」。两种表示只差一次 dumps，
        所以这里顺手做格式归一，让后面 run_case 的代码不用再分叉。
    """
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'  # \s* 容忍标签内外的空格/换行
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",  # 自己造一个 id，服务端只要求同一轮内唯一
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    """真正执行一次工具调用（这里执行的是 MOCK_RESULTS 里的假实现）。

    作用：把两种调用形态统一成「(工具名, 参数字典)」，查表执行，并保证永不抛异常。
    参数：
        call: 本地后端传 dict（{"name":..., "arguments":...}），API 后端传工具名字符串。
        arguments: API 后端传的参数字符串（本地后端为 None，因为参数在 call 里）。
    返回：
        dict，工具返回值；出错时返回 {"error": ...} 而不是抛出。
    设计理由：
        1. 参数可能已经是 dict，也可能是 JSON 字符串（本地/API 两种来源），所以这里现解析。
        2. ⚠️ 工具异常必须被吞掉：一次调用失败不应该让整场评测中断，
           返回 error 文本回灌给模型反而更真实 —— 真实 Agent 也要学会从错误里恢复。
        3. 错误信息截断到 80 字符，避免一条超长报错把上下文撑爆。
    """
    name = call.get("name", "") if isinstance(call, dict) else call  # 两种形态取名字
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args  # 字符串就解析，dict 直接用
    except Exception:
        args = {}  # 参数都解析不了，就当空参调一次，让工具自己报「缺参」
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"未知工具: {name}"}  # 模型幻觉出一个不存在的工具名：如实回报
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):
    """本地后端的一次生成：渲染 chat template → model.generate → 返回新生成的文本。

    作用：把 messages + 工具清单变成一段模型可读的 prompt，自回归解码，流式打印。
    参数：
        model: MiniMindForCausalLM（已 eval、已 half）。
        tokenizer: 与模型配套的 tokenizer。
        messages: List[dict]，完整多轮历史（每轮都会重新渲染整段 prompt）。
        tools: List[dict]，本轮开放的工具 schema。
        args: argparse.Namespace（max_new_tokens / top_p / temperature / device / show_speed）。
    返回：
        str，只包含「新生成」的那段文本（不含 prompt）。
    形状：
        输入 input_ids [1, T] → 生成后 generated_ids [1, T + N]；
        取 [0][len(input_ids[0]):] 得到 [N] 才是新 token。
    设计理由：
        1. tools=tools 会让 chat template 走 tools 分支，把工具签名（JSON Schema）
           写进 system prompt —— 模型「知道有哪些工具」全靠这一步，漏传就完全不会调用。
        2. open_thinking=False：关闭 <think> 思考段输出（对应 reason 模型的开关），
           评测工具调用时不需要长链推理，关掉能省很多 token。
        3. do_sample=True + top_p/temperature：用采样而不是贪心，更接近真实产品表现；
           但也意味着同一用例每次结果可能不同（main 里每条用例会重新随机播种）。
        4. TextStreamer(skip_prompt=True) 边生成边打印，长回答时不会让人干等。
    """
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # skip_prompt：不回显输入
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    # add_generation_prompt=True 会在末尾补上 <|im_start|>assistant\n，让模型接着写「回答/工具调用」。
    # truncation=True：历史变长后截断到 tokenizer/model 允许的最大长度，防止超长报错。
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        # pad_token_id 与 eos_token_id 必须显式给：否则 transformers 会去 config 里找，
        # 找不到时在 batch 生成场景会告警甚至报错。
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    # generate 返回的是「prompt + 新 token」的拼接，必须切掉前 T 个才是真正的输出。
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])  # 新生成的 token 数
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    """API 后端的一次生成：调用 OpenAI 兼容接口，返回 (正文, 工具调用列表)。

    作用：把 messages + tools 丢给服务端，并把「流式分片」拼回完整结果。
    参数：
        client: openai.OpenAI 实例。
        messages: List[dict]，OpenAI 协议的对话历史。
        tools: List[dict]，本轮开放的工具 schema。
        args: argparse.Namespace（api_model / temperature / top_p）。
        stream: bool，是否流式输出。
    返回：
        (content: str, tool_calls: List[dict] or None)
    形状：无张量；流式下 tool_calls 的 arguments 是「逐片拼接出来的字符串」。
    设计理由：
        1. 非流式时优先用服务端返回的原生 tool_calls；若服务端只会吐纯文本，
           再用 parse_tool_call_from_text() 从正文里兜底解析。
        2. 流式时 tool_calls 按 index 分片到达：第一片可能只有 id，第二片才有 name，
           后面几片是 arguments 的片段。必须按 index 累积拼接，否则拿到的是残缺 JSON。
           ⚠️ 这是对接第三方 OpenAI 兼容服务时最容易写错的地方。
    """
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p  # 上限写死 8192：多步推理时要留足空间
    )
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""  # content 可能是 None（模型只调工具、不说话）
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)  # 服务端不支持原生 tool_calls 时的兜底
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)  # 分片自带的槽位号
                # 先补足占位元素，保证 idx 可以当下标直接用（前面的槽位可能还没到）
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                # 下面三处都用 += ：协议里每一片都是「增量片段」，不是完整快照
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)  # 流完了也没有原生调用：再从正文捞一次
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    """跑一条用例的完整多轮流程（Agent 的 ReAct 骨架：想 → 调 → 看 → 再想）。

    作用：反复「生成 → 若含工具调用就执行并回填结果」直到模型不再调用工具为止。
    参数：
        prompt: str，用户问题。
        tools: List[dict]，本轮开放的工具 schema。
        args: argparse.Namespace，含 backend / stream 与生成参数。
        model, tokenizer: 本地后端的模型与分词器。
        client: API 后端的 OpenAI 客户端。
    返回：None（对话直接打印到终端；一次运行是否「成功」由人看日志判断）。
    形状：
        messages 每轮增长 1~2 条；input_ids 的 T 随之变长（每次都重新渲染整段历史）。
    设计理由：
        1. 为什么每次都要重渲染整段历史？因为 generate() 是无状态的：它不知道上一轮发生过什么，
           上下文只能靠 messages 传进去。（也正因为如此，多轮场景没法直接复用上一轮的 KV cache。）
        2. ⚠️ while True 没有最大轮数上限。模型若反复调用同一个工具（很常见的失败模式），
           这里会一直转下去。训练侧的 rollout_single() 反而有 max_turns=3 的保护，
           评测脚本为了看到「模型到底想循环几次」，故意不设上限。
        3. 成功判定很朴素：看工具有没有被调对、参数对不对、最终回答有没有用上工具结果。
           想做自动打分可以照抄 trainer/train_agent.py 里 calculate_rewards() 的规则。
    """
    messages = [{"role": "user", "content": prompt}]
    while True:  # 每一轮 = 一次「模型生成 + 可能的工具执行」
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)  # 从 <tool_call> 标签里解析
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls:
            break  # 模型不再要求调用工具 ⇒ 本用例结束（它的最终回答就是 content）
        # 把两种后端的 tool_calls 归一成同一种 dict：
        #   API 后端拿到的是 OpenAI 的对象（有 .id / .function.name / .function.arguments），
        #   本地后端拿到的是 parse_tool_calls() 的 dict（{"name":..., "arguments":...}）。
        # 用 hasattr 区分「对象」与「dict」，这样下面的循环只需处理一种结构。
        # ⚠️ 注意这条语句整句是一个条件表达式：只有 backend == 'api' 时才真的做转换。
        tool_calls = [{
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls
        # 回填 assistant 消息时两种后端要求不同：
        #   本地后端只放 content —— 因为 <tool_call>...</tool_call> 本来就在正文里，
        #     模板会原样拼进历史，模型下一轮「看得见」自己刚才的调用；
        #   API 后端必须带结构化的 tool_calls 数组，否则服务端会认为「工具结果没有来源」。
        # 这也是本地后端不需要 tool_call_id 的原因（见下面 role=tool 那一步）。
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        # 逐个执行本轮的所有调用（模型可能一次给多个，例如先查时间再算数）。
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            # 本地后端传整个 dict（execute_tool 内部再取 name / arguments）；
            # API 后端传工具名，参数从 arguments 单独传进去。
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            # 工具结果回填的两种写法（⚠️ 混用会直接报错）：
            #   本地后端只给 role=tool：MiniMind 的 chat template 会把这条消息渲染成
            #     <|im_start|>user 换行 <tool_response> 换行 {json} 换行 </tool_response> <|im_end|>，
            #     即「工具结果以 XML 标签的形式塞进 user 轮」，不需要 id 来关联；
            #   API 后端必须额外带 tool_call_id：服务端靠它把结果对上刚才那次调用，
            #     缺了会直接返回 400 invalid_request_error。
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})


def main():
    """命令行入口：解析参数 → 初始化后端 → 选自动/手动模式 → 逐条跑用例。

    作用：把「两个后端 × 两种输入模式」串起来。
    参数：无（全部来自 argparse）。
    返回：None。
    设计理由：
        1. 后端二选一：local 直接用本地权重（验证训练效果），api 打 OpenAI 兼容服务
           （验证部署链路 —— 比如 vLLM / Ollama 起服务后，工具调用协议是否还能正确往返）。
        2. 手动模式用 iter(callable, sentinel) 实现「一直读输入，直到读到哨兵值」，
           比 while True + break 更紧凑（见下面 cases 的构造）。
        3. 每条用例前重新随机播种：让不同用例的采样互不干扰；代价是整场评测不可复现，
           想复现就把 setup_seed 的参数换成固定值。
    设计取舍：
        这里只打印、不自动打分。工具调用质量很依赖语义（选对工具 / 参数合理），
        用眼睛看日志反而比硬编码阈值更可靠；要做批量回归再照抄 train_agent.py 的规则奖励。
    """
    parser = argparse.ArgumentParser(description="MiniMind ToolCall评测")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="推理后端（local=本地模型，api=OpenAI兼容接口）")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='../out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.9, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--show_speed', default=0, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="OpenAI兼容接口的base_url")
    parser.add_argument('--api_key', default='sk-123', type=str, help="OpenAI兼容接口的api_key")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API请求时使用的模型名称")
    parser.add_argument('--stream', default=1, type=int, help="API模式下是否流式输出（0=否，1=是）")
    args = parser.parse_args()

    # 两个后端在这里分家：local 需要真的把权重读进显存，api 只需要一个 HTTP 客户端。
    # 三个变量都可能保持 None，run_case 会按 args.backend 只取需要的那两个。
    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)  # 任何 OpenAI 兼容服务都能接

    # 交互式选模式：0 = 跑 TEST_CASES 里的固定用例（可回归、可对比）；
    # 1 = 手动输入，工具给全集，用来随手试模型。
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})
    # 上面构造的 cases 有两种来源：
    #   自动模式：把 TEST_CASES 里的工具名换成完整 schema，并额外记住 tool_names 供打印；
    #   手动模式：iter(callable, sentinel) —— 不断调用 lambda 读一行输入，直到返回「空 prompt 的 dict」
    #             这个哨兵值时迭代结束（这是 Python 里做「读到 EOF 为止」的惯用手法）。
    # 注意：自动模式只开放用例指定的 2~3 个工具，手动模式才给全部 8 个。
    for case in cases:
        if not case["prompt"]: break  # 手动模式下直接回车 ⇒ 哨兵 ⇒ 提前结束
        setup_seed(random.randint(0, 31415926))  # 每条用例换一个随机种子，避免所有用例共用同一串采样
        if input_mode == 0:
            print(f'📦 可用工具: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')  # 用例之间画分隔线，方便在长日志里定位


if __name__ == "__main__":
    # 典型用法：
    #   本地权重：python eval_toolcall.py --backend local --load_from ../model --weight full_sft
    #   本地目录：python eval_toolcall.py --backend local --load_from ../minimind-3
    #   API 服务：python eval_toolcall.py --backend api --api_base_url http://localhost:8000/v1
    # ⚠️ 本文件里的 parse_tool_calls / execute_tool 与 trainer/train_agent.py 里是两套独立实现，
    #    改动 <tool_call> 协议（比如换字段名）时必须两边一起改，否则训练与评测会各说各话。
    main()

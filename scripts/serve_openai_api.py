# ==============================================================================
# 文件：scripts/serve_openai_api.py  —— MiniMind 的 OpenAI 兼容 HTTP 服务
# ==============================================================================
# 作用：
#   用 FastAPI 把 MiniMind 包成一个「长得像 OpenAI」的服务，只实现一个核心接口
#   POST /v1/chat/completions，但支持流式 SSE、reasoning_content（思考内容）、
#   tool_calls（工具调用）、open_thinking 开关。
#
# 在流水线中的位置：
#   上游 ← ../out/{weight}_{hidden_size}{_moe}.pth（原生 torch 权重）或 transformers 目录；
#          tokenizer 里内置的 chat_template（由 trainer/train_tokenizer.py 写入）
#   下游 → 任何 OpenAI 兼容客户端：scripts/chat_api.py（最小调用示例）、
#          OpenWebUI / FastGPT / Dify 等第三方 UI、scripts/eval_toolcall.py（工具调用评测）
#
# 核心思想：
#   1. 兼容层只做「协议翻译」：HTTP JSON → chat template 文本 → generate →
#      再把文本拆成 content / reasoning_content / tool_calls 三类字段返回。
#      模型本身完全不知道有 HTTP 这回事，协议细节全部被关在这一层里。
#   2. 流式输出不能「生成完再切」：必须在 token 流上一边收一边切，
#      而且要用缓冲区兜住被切两半的 <tool_call> / </think> 标记，否则前端会收到残缺标签。
#   3. 线程模型是这份代码最难读的地方：generate 是同步阻塞调用，直接放在 async 路由里会
#      卡死事件循环，所以要丢到子线程跑，再用 Queue 把「子线程的生产」变成
#      「生成器的消费」；而 FastAPI 会把同步生成器丢进线程池迭代，于是阻塞的 queue.get()
#      不会阻塞 asyncio 的 event loop。
#
# 关键张量/数据：
#   messages / tools:  请求体里的 JSON，纯文本，还没 tokenize
#   new_prompt:        str，chat template 渲染出的一整段对话文本
#   inputs.input_ids:  [1, T_prompt]（B=1，只服务单条对话）
#   generated_ids:     [1, T_prompt + T_new]，切掉前 T_prompt 个才是新生成的
#   SSE chunk（流式单位）: {"choices": [{"delta": {"content"|"reasoning_content"|"tool_calls": ...}}]}
#   tool_calls[i]:     {"id", "index", "type": "function",
#                       "function": {"name": str, "arguments": JSON 字符串}}
#
# 阅读建议：
#   1. 先看文件最末尾的 __main__：搞清楚 model / tokenizer / device 这三个全局变量
#      是在哪里被赋值的，以及服务是怎么起来的（这决定了它只能被直接运行，不能被 import）。
#   2. 再看 @app.post("/v1/chat/completions") 的两个分支，非流式分支最直观，先读它。
#   3. 最后啃 generate_stream_response 的状态机：pending 缓冲 + in_tool_call /
#      thinking_ended / strip_newlines 三个标志位，是整个文件的技术核心。
#   4. 可以跳过：init_model（和 eval_llm.py 的几乎一样，只多了一个 max_seq_len）。
# ==============================================================================
import argparse
import json
import re
import os
import sys

# 把仓库根目录塞进 sys.path，这样从 scripts/ 目录直接运行时也能 import 到 model.xxx；
# __package__ 显式声明成 "scripts" 是为了让基于包的相对导入/模块解析不报错
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

# FastAPI 实例在模块级创建，路由用装饰器挂在它上面；
# 真正启动是在文件末尾 uvicorn.run(app, ...)，所以这个模块既是「应用定义」又是「启动脚本」
app = FastAPI()


def init_model(args):
    """加载模型与分词器，返回 (model, tokenizer)。

    参数：
        args: argparse.Namespace，字段同文件末尾的命令行参数。
    返回：
        model:     MiniMindForCausalLM，fp16 + eval() 模式，已放到全局 device 上
        tokenizer: AutoTokenizer，带 chat_template
    设计理由：
        与 eval_llm.py 同构：目录名恰好是 model 时按「原生 torch 权重」处理，
        否则当成 transformers 目录整体加载。⚠️ 这里 ckp 用的是 '../' 开头的相对路径，
        因为约定要在 scripts/ 目录下执行 `python serve_openai_api.py`。
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            # 服务端比 eval_llm.py 多传了 max_seq_len：它决定 RoPE 预计算表的长度，
            # 也就是这个服务「理论上」能吃下的上下文上限
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        # 注意这里用的是模块级全局 device（只在 __main__ 里赋值），而不是参数里的 args.device
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        if args.lora_weight != 'None':
            # 同 eval_llm.py：先 apply_lora 插结构，再 load_lora 灌数值，两步缺一不可
            apply_lora(model, rank=args.lora_rank)
            load_lora(model, f'../{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    # 这里没用 get_model_params()，而是直接求和后除以 1e6 打印成 M(illion)，
    # 服务端只关心「到底加载了多少参数」，不需要像训练脚本那样区分 MoE 的激活参数量
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    # half() 省显存、eval() 关 dropout、to(device) 上卡；服务端常驻，只做一次
    return model.half().eval().to(device), tokenizer


class ChatRequest(BaseModel):
    """POST /v1/chat/completions 的请求体模型（Pydantic）。

    字段：完全按 OpenAI Chat Completions 的命名，外加两个私货（tools 之外的
          open_thinking / chat_template_kwargs），这样第三方 UI 不改代码就能接。
    设计理由：
        没有声明在模型里的字段（例如 reasoning_effort）会被 Pydantic 默认忽略，
        所以客户端多传参数不会把服务打挂——这正是「兼容层」该有的宽容度。
    注意：
        model 和 messages 没有默认值 → 是必填字段；其中 model 服务端其实不使用，
        只是 OpenAI 协议要求必须带。
    """
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    # 💡 作者巧思：用 Field(default_factory=list) 而不是直接写 `= []`。
    #    Python 的可变默认值在函数定义时只创建一次，会被所有请求共享，
    #    一旦哪里改了它就会出现「上一个请求的 tools 泄漏到下一个请求」的诡异 bug。
    tools: list = Field(default_factory=list)
    open_thinking: bool = False
    chat_template_kwargs: dict = None
    
    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking，返回最终是否启用思考。

        参数：无（读自身字段）。
        返回：bool —— True 表示在 chat template 里插入 <think>，让模型先思考再回答。
        设计理由：
            不同推理框架/UI 传这个开关的写法不一样：
              1) 本服务自定义的顶层 open_thinking；
              2) OpenAI 生态里常见的 chat_template_kwargs: {"open_thinking": true}；
              3) vLLM / SGLang 的写法 chat_template_kwargs: {"enable_thinking": true}。
            三种都认一遍，客户端写哪种都能生效；chat_template_kwargs 为空时直接返回 False。
        """
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False


class CustomStreamer(TextStreamer):
    """把 transformers 的 TextStreamer 改造成「往 Queue 里塞文本」的生产者。

    参数：
        tokenizer: 用于增量 detokenize 的分词器
        queue:     queue.Queue，子线程（generate）写入、主线程（生成器）读取
    设计理由：
        TextStreamer 已经帮我们处理了「一个 token 可能只解出半个汉字 / 半个 UTF-8 字符」
        的增量拼接问题，我们只需要重写它吐最终文本的那个钩子，把结果换个去向（队列）
        而不是打印到 stdout。这样既复用了 transformers 的增量解码逻辑，又能跨线程传递。
    """
    def __init__(self, tokenizer, queue):
        """初始化：沿用父类的 skip_prompt / skip_special_tokens 行为，并保存队列与分词器。

        参数：
            tokenizer: 分词器
            queue:     线程安全队列，on_finalized_text 往里 put 字符串
        返回：None
        说明：
            skip_prompt=True 表示不把 prompt 部分再吐一遍（否则前端会先收到一遍用户问题）；
            skip_special_tokens=True 表示不吐 <|im_start|> / <|im_end|> 这些控制符。
        """
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """父类每当攒够一段可读文本就回调这里：把文本推进队列。

        参数：
            text:       本次新解出的文本片段（可能为空串）
            stream_end: 是否已到流末尾（generate 结束时父类会带 True 调用一次）
        返回：None
        设计理由：
            ⚠️ 用 None 当「结束哨兵」而不是另设标志位：Queue 只能传数据，
            消费端需要一个明确的「不会再有数据了」的信号，None 是最省事的约定，
            主循环读到 None 就 break（读到 dict 则是错误对象，见 generate_stream_response）。
        """
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def parse_response(text):
    """把模型输出的一段原始文本拆成 (正文, 思考内容, 工具调用列表)。

    参数：
        text: str，模型生成的完整文本（不含 prompt）
    返回：
        text:              str，清掉 think 段与 tool_call 段之后的正文
        reasoning_content: str | None，<think> 里的思考内容
        tool_calls:        list | None，符合 OpenAI 结构的工具调用列表（没有则为 None）
    设计理由：
        模型只会「输出文本」，而 OpenAI 协议要的是「结构化字段」，这一层就是文本→结构的翻译：
          - 思考段用 <think>...</think> 包裹 → 抽出来放进非标准的 reasoning_content 字段；
          - 工具调用用 <tool_call>{json}</tool_call> 包裹 → 抽出来放进 tool_calls 字段。
    """
    reasoning_content = None
    # 情况一：完整的 <think>...</think>。re.DOTALL 让 . 能跨行匹配，
    # 否则思考内容里有换行就匹配不到；.*? 是「非贪婪」，避免把后面多个 think 段一口气吃掉
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        # 抽取后要从正文里删掉整段思考（连同后面的空白），否则前端会把思考再显示一遍
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # 情况二：只有 </think> 没有 <think>。这非常常见——因为开启思考时，
    # chat template 已经把 "<|im_start|>assistant\n<think>" 写进了 prompt，
    # 所以模型自己生成的文本是从思考正文开始的，只有结尾那个 </think> 是它吐出来的。
    # 这时用 split('</think>', 1) 一切两半：左半是思考，右半是正式回答
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    tool_calls = []
    # 一段回复里可能有多个工具调用，逐个抽出来转成 OpenAI 的 tool_calls 结构
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            # ⚠️ 两个细节都不能省：
            #   1) arguments 必须是「JSON 编码后的字符串」而不是 dict —— 这是 OpenAI 协议的规定，
            #      很多客户端拿到 dict 会直接崩；
            #   2) ensure_ascii=False 保证中文参数不会被转成 \\uXXXX 转义，日志里才看得懂。
            # id 用「时间戳_序号」拼出来，只要同一轮里唯一即可（协议不要求全局唯一）
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            # 模型偶尔会吐出不合法的 JSON（少括号、多了自然语言）。这里静默跳过，
            # 让这段文本继续留在正文里给用户看，而不是让整个请求 500
            pass
    if tool_calls:
        # 只要抽出了至少一个合法调用，就把所有 <tool_call> 块从正文里删干净
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # 返回值的第三项用 `or None`：空列表在 JSON 里会变成 []，而 OpenAI 协议里
    # 「没有工具调用」应该是字段缺失/null，两种客户端都更安全
    return text.strip(), reasoning_content, tool_calls or None


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    """流式生成器：产出一串 JSON 字符串，每个对应一个 SSE chunk 的 data 内容。

    参数：
        messages:     list，OpenAI 风格的消息列表
        temperature:  float，采样温度（内部把 logits 除以它，越小越确定）
        top_p:        float，nucleus 采样阈值
        max_tokens:   int，最大新增 token 数
        tools:        list | None，工具定义（会原样喂给 chat template）
        open_thinking: bool，是否在 assistant 前缀后插入 <think>
    返回：
        generator[str]：每次 yield 一个 JSON 字符串（不含 "data: " 前缀，前缀由路由层拼）
    设计理由：
        「模型在子线程里吐 token」和「HTTP 响应在生成器里吐 chunk」是两个节奏，
        中间必须有一个解耦层：Queue + 一个状态机，把不定长的文本流切成
        content / reasoning_content / tool_calls 三类增量。
    """
    try:
        # 第一步：消息列表 → 一整段纯文本。tools 不为空时模板会把工具定义渲染进 system 段，
        # 并告诉模型「要用 <tool_call>{json}</tool_call> 的格式发起调用」。
        # open_thinking=True 则在最后的 <|im_start|>assistant 后面追加 <think>
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # 生产者/消费者之间的通道：CustomStreamer 往里放文本，下面主循环 get 出来切块
        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            """子线程任务：跑一次完整的 generate，把文本流式推进队列。

            参数：无（闭包捕获 model / inputs / streamer 等）。
            返回：None。
            设计理由：
                generate 是同步阻塞的（它内部是一个逐 token 的采样循环），
                直接写在 async 路由或生成器里会把整个服务卡住；丢进 Thread 后，
                主线程只管从队列里读，两边通过 Queue 天然同步。
            """
            try:
                model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    # attention_mask 一定要传：pad 位置必须被屏蔽，否则模型会把补齐位当上下文
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
            except Exception as e:
                # ⚠️ 子线程里抛异常不会传到主线程，只会静静死掉，主线程会永远卡在 queue.get()。
                # 所以这里必须自己兜住：把错误塞进队列，再补一个 None 哨兵把消费端放行
                queue.put({"error": str(e)})
                queue.put(None)

        Thread(target=_generate).start()

        # ===== 以下是一个「流式标记解析」状态机 =====
        # 🔍 可以和 nanoGPT 对照的地方：nanoGPT 的解码循环里没有「协议」概念，采到 token 就完事；
        #    这里要多做一层——把 token 流按 <think> / <tool_call> 切开，映射到 OpenAI 的 delta 字段。
        #    多出来的复杂度全部来自「服务化」，不是来自模型本身
        # pending：已经收到但还不知道该发到哪个字段、或者可能被截断的文本缓冲
        pending = ""
        # in_tool_call：已经看到 <tool_call>，正在等 </tool_call> 收齐，期间不往外发
        in_tool_call = False
        # tool_count：已经发出的工具调用序号，用于 delta.tool_calls[].index
        tool_count = 0
        # 没开思考时，一上来就是正式回答；开了思考则先进入 reasoning_content 模式
        thinking_ended = not bool(open_thinking)
        # </think> 后面通常跟两个换行，需要吃掉，否则正文开头会多出空行
        strip_newlines = False

        while True:
            # 阻塞读：子线程每解出一段文本就 get 一次；读到 None 说明生成结束
            text = queue.get()
            if text is None:
                break
            # dict 是子线程塞进来的错误对象，直接原样转成 JSON 推给前端
            if isinstance(text, dict):
                yield json.dumps(text, ensure_ascii=False)
                continue
            pending += text

            # 内层循环：把 pending 里的内容尽量切出去，直到「没东西可发」或「需要等更多字符」
            while pending:
                if strip_newlines:
                    # 吃掉 </think> 之后的换行；如果整段都是换行，pending 会变空，直接跳出等下一块
                    pending = pending.lstrip('\n')
                    if not pending:
                        break
                    strip_newlines = False
                if in_tool_call:
                    # 工具调用必须「等收齐再解析」：JSON 只解析半个一定是错的，
                    # 所以这里一直攒到 </tool_call> 出现为止（找不到就 break 回去继续 get）
                    end = pending.find('</tool_call>')
                    if end < 0:
                        break
                    end += len('</tool_call>')
                    # 切出完整的一块 <tool_call>...</tool_call>，剩下的留给下一轮
                    block, pending = pending[:end], pending[end:]
                    # 复用 parse_response 解析；只取第一个调用（模型一轮一般只发一个）
                    _, _, tool_calls = parse_response(block)
                    if tool_calls:
                        call = tool_calls[0]
                        # delta.tool_calls 里每项都要带 index，客户端靠它把多个调用拼成数组
                        call['index'] = tool_count
                        call['id'] = f"call_{int(time.time())}_{tool_count}"
                        tool_count += 1
                        yield json.dumps({"choices": [{"delta": {"tool_calls": [call]}}]}, ensure_ascii=False)
                    else:
                        # JSON 不合法、解析不出调用：那就当成普通文本发出去，不能让内容凭空消失
                        yield json.dumps({"choices": [{"delta": {"content": block}}]}, ensure_ascii=False)
                    in_tool_call = False
                    continue

                # 当前该盯哪个标记：没结束思考时盯 </think>，否则盯 <tool_call>
                marker = '<tool_call>' if thinking_ended else '</think>'
                # 对应的 SSE 字段名：思考阶段的内容进 reasoning_content，之后进 content
                field = 'content' if thinking_ended else 'reasoning_content'
                pos = pending.find(marker)
                if pos < 0:
                    # 没找到标记，但可能只是「还差几个字符」：
                    # 💡 作者巧思：如果 pending 的结尾恰好是 marker 的某个前缀
                    #   （比如 marker='</think>'，而 pending 以 '</thi' 结尾），
                    #   就把这最后几个字符扣住不发（keep），等下一块文本补齐后再判断。
                    #   否则标记被两个 chunk 切开会永远匹配不上，前端就会看到裸露的 </thi>。
                    keep = next((n for n in range(len(marker) - 1, 0, -1)
                                 if pending.endswith(marker[:n])), 0)
                    end = len(pending) - keep
                    if end:
                        yield json.dumps({"choices": [{"delta": {field: pending[:end]}}]}, ensure_ascii=False)
                        pending = pending[end:]
                    break
                # 标记前面还有正文：先把正文按当前 field 发出去（pos>0 才发，避免空 chunk）
                if pos:
                    yield json.dumps({"choices": [{"delta": {field: pending[:pos]}}]}, ensure_ascii=False)
                pending = pending[pos:]
                if thinking_ended:
                    # 已经能干活了又碰到 <tool_call>：进入「攒工具调用」状态，
                    # 标记本身保留在 pending 里，留给上面的 in_tool_call 分支去切
                    in_tool_call = True
                else:
                    # 碰到 </think>：说明思考结束。把标记丢掉，切到正文模式，
                    # 并打开 strip_newlines 清掉紧跟其后的空行
                    pending = pending[len(marker):]
                    thinking_ended = True
                    strip_newlines = True

        # 生成结束后 pending 里可能还剩一点没发的内容，按当前所处模式兜底发出去
        if pending:
            field = 'content' if thinking_ended else 'reasoning_content'
            yield json.dumps({"choices": [{"delta": {field: pending}}]}, ensure_ascii=False)
        # 最后补一个空 delta + finish_reason：有工具调用就是 "tool_calls"，否则 "stop"。
        # 客户端（以及多轮 Agent 循环）就靠这个字段决定「继续执行工具」还是「结束回合」
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_count else "stop"}]}, ensure_ascii=False)

    except Exception as e:
        # 走到这里说明响应头已经发出去了，没法再改成 500，只能在流里塞一条 error 让客户端自己判断
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """唯一的业务路由：OpenAI 兼容的对话补全接口。

    参数：
        request: ChatRequest，FastAPI 依据类型注解自动从 JSON body 反序列化并做字段校验。
    返回：
        stream=True  → StreamingResponse，媒体类型 text/event-stream（SSE）
        stream=False → dict，一个完整的 ChatCompletion 对象
    设计理由：
        一个接口两种返回形态，是 OpenAI 协议本身的设计：客户端用 stream 字段选择
        「一次性拿全」还是「边生成边拿」。
    """
    try:
        if request.stream:
            # ⚠️ 标准 OpenAI 流在结束时会再发一行 `data: [DONE]`，本实现没有发：
            #    流在生成器耗尽、连接关闭时结束。OpenAI SDK 把「连接关闭」也当作流结束，
            #    所以 scripts/chat_api.py 能正常工作；但个别严格校验 [DONE] 的客户端可能一直等。
            # 每个 JSON 前面拼 "data: "、后面拼两个换行，这是 SSE（Server-Sent Events）的帧格式：
            # 一个空行代表一帧结束，客户端按帧解析。
            # 💡 注意 generate_stream_response 是「普通同步生成器」，StreamingResponse 会把它
            #    放进线程池里迭代，所以里面阻塞的 queue.get() 不会卡住 asyncio 事件循环。
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"
            )
        else:
            # 非流式：一次 generate 拿到全部结果，再统一解析成结构化字段。
            # 这里是同步的阻塞调用，但路由是 async def —— 严格来说会短暂占用事件循环；
            # 单机自用场景可以接受，真要扛并发应该改成 run_in_threadpool
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            # 双保险：MiniMind 的 generate 本身已带 @torch.inference_mode()，
            # 这里再包一层 no_grad 是为了防止换了别的模型实现时误建计算图、白吃显存
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                # 切片起点是 prompt 的长度（shape[1] = T_prompt），只 decode 新生成的部分
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            content, reasoning_content, tool_calls = parse_response(answer)
            # 先摆好必填字段，再按需追加可选字段：协议里「没有思考/没有工具调用」应当是
            # 字段缺失，而不是塞一个空字符串或空数组（不少客户端会因此误判）
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            # 响应体字段名全部照抄 OpenAI：id / object / created / model / choices
            # 其中 model 固定回 "minimind"（服务端不区分客户端传的模型名）
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except Exception as e:
        # 走到这里说明还没开始写响应，可以正常返回 500 + detail
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA rank，需与训练时一致（默认16）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    # ⚠️ device / model / tokenizer 都是「模块级全局变量」，只在这里被赋值。
    #    chat_completions 和 generate_stream_response 直接引用它们（没有走参数传递）。
    #    后果：这个文件只能 `python serve_openai_api.py` 直接运行；
    #    如果用 `uvicorn scripts.serve_openai_api:app` 启动，__main__ 不会执行，
    #    全局变量全是未定义的，第一个请求就会 NameError。这是本文件最隐蔽的坑。
    device = args.device
    model, tokenizer = init_model(args)
    # host="0.0.0.0" 表示监听所有网卡（容器/局域网内其他机器也能访问，公网部署请自行加鉴权）；
    # 8998 是本仓库约定的服务端口，chat_api.py / eval_toolcall.py 的 base_url 要与此对齐
    uvicorn.run(app, host="0.0.0.0", port=8998)

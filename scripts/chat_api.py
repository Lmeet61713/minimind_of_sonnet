# ==============================================================================
# 文件：scripts/chat_api.py  —— 最小可用的 OpenAI 兼容客户端（终端对话 Demo）
# ==============================================================================
# 作用：
#   用官方 openai SDK 连一个「本地部署的 MiniMind 服务」，在终端里做多轮流式对话。
#   它本身不含任何模型代码，全部推理都在服务端完成，是这个仓库里最短的一份「调用方」示例。
#
# 在流水线中的位置：
#   上游 ← scripts/serve_openai_api.py 起的 HTTP 服务（默认 8998 端口，实现了
#          /v1/chat/completions）；也可以是 Ollama / vLLM / SGLang 等任何 OpenAI 兼容后端
#   下游 → 只有终端输出，不产出文件。它的价值是：验证服务端字段（stream / reasoning_content /
#          tool_calls / open_thinking）是否按 OpenAI 协议正确返回，再拿去接 OpenWebUI、Dify 等前端。
#
# 核心思想：
#   1. 生态复用的关键不是「实现完整的 OpenAI API」，而是「实现它的字段子集」：
#      只要 /v1/chat/completions 的 messages / stream / delta 结构对得上，
#      OpenAI SDK 和一大堆第三方 UI 都能直接连过来，一行客户端代码都不用改。
#   2. 流式（SSE）不是「一次返回半句话」，而是服务端把生成切成很多个小 chunk，
#      每个 chunk 只带增量 delta；客户端负责把 delta 拼成完整回复。
#   3. 思考内容走独立字段 reasoning_content，与正式回答 content 分开传输，
#      前端就可以把思考折叠展示，而不用去正则解析正文里的标签。
#
# 关键张量/数据：
#   messages: List[{"role": "user"/"assistant", "content": str}]，纯文本，还没有 tokenize
#   chunk:    服务端推来的一个 SSE 片段；chunk.choices[0].delta.content 是新增文本（str）
#             chunk.choices[0].delta.reasoning_content 是新增思考文本（str，非标准字段）
#   历史切片：conversation_history[-(N or 1):]，N 必须是偶数（一问一答成对）
#
# 阅读建议：
#   1. 先看 client 的 base_url，确认它指向哪个服务（端口/模型名必须和服务端对齐）。
#   2. 再看 if not stream / else 两个分支：非流式读 message.content，流式拼 delta。
#   3. 可以跳过：conversation_history_origin 这种「预留重置」的写法。
# ==============================================================================
from openai import OpenAI

# api_key：本地服务通常根本不校验鉴权，但 OpenAI SDK 强制要求这个字段非空，所以随便填一个
# base_url：⚠️ 11434 是 Ollama 的默认端口，不是本仓库 serve_openai_api.py 的端口（那个是 8998）。
#   直连本仓库服务时要改成 "http://localhost:8998/v1"；
#   连 Ollama 时保持 11434，并保证下面的 model 名是 ollama 里真实存在的模型名
client = OpenAI(
    api_key="sk-123",
    base_url="http://localhost:11434/v1"
)
# stream=True 走 SSE 流式；改成 False 就退化成「等全部生成完再一次性返回」的非流式调用
stream = True
# 保存一份「空的历史」，方便将来做 /clear 之类的重置；.copy() 是浅拷贝，
# 里面存的是 dict 引用，不过本脚本只 append 不原地修改 dict，所以不会互相污染
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
# 携带多少条历史消息（不是「轮」）：必须为偶数才能保持 user/assistant 成对；0 表示不带历史
history_messages_num = 0  # 必须设置为偶数（Q+A），为0则不携带历史对话
# 裸 while True + input()：没有退出条件，靠 Ctrl-C / Ctrl-D 结束进程
while True:
    query = input('[Q]: ')
    # 先把本轮问题追加进完整历史，这样「完整历史」始终是对话的唯一事实来源
    conversation_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        # 服务端的 ChatRequest.model 是必填字段但不会被使用（serve_openai_api.py 忽略它），
        # 所以接本仓库服务时这里写什么都行；接 Ollama 时必须与已拉取的模型名一致
        model="minimind-local:latest",
        # 💡 作者巧思：history_messages_num=0 时，`0 or 1` 求值为 1，切片变成「只取最后 1 条」，
        #   刚好就是刚刚 append 的这条 user 问题（等价于不带历史）；
        #   非 0 时取最后 N 条。一个 or 同时表达了「不带历史」和「带 N 条历史」两种语义。
        messages=conversation_history[-(history_messages_num or 1):],
        stream=stream,
        temperature=0.8,
        max_tokens=2048,
        top_p=0.8,
        # extra_body 是 OpenAI SDK 的「逃生舱」：把非标准字段原样塞进 HTTP JSON body。
        # open_thinking 会被服务端 ChatRequest.get_open_thinking() 读到，从而在 chat template
        # 里插入 <think>，让模型先思考再回答。
        # reasoning_effort 是 OpenAI o 系列的参数，本仓库服务端不消费它；Pydantic 默认忽略多余字段，
        # 所以留着也不会报错（它只是「和 OpenAI 接口长得一样」的装饰）
        extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"} # 思考开关
    )
    if not stream:
        # 非流式：response 是一个完整的 ChatCompletion 对象，直接取第一条 choice 的 message
        # 这里的两层判空是防御「模型什么都没生成」或「内容被安全过滤」的情况，
        # 否则下面取 .message.content 会抛 AttributeError 而不是给出可读的错误
        if not response.choices or response.choices[0].message is None:
            raise ValueError("LLM returned empty or filtered response")
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        print('[A]: ', end='', flush=True)
        # 流式：response 是个迭代器，每次吐出一个 ChatCompletionChunk
        assistant_res = ''
        # 🔍 可以和 nanoGPT 对照的地方：nanoGPT 的 sample() 在本地一边采样一边 print，
        #    这里「逐 token 产出」这件事被搬到了服务端，客户端只负责把片段拼回去；
        #    网络把原本紧挨着的采样循环拉成了一条跨进程的数据流
        for chunk in response:
            # 结尾的统计 chunk（带 usage）常常没有 choices；服务端实现也可能推空 choices，
            # 所以必须先判空再索引，否则会 IndexError
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            # reasoning_content 是非标准字段：有的 SDK 版本会把它收进 model_extra 而不是
            # 生成属性，所以用 getattr 兜底取，取不到就当空串（不会抛异常）
            r = getattr(delta, 'reasoning_content', None) or ""
            c = delta.content or ""
            if r:
                # \033[90m ... \033[0m 是 ANSI 转义：90=亮黑/灰色，让思考内容在视觉上弱化；
                # 结尾的 \033[0m 负责复位颜色，漏掉它终端后面会一直是灰的
                print(f'\033[90m{r}\033[0m', end="", flush=True)
            if c:
                print(c, end="", flush=True)
            # ⚠️ 只累加 content、不累加 r（思考内容）：这样下一轮把历史发回服务端时，
            #    上下文里只有正式回答，不会把上一轮的思考草稿也喂回去。
            #    若想保留思考过程，就得改成带 reasoning_content 字段的消息再 append
            assistant_res += c

    # 把回答写回历史，构成完整的一问一答，供下一轮按 history_messages_num 截取
    conversation_history.append({"role": "assistant", "content": assistant_res})
    print('\n\n')

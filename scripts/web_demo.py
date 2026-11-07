# ==============================================================================
# 文件：scripts/web_demo.py  —— MiniMind 的 Streamlit 聊天 WebUI（含工具调用）
# ==============================================================================
# 作用：
#   用 Streamlit 搭一个浏览器聊天页面：选模型目录、调温度/历史轮数、开关 thinking、
#   勾选工具（最多 4 个），流式显示回答，并把模型的 <think> 思考段、<tool_call> 工具调用
#   渲染成可折叠的卡片；工具调用会真的在本地执行，把结果回填给模型继续生成。
#
# 在流水线中的位置：
#   上游 ← scripts/ 目录下 transformers 格式的模型文件夹（含 config.json / 权重 / tokenizer）；
#          脚本会自动扫描子目录，所以要先 `cp -r minimind-3 ./scripts/` 之类的拷进来
#   下游 → 浏览器页面。它是「Agent 能力」的推理侧对照：
#          trainer/train_agent.py 训练时用的也是「生成 <tool_call> → 执行 → 回填」这套循环，
#          这里用 Streamlit 把它手工复现了一遍。
#
# 核心思想：
#   1. Streamlit 的执行模型是「每次用户交互都把整个脚本从上到下重跑一遍」，
#      所以任何需要跨交互保留的东西（聊天记录、开关状态、模型对象）都必须放进
#      st.session_state 或 @st.cache_resource，不能指望普通变量活下来。
#   2. 用「双轨历史」：messages 只负责页面渲染（可以把工具结果拼成 HTML 卡片），
#      chat_messages 才是真正喂给 chat template 的消息列表（含 role="tool"）。
#      分开的好处是展示层可以随意美化，而不会污染模型看到的上下文。
#   3. 多轮工具调用就是一个 for 循环：生成 → 正则找 <tool_call> → 本地执行 →
#      把 assistant 原文和 tool 结果写回 chat_messages → 用新上下文再生成一次。
#      循环上限 16 轮，防止模型无限调工具把页面卡死。
#
# 关键张量/数据：
#   new_prompt:   str，chat template 渲染出的文本（含 tools 段 / 可能的 <think> 前缀）
#   input_ids:    [1, T_prompt]
#   streamer:     TextIteratorStreamer，按「文本增量」吐出来，直接 for 循环消费
#   answer:       str，本轮生成的原始文本（可能含 <think> / <tool_call> 标签）
#   full_answer:  str，多轮工具调用累积的「展示用」文本（已经混入 HTML 卡片）
#   chat_messages: List[dict]，发给模型的消息；messages: List[dict]，页面上的消息
#
# 阅读建议：
#   1. 先读 main()：prompt 从哪来 → 怎么拼模板 → 怎么起线程生成 → 工具调用循环。
#   2. 再读 process_assistant_content() 的四个 think 分支，理解「流式输出时标签还没闭合」
#      这种中间态为什么必须单独处理。
#   3. 可以跳过：开头那一大段 CSS、LANG_TEXTS 多语言表、TOOLS 的 8 个工具定义（看名字即可）。
# ==============================================================================
import random
import re
import json
import os
import sys
from threading import Thread

import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 同 serve_openai_api.py：把仓库根目录加进 sys.path，才能 import 到 trainer / model 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trainer.trainer_utils import safe_math_eval

# 必须是第一个 st.* 调用：设置页面标题、把侧边栏默认折叠
st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

# 一大段 CSS 注入（unsafe_allow_html=True 才生效）：
# 主要做两件事：一是把 st.button 的默认样式 all: unset 掉再重画成 18px 的圆形小图标按钮，
# 二是用负 margin 把页面顶部/底部的默认留白挤掉，让聊天区更满。
# ⚠️ 这些选择器（.stButton、.stMainBlockContainer）依赖 Streamlit 的 DOM 结构，
#    升级 Streamlit 大版本后样式可能失效——页面还能用，只是按钮变回默认样子
st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
""", unsafe_allow_html=True)

# 设备选择：有卡用卡，没卡退回 CPU（64M 模型在 CPU 上也能跑，只是慢）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 多语言文本
# 一个极简的 i18n 方案：key → {lang: text} 的扁平字典，比引入 gettext 轻得多，
# 对这个只有十来个字符串的页面足够了
LANG_TEXTS = {
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'thinking': '思考',
        'tools': '工具',
        'language': '语言',
        'send': '给 MiniMind 发送消息',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '自适应思考，目前多轮对话或Tool Call共存时思考不稳定',
        'tool_select': '工具选择（最多4个）',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to MiniMind',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 4)',
    }
}

def get_text(key):
    """取当前语言下的文案，缺语言/缺 key 时逐级回退。

    参数：
        key: str，LANG_TEXTS 里的键名。
    返回：
        str，对应语言的文案；都找不到时原样返回 key 本身（页面不会崩，只会露出键名）。
    设计理由：
        两级 .get() 保证「语言没配对」和「key 拼错」都不会抛 KeyError，
        这在 Streamlit 这种「一次小错就整页白屏」的环境里很实用。
    """
    lang = st.session_state.get('lang', 'en')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))

# 工具定义
# 标准的 OpenAI function calling schema：type=function + name/description/parameters(JSON Schema)。
# 这段列表会被原样序列化进 chat template 的 <tools> 段，模型据此学会「有哪些工具、参数长什么样」。
# 💡 注意：这里只有「说明书」，没有实现；真正的执行在下面的 execute_tool() 里
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位转换", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "获取汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
]

# 侧边栏空间有限，给每个工具配一个 2 字中文短名（checkbox 上显示这个，而不是英文函数名）
TOOL_SHORT_NAMES = {
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}

def execute_tool(tool_name, args):
    """本地执行一个工具调用，返回给模型的观测结果。

    参数：
        tool_name: str，工具名（模型在 <tool_call> 里给出的 name）。
        args:      dict，模型给出的 arguments（已从 JSON 解析成 dict）。
    返回：
        dict，形如 {"result": ...}；出错时是 {"error": str}。
    设计理由：
        1) 所有分支都返回 dict，方便统一 json.dumps 后塞进 role="tool" 的消息；
        2) 用 try/except 兜住全部异常——模型可能给出缺参数、类型不对的调用，
           工具崩溃不应该让整个对话崩，把错误当成「观测」回填给模型反而更有教学价值；
        3) 这里全是「假实现」（天气固定晴、汇率固定 7.2），因为没有联网必要，
           重点是跑通「调用 → 观测 → 再生成」的闭环。
    """
    # 函数内 import：只有这个函数用到 datetime，放全局会拖慢脚本首屏（Streamlit 每次交互都重跑）
    import datetime
    try:
        if tool_name == 'calculate_math':
            # 走 trainer_utils.safe_math_eval，它用 ast 白名单求值而不是 eval()，
            # 避免模型生成 __import__('os').system(...) 这种把本机搞挂的表达式
            return {"result": safe_math_eval(args.get('expression', '0'))}
        elif tool_name == 'get_current_time':
            # ⚠️ 取了 timezone 却没用：真实现要 datetime.now(ZoneInfo(tz))，这里演示为主
            tz = args.get('timezone', 'Asia/Shanghai')
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        elif tool_name == 'random_number':
            return {"result": random.randint(args.get('min', 0), args.get('max', 100))}
        elif tool_name == 'text_length':
            return {"result": len(args.get('text', ''))}
        elif tool_name == 'unit_converter':
            # 假装做了换算，实际把输入原样拼成一句「=?」，说明这里是占位实现
            return {"result": f"{args.get('value', 0)} {args.get('from_unit', '')} = ? {args.get('to_unit', '')}"}
        elif tool_name == 'get_current_weather':
            return {"result": f"{args.get('city', 'Unknown')}: 晴, 7~10°C"}
        elif tool_name == 'get_exchange_rate':
            return {"result": f"1 {args.get('from_currency', 'USD')} = 7.2 {args.get('to_currency', 'CNY')}"}
        elif tool_name == 'translate_text':
            return {"result": f"[翻译结果]: hello world"}
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}


def process_assistant_content(content, is_streaming=False):
    """把模型原始文本渲染成可展示的 HTML：think 段折叠、tool_call 变卡片。

    参数：
        content:      str，模型输出的原始文本（可能含 <think> / <tool_call> 标签）
        is_streaming: bool，是否处于流式生成中。为 True 时标签可能还没闭合，
                      需要走「未闭合」的那几个分支，否则页面会在生成过程中闪来闪去
    返回：
        str，可以直接交给 st.markdown(..., unsafe_allow_html=True) 的 HTML
    设计理由：
        模型输出的是**带标记的纯文本**，而用户想看到的是「可折叠的思考 + 一张工具卡片 + 正文」。
        这层转换完全在前端做，好处是原始文本可以直接存进历史、也可以原样再喂回模型，
        不需要在存储时就把结构揉碎。
    """
    # 处理tool_call标签，格式化显示
    if '<tool_call>' in content:
        def format_tool_call(match):
            """把一个 <tool_call>{json}</tool_call> 替换成蓝色的工具卡片 HTML。

            参数：
                match: re.Match，group(1) 是标签内的 JSON 文本。
            返回：
                str，HTML 片段；JSON 解析失败时原样返回匹配到的文本（宁可显示原文也不吞内容）。
            """
            try:
                tc = json.loads(match.group(1))
                name = tc.get('name', 'unknown')
                args = tc.get('arguments', {})
                # 卡片配色走半透明蓝灰（rgba），这样在 Streamlit 的亮/暗主题下都不会太突兀
                return f'<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalling</div><div><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div></div>'
            except:
                return match.group(0)
        # re.sub 的替换函数形式：每匹配到一个 <tool_call> 块就调用一次 format_tool_call，
        # DOTALL 让 JSON 里的换行也能被 .*? 匹配到
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
    
    # 流式生成且开启思考时，一开始就放到折叠里
    # 这个分支专门处理「开了 thinking、但模型还没吐出 </think>」的中间态：
    # 此时没有标签可用，只能靠「正文通常以 \n\n + 我是/您好/你好 开头」这个启发式来切分
    if is_streaming and st.session_state.get('enable_thinking', False) and '</think>' not in content and '<think>' not in content:
        m = re.search(r'(\n\n(?:我是|您好|你好)[^\n]*)', content)
        if m and m.start(1) > 5:
            # 找到了「正文开头」：前面是思考，后面是回答，把思考塞进折叠框、
            # 回答部分原样留在外面。m.start(1) > 5 是为了避免把整段很短的内容误判成思考
            i = m.start(1)
            think_part = content[:i]
            answer_part = content[i:]
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_part.strip()}</div></details>{answer_part}'
        elif len(content) > 5:
            # 还没出现正文特征：整段都当思考，边生成边展示。
            # 💡 作者巧思：外层用 flex-direction: column-reverse，内层给 margin-bottom: auto，
            #    效果是内容变长时视口始终贴着「最新出现的文字」，不用写 JS 就能自动滚动跟随。
            #    配合 max-height:100px + overflow-y:auto，思考再长也只占一小块地方。
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{content.strip().replace(chr(10), "<br>")}</div></div></details>'

    # 分支一：<think> 和 </think> 都在 —— 最理想的情况，直接把中间那段抽出来折叠
    if '<think>' in content and '</think>' in content:
        def format_think(match):
            """把完整的一对 <think>...</think> 替换成「已思考」折叠框。

            参数：
                match: re.Match，group(2) 是思考正文。
            返回：
                str，折叠框 HTML；思考内容只有换行（模型吐了空 think）时返回空串把它整个删掉。
            """
            think_content = match.group(2)
            # 模型有时会产出 "<think>\n\n</think>" 这样的空思考，这种折叠框毫无信息量，直接删掉
            if think_content.replace('\n', '').strip():  # 不是全换行
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)

    # 分支二：只有 <think> 没有 </think> —— 正在思考中（流式到一半）
    if '<think>' in content and '</think>' not in content:
        def format_think_in_progress(match):
            """把「还没闭合」的 <think> 之后的内容全部渲染成「思考中」折叠框。

            参数：
                match: re.Match，group(1) 是 <think> 之后的所有文本。
            返回：
                str，折叠框 HTML。
            """
            tc = match.group(1)
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">思考中...</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto; display: flex; flex-direction: column-reverse;"><div style="margin-bottom: auto;">{tc.strip().replace(chr(10), "<br>")}</div></div></details>'
        # 用 $ 锚定文本结尾：把开始标签之后的全部内容当成「进行中的思考」
        content = re.sub(r'<think>(.*?)$', format_think_in_progress, content, flags=re.DOTALL)

    # 分支三：只有 </think> 没有 <think> —— 最常见的情况
    # 因为开启 thinking 时，<think> 已经被 chat template 写进 prompt 里了，
    # 模型自己生成的部分只剩下「思考正文 + </think> + 回答」，
    # 所以前端看到的永远是一个没有开标签的 think 段
    if '<think>' not in content and '</think>' in content:
        def format_think_no_start(match):
            """把 </think> 之前的所有文本渲染成「已思考」折叠框。

            参数：
                match: re.Match，group(1) 是 </think> 之前的思考正文。
            返回：
                str，折叠框 HTML；思考正文为空时返回空串。
            """
            think_content = match.group(1)
            if think_content.replace('\n', '').strip():
                return f'<details open style="border-left: 2px solid #666; padding-left: 12px; margin: 8px 0;"><summary style="cursor: pointer; color: #888;">已思考</summary><div style="color: #aaa; font-size: 0.95em; margin-top: 8px; max-height: 100px; overflow-y: auto;">{think_content.strip()}</div></details>'
            return ''
        content = re.sub(r'(.*?)</think>', format_think_no_start, content, flags=re.DOTALL)

    return content


@st.cache_resource
def load_model_tokenizer(model_path):
    """加载并缓存模型/分词器（全页面只加载一次）。

    参数：
        model_path: str，scripts/ 下的子目录名（transformers 格式）。
    返回：
        (model, tokenizer)：model 已 half() + eval() + 搬到 device。
    设计理由：
        ⚠️ Streamlit 每次交互都会重跑整个脚本，如果不加 @st.cache_resource，
        每发一条消息都要重新从磁盘加载一遍 64M 权重，页面会卡到没法用。
        cache_resource 按参数缓存「资源型」对象（模型、连接、客户端），
        区别于 cache_data（缓存可序列化的数据）。
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    # half()：fp16 省显存；eval()：关 dropout，保证推理结果稳定
    model = model.half().eval().to(device)
    return model, tokenizer


def clear_chat_messages():
    """清空两轨历史（展示用 messages + 模型用 chat_messages）。

    参数：无。
    返回：None。
    说明：
        直接 del 掉 session_state 里的键，下一次渲染时 main() 里的
        "if messages not in st.session_state" 分支会重新建空列表。
    """
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    """回放一遍历史消息（用于页面刷新后还原聊天记录）。

    参数：无。
    返回：
        list，st.session_state.messages。
    说明：
        ⚠️ 历史遗留函数：main() 里已经内联了同样的渲染逻辑，当前流程并不调用它。
        保留下来是为了说明「重放历史」这件事必须做——Streamlit 重跑脚本后
        页面是空的，只有把 session_state 里的消息重新 markdown 一遍才能「看起来没丢」。
    """
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                # assistant 消息存的是模型原始文本（可能含标签），要先过一遍渲染
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
            else:
                # user 消息套一个右对齐的深色气泡（纯 CSS 实现，不用额外组件）
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

def regenerate_answer(index):
    """重新生成：把最后一条 assistant 回答和它对应的用户消息一起弹掉，然后重跑页面。

    参数：
        index: 预留参数，当前实现没有使用（弹的是栈顶，不是任意位置）。
    返回：None。
    说明：
        st.rerun() 会立刻中断当前脚本并从头重跑，所以「删除 + 重跑」就等于
        「页面回到上一步，等待重新生成」。
    """
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


# 动态扫描模型目录
# 约定：把 transformers 格式的模型文件夹直接拷进 scripts/ 下，这里自动发现它们。
# 判定条件是「含 .bin/.safetensors/.pt 权重文件，或者有分片索引 model.safetensors.index.json」
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {}
# reverse=True 让新拷进来的目录（名字通常更新/更大）排在前面，方便默认选中最新模型
for d in sorted(os.listdir(script_dir), reverse=True):
    full_path = os.path.join(script_dir, d)
    # 跳过隐藏目录（.cache 之类）和下划线开头的目录（__pycache__ 之类）
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):
        if any(f.endswith(('.bin', '.safetensors', '.pt')) or os.path.exists(os.path.join(full_path, 'model.safetensors.index.json')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):
            MODEL_PATHS[d] = [d, d]
if not MODEL_PATHS:
    # 一个都没找到时也放一个占位项，否则 st.sidebar.selectbox 会因为空列表直接报错，
    # 用户就只能看到一片红色 traceback 而不知道「该拷模型进来」
    MODEL_PATHS = {"No models found": ["", "No models"]}

# 模型选择
selected_model = st.sidebar.selectbox('Model', list(MODEL_PATHS.keys()), index=0)
model_path = MODEL_PATHS[selected_model][0]
# 标题里的口号按当前语言切换（中文用「我是 XXX，有什么可以帮你的？」，英文用 I am ...）
slogan = f"我是 {MODEL_PATHS[selected_model][1]}，有什么可以帮你的？" if st.session_state.get('lang', 'en') == 'zh' else f"I am {MODEL_PATHS[selected_model][1]}, how can I help you?"

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 语言选择
# radio 的 index 必须由 session_state 反推，否则每次重跑都会跳回第一项
lang_options = {'中文': 'zh', 'English': 'en'}
current_lang = st.session_state.get('lang', 'en')
lang_index = 0 if current_lang == 'zh' else 1
lang_label = st.sidebar.radio('Language / 语言', list(lang_options.keys()), index=lang_index, horizontal=True)
if lang_options[lang_label] != current_lang:
    # 语言变了：写回 session_state 并立刻 rerun，让整页文案用新语言重新渲染一遍
    st.session_state.lang = lang_options[lang_label]
    st.rerun()

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 参数设置
# 直接把 widget 的返回值赋给 session_state 的字段，后面的生成循环就能直接读到。
# history_chat_num 是「轮数」且 step=2，保证永远是偶数（一问一答成对，不会出现半轮历史）
st.session_state.history_chat_num = st.sidebar.slider(get_text('history_rounds'), 0, 8, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider(get_text('max_length'), 256, 8192, 8192, step=1)
# temperature 下限给 0.6：太小会让 64M 模型陷入复读，太大又会胡言乱语，0.6~1.2 是可用的区间
st.session_state.temperature = st.sidebar.slider(get_text('temperature'), 0.6, 1.2, 0.90, step=0.01)

st.sidebar.markdown('<hr style="margin: 12px 0 16px 0;">', unsafe_allow_html=True)

# 功能开关
st.session_state.enable_thinking = st.sidebar.checkbox(get_text('thinking'), value=False, help=get_text('think_tip'))
st.session_state.selected_tools = []
with st.sidebar.expander(get_text('tools')):
    st.caption(get_text('tool_select'))
    # 先数一遍已经勾了几个，用来把「第 5 个」以后的 checkbox 置灰（tools 上限 4 个）
    selected_count = sum(1 for tool in TOOLS if st.session_state.get(f"tool_{tool['function']['name']}", False))
    for tool in TOOLS:
        name = tool['function']['name']
        short_name = TOOL_SHORT_NAMES.get(name, name)
        # key=f"tool_{name}" 让 Streamlit 自己把勾选状态存进 session_state，重跑脚本也不会丢；
        # disabled 里再判一次「自己是否已勾选」，否则勾满 4 个后连取消勾选都点不动了
        checked = st.checkbox(short_name, key=f"tool_{name}", disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False)))
        # 这里重建 selected_tools，而不是原地 append，保证顺序和 TOOLS 定义一致
        if checked and len(st.session_state.selected_tools) < 4:
            st.session_state.selected_tools.append(name)

# 页头 logo 走 ModelScope 的直链（⚠️ 需要联网才能显示，离线环境会是一个破图占位）
image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

# 页头：logo + 口号 + 免责声明，全部用内联 HTML 拼出来，方便精确控制排版
st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 40px; height: 40px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    f'<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">{get_text("disclaimer")}</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    """固定 Python / NumPy / PyTorch / cuDNN 的随机种子。

    参数：
        seed: int，随机种子。
    返回：None。
    说明：
        功能与 trainer/trainer_utils.py 里的 setup_seed 完全一样，属于重复实现。
        四个种子都要设：random 管 Python 层采样，np 管可能出现的数据处理，
        torch 管 CPU 侧随机数，torch.cuda 管 GPU 侧随机数（多卡还要 manual_seed_all）。
        最后把 cudnn 设成 deterministic 并关掉 benchmark，是为了让同一份输入
        每次都走完全相同的 kernel，结果可复现（代价是可能慢一点）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """页面主流程：渲染历史 → 读输入 → 生成 → 工具调用循环 → 落库。

    参数：无。
    返回：None。
    执行顺序（Streamlit 每次交互都会从头跑一遍）：
        1) 拿模型（走 cache_resource，实际只加载一次）；
        2) 把历史消息重新渲染出来（页面重跑后 DOM 是空的，必须重放）；
        3) st.chat_input 拿本次输入；如果是「重新生成」，则从 session_state 里取回上一轮的输入；
        4) 拼 chat template → 起线程生成 → TextIteratorStreamer 逐字渲染；
        5) 若输出里有 <tool_call>，执行工具、把结果写回 chat_messages、再生成，最多 16 轮；
        6) 把最终回答写回两轨历史。
    """
    model, tokenizer = load_model_tokenizer(model_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    # 重放历史：Streamlit 重跑脚本后页面是干净的，不重放就「聊天记录全没了」
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder=get_text('send'))

    # 「重新生成」分支：上一轮点了重新生成按钮，按钮回调把原问题存进了 session_state，
    # 这里取回来当作本次 prompt，并把三个临时键删掉（否则下一轮又会重复触发）
    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        # 先把用户气泡画出来（用户消息不进 chat_messages 之前就要有反馈）
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; background-color: #3d4450; border-radius: 22px; color: white;">{prompt}</div></div>',
            unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # 占位符：后面每一段文本都 markdown 覆盖到同一个位置，视觉上就是「一个字一个字蹦出来」
        placeholder = st.empty()

        # 每轮换随机种子：让同一句话每次生成都不一样；牺牲了可复现性换取多样性
        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)

        # 只把用户勾选的工具传给模板；一个都没勾就传 None（模板会走「无 tools」分支）
        tools = [t for t in TOOLS if t['function']['name'] in st.session_state.get('selected_tools', [])] or None
        # 💡 作者巧思：有 tools 时不加 system 人设。因为 chat template 的 tools 分支
        #    自己会生成一段 system（说明工具怎么调用），再塞一个 system 会重复且互相干扰；
        #    没有 tools 时才补一句人设，让回答更像聊天助手
        sys_prompt = [] if tools else [{"role": "system", "content": "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"}]
        # 双轨历史在这里分叉：真正发模型的 chat_messages 只保留最近 history_chat_num+1 条。
        # +1 是因为刚 append 的这条 user 消息也要算在「历史」里，否则每轮都会把当前问题丢掉
        st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
        # 模板参数先攒成 dict 复用：工具调用循环里要用同一套参数再渲染好几次
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if st.session_state.get('enable_thinking', False):
            # 开启思考：模板会在 assistant 前缀后插入 <think>，模型先想再答
            template_kwargs["open_thinking"] = True
        if tools:
            template_kwargs["tools"] = tools
        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)

        # 文本 → 张量 [1, T_prompt]；truncation=True 防止历史太长超过模型上下文
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # TextIteratorStreamer：把 generate 的输出变成「可以 for 循环的文本迭代器」，
        # 与 eval_llm.py 里打印到 stdout 的 TextStreamer 是兄弟类，区别只在输出方式
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # ⚠️ 这里用的是 transformers 原生 generate，所以长度参数叫 max_length（总长度），
        #    必须写成 T_prompt + max_new_tokens；而 eval_llm.py 用的是仓库自己写的 generate，
        #    那里面参数叫 max_new_tokens（新增长度）。两个定义混用是新手最常见的坑
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": st.session_state.temperature,
            "top_p": 0.85,
            "streamer": streamer,
        }

        # 生成放子线程：主线程要立刻开始消费 streamer，否则「先阻塞生成完」就退化成非流式了
        Thread(target=model.generate, kwargs=generation_kwargs).start()

        answer = ""
        # iter(streamer) 会一直阻塞到有新文本；生成结束时迭代自然结束
        for new_text in streamer:
            answer += new_text
            # is_streaming=True：告诉渲染函数「标签可能还没闭合」，走对应的中间态分支
            placeholder.markdown(process_assistant_content(answer, is_streaming=True), unsafe_allow_html=True)

        full_answer = answer
        # ===== 多轮工具调用循环 =====
        # 🔍 可以和 nanoGPT 对照的地方：nanoGPT 只有「一次自回归生成」这件事；
        #    这里把它整个包进外层循环，每轮之间插入「执行工具 + 回填观测」，
        #    模型权重一点没变，行为却从「续写」变成了「会查工具」
        # 上限 16 轮：模型有可能反复调工具不停下来，必须有个硬性熔断，否则页面永远转圈
        for _ in range(16):
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
            if not tool_calls:
                break
            # 把模型这次的原始输出（含 <tool_call> 标签）作为 assistant 消息存档，
            # 后面的 tool 消息才是对它的「回应」——这样多轮上下文在模板里才拼得对
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            tool_results = []
            for tc_str in tool_calls:
                try:
                    tc = json.loads(tc_str.strip())
                    result = execute_tool(tc.get('name', ''), tc.get('arguments', {}))
                    # 以 role="tool" 的形式回填观测结果，模型下一轮会把它当作「工具返回值」读到
                    st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                    # 同时拼一张绿色的「已执行」卡片，追加到展示文本里（只影响页面，不进模型上下文）
                    tool_results.append(f'<div style="background: rgba(90, 130, 110, 0.20); border: 1px solid rgba(150, 200, 170, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><div style="font-size:12px;opacity:.75;display:block;margin:0 0 6px 0;line-height:1;">ToolCalled</div><div><b>{tc.get("name", "")}</b>: {json.dumps(result, ensure_ascii=False)}</div></div>')
                except:
                    # 某一条调用解析失败就跳过它，不影响其它调用；模型下一轮会自己发现「没有观测结果」
                    pass
            full_answer += "\n" + "\n".join(tool_results) + "\n"
            placeholder.markdown(process_assistant_content(full_answer, is_streaming=True), unsafe_allow_html=True)
            # 用「历史 + 工具结果」重新渲染模板，再生成一次：这就是 Agent 的「思考-行动-观察」循环
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, **template_kwargs)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            # 重新起一个 streamer（旧的已经耗尽），并复用同一个 generation_kwargs 字典，
            # 只替换输入相关的几项，避免重复构造
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["input_ids"] = inputs.input_ids
            generation_kwargs["attention_mask"] = inputs.attention_mask
            generation_kwargs["max_length"] = inputs.input_ids.shape[1] + st.session_state.max_new_tokens
            generation_kwargs["streamer"] = streamer
            Thread(target=model.generate, kwargs=generation_kwargs).start()
            answer = ""
            for new_text in streamer:
                answer += new_text
                # 注意这里渲染的是 full_answer + answer：之前几轮的卡片 + 本轮的实时输出，
                # 保证页面上越来越长而不是每次覆盖掉历史
                placeholder.markdown(process_assistant_content(full_answer + answer, is_streaming=True), unsafe_allow_html=True)
            full_answer += answer
        answer = full_answer

        # 落库：展示轨存的是「已经混入 HTML 卡片」的字符串，模型轨存的是最终回答。
        # ⚠️ 注意 messages 里存的是带 HTML 的版本，所以重放历史时可以直接当 HTML 渲染；
        #    但它已经不再是「模型原始输出」了，想要干净的原文应看 chat_messages
        messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

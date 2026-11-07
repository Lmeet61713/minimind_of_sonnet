# ==============================================================================================
# 作用：用 GRPO 训练一个「会多轮调用工具」的 MiniMind Agent（64M）。
#       模型自己决定「调不调工具、调哪个、传什么参数」；工具返回结果后模型继续推理；
#       最后用「规则奖励 + Reward Model」给整条轨迹打一个标量分，再做组内相对优势的策略梯度更新。
# 在流水线中的位置：
#   上游 ← trainer/train_full_sft.py 产出的 SFT 权重（--from_weight full_sft），
#          数据 ../dataset/agent_rl.jsonl（每条样本 = messages + tools + gt 参考答案）；
#   下游 → ../out/agent_768.pth，可直接用 eval_llm.py 推理，也可以继续做 DPO / 蒸馏。
#   同族脚本：train_grpo.py 是「单轮、无工具」的 GRPO；本文件是它的「多轮 + 工具」版，loss 部分几乎一样。
# 核心思想：
#   1) 多轮轨迹必须「只训练模型自己说的话」。一条轨迹里交错着模型输出和工具返回，
#      工具返回是环境给的、不是策略生成的；对它做梯度上升等于教模型「复述工具结果」。
#      所以有 response_mask：模型输出=1，工具观察/模板 token=0。
#   2) 奖励要稠密，但每一项都必须「当场可判定」。最终答对才给的分太稀疏（随机策略几乎永远拿不到），
#      于是用格式/长度/工具数量/GT 命中这些规则信号做 reward shaping；
#      同时每项都有上界、总分再 clip 到 [-3,3]，防止某一条把组内标准差拉爆。
#   3) 不需要 critic。GRPO 用「同一 prompt 采 G 条 + 组内标准化」代替 value 函数估计 baseline，
#      省掉一个和策略同规模的模型（PPO 要 actor+critic+RM 三个模型，GRPO 只要 actor+RM）。
#   4) 多轮 = 多次单轮拼起来。每一轮都重新跑一次 rollout，把新生成的 token 追加到同一条 response 里，
#      所以「一条轨迹」= 若干段模型输出 + 若干段工具观察，拼成一条 token 序列后再统一算 loss。
# 关键张量/数据（B=batch, G=num_generations=4, L ≤ max_total_len=2500）：
#   input_ids            [B*G, L]      prompt + 交错轨迹，右侧 pad 到本 batch 最长
#   full_response_masks  [B*G, L]      1=模型生成，0=prompt/工具观察/pad
#   completion_mask      [B*G, L-1]    full_response_masks 右移一位，与 logits 对齐
#   per_token_logps      [B*G, L-1]    当前策略的 log pi，logits [B*G, L-1, V]
#   old_per_token_logps  [B*G, L-1]    rollout 时策略的 log pi（PPO ratio 的分母）
#   ref_per_token_logps  [B*G, L-1]    冻结的 SFT 参考模型的 log pi（KL 用）
#   rewards / advantages [B*G]         一条轨迹一个标量
# 阅读建议：
#   1) 先读 rollout_single —— 全场最难的一段（marker 技巧），读懂它整条链路就通了；
#   2) 再读 calculate_rewards —— 理解「奖励怎么设计」比理解 loss 更重要；
#   3) 然后读 rl_train_epoch 的「打包」和「loss」两段；
#   4) TOOLS / WEATHER_DATA / MOCK_RESULTS / CHECK_ARGS 只是「模拟环境」，扫一眼知道有 6 个工具即可。
# ==============================================================================================
import os
import sys

# 与其它 train_*.py 相同：允许 `python trainer/train_agent.py` 直接运行时，
# 也能 `from model.model_minimind import ...`（把仓库根目录加进 sys.path）。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
# 上面这行的意思：在 Windows 上，pyarrow 和 torch 的 DLL 加载顺序会冲突（issue #771），
# 先 import datasets 把 pyarrow 拉起来就能规避。noqa: F401 是告诉 flake8「这个 import 看似没用到，别报错」。
import re
import gc
import json
import math
import random
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel, safe_math_eval
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

# 屏蔽 transformers / torch 的各种 UserWarning：这个脚本会反复调 generate 和 chat template，
# 不屏蔽的话日志会被「padding token 未设置」「attention mask 未传」之类的警告刷满，看不到真正的 reward 曲线。
warnings.filterwarnings('ignore')

# ================================ 工具与 Reward = Start ================================

def rep_penalty(text, n=3, cap=0.5):
    """n-gram 重复率惩罚：专治「复读机」式的刷分行为。

    参数：
        text: 待检查的文本（通常只传「最终答案」部分，不要把 think 也算进来）
        n:    n-gram 的窗口大小，默认 3（3-gram 重复已经能抓住「车轱辘话」）
        cap:  单项惩罚上限，默认 0.5

    返回：
        [0, cap] 之间的 float，越大表示重复越严重。

    公式：
        penalty = min(cap, (总 n-gram 数 - 去重后 n-gram 数) * cap * 2 / 总 n-gram 数)

    设计理由：
        1) 只数「重复的 n-gram 占比」，不数「某个词出现几次」——后者会误伤 "0"、"元" 这类本来就该反复出现的词。
        2) 乘 2 是让「50% 的 3-gram 都重复」就能吃满 cap，不用等到 100%。
        3) 用 min(cap, ...) 封顶：重复惩罚只是「别刷分」，不应该盖过 GT 命中（+2.5）这个主奖励，
           否则模型会学到「只要不重复，答什么都行」。
        4) grams 为空（文本太短，凑不出一个 3-gram）时返回 0.0，避免除零。

    分词方式：
        \\w+ 抓词/数字，[^\\w\\s] 抓标点 —— 这样 "1 + 1 = 2。" 会切成 ['1','+','1','=','2','。']，
        中文会被 \\w+ 整段吃掉（连续汉字算一个匹配），所以中文文本的 3-gram 更像是
        「连续 3 个词块」的重复率，而不是「连续 3 个字」—— 对判复读已经够用。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义 ========
# 6 个工具的 schema，格式遵循 OpenAI function-calling 约定（type/function/name/description/parameters）。
# ⚠️ 这些 dict 会被 chat template 用 tojson 直接塞进 system prompt（见 model/tokenizer_config.json 的 chat_template），
#    所以字段名不能改（模型见过的就是这套格式），description 写得越清楚，模型调用的准确率越高。
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 模拟数据 ========
# 所有工具都返回「写死的假数据」，这是刻意为之：
#   1) 训练要可复现 —— 真实 API 每次返回不同结果，同一个 prompt 的 G 条样本奖励就没有可比性；
#   2) 训练要快 —— 一条轨迹 3 轮、每 batch 4 条，走一次网络请求就废了；
#   3) 答案要可验证 —— gt 里写的答案就是按这些固定值算出来的（比如时间永远是 2025-03-07 14:30:00）。
# ⚠️ 换成真实 API 时，记得同步更新 dataset 里的 gt，否则 GT 命中分会全 0，模型学不到东西。
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

# ======== 模拟执行 ========
# 工具名 → 执行函数（lambda）。这里的 lambda 全部对参数做了兜底（.get 默认值 / or 0 / float()），
# 因为模型可能传错类型、漏字段、传字符串；工具在训练里崩一次，整条 rollout 就白跑了。
# 💡 注意 calculate_math：用 safe_math_eval（AST 白名单求值）而不是 eval。
#    模型生成的表达式完全可能包含 __import__('os').system('rm -rf /')，eval 就是现成的 RCE。
MOCK_RESULTS = {
    # 数学表达式 → 字符串结果。转 str 是为了 JSON 可序列化（json.dumps 不认 Decimal / 复数）。
    "calculate_math": lambda args: {"result": str(safe_math_eval(args.get("expression", "0")))},
    # 单位换算：key 拼成 "km_miles" 这种形式查表；查不到时 .get(..., 1) 乘 1 等于「原样返回」，
    # 仍然是一段合法 JSON，模型能看懂「没换成」，而不是拿到一个异常。
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    # 天气：内层 (lambda w: ...) 是为了把 WEATHER_DATA 里的 (温度, 天气) 元组就地展开成 dict。
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验 ========
# 工具名 → 「必填字段是否齐全」的判定函数。只校验结构、不校验语义。
# 用途：算「有效调用数」（valid_call_count），进而算 tool_gap —— 这是奖励的一部分。
# ⚠️ 这里的 key 集合必须和 MOCK_RESULTS 完全一致，否则某个工具会被判成「参数永远不合法」。
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,   # 没有必填参数（timezone 有默认值）
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    """从模型输出里解析出所有 <tool_call>...</tool_call> 块。

    参数：
        text: 模型某一轮的输出文本（可能包含思考、正文、多个工具调用）

    返回：
        List[dict]，每项形如 {"name": "calculate_math", "arguments": {...}}；
        解析失败的块直接丢掉（返回的 list 可能为空）。

    设计理由：
        1) 模板约定（见 tokenizer_config.json 的 chat_template）要求模型输出
           <tool_call>{"name": ..., "arguments": {...}}</tool_call>，所以用正则抓标签最稳。
        2) (.*?) 用非贪婪 + re.DOTALL：一条回复里可能有多组 tool_call，非贪婪才能逐组匹配；
           DOTALL 让 JSON 里的换行也能被匹配到。
        3) json.loads 失败就 continue，不抛异常：模型输出半截 JSON 是常态（尤其训练初期），
           一条坏样本不能把整个训练脚本炸掉 —— 它会在奖励里被「标签扣分」自然惩罚。
    """
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: call = json.loads(m.strip())
        except: continue
        if isinstance(call, dict): calls.append(call)
    return calls

def execute_tool(name, args):
    """执行一个工具调用，永远返回「可序列化的结果或 None」。

    参数：
        name: 工具名（来自模型输出，不可信，必须白名单查表）
        args: dict 参数

    返回：
        dict 结果；工具不存在或执行抛异常时返回 None（调用方会把 None 渲染成 {"error": "tool not found"}）。

    设计理由：
        - 用 MOCK_RESULTS.get(name) 做白名单，模型输出什么名字都不会真的调到别的东西；
        - 吞掉所有异常：safe_math_eval 会对超长/异形表达式抛 ValueError，模型给的参数也可能类型不对，
          这些都应该变成「一次失败的调用」，而不是「一次训练中断」。
    """
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        return fn(args)
    except Exception:
        return None

# ======== 多轮 Rollout ========
# -------------------------- 多轮 Agent RL 和单轮 GRPO 的本质区别 --------------------------
# 单轮 GRPO（train_grpo.py）：
#   prompt → 模型生成一整段 response → 打分 → 组内标准化 → 更新。
#   一条样本的每个 token 都是模型生成的，completion_mask 全 1，rollout 返回的 per_token_logps 直接能当 old_logps。
# 多轮 Agent RL（本文件）：
#   prompt → 模型生成 turn1 → 执行工具 → 工具结果塞回上下文 → 模型继续 turn2 → ...
#   一条轨迹的 token 序列是「模型输出」和「环境返回」交错拼起来的：
#     [prompt] [assistant turn1][<|im_end|>][user: <tool_response>...</tool_response><|im_end|>][assistant turn2]...
#                ^^^^^^^^^^^^^ 算 loss                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 不算 loss
#   于是引出三个单轮里不存在的问题：
#     (a) mask：工具观察的 token 必须置 0，否则模型会被训练成「复述工具返回」；
#     (b) 对齐：奖励是打在「整条轨迹」上的，所以 G 条样本要对应 G 条完整轨迹，而不是 G 段文本；
#     (c) 拼接：中途要往 response 里追加「工具观察 + chat template 插入的结构 token」，
#         而且拼出来的 token 序列必须和「把整段上下文重新 tokenize 一遍」完全一致 ——
#         这就是下面 rollout_single 里 marker 技巧存在的唯一原因。
# ----------------------------------------------------------------------------------------
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """跑完一条完整的多轮轨迹，返回「可以直接拿去训练」的交错 token 序列。

    参数：
        rollout_engine: RolloutEngine 实例（torch 或 sglang）
        tokenizer:      用来渲染 chat template、tokenize 观察文本、取 eos id
        messages:       List[dict]，初始对话历史（不含 assistant 回复）。⚠️ 本函数会原地 append，调用方要传副本
        tools:          这道题允许的工具 schema 列表（会渲染进 system prompt，也决定 valid_names）
        max_turns:      最多几轮「模型输出 + 工具调用」
        max_new_tokens: 每一轮最多生成多少 token
        thinking_ratio: 以多大概率开启 think（open_thinking）。同一条 prompt 的 G 条样本会独立随机，
                        这样「想 vs 不想」的差异会体现在组内 advantage 上，模型自己学出什么时候值得想
        device:         张量放哪

    返回（8 元组）：
        final_output      str          最后一轮的输出文本（奖励主要看它）
        final_context     str          最后一轮渲染出的完整上下文文本（只用于 debug）
        prompt_ids        List[int]    第一轮 prompt 的 token id（不含任何模型输出）
        response_ids      List[int]    交错轨迹的 token id = 模型输出 + 工具观察 + 模板 token
        response_mask     List[int]    与 response_ids 等长，1=模型生成，0=工具观察/模板
        response_old_logps List[float] 与 response_ids 等长，模型生成位置是 rollout 时的 logprob，其余补 0.0
        all_turn_outputs  List[str]    每一轮模型输出的文本（数 tool_call、算分都要用）
        unfinished        bool         轮数用尽但最后一步还在调工具（轨迹被硬截断）

    形状约定：
        prompt_ids + response_ids 就是训练用的完整序列（长度可能超过 max_total_len，由训练侧截断）。

    为什么返回值这么多：
        训练侧需要「token 序列」算 loss，又需要「文本」算奖励，还需要「每轮的文本」判断工具调用次数，
        三类信息都只能在这里拿到（一旦拼成 token 序列，轮次边界就丢了）。
    """
    all_outputs = []            # 每一轮模型输出的文本，最后用于多轮 reward
    prompt_ids = None           # 第一轮 prompt 的 token id，只算一次，后面所有轮都复用它
    response_ids = []           # 交错轨迹：模型输出 + 工具观察 + 模板 token
    response_mask = []          # 1=模型生成（算 loss），0=工具/模板（不算 loss）
    response_old_logps = []     # 与 response_ids 对齐的 old logprob（PPO ratio 的分母）
    final_context = ""          # 最终上下文文本（debug 打印用）
    unfinished = False          # 是否「轮数用完还在调工具」
    # 每条轨迹独立随机决定要不要开 think：True 时模板只预填 <think>，让模型自己写推理；
    # False 时模板直接塞 <think>\n\n</think>\n\n（空思考），逼模型直接给答案。
    # 💡 让同一 prompt 的 G 条样本随机走两种模式，组内对比就能告诉模型「这道题值不值得想」。
    open_thinking = random.random() < thinking_ratio
    for turn in range(max_turns):
        # 1) 把当前消息列表渲染成文本上下文。
        #    tools=tools 会让模板在 system prompt 里插入 6 个工具的 schema；
        #    add_generation_prompt=True 补上 <|im_start|>assistant\n<think>...，为模型生成做准备。
        #    这里写死 True 是安全的：unfinished 只可能在「最后一轮」被置 True，
        #    而那时循环已经结束，不会再回到这一行（下面 143/151 行才需要 not unfinished）。
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        if prompt_ids is None:
            # 只在第一轮 tokenize 一次，作为整条轨迹不变的「前缀」。
            # ⚠️ add_special_tokens=False：模板文本里已经手写了 <|im_start|>/<|im_end|> 这些特殊 token，
            #    再加 BOS 会重复，序列就和推理时不一致了。
            prompt_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        # 2) 模型真正的输入 = 首轮 prompt + 到目前为止的全部交错 token。
        #    形状 [1, P+len(response_ids)]，batch=1（一条轨迹一次前向）。
        #    ⚠️ 用 response_ids 而不是「重新 tokenize 整段 context」：多轮拼接必须保证 token 边界一致，
        #       重新 tokenize 会把已经确定的历史 token 重新切一遍，BPE 的合并可能给出不同结果。
        input_ids = torch.tensor([prompt_ids + response_ids], device=device)
        rollout_result = rollout_engine.rollout(
            prompt_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),   # 没有 padding，全 1
            num_generations=1,                            # 多样性由外层循环 G 次提供，这里不展开
            max_new_tokens=max_new_tokens,
            temperature=0.8,                              # 0.8：兼顾多样性与可读性（太高会胡言乱语，太低 G 条全一样）
        )
        # 3) 取出「这一轮新增」的 token 和它们的 old logprob。
        #    completion_mask 里 1 的个数就是真实生成长度（sglang 引擎会在右侧补 pad，torch 引擎全 1）。
        valid_len = int(rollout_result.completion_mask[0].sum().item())
        new_ids = rollout_result.completion_ids[0, :valid_len].tolist()
        new_logps = rollout_result.per_token_logps[0, :valid_len].tolist()
        # 两个引擎的补位策略不同，长度不一致说明有一侧有 bug；早失败早发现，别等 loss 算出一堆 nan 再回头查。
        if len(new_ids) != len(new_logps):
            raise RuntimeError(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        new_text = rollout_result.completions[0]
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        # ⚠️ eos 位置 mask=0：<|im_end|> 是模板自己会补的结构 token（下一轮渲染时还会再出现一次），
        #    不需要模型学它，也不该计入长度归一的分子分母。
        #    顺便，MiniMind 的 generate 会给「提前结束的行」反复补 eos，这一条规则也顺手把它们屏蔽掉了。
        response_mask.extend([int(t != tokenizer.eos_token_id) for t in new_ids])
        response_old_logps.extend(new_logps)
        # 仅用于 debug：这是「文本拼接」，decode 时特殊 token 被吃掉了，并不等于真实的 token 序列。
        final_context = context + new_text
        calls = parse_tool_calls(new_text)
        if not calls:
            break        # 模型自己决定不调工具了 → 轨迹正常终止
        unfinished = turn == max_turns - 1   # 最后一轮还在调工具 → 轨迹被硬截断，最后没有答案
        # 4) 把「模型的 assistant 发言」和「每个工具的执行结果」写进 messages，
        #    下一轮渲染模板时它们会变成正式的对话历史。
        assistant_message = {"role": "assistant", "content": new_text}
        messages.append(assistant_message)
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                # 有些模型会把 arguments 输出成一个 JSON 字符串而不是对象，这里补一次解析。
                try: raw = json.loads(raw)
                except: raw = {}     # 解析失败就当空参数，让 CHECK_ARGS 判「无效调用」，而不是抛异常
            if not isinstance(raw, dict): raw = {}
            result = execute_tool(name, raw)
            # 工具结果统一序列化成 JSON 文本；[:2048] 截断防止「10**100000」这类结果撑爆 tokenizer / 上下文。
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
            messages.append({"role": "tool", "content": result_str})

        # ------------------------------------------------------------------------------------
        # 5) 【marker 技巧】把「工具观察新增的 token」精确地切出来
        # ------------------------------------------------------------------------------------
        # 问题：模板渲染 tool 消息时并不是简单拼字符串，而是插入一整段结构 token：
        #       <|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n
        #   如果直接 tokenizer(result_str) 再接到 response_ids 后面：
        #     (a) 会漏掉上面这些模板 token，训练序列和推理序列从此错位；
        #     (b) BPE 的合并依赖相邻字符，单独 tokenize 出的 id 和「整段重新 tokenize」可能不同，
        #         哪怕只差一个 token，模型看到的分布就和你以为的不一样。
        # 解法：
        #   a) 先在 assistant content 末尾塞一个绝不可能被模型自己生成的 marker；
        #   b) 用模板渲染整段上下文 —— 此时 marker 正好卡在「assistant 正文」和「模板插入的 <|im_end|>」之间；
        #   c) 按 marker 把文本切成两半，后半段就是这一轮新增的全部文本（含模板 token）；
        #   d) 单独 tokenize 后半段，追加到 response_ids，mask 全 0。
        # 这样拼出来的 prompt_ids + response_ids，与「把整段上下文重新 tokenize 一遍」逐 token 相同。
        # marker 里带 id(messages) 和当前长度：同一进程内每次都不一样，不会撞上模型输出或工具返回里的已有文本。
        marker = f"<|agent_observation_{id(messages)}_{len(response_ids)}|>"
        assistant_message["content"] += marker
        # add_generation_prompt=not unfinished：
        #   没被截断 → 补上 <|im_start|>assistant\n<think>...，下一轮直接接着生成；
        #   被截断   → 不补，因为不会再有下一轮，补了就等于在轨迹尾部塞一段「没人回答的提问」。
        marked_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        # 立刻还原 messages：不能让 marker 留在里面污染下一次渲染（也会污染 debug 输出）。
        assistant_message["content"] = new_text
        # partition 返回 (marker 之前, marker 本身, marker 之后)；我们只要后半段。
        _, found, observation = marked_context.partition(marker)
        if not found:
            # 某些模板会对 content 做 strip/escape，marker 被改掉后整个方案就失效了。
            # 这时必须硬失败：静默继续的话，拼出来的训练序列会从这一刻起全部错位，而且训练不会报错，只是学不动。
            raise RuntimeError("chat template did not preserve the assistant content boundary")
        obs_delta = tokenizer(observation, add_special_tokens=False)["input_ids"]
        # ⚠️ 去重 eos：模型生成结束时末尾会带一个 <|im_end|>（就是 eos，id=2），
        #    而模板渲染 assistant 分支时自己也会补一个 <|im_end|>；
        #    observation 恰好从「模板补的那个 eos」开始，于是和 new_ids 末尾的那个重复。
        #    去掉一个，拼出来的 ids 才和「整段重新 tokenize」严格一致。
        #    注意条件同时要求「new_ids 末尾确实是 eos」——如果模型是撞到 max_new_tokens 停的（没有 eos），
        #    那这个前导 eos 是模板补的、必须保留，不能删。
        if new_ids and new_ids[-1] == tokenizer.eos_token_id and obs_delta[:1] == [tokenizer.eos_token_id]:
            obs_delta = obs_delta[1:]
        # 再渲染一次（此时 messages 已还原、没有 marker），只用于 debug / final_context。
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        response_ids.extend(obs_delta)
        # 工具观察 + 模板 token：mask 全 0。整段轨迹里凡是「不是模型自己生成」的位置都要置 0。
        response_mask.extend([0] * len(obs_delta))
        # old_logps 补 0.0 占位：这些位置 mask=0，永远不会进 loss，值是什么无所谓（但长度必须对齐）。
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    # 6) 收尾。all_outputs 为空说明一轮都没跑成（正常情况下不会发生），兜底成空串而不是抛异常。
    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []   # prompt_ids 为 None 时兜底成 []，避免 None 参与列表拼接
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """对 batch 里每条样本各跑 num_gen 条轨迹，把结果摊平成「同 prompt 的 G 条连续相邻」的列表。

    参数：
        messages_batch: List[List[dict]]，长度 B
        tools_batch:    List[list]，长度 B
        num_gen:        G，每条 prompt 采几条
        其余参数透传给 rollout_single

    返回：
        8 个 list，长度全部是 B*G，顺序为 sample-major（idx = sample_idx * G + j）。

    设计理由 / 坑：
        💡 msgs_copy = [dict(m) for m in messages]：rollout_single 会往 messages 里 append assistant/tool 消息。
           不拷贝的话，同一条 prompt 的第 2 次生成会「继承」第 1 次的对话历史 —— 这是多轮 RL 里最隐蔽的 bug 之一，
           表现是第 2~G 条轨迹莫名其妙地「已经知道答案」。
        ⚠️ 顺序必须和训练侧约定一致：rewards.view(-1, G) 和 idx // num_gen 都假设「同 prompt 连续相邻」。
        ⚠️ 这里是纯串行 for 循环（B*G 次 rollout_single），batch 并行度为 1：
           多轮轨迹轮数不同、长度不同，凑 batch 很麻烦，而且 batch=1 的 generate 本来就利用率低。
           想要快就换 --rollout_engine sglang，让服务端做连续批处理。
    """
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            # 深一层的浅拷贝：list 是新的，里面的 dict 是同一个——正好够用，因为 rollout_single 只 append 新 dict，
            # 不会修改已有的 message dict。
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished

# ======== Reward 计算 ========
def validate_gt_in_text(text, gt_list):
    """判断参考答案（gt）里的哪些项出现在了模型回答里，返回命中的子集。

    参数：
        text:    模型最终回答的文本
        gt_list: 参考答案列表（可能混着「数字」和「字符串/日期」两类答案）

    返回：
        set，命中项的原值集合。

    两路匹配：
        1) 字符串匹配：s.lower() in text.lower()（大小写不敏感的子串）
           —— 适合 "Hello World"、"28°C"、"2025-03-07 14:30:00" 这类非纯数字答案；
        2) 数值匹配：gt 本身是纯数字时，用正则从文本里抠出所有数字，再逐个比较。
           —— 适合 "7.21"、"1440" 这类答案，能兼容模型写成 "7.210000"、"约 7.21 元" 的情况。

    为什么需要数值匹配这一路：
        模型算出来的浮点数和 gt 很难「字符串相等」（7.2099999 vs 7.21），
        只靠字符串匹配会让 GT 分永远是 0，训练直接失去主信号。

    正则 r'(?<![\\w.])[-+]?\\d+(?:\\.\\d+)?(?![\\w.])' 逐项拆解：
        (?<![\\w.])   左边不能是字母/数字/下划线，也不能是小数点
                      → 否则 "1234" 里会抠出 "234"，"3.14" 里会抠出 "14"；
        [-+]?         允许正负号（"−5" 这种 Unicode 负号不在内，所以外面还有一次 translate？没有——
                      这里只处理 ASCII 符号；模型输出中文负号时会漏，属于可接受误差）；
        \\d+           整数部分；
        (?:\\.\\d+)?  可选的小数部分（不支持科学计数法 1e5，够用）；
        (?![\\w.])    右边同理，避免把 "1.5" 抠成 "1"、把 "3x" 抠成 "3"。
        text_num 先 replace(',', '') 去掉千分位：否则 "1,234" 会被切成 123 和 4 两个数。

    浮点比较用 abs(a - b) < 1e-6 而不是 ==：
        0.1 + 0.2 != 0.3，模型算出的 7.209999 和 gt 的 7.21 必须算命中。

    ⚠️ 命中率按「gt 项数」算（见 calculate_rewards）：3 项命中 2 项 → 0.667 * 2.5 分，
       部分正确也有梯度，比「全对才给分」好学得多。
    """
    text, text_num = str(text), str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}

def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    """对 B*G 条轨迹逐条算标量奖励，返回 [B*G]。

    参数：
        prompts:             List[str]，长度 B，渲染好的 prompt 文本（给 RM 当上下文用）
        completions:         List[str]，长度 B*G，最后的输出文本
        gt_batch:            List[list]，长度 B，参考答案
        tools_batch:         List[list]，长度 B，该题允许的工具 schema
        num_gen:             G
        reward_model:        LMForRewardModel 或 None（None 就跳过 RM 分，方便小机器跑）
        turn_outputs_batch:  List[List[str]]，长度 B*G，每轮的输出文本
        unfinished_batch:    List[bool]，长度 B*G，是否被轮数截断

    返回：
        [B*G] 的 float32 tensor，每个元素都被 clip 到 [-3, 3]。

    奖励总设计（为什么长这样）：
        GT 命中（+2.5）是「任务奖励」，但它太稀疏：SFT 刚出来的模型几乎不可能一次就调对工具并算对，
        全 0 奖励 → 组内 std=0 → advantage 全 0 → 一点梯度都没有。
        所以先给一堆「规则能当场判定」的稠密信号（格式、长度、工具数量），让模型先学会「按格式调工具」，
        再靠 GT 分把「算对」这件事慢慢学出来。这就是 reward shaping。
        ⚠️ 注意：shaping 项的量级都刻意小于 GT 分（0.25 ~ 1.0 vs 2.5），
           否则模型会去刷「长度分」「思考分」而不是答题。

    两类轨迹分开处理：
        A) 没调工具：长度分 + 思考长度分 + 思考闭合分 + RM 分 + 重复惩罚
        B) 调了工具：工具数量对齐分 + GT 命中分 + 未完成扣分 + 重复惩罚
        （为什么分开：A 类没有工具调用，用 RM 评「答得好不好」；B 类有工具，
          工具参数里的数字很容易污染答案文本，所以只从「最后一个 </tool_call> 之后」取答案。）
    """
    rewards = torch.zeros(len(completions), device=device)   # [B*G]，每条轨迹一个标量
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen   # 反推这条轨迹属于哪个 prompt（排列顺序见 rollout_batch）
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        # 每轮输出都砍掉 </think> 之前的思考部分，只留「最终答案」。
        # 为什么要剥离：模型可能在思考里写「我先调 calculate_math」「答案是 42」这类中间结论，
        # 如果把它们算进 GT 匹配，模型只要在思考里瞎猜一个数字就能骗到分。
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()   # 最后一轮的答案才是最终答案
        valid_names = {t['function']['name'] for t in tools} if tools else set()   # 这道题允许调用的工具名
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 解析tool调用
        # 标签扣分：<tool_call> 与 </tool_call> 数量不等，说明模型吐了半截 JSON。
        # 每处扣 0.5，这是负奖励 —— 让它学会「要么完整闭合，要么别开」。
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 标签扣分
        # -------- 无工具调用：格式+reward奖励 --------
        if not tool_calls:
            # 长度分：<5 字符多半是「好的」「不知道」这种摆烂；>800 多半是复读。
            # ±0.5 的量级比 GT 分小一个数量级，避免模型为了刷长度而答题。
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5  # 长度分
            if '</think>' in response:
                # split(maxsplit=1)：只切第一个 </think>，答案里再出现该字符串也不影响。
                think, answer = response.split('</think>', 1)
                # 思考长度分：20~300 字符是「有推理但不啰嗦」的区间；空思考（<20）和长篇大论（>300）都扣分。
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 思考长度分
                # 闭合分：只允许一个 </think>，多了说明格式坏了（模型可能把思考段重复输出）。
                reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考闭合分
                answer = answer.strip()
            if reward_model is not None:
                prompt = prompts[sample_idx]   # 同一 prompt 的 G 条样本共享同一个 prompt 文本
                # 从渲染好的 chat template 文本里把 (role, content) 抠回来，拼成 messages 喂给 RM。
                # ⚠️ 这里强依赖 MiniMind 的模板格式 <|im_start|>role\n content <|im_end|>；
                #    换模板（或换 tokenizer）必须同步改这个正则，否则 matches 为空、RM 分恒为 0。
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                # RM 分：LMForRewardModel.get_score 内部已经把分数 clamp 到 [-3,3]。
                score = reward_model.get_score(messages, answer)
                reward += score  # RM分
            # 重复惩罚：3-gram 重复率（最高 0.5），专治「1+1=2。1+1=2。...」这种刷分行为。
            reward -= rep_penalty(answer)
            # 总分 clip 到 [-3,3]：防止某条轨迹的奖励特别离谱，把组内 std 拉大、
            # 让其它样本的 advantage 全被压扁（advantage 是相对值，一个 outlier 能毁掉整组）。
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
        # -------- 有工具调用：执行结果奖励 --------
        else:
            gt = gt_batch[sample_idx]
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):     # arguments 可能是 JSON 字符串
                    try: raw = json.loads(raw)
                    except: raw = {}
                if not isinstance(raw, dict): raw = {}
                check = CHECK_ARGS.get(name)   # 每个工具一个「必填字段是否齐全」的判定器
                # 有效 = 工具名在本题允许列表内 且 参数校验通过；非法调用不计入有效数（但下面会额外扣分）。
                valid_call_count += int(bool(name in valid_names and check and check(raw)))
            # tool_gap = |有效调用数 - GT 需要的调用数| + 非法调用数
            #   前半：惩罚「该调 2 次只调了 1 次」和「多调了没用的」；
            #   后半：惩罚「调了不存在的工具 / 参数缺字段」——这类调用在 valid_call_count 里已经被扣掉，
            #         但光「不计分」不够，还得显式扣分，否则「乱调一堆 + 蒙对一个」和「精确调一个」得分相同。
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # tool数差值
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap  # tool对齐分
            # 只从「最后一个 </tool_call> 之后」的文本里找答案。
            # 为什么：工具参数里常常就带着正确答案的数字（比如汇率 7.21 写在 arguments 里），
            # 如果拿整段 response 去匹配，模型只要「把答案抄进参数」就能拿满分，完全学不会用它。
            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            # GT 分：主奖励。按命中比例给，2/3 命中 → 1.67 分，部分正确也有梯度。
            if gt: reward += 2.5 * len(verified) / len(gt)  # GT分
            # 未完成扣分：轮数用完还在调工具，说明它「停不下来」，最终没给答案，这样的轨迹不能和高分轨迹同等待遇。
            if unfinished: reward -= 0.5  # 未完成扣分
            # 重复惩罚：优先罚「最终答案」；如果最终答案为空（未完成 / 没写答案），退回罚整段 answer，
            # 至少还能压住工具调用部分的复读。
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
    return rewards

# ================================ 工具与 Reward = End ================================
def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False):
    """跑一个 epoch 的 Agent RL 训练（rollout → 打包 → 奖励 → 优势 → loss → 反传 → 保存/同步）。

    参数：
        epoch:        当前 epoch（从 0 开始）
        loader:       DataLoader，每 batch 给出 {'messages', 'tools', 'gt'}
        iters:        这个 epoch 的总 step 数（含断点续训时被 skip 掉的部分），用于「最后一步」判断和日志分母
        rollout_engine: RolloutEngine
        ref_model:    冻结的 SFT 参考模型（算 KL 用，整个训练不更新）
        reward_model: LMForRewardModel 或 None
        start_step:   断点续训跳过的步数
        wandb:        日志器（这里实际是 swanlab）
        use_sglang:   目前未使用，仅为向后兼容保留（真正的分支在 rollout_engine 内部）

    ⚠️ 阅读提示：这个函数没有把 tokenizer / args / model / optimizer / scheduler / autocast_ctx / lm_config
       当参数传进来 —— 它们都是「模块级 global」。因为它们在文件末尾 `if __name__ == "__main__":` 里被赋值，
       而 __main__ 的赋值同样落在模块命名空间里，所以函数内部能直接读到。
       好处是签名短，坏处是可测试性差 —— 读代码时找不到参数，就往下翻到 main 段。

    关键形状：
        input_ids [B*G, L] → logits [B*G, L-1, V] → per_token_logps [B*G, L-1]
        advantages [B*G] → unsqueeze(1) → [B*G, 1] 广播到 [B*G, L-1]
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']

        # ===== 第一阶段：采样（rollout）=====
        # 外层再包一层 no_grad 是防御性的（引擎内部自己也有 no_grad）：rollout 只「生产数据」，不产生可微路径。
        # ⚠️ 这一步通常是最慢的：B*G 条轨迹串行跑，每条最多 3 轮、每轮一次完整前向。
        #    你在日志里看到 step 很久才 +1，大概率不是 backward 慢，而是耗在这里。
        with torch.no_grad():
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)

        # prompts 单独渲染一遍纯文本，只给 reward_model 用（RM 只关心 system/user 的文本内容）。
        # ⚠️ 这里没传 open_thinking，渲染结果和 rollout 时用的 context 不一定逐字相同；
        #    但它只影响 RM 看到的 prompt 文本，不影响任何训练 token。
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
        # ===== 第二阶段：把变长轨迹打包成定长张量 =====
        # 每条轨迹：ids = prompt_ids + response_ids，mask = [0]*len(prompt) + response_mask
        #   - mask 前面补 0：prompt 是「给定条件」，不是策略生成的，不参与 loss；
        #   - old_logps 前面补 0，且整体长度 = len(ids) - 1：
        #     因为 logprob[j] 是「用位置 j 预测位置 j+1」的值，最后一个位置没有「下一个 token」，所以少一个；
        #   - 超长从【左侧】截断：保留轨迹尾部。多轮 Agent 的「答案」在最后，砍头比砍尾损失小；
        #     代价是 prompt 会被截掉一部分（模型看到的题目变短）。
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            ids = p + r
            mask = [0] * len(p) + m
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp    # max(...,0)：极端情况下 prompt 为空也不会负长度
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                # 截断后 old_logps 必须跟着右对齐：它对应的是「新序列的 ids[1:]」，
                # 所以取最后 len(ids)-1 个元素（右对齐 = 丢掉最左边那些已经不对应任何位置的部分）。
                old_logps = old_logps[-(len(ids) - 1):]
            # prompt_len = 第一个 mask==1 的位置 = 模型开始生成的地方。
            # ⚠️ mask 中间也有 0（工具观察、eos），所以不能 sum、也不能用 rfind，只能找「第一个 1」。
            #    截断之后这个值可能变小甚至变 0，它只用于 debug 打印切分点，不影响 loss。
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
        # 下面几个张量统一「右侧 pad 到本 batch 的最长序列」（不是 pad 到 max_total_len，省显存），
        # 再用 full_mask 在 attention 里屏蔽 pad 位置。
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)   # [B*G] 每条真实长度
        max_len = seq_lens.max().item()
        # input_ids：[B*G, L]，pad 用 pad_token_id（id=0，<|endoftext|>）。
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)   # [B*G]
        # full_response_masks：float32 而不是 bool，因为后面要直接乘 float 的 loss。
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        # ⚠️ old_per_token_logps 长度是 max_len - 1（对应 input_ids[:, 1:]），不是 max_len。
        #    内层补 0 到 (max_len-1) - len(old_logps)，补的那部分对应 pad 位置，反正 mask=0。
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
        # full_mask：[B*G, L] long。位置索引 < 该行真实长度 → 1，其余 → 0。
        # 这就是「三角形 + 行长度」的写法：arange(L) 广播成 [1,L]，seq_lens 广播成 [B*G,1]。
        full_mask = (torch.arange(max_len, device=args.device).unsqueeze(0) < seq_lens.unsqueeze(1)).long()

        # ===== 第三阶段：算奖励 =====
        # 注意：奖励是「轨迹级」的标量，不是 token 级 —— 一条轨迹一个数，这是 RL 和 SFT 最本质的区别。
        rewards = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=args.device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch)

        # ===== 第四阶段：策略前向 + KL 参考前向 =====
        with autocast_ctx:
            res = model(input_ids, attention_mask=full_mask)
            # MoE 的负载均衡辅助损失：让专家使用率均匀，不加会塌缩到少数几个专家。非 MoE 时是 0 张量。
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]   # [B*G, L-1, V]：丢最后一列，因为它没有「下一个 token」可预测
            # 一次性 gather 目标 token 的 logprob：
            #   input_ids[:, 1:] 是每个位置的「正确答案」，形状 [B*G, L-1] → unsqueeze(-1) 变 [B*G, L-1, 1]
            #   gather(dim=2) 取每个位置对应的那一列 → squeeze(-1) → [B*G, L-1]
            # 🔍 对照 nanoGPT：nanoGPT 是 F.cross_entropy(logits, targets, ignore_index=-100)；
            #    这里不用 ignore_index，而是先 gather 出每个 token 的 logprob，再用 mask 乘掉不要的位置 ——
            #    因为 RL 里「不算 loss」的原因不是「标签缺失」，而是「这个 token 不是模型生成的」，
            #    而且算完 logprob 还要和 old_logps / ref_logps 逐 token 做运算，必须先展开成向量。
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # ref_model 冻结：整个训练都不回传，显存只在前向期间占用。
            # n_keep = L-1：只要最后 L-1 个位置的对数概率，正好对应 input_ids[:, 1:]。
            # ⚠️ 这里 n_keep+1 == L，等于还是全量前向 —— logits_to_keep 的省算力效果只在
            #    「n_keep 远小于 T」时体现（比如 rollout 里只重算新生成的 R 个 token）。
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

        # ===== 第五阶段：把 mask 和 logits 对齐 =====
        # full_response_masks[i] 标记的是「位置 i 的 token 是不是模型生成的」，
        # 但 per_token_logps[i] 是「用位置 i 预测位置 i+1」的结果 → 需要右移一位。
        # completion_mask[j] = full_response_masks[j+1]，正好和 per_token_logps[j] 描述同一个 token。
        completion_mask = full_response_masks[:, 1:]
        # 找每行「第一个没有被 mask 掉的 eos」，把它后面的位置全部清零。
        # ⚠️ 说明：在标准路径里，rollout_single 已经把 eos 位置置 0，所以 is_eos 恒为 False、这段是 no-op。
        #    它是一道防御：万一以后改了 mask 规则（或者换了引擎），也不会把 eos 之后的多余 token 算进 loss。
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)   # 默认「没有 eos」→ 保留到最后一个位置
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]   # bool→int 后 argmax 就是「第一个 True」的下标
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)   # [1, L-1]
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()      # [B*G, L-1]，eos 及其之前保留
        token_counts = completion_mask.sum(dim=1)   # [B*G] 每条轨迹的有效 token 数 = 长度归一的分子分母
        valid_rows = token_counts > 0               # 整行都被 mask 掉的样本（例如截断后只剩 prompt）要跳过

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            # 人工核对用：把每条轨迹的上下文、prompt/completion 切分点、解码文本、奖励全打出来。
            # 💡 多轮 Agent RL 调不动的时候，90% 的问题能在这里一眼看出来
            #    （工具观察没拼进去、mask 切错位、答案被截断、奖励恒为 0……）。
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                Logger('-'*100)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j   # 和 rollout_batch 的排列顺序一致
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(contexts[idx])
                    Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                    tokens = input_ids[idx, plen:slen].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)   # 这里不跳特殊 token，方便看 <|im_end|> 在哪
                    Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                    Logger(text)
                    Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                    Logger('='*100)

        # ===== 第六阶段：组内相对优势（GRPO 的核心，不需要 critic）=====
        # rewards 的排列是 [s0_g0, s0_g1, ..., s1_g0, ...] → view(-1, G) 正好把同一 prompt 的 G 条分到一行。
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)   # 每组均值，展开回 [B*G]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)   # 每组标准差，展开回 [B*G]
        # A_i = (r_i - mean(group)) / (std(group) + 1e-4)
        #   - 用「组内相对」而不是绝对奖励：奖励的绝对值没有意义（2.5 分算好还是坏，取决于同组别人得了几分）；
        #   - unbiased=False（除以 G 而不是 G-1）：G 只有 4~8，无偏估计会让 std 偏大、优势被系统性压扁；
        #   - eps=1e-4：G 条奖励完全相同时 std=0，加 eps 避免除零。此时分子也是 0 → 优势=0
        #     → 这条样本不产生梯度，这是合理的：一组里没有好坏之分，就没有可学的信号。
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # ===== 第七阶段：KL 惩罚 + 策略梯度 =====
        # 无偏 KL 估计（k3 估计量）：KL(pi_theta || pi_ref) ≈ exp(d) - d - 1，其中 d = log pi_ref - log pi_theta。
        #   性质 1：e^x - x - 1 恒 >= 0（最小值在 x=0 处取 0），所以可以直接当惩罚项，不会出现「负 KL」；
        #   性质 2：在 d≈0 处二阶展开 ≈ d^2/2，梯度平滑，方差比直接估 KL 小得多（PPO 论文里的 k3 技巧）。
        # 🔍 对照 nanoGPT：SFT 只有「拟合」，没有「别离参考模型太远」这一项；RL 里不加 KL，
        #    模型会很快钻奖励漏洞（输出一堆能骗分但不成人话的东西），这叫 reward hacking。
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        # 重要性采样比：ratio = pi_theta(y) / pi_old(y) = exp(log pi_theta - log pi_old)
        #   pi_old 是「采样这份数据时」的策略，值来自 old_per_token_logps（rollout 时引擎返回的 logprob）。
        # ⚠️ 陷阱：old_logps 在「工具观察 / eos / pad」这些位置填的是 0.0，
        #    所以那些位置的 ratio = exp(logp_theta - 0) = exp(logp_theta) ≠ 1，看着很吓人。
        #    正常情况下 completion_mask=0 会把它们全部乘掉；一旦 mask 写错，ratio 会瞬间爆到 1e30，loss 变 nan。
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            # CISPO：把 ratio 截断后【detach】，只当权重用，梯度只从 per_token_logps 流过。
            #   loss = -( clamp(ratio, max=epsilon_high) * A * log pi_theta - beta * KL )
            #   - 只有上界（epsilon_high=5.0），没有下界：ratio 小意味着这个 token 现在比采样时更不可能，
            #     但 detach 之后它只是个常数权重，小就小，不需要像 PPO 那样「剪掉」。
            #   - 💡 和 PPO 的 min() 对比：PPO 在截断侧会把梯度整个清零（min 选中常数项 → 该 token 不再学习），
            #     大量 token 被 clip 时训练会变慢甚至停滞；CISPO 保留每个 token 的梯度、只重新加权，
            #     在「长序列 + 多轮工具交互」这种 ratio 容易飘的场景下更稳。
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # 标准 GRPO / PPO clip：
            #   loss = -min( ratio*A, clamp(ratio, 1-eps, 1+eps)*A ) + beta*KL
            #   A>0（好轨迹）：ratio 涨到 1+eps 就被封顶，防止一次更新走太远；
            #   A<0（坏轨迹）：ratio 降到 1-eps 就被封顶，同理。
            #   取 min 保证「保守」：只享受对自己有利的那一侧，另一侧一律截断。
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)          # [B*G, L-1] * [B*G, 1] → 逐 token 乘同一个优势
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        # 长度归一化：先在每条轨迹内部对所有有效 token 取平均（/ token_counts），再对轨迹取平均。
        #   - 不直接 sum：长轨迹 token 多，sum 会让它主导梯度（一条 2000 token 的轨迹 = 四条 500 token 轨迹）；
        #   - 也不做「全局 token 平均」：那样短轨迹的每个 token 权重会被放大，效果一样偏。
        #     逐轨迹平均最公平：每条轨迹对梯度的贡献一样大。
        #   - clamp(min=1) 防除零；valid_rows 过滤掉整行被 mask 掉的样本。
        #   - 全部无效时返回 per_token_loss.sum() * 0.0，而不是常数 0：
        #     这样保留了计算图和 dtype/device，DDP 下不会因为「本 rank 没有梯度」而报
        #     "Expected to have finished reduction in the prior iteration"。
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        # 梯度累积：除以 accumulation_steps，让 N 步的梯度之和 = N 步的平均梯度（等价于大 batch）。
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0 or step == iters:
            # clip 放在 step 之前：累积期间梯度是一直累加的，等真正 step 时再 clip 才是对的
            # （如果在 backward 之后就 clip，后面几步的梯度会把裁掉的部分又加回来）。
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            # 指标怎么读（调参时按这个顺序看）：
            #   Reward   平均奖励：应该缓慢上升；一直在 0 附近说明奖励没有区分度（检查 GT 分是否恒为 0）
            #   KL       和 ref 的平均 log-ratio：缓慢上升是正常的；突然飙升 → lr 太大或策略崩了
            #   GrpStd   组内奖励标准差：趋近 0 表示同组 G 条几乎一样，advantage 全为 0，学不到东西
            #            （说明 temperature 太低，或者这个 prompt 太简单/太难）
            #   AdvStd   advantage 的标准差：做过组内标准化，正常情况下接近 1
            #   AvgLen   平均有效 token 数：多轮任务里会随工具调用次数波动
            pl = loss.item() * args.accumulation_steps   # 还原成未除 accumulation_steps 的 loss，便于跨配置对比
            ar = rewards.mean().item()
            al = token_counts.float().mean().item()
            # 注意这里的 kl 是「原始 log-ratio 的加权平均」（可正可负），不是 loss 里那个恒非负的 k3 估计量。
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            am, ast = advantages.mean().item(), advantages.std().item()
            lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward":ar,"kl_ref":kl,"group_reward_std":gs,"advantages_std":ast,"policy_loss":pl,"avg_response_len":al,"advantages_mean":am,"learning_rate":lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()     # eval 只影响 dropout / MoE 的 aux_loss 计算路径；保存前后切换是习惯写法
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model   # 脱 DDP
            raw_model = getattr(raw_model, '_orig_mod', raw_model)                               # 脱 torch.compile
            state_dict = raw_model.state_dict()
            # 存 fp16：推理/发布用的权重，体积减半（训练用的一半精度对这个规模影响很小）。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 同时存一份「完整断点」：模型 + 优化器 + scheduler + epoch/step，可 --from_resume 1 续训。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            # del state_dict：state_dict 里是参数的引用，但 .half().cpu() 生成的 dict 是一份新的 CPU 副本，
            # 不清掉会一直占着内存，直到下一次分配触发 GC。
            del state_dict

        # 🔍 关键：update_policy 在【所有 rank】上调用（注意它在 is_main_process() 之外）。
        #   torch 引擎：只是换个引用，幂等，随便调；
        #   sglang 引擎：rank0 负责落盘 + HTTP 通知，其它 rank 在 dist.broadcast 上等 ok 标志。
        #   → 如果这里也写成「只在 rank0 调用」，其它 rank 永远不会进 broadcast，rank0 卡死 = 整个训练死锁。
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # 显式释放这一轮的大张量。
        # 多轮 rollout 很吃显存，而 Python 的引用计数虽然会立刻回收，但 GPU 缓存块要等 allocator 复用；
        # 手动 del 能让缓存尽早可复用，减少 OOM 概率（尤其是 per_token_logps 和 ref_per_token_logps，
        # 都是 [B*G, L-1] 的 float32）。
        del per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    # ================================ 主程序：参数与初始化 ================================
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    # ⚠️ 学习率 3e-7，比 SFT 的 1e-5 小两个数量级：RL 的策略梯度方差很大，lr 大了几步就把模型带崩。
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    # max_total_len 是「打包后单条序列」的上界，可能要 > max_seq_len（多轮拼接会超过单轮长度）。
    # ⚠️ 注意默认配置下 2500 > 1024+768=1792，超出部分靠 RoPE 外推硬撑，效果会掉。
    #    真要训长轨迹，应该把 --max_seq_len 一起调大，而不是只调这个。
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    # ⚠️ 注意这里默认 0.1，而 rollout_single 的形参默认是 0.5。实际生效的是 0.1
    #    （rollout_batch 显式传了 args.thinking_ratio），函数默认值只在被单独调用时才起作用。
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    args = parser.parse_args()

    # 初始化分布式（单卡时是 no-op，返回 0）。之后 args.device 会被改成 "cuda:{local_rank}"，
    # 因为每个进程必须绑定到自己那张卡上，否则 8 个进程会挤在 cuda:0 上 OOM。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 每个 rank 用不同的种子（42+rank），否则 8 个进程会采到一模一样的轨迹，等于 batch 白扩了。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    # 💡 max_seq_len 要取 max_seq_len + max_gen_len：模型的位置编码 / KV 缓存必须能同时装下
    #    「最长 prompt」和「最长生成」。只给 1024 的话，多轮拼接后超长会直接 shape 报错。
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    # 只在 --from_resume 1 时去读断点（权重 + 优化器 + scheduler + 进度），否则从 from_weight 重新开始。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 上不能 autocast cuda（会直接报错），所以显式用 nullcontext。
    # ⚠️ 这个脚本没有 GradScaler：如果你把 --dtype 改成 float16，fp16 的梯度很容易 underflow，
    #    要么保持 bfloat16，要么自己补上 torch.cuda.amp.GradScaler。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        # 名字叫 wandb 但实际是 swanlab（API 兼容 wandb），所以变量名沿用了 wandb。
        import swanlab as wandb
        # wandb_id 从断点里恢复：续训时日志能接上同一条曲线，而不是新建一个 run。
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}", id=wandb_id, resume=resume)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # 参考模型：和策略模型同一份 SFT 权重，但整个训练过程「冻结」——
    # eval() 关掉 dropout，requires_grad_(False) 让 PyTorch 不建计算图。
    # 💡 为什么必须要 ref：KL 惩罚是唯一阻止模型「为了奖励胡说八道」的约束（reward hacking）。
    #    代价是显存里多一份 64M 模型；显存实在不够可以调小 beta 甚至去掉 KL，但训练会明显更不稳。
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # Reward Model：外部的 1.8B 打分器，只在「没调用工具」的分支里用，不参与训练。
    # 用 fp16 是因为它只做前向，不需要梯度精度。
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    # Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # DDP 下必须用 DistributedSampler：否则 8 个进程会把整个数据集各跑一遍（等于 batch 缩小 8 倍、重复训练）。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 手写 collate：messages / tools / gt 都是变长的 Python 对象，默认的 default_collate 处理不了（它会尝试 stack）。
    # 所以这里干脆原样包成 list 交给训练循环，真正的张量化在 rl_train_epoch 的「打包」阶段做。
    def collate_fn(batch): return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    # 只为算 iters 而建的一个 loader（真正的 loader 用 batch_sampler，len() 不好直接拿）。
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    # scheduler 的 T_max 是「optimizer.step 次数」，不是 batch 数：要除以梯度累积步数。
    # eta_min=lr/10：退火到底也保留 10% 的学习率，后期还要能继续微调。
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复训练状态：模型权重 + 优化器动量 + lr 进度 + 从哪个 epoch/step 继续。
        # 注意 ref_model 不恢复 —— 它永远是 SFT 权重，不随训练变化。
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        # torch.compile：把前向编译成融合 kernel，64M 模型在 4090 上大概能快 20~30%。
        # ⚠️ 编译后模型被包成 OptimizedModule，state_dict 的 key 会带 "_orig_mod." 前缀，
        #    所以保存和 update_policy 里都要 getattr(m, '_orig_mod', m) 脱掉这一层。
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # 同 train_ppo：RoPE buffer 各 rank 一致，每步广播纯属浪费
        # （broadcast_buffers=False 关掉每步的 buffer 广播；MiniMind 只有 RoPE 的 freqs 是 buffer）
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    # DDP 包装之后再同步一次：让引擎拿到「最终形态」的模型（rollout 内部会 .module 解包）。
    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        # DDP 下每个 epoch 必须 set_epoch，否则 DistributedSampler 的 shuffle 种子不变，
        # 各个 epoch 的样本顺序完全一样（而且各 rank 的划分方式也固定）。
        train_sampler and train_sampler.set_epoch(epoch)
        # 单卡/非分布式时自己打乱：42+epoch 让每个 epoch 顺序不同，但整体可复现。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 断点续训时跳过已经训过的 batch（只在第一个 epoch 生效）。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: skip {start_step} steps')
            # iters 要传 len(loader) + skip：len(loader) 已经是「跳过之后」的长度，
            # 加回来才能让日志里的 (step/iters) 和 lr 进度算对。
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

    if dist.is_initialized():
        # barrier 保证所有 rank 都跑完再销毁进程组，否则先跑完的 rank 退出会让别的 rank 的
        # allreduce 永久 hang（NCCL 检测到对端消失时通常直接报错，但有时要等很久）。
        dist.barrier()
        dist.destroy_process_group()

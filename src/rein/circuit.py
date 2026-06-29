"""熔断四道闸(circuit breaker)+ 工具调用签名。

生产环境里 agent 最危险的故障是「失控」:死循环狂调工具、烧光 token 预算、
卡死不返回。这一模块就是兜底的安全闸 —— loop 每推进一步都来问一句:
「该停了吗?」任一道闸触顶,就安全终止,并在 stop_reason 里说明是哪道闸。

四道闸(对应 LoopConfig 的四个字段):
1. max_iterations —— 轮数闸:防止无限轮次。
2. max_tokens     —— 成本闸:累计 token 触顶即停(生产头号事故:死循环烧钱)。
3. timeout_s      —— 墙钟闸:从 run 开始计时,超时即停。
4. detect_loops   —— 重复闸:连续多轮调用「完全相同的工具(同名 + 同参)」判定卡死。

设计要点:
- check_circuit 是一个【纯函数】:输入 (session, config, start_time),输出
  stop_reason 字符串或 None。不改任何状态 —— 这样它好测、可在 loop 任意位置调用。
- 重复检测所需的「计数 / 上次签名」都存在 Session 里(可序列化),本模块只读不写;
  累加逻辑由 loop 在每轮用 signature() 比对后更新 Session。
"""

import json
import time

from rein.config import LoopConfig
from rein.ir import ToolCall
from rein.session import Session


def signature(tool_calls: list[ToolCall]) -> str:
    """计算一组工具调用的「签名」—— 同名 + 同参得到相同字符串,用于重复检测。

    刻意【忽略 ToolCall.id】(每次调用 id 都不同),只看「调了哪个工具、传了什么参数」。
    用 sort_keys=True 把参数 dict 规范化,保证 {"a":1,"b":2} 与 {"b":2,"a":1} 同签名。

    Args:
        tool_calls: 本轮模型发起的工具调用列表。

    Returns:
        规范化后的签名字符串;两轮签名相等 ⇒ 调用完全相同。
    """
    items = [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
    return json.dumps(items, sort_keys=True, ensure_ascii=False)


def check_circuit(session: Session, config: LoopConfig, start_time: float) -> str | None:
    """检查四道熔断闸,返回触顶的闸名(stop_reason)或 None(都没触顶)。

    Args:
        session:    当前会话状态(读取 iteration / usage / repeat_count)。
        config:     loop 配置(各道闸的阈值;None 表示该闸不限制)。
        start_time: run 开始时的 time.monotonic() 时间戳,用于墙钟超时判断。

    Returns:
        - "max_iterations" / "max_tokens" / "timeout" / "loop_detected":某道闸触顶。
        - None:四道闸都未触顶,可以继续。
    """
    # ① 轮数闸:已完成轮数达到上限即停。
    if session.iteration >= config.max_iterations:
        return "max_iterations"

    # ② 成本闸:累计(输入 + 输出)token 达到上限即停;None 表示不限制。
    if config.max_tokens is not None:
        total_tokens = session.usage.input_tokens + session.usage.output_tokens
        if total_tokens >= config.max_tokens:
            return "max_tokens"

    # ③ 墙钟闸:从 start_time 起的耗时达到上限即停;None 表示不限制。
    #    用 monotonic(单调时钟)而非 wall clock,避免系统改时间导致误判。
    if config.timeout_s is not None:
        if time.monotonic() - start_time >= config.timeout_s:
            return "timeout"

    # ④ 重复闸:连续相同调用次数达到阈值即停;仅在 detect_loops 开启时生效。
    if config.detect_loops and session.repeat_count >= config.repeat_threshold:
        return "loop_detected"

    return None

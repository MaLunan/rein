"""Loop 运行配置(LoopConfig)。

LoopConfig 描述「这个 agent 的循环该怎么跑」—— 它是 Agent 蓝图的一部分,
而不是会话状态,因此【不会】进入可序列化的 Session。

M0 的核心是两件事:
1. 熔断四道闸 —— 防止 agent 失控(死循环、烧光预算、卡死)。
2. 权限模式 —— 工具执行前的放行策略。
"""

from typing import Literal

from pydantic import BaseModel


class LoopConfig(BaseModel):
    """Agentic loop 的运行配置(熔断 + 权限)。"""

    # ---- 熔断四道闸:任意一道触顶,loop 就安全停止,并在 RunResult.stop_reason 说明原因 ----

    max_iterations: int = 50
    """第①道闸:最大轮数。一轮 = 一次「调模型 → 执行工具」。
    防止模型无限循环调用工具。默认 50:足够复杂任务,又不至于失控。"""

    max_tokens: int | None = 200_000
    """第②道闸:累计 token 上限(成本闸)。每轮把 Completion 的 token 累加到
    Session.usage,一旦总量超过这个值就停。None 表示不限制。
    默认 20 万 —— 生产头号事故就是 agent 死循环烧光预算,这道闸是兜底。"""

    timeout_s: float | None = 120
    """第③道闸:墙钟超时(秒)。从 run 开始计时,超过即停。None 表示不限制。
    默认 120 秒。"""

    detect_loops: bool = True
    """第④道闸开关:是否开启「重复调用检测」。"""

    repeat_threshold: int = 3
    """第④道闸阈值:连续多少轮「调用完全相同的工具(同名 + 同参)」就判定为
    卡死并停止。仅在 detect_loops=True 时生效。默认 3。"""

    # ---- 权限 ----

    permission: Literal["allow", "ask", "deny"] = "allow"
    """工具执行前的放行策略:
    - "allow":直接放行(M0 默认,保证「5 行示例」一路跑完、不被打断)。
    - "deny": 直接拒绝,以错误结果(is_error=True)回填,让模型自己应对。
    - "ask":  执行前请人确认 —— 需要「暂停 / 恢复」机制,留到 M2 实现;
              M0 若设为 ask 会在执行时抛 NotImplementedError,绝不假装支持。
    """

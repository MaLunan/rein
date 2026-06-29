"""运行结果(RunResult)、单步记录(Step)与中断态(Interrupt)。

这是 agent 跑完一次后「交回给你」的东西:
- RunResult:最终结果报告(答案 + 过程 + 用量 + 完整会话状态)。
- Step:    运行过程中每一步的流水账,用于事后回看 / 调试(可观测的最小形态)。
- Interrupt:中断态的结构,M2 实现「暂停 / 恢复」时才真正用到;M0 先占好形状。
"""

from typing import Literal

from pydantic import BaseModel, Field

from rein.ir import ToolCall, Usage
from rein.session import Session


class Step(BaseModel):
    """运行过程中的「一步」记录 —— 要么是问了一次模型,要么是执行了一个工具。

    这是「可观测」的最小形态:把每一步留痕,事后能回放整个运行过程、定位问题。
    """

    index: int
    """第几步(从 0 开始)。"""

    kind: Literal["model", "tool"]
    """这一步是什么:"model"=调用了一次大模型;"tool"=执行了一个工具。"""

    summary: str
    """这一步的简短摘要:模型步是模型说的话(截断),工具步是「工具名 → 结果摘要」。"""

    tool_calls: list[ToolCall] | None = None
    """仅当模型步且模型决定调用工具时有值:这一步模型请求了哪些工具。"""

    is_error: bool | None = None
    """仅当工具步:这个工具是否执行出错。"""

    duration_s: float | None = None
    """这一步耗时(秒)。"""

    usage: Usage | None = None
    """这一步的 token / 成本用量(主要是模型步有)。"""


class Interrupt(BaseModel):
    """中断态 —— agent 在工具执行前需要「外部介入」时产生。

    M0 不会产生中断(permission=allow 一路到底);此结构先占好形状,
    供 M2 的 HITL(人工确认)/ 断点续跑使用。
    """

    type: Literal["need_approval", "need_input", "error"]
    """中断原因:
    - "need_approval":有危险工具待人批准
    - "need_input":   需要人补充信息
    - "error":        出现了需要外部决策(重试 / 放弃)的错误"""

    tool_call: ToolCall | None = None
    """触发中断的工具调用(如待批准的那个)。"""

    message: str = ""
    """给调用方 / 人看的说明文字。"""


class RunResult(BaseModel):
    """agent 跑完一次的「结果报告」。"""

    status: Literal["done", "interrupted"]
    """整体状态:done = 正常跑完;interrupted = 中途中断、等待恢复(M2)。"""

    output: str | None
    """最终的助手回答文本(没有则为 None)。"""

    session: Session
    """完整的会话状态。放进结果里,是因为「结果」不只是那句答案,
    还包括整个过程状态 —— 它可被序列化存档、(M2 起)可被恢复继续。"""

    usage: Usage = Field(default_factory=Usage)
    """本次运行的总 token / 成本用量。"""

    elapsed_s: float | None = None
    """本次运行的总墙钟耗时(秒)。可观测用;未测得时为 None。"""

    stop_reason: str = "done"
    """结束原因:done / max_iterations / max_tokens / timeout / loop_detected / interrupted。"""

    steps: list[Step] = Field(default_factory=list)
    """逐步流水账(见 Step)。"""

    interrupt: Interrupt | None = None
    """若 status="interrupted",这里说明中断详情(M2 用);否则为 None。"""

    def __str__(self) -> str:
        """让 print(agent.run(...)) 直接显示答案 —— 小白只想看结果就 print;
        需要细节再取 .steps / .usage / .stop_reason / .session(渐进式暴露)。"""
        return self.output or ""

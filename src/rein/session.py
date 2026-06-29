"""会话状态(Session)与状态机阶段(Stage)。

Session 是整个框架的「灵魂」:它是一份【完全可序列化的纯数据】,装着一次
agent 运行的全部状态。配合无状态的 loop(见 loop.py),实现:

    暂停 = 把 Session 存盘   /   恢复 = 把 Session 喂回 loop

M0 不实现恢复(resume),但状态全部收进 Session,为 M2 的断点续跑留好「形状」。

设计铁律:Session 只有数据字段,没有任何业务方法 —— 严守「数据与行为分离」。
所有推进逻辑都放在 loop.py。这样 Session 永远是一份干净、可序列化的状态快照。
"""

from enum import Enum

from pydantic import BaseModel, Field

from rein.ir import Message, ToolCall, Usage


class Stage(str, Enum):  # noqa: UP042
    """状态机的三个阶段。

    继承 str,使其能直接序列化成字符串(如 "call_model"),存盘 / 恢复无障碍。

    流转关系:
        CALL_MODEL --(模型没有要调工具)--> DONE
        CALL_MODEL --(模型要调工具)------> RUN_TOOLS
        RUN_TOOLS  --(工具执行完)--------> CALL_MODEL
    """

    CALL_MODEL = "call_model"
    """阶段一:调用模型,拿到一次 Completion。"""

    RUN_TOOLS = "run_tools"
    """阶段二:执行模型请求的工具,把结果回填。"""

    DONE = "done"
    """终态:本次运行结束。"""


class Session(BaseModel):
    """一次 agent 运行的全部状态(可序列化的状态快照)。"""

    messages: list[Message] = Field(default_factory=list)
    """对话历史:系统提示 / 用户输入 / 模型回答 / 工具结果,按时间顺序排列。"""

    stage: Stage = Stage.CALL_MODEL
    """当前处于状态机的哪个阶段。新会话从 CALL_MODEL 开始。"""

    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    """【待执行】的工具调用队列。
    - CALL_MODEL 阶段:模型要调工具时,把工具调用存到这里,然后转入 RUN_TOOLS。
    - RUN_TOOLS 阶段:逐个执行并清空。
    放进 Session(而非 loop 的局部变量)的原因:未来的中断点正好在「工具执行前」,
    那一刻这些「待办」必须能随 Session 一起存盘、恢复 —— 这是为恢复留形状的关键。"""

    iteration: int = 0
    """已完成的轮数(一轮 = 一次「模型→工具」)。熔断第①道闸(max_iterations)用。"""

    usage: Usage = Field(default_factory=Usage)
    """累计的 token / 成本。每轮把 Completion 的用量加进来。熔断第②道闸(max_tokens)用。"""

    repeat_count: int = 0
    """连续「调用完全相同工具」的次数。熔断第④道闸(detect_loops)用。"""

    last_signature: str | None = None
    """上一轮工具调用的「签名」(工具名 + 参数的规范化字符串)。
    与本轮签名比较,相同则 repeat_count 累加。熔断第④道闸用。"""

    done: bool = False
    """是否已结束。loop 的驱动循环以它作为终止条件。"""

    stop_reason: str | None = None
    """结束原因:done(正常)/ max_iterations / max_tokens / timeout / loop_detected。
    None 表示尚未结束。"""

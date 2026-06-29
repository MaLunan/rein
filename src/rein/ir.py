"""统一内部表示(IR / Intermediate Representation)。

整个框架内部「只」流通这一套数据类型;各家大模型厂商(经 LiteLLM 接入)的
格式差异,全部在 Provider 边界翻译成这些类型 —— 核心层永远只跟 IR 打交道。

设计要点:
- 所有类型都是 pydantic BaseModel —— 因为「可序列化」是整个框架的地基:
  Session 要能存盘、断点恢复,就要求它内部每个对象都能 model_dump_json() 序列化、
  再 model_validate_json() 还原(这一点已用冒烟测试验证过)。
- M0 只覆盖「纯文本 + 工具调用」场景;流式增量(StreamChunk)、多模态等留到后续阶段。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """模型发起的「一次工具调用」请求。

    当模型在回答里表示「我要调用 search(query='天气')」时,就会产生一个 ToolCall。
    """

    id: str
    """这次调用的唯一标识。工具执行完后,结果(ToolResult)用同一个 id 指回来,
    让模型知道「这个结果对应的是哪一次调用」。"""

    name: str
    """要调用的工具名(对应 @tool 装饰的函数名)。"""

    arguments: dict[str, Any]
    """调用参数,已经是解析好的 dict(不是 JSON 字符串)。
    模型原始吐回的是 JSON 文本,由 Provider 负责解析成 dict 再装进这里。
    约束:内容必须 JSON 可序列化(str/int/float/bool/list/dict/None)。"""


class ToolResult(BaseModel):
    """一次工具调用的执行结果,准备回填给模型。"""

    tool_call_id: str
    """指回对应的 ToolCall.id,告诉模型这是哪次调用的结果。"""

    content: str
    """结果文本。注意:这里永远是「字符串」——
    工具即使返回了对象 / 字典,也会先被序列化器(见 tools.py)转成文本再放进来。
    原因有二:① 喂给模型的只能是文本;② Session 要保持可序列化。"""

    is_error: bool = False
    """工具是否执行出错。出错时我们「不抛异常中断 loop」,而是把错误信息
    当作普通结果回填(is_error=True),让模型读到错误、自己想办法补救(自愈)。"""


class Message(BaseModel):
    """一条对话消息 —— 对话历史就是一个 Message 列表。

    通过 role + 不同字段的组合,表达以下几种消息:
    - 系统提示:     role="system",   content="你是..."
    - 用户输入:     role="user",     content="帮我..."
    - 模型纯回答:   role="assistant", content="答案..."
    - 模型要调工具: role="assistant", tool_calls=[ToolCall, ...]
    - 工具结果回填: role="tool",     tool_call_id=..., content="结果..."

    这套结构刻意对齐 OpenAI / LiteLLM 的通用格式,这样 Provider 在 IR 与
    厂商格式之间互转时几乎零成本(我们 wrap LiteLLM,见 DESIGN D2)。
    """

    role: Literal["system", "user", "assistant", "tool"]
    """消息角色,决定这条消息是「谁」说的。"""

    content: str = ""
    """文本内容。默认空串(而非 None)—— M0 只处理纯文本,默认空串可以省掉
    到处判 None 的麻烦。当 assistant 只是要调工具时,content 可能为空。"""

    tool_calls: list[ToolCall] | None = None
    """仅当 role="assistant" 且模型决定调用工具时有值;否则为 None。"""

    tool_call_id: str | None = None
    """仅当 role="tool"(工具结果消息)时有值,指回对应的 ToolCall.id。"""


class Usage(BaseModel):
    """一次 / 累计的 token 与成本统计。"""

    input_tokens: int = 0
    """输入(提示)消耗的 token 数。"""

    output_tokens: int = 0
    """输出(生成)消耗的 token 数。"""

    cost_usd: float | None = None
    """折算成美元的成本;拿不到精确成本时为 None。"""

    def __add__(self, other: "Usage") -> "Usage":
        """让两个 Usage 直接相加,方便在 loop 里逐轮累计:
            session.usage = session.usage + completion.usage
        成本只有在至少一边拿得到时才相加,否则保持 None。"""
        if not isinstance(other, Usage):
            return NotImplemented
        cost = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )


class Completion(BaseModel):
    """模型「一次返回」的完整结果(对应一轮 CALL_MODEL)。"""

    message: Message
    """模型这次产出的消息(可能是纯文本回答,也可能带 tool_calls)。"""

    usage: Usage = Field(default_factory=Usage)
    """这次调用的 token / 成本消耗。用 default_factory 避免可变默认值被共享。"""

    finish_reason: Literal["stop", "tool_calls", "length", "error"]
    """模型为什么停下,决定 loop 的下一步:
    - "stop":       正常说完了        → loop 结束
    - "tool_calls": 要调用工具        → 去执行工具,再继续下一轮
    - "length":     达到长度上限      → (后续阶段可据此触发上下文压缩)
    - "error":      出错"""


class ToolSpec(BaseModel):
    """喂给模型的「工具定义」—— 告诉模型有哪些工具可用、各自怎么调。

    由 tools.py 从被 @tool 装饰的函数(签名 + 类型注解 + docstring)自动生成。
    """

    name: str
    """工具名。"""

    description: str
    """工具用途说明(通常来自函数 docstring),模型据此决定何时调用它。"""

    parameters: dict[str, Any]
    """参数的 JSON Schema(描述每个参数的类型 / 是否必填),
    由 build_schema() 从函数注解生成。"""


class StreamChunk(BaseModel):
    """流式输出的「一个增量片段」(M1)。

    设计要点(对应 M1 注意点 1/2):
    - 流式是【旁路观测】:它实时把模型吐字发出来给你看,但【不参与状态推进】——
      状态机仍按「算完的完整 Completion」一步步走(见 loop.astream)。
    - 工具调用的逐字分片在 Provider 内部累积拼装,【完整后】才作为一次 tool_calls
      暴露出来 —— 各厂商分片格式的差异不泄漏到核心。
    """

    type: Literal["text", "tool_calls", "done"]
    """片段类型:
    - "text":      模型新吐出的一段文本(增量)。
    - "tool_calls":模型这一轮请求的工具调用(已拼装完整,一次性给出)。
    - "done":      本次流式结束(携带累计用量与结束原因)。"""

    text: str = ""
    """type="text" 时的增量文本;其它类型为空串。"""

    tool_calls: list[ToolCall] | None = None
    """type="tool_calls" 时,这一轮完整的工具调用列表;其它类型为 None。"""

    usage: Usage | None = None
    """type="done" 时,本次运行的累计 token / 成本用量。"""

    finish_reason: str | None = None
    """type="done" 时的结束原因(done / stop / max_iterations / ...)。"""

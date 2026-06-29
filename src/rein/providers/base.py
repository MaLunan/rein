"""Provider 接口:模型接入层的统一契约。

Provider 负责「把一段对话发给某个大模型,拿回一次结果」—— 它是 IR 与各家
厂商之间的唯一边界:进去的是 IR(Message / ToolSpec),出来的也是 IR(Completion)。
核心层(loop)只认这个接口,不关心背后到底是 Mock、LiteLLM 还是别的实现。

为什么用 Protocol(结构化类型)而不是抽象基类:
- 任何对象只要「长得对」(有一个签名匹配的 async complete),就自动算 Provider,
  无需显式继承 —— 这让用户能轻松塞进自己的实现(鸭子类型),契合「极薄」哲学。
- 加 @runtime_checkable,可在需要时用 isinstance(x, Provider) 做运行期校验。

M0 只定义 complete(一次性返回);流式 stream() 留到 M1 再加,不在此处占位。
"""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from rein.ir import Completion, Message, StreamChunk, ToolSpec


@runtime_checkable
class Provider(Protocol):
    """模型接入层的统一接口。"""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> Completion:
        """把对话历史(+ 可选工具定义)发给模型,返回一次完整结果。

        Args:
            messages: 对话历史(IR Message 列表)。
            tools:    本次可用的工具定义(IR ToolSpec 列表);无工具时为 None。
            **kwargs: 透传给底层实现的额外参数(temperature 等)。

        Returns:
            Completion:模型这次的产出(纯文本回答或带 tool_calls),含 token 用量。
        """
        ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """流式版的 complete:逐段产出 StreamChunk(M1)。

        实现为异步生成器(`async def ...: yield ...`),依次吐出:
        - 若干 type="text" 片段(模型实时吐字),
        - 可选一个 type="tool_calls" 片段(拼装完整的工具调用),
        - 最后一个 type="done" 片段(累计用量 + 结束原因)。

        注意:工具调用的逐字分片必须在实现内部累积拼完整,完整后才作为
        type="tool_calls" 暴露 —— 核心层只认完整 ToolCall。
        """
        ...

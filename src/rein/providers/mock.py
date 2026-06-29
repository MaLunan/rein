"""MockProvider —— 不联网、确定性的测试用 Provider(核心模块)。

它不调用任何真实模型,而是按【预设脚本】依次返回结果,用来测试 loop、
工具编排逻辑:无需 API key、不烧 token、结果可复现。

用法示例:
    provider = MockProvider([
        [ToolCall(id="1", name="search", arguments={"q": "天气"})],  # 第1次:让模型"调工具"
        "北京今天晴。",                                              # 第2次:模型"纯文本回答"
    ])

每次 complete() 按顺序消费列表里的一项:
- 是 str            → 当作模型的纯文本回答(finish_reason="stop")
- 是 list[ToolCall] → 当作模型发起工具调用(finish_reason="tool_calls")
"""

from collections.abc import AsyncIterator

from rein.ir import Completion, Message, StreamChunk, ToolSpec, Usage


class MockProvider:
    """按预设脚本依次返回结果的假 Provider,用于测试。"""

    def __init__(self, responses: list, usage_per_call: Usage | None = None):
        """
        Args:
            responses:      预设回应列表,每项为 str(文本)或 list[ToolCall](工具调用)。
            usage_per_call: 每次调用计入的 token 用量(默认 input=1/output=1,
                            方便测试熔断的 token 累计)。
        """
        self._responses = list(responses)
        self._index = 0
        self._usage_per_call = usage_per_call or Usage(input_tokens=1, output_tokens=1)

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> Completion:
        """返回脚本里的下一项;脚本用尽后回一句结束语,避免测试里 loop 卡死。"""
        if self._index >= len(self._responses):
            return Completion(
                message=Message(role="assistant", content="[MockProvider:脚本已用尽]"),
                finish_reason="stop",
                usage=self._usage_per_call,
            )

        item = self._responses[self._index]
        self._index += 1

        if isinstance(item, str):
            # 纯文本回答 → 正常结束
            return Completion(
                message=Message(role="assistant", content=item),
                finish_reason="stop",
                usage=self._usage_per_call,
            )

        # 否则视为一组 ToolCall → 模型要调工具
        tool_calls = list(item)
        return Completion(
            message=Message(role="assistant", tool_calls=tool_calls),
            finish_reason="tool_calls",
            usage=self._usage_per_call,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """流式版:消费同一份脚本,但把文本【逐字】发出来模拟分片(便于测试)。

        - 文本项  → 逐字 yield type="text",最后 yield type="done"。
        - 工具项  → yield 一个完整 type="tool_calls",再 yield type="done"。
        - 脚本用尽 → 发一句结束文本 + done,避免流式 loop 卡死。
        """
        if self._index >= len(self._responses):
            yield StreamChunk(type="text", text="[MockProvider:脚本已用尽]")
            yield StreamChunk(type="done", finish_reason="stop", usage=self._usage_per_call)
            return

        item = self._responses[self._index]
        self._index += 1

        if isinstance(item, str):
            for ch in item:  # 逐字符发,模拟真实的 token 分片
                yield StreamChunk(type="text", text=ch)
            yield StreamChunk(type="done", finish_reason="stop", usage=self._usage_per_call)
            return

        # 工具调用:分片已在(假想的)Provider 内拼完整,这里一次性给出
        yield StreamChunk(type="tool_calls", tool_calls=list(item))
        yield StreamChunk(type="done", finish_reason="tool_calls", usage=self._usage_per_call)

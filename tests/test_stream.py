"""流式测试(M1):astream 实时分片 / 工具调用拼装 / done 用量 / 旁路不改主干。

全部用 MockProvider 模拟分片,不联网。
"""

import asyncio

from rein.agent import Agent
from rein.ir import Message, ToolCall
from rein.loop import arun, astream
from rein.providers import MockProvider
from rein.session import Session, Stage
from rein.tools import Tool, ToolRegistry


def _add_registry() -> ToolRegistry:
    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    reg = ToolRegistry()
    reg.add(Tool.from_function(add))
    return reg


def _tool_script() -> list:
    """一段「先调 add 再给文本」的脚本(每次调用都新建,避免共享消费)。"""
    return [
        [ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})],
        "结果是 3",
    ]


def _collect(agen_factory):
    """把一个异步生成器跑完,收集所有 chunk。"""

    async def go():
        out = []
        async for c in agen_factory():
            out.append(c)
        return out

    return asyncio.run(go())


def test_astream_文本逐片实时输出():
    agent = Agent(provider=MockProvider(["你好世界"]))
    chunks = _collect(lambda: agent.astream("hi"))

    texts = [c.text for c in chunks if c.type == "text"]
    assert "".join(texts) == "你好世界"
    assert len(texts) >= 2  # 确实被切成了多片(逐字)

    done = [c for c in chunks if c.type == "done"]
    assert len(done) == 1
    assert done[0].finish_reason == "done"
    assert done[0].usage.output_tokens == 1  # 一次模型调用


def test_astream_工具调用拼装完整且多轮():
    agent = Agent(provider=MockProvider(_tool_script()), tools=[])

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    chunks = _collect(lambda: agent.astream("帮我加一下"))
    types = [c.type for c in chunks]

    # 出现一次完整的 tool_calls 片段,且工具名/参数正确
    tc_chunks = [c for c in chunks if c.type == "tool_calls"]
    assert len(tc_chunks) == 1
    assert tc_chunks[0].tool_calls[0].name == "add"
    assert tc_chunks[0].tool_calls[0].arguments == {"a": 1, "b": 2}

    # 工具回填后,模型最终文本被流式吐出
    final_text = "".join(c.text for c in chunks if c.type == "text")
    assert final_text == "结果是 3"
    assert types[-1] == "done"


def test_astream_done携带累计用量():
    """两次模型调用 → done 的累计用量 output=2。"""
    agent = Agent(provider=MockProvider(_tool_script()))

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    chunks = _collect(lambda: agent.astream("加"))
    done = [c for c in chunks if c.type == "done"][0]
    assert done.usage.output_tokens == 2
    assert done.usage.input_tokens == 2


def test_流式与非流式_最终session状态一致():
    """旁路证明:astream 跑完的 session 与 arun 跑完的 session 关键状态完全一致。"""
    # 非流式
    s1 = Session(messages=[Message(role="user", content="x")])
    asyncio.run(arun(s1, MockProvider(_tool_script()), _add_registry()))

    # 流式(把生成器跑干)
    s2 = Session(messages=[Message(role="user", content="x")])

    async def drain():
        async for _ in astream(s2, MockProvider(_tool_script()), _add_registry()):
            pass

    asyncio.run(drain())

    # 关键状态逐项比对
    assert s2.done == s1.done is True
    assert s2.stage == s1.stage == Stage.DONE
    assert s2.iteration == s1.iteration
    assert s2.stop_reason == s1.stop_reason
    assert s2.usage.output_tokens == s1.usage.output_tokens
    # 对话历史(角色 + 内容)一致
    assert [(m.role, m.content) for m in s2.messages] == [(m.role, m.content) for m in s1.messages]


def test_astream_无工具纯文本结束():
    """没有任何工具时,纯文本也能正常流式结束。"""
    chunks = _collect(
        lambda: astream(
            Session(messages=[Message(role="user", content="hi")]),
            MockProvider(["答案"]),
            ToolRegistry(),
        )
    )
    assert "".join(c.text for c in chunks if c.type == "text") == "答案"
    assert chunks[-1].type == "done"

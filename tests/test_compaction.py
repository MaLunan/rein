"""上下文压缩测试(M3 批1):token 估算 / 滑窗 / 摘要 / 可序列化 / 自动触发 / 不破坏 resume。"""

import asyncio

from rein.agent import Agent
from rein.compaction import SlidingWindow, SummarizeCompaction, estimate_tokens
from rein.config import LoopConfig
from rein.ir import Completion, Message, ToolCall, Usage
from rein.loop import arun
from rein.providers import MockProvider
from rein.session import Session
from rein.tools import ToolRegistry

# ---------- token 估算 ----------


def test_estimate_tokens_中文比同长英文重():
    zh = estimate_tokens("你好世界")  # 4 个非 ASCII ≈ 4+
    en = estimate_tokens("hello world")  # 11 个 ASCII ≈ 11*0.25
    assert zh > en


def test_estimate_tokens_接受消息列表():
    msgs = [Message(role="user", content="abcd"), Message(role="assistant", content="中文")]
    assert estimate_tokens(msgs) >= 1


# ---------- SlidingWindow ----------


def test_sliding_window_保留system与最近N():
    msgs = [Message(role="system", content="sys")] + [
        Message(role="user", content=f"u{i}") for i in range(10)
    ]
    out = asyncio.run(SlidingWindow(max_messages=3).compact(msgs))
    assert out[0].role == "system"
    assert [m.content for m in out[1:]] == ["u7", "u8", "u9"]


def test_sliding_window_未超原样返回():
    msgs = [Message(role="user", content="a"), Message(role="user", content="b")]
    out = asyncio.run(SlidingWindow(max_messages=5).compact(msgs))
    assert out == msgs


def test_sliding_window_去掉开头孤儿tool():
    msgs = [
        Message(role="user", content="a"),
        Message(role="assistant", content="b"),
        Message(role="tool", tool_call_id="x", content="r"),
        Message(role="user", content="c"),
        Message(role="assistant", content="d"),
    ]
    # 最近 3 = [tool, c, d] → 开头的孤儿 tool 被丢 → [c, d]
    out = asyncio.run(SlidingWindow(max_messages=3, keep_system=False).compact(msgs))
    assert [m.content for m in out] == ["c", "d"]


# ---------- SummarizeCompaction ----------


def _long_history(n: int = 10) -> list[Message]:
    return [Message(role="system", content="你是助手")] + [
        Message(
            role="user" if i % 2 == 0 else "assistant",
            content="这是一段比较长的中文历史消息内容" * 2,
        )
        for i in range(n)
    ]


def test_summarize_折叠旧历史_保留近期():
    msgs = _long_history(10)
    before = len(msgs)
    out = asyncio.run(SummarizeCompaction(max_tokens=50, keep_recent=3).compact(msgs))

    assert out[0].role == "system" and out[0].content == "你是助手"  # system 保留
    assert any("[历史摘要]" in m.content for m in out)  # 有摘要
    assert len(out) < before  # 确实变短
    assert out[-1].content == msgs[-1].content  # 近期原文保留


def test_summarize_未超不动():
    msgs = [Message(role="user", content="短")]
    out = asyncio.run(SummarizeCompaction(max_tokens=10000).compact(msgs))
    assert out == msgs


def test_summarize_注入provider做真摘要():
    class FakeSummarizer:
        async def complete(self, messages, tools=None, **kwargs):
            return Completion(
                message=Message(role="assistant", content="这是摘要"),
                finish_reason="stop",
                usage=Usage(),
            )

    msgs = [Message(role="user", content="长内容" * 50) for _ in range(8)]
    out = asyncio.run(
        SummarizeCompaction(max_tokens=10, keep_recent=2, summarizer=FakeSummarizer()).compact(msgs)
    )
    assert any(m.content == "[历史摘要] 这是摘要" for m in out)


# ---------- 压缩后可序列化(不破坏 M2 恢复) ----------


def test_压缩后仍可序列化往返():
    msgs = _long_history(8)
    out = asyncio.run(SummarizeCompaction(max_tokens=20, keep_recent=2).compact(msgs))
    s = Session(messages=out)
    back = Session.model_validate_json(s.model_dump_json())
    assert len(back.messages) == len(out)


# ---------- 自动触发(loop 在问模型前压缩) ----------


def test_loop自动压缩超长历史():
    msgs = (
        [Message(role="system", content="sys")]
        + [Message(role="user" if i % 2 == 0 else "assistant", content=f"m{i}") for i in range(20)]
        + [Message(role="user", content="最后问题")]
    )
    s = Session(messages=msgs)

    r = asyncio.run(
        arun(s, MockProvider(["答案"]), ToolRegistry(), compaction=SlidingWindow(max_messages=3))
    )
    assert r.status == "done"
    # 压缩在问模型前生效:历史被大幅裁剪(远小于原来的 22 条)
    non_system = [m for m in r.session.messages if m.role != "system"]
    assert len(non_system) <= 5
    assert any(m.role == "system" and m.content == "sys" for m in r.session.messages)


# ---------- 压缩与 resume 共存(不破坏 M2) ----------


def test_压缩与ask审批resume共存():
    agent = Agent(
        provider=MockProvider(
            [[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "等于 3"]
        ),
        config=LoopConfig(permission="ask"),
        compaction=SlidingWindow(max_messages=10),
    )

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    r = agent.run("加")
    assert r.status == "interrupted"
    r2 = agent.resume(r.session, approve=True)
    assert r2.status == "done"
    assert r2.output == "等于 3"

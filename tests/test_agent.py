"""agent 测试:并发不串台(核心)/ 工具注册 / tools 包装 / 懒建报错 / system / Chat 多轮 / run 事件循环保护。"""

import asyncio

import pytest

from rein.agent import Agent, Chat, tool
from rein.ir import Completion, Message, ToolCall, Usage
from rein.providers import MockProvider


class EchoProvider:
    """无状态的回显 Provider:把「最后一条 user 消息」原样回出来。

    无状态是关键 —— 它能被并发的多个会话安全共享,从而专门用来检验
    「Session 是否串台」(若串台,甲会话会读到乙的 user 消息)。
    """

    async def complete(self, messages, tools=None, **kwargs) -> Completion:
        last_user = [m for m in messages if m.role == "user"][-1].content
        return Completion(
            message=Message(role="assistant", content=f"echo:{last_user}"),
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


def test_并发多session不串台():
    """同一个 Agent 并发跑两个 prompt,各自结果互不污染(D3 并发安全)。"""
    agent = Agent(provider=EchoProvider())

    async def go():
        return await asyncio.gather(agent.arun("甲"), agent.arun("乙"))

    a, b = asyncio.run(go())
    assert a.output == "echo:甲"
    assert b.output == "echo:乙"


def test_agent_tool装饰器_注册且返回原函数():
    agent = Agent(provider=MockProvider(["ok"]))

    @agent.tool
    def search(q: str) -> str:
        "搜索"
        return q

    assert callable(search)  # 返回原函数,仍可直接调用
    assert search("x") == "x"
    assert "search" in agent.registry  # 已登记


def test_agent_tool实际被调用():
    """模型请求 add 工具,Agent 能找到并执行,最终给出答案。"""
    script = [[ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})], "和是 5"]
    agent = Agent(provider=MockProvider(script))

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    r = agent.run("帮我加一下")
    assert r.output == "和是 5"
    assert any(m.role == "tool" and m.content == "5" for m in r.session.messages)


def test_构造器tools_接受裸函数与Tool():
    def f1(x: int) -> int:
        "原样返回"
        return x

    agent = Agent(provider=MockProvider(["ok"]), tools=[f1, tool(lambda: "y")])
    assert "f1" in agent.registry
    assert len(agent.registry) == 2


def test_既无model又无provider_报错():
    agent = Agent()
    with pytest.raises(ValueError):
        agent.run("hi")


def test_system提示进入历史():
    agent = Agent(provider=EchoProvider(), system="你是助手")
    r = agent.run("在吗")
    assert r.session.messages[0].role == "system"
    assert r.session.messages[0].content == "你是助手"


def test_chat_多轮保留历史():
    """Chat 跨轮保留历史:第二轮的消息列表应包含第一轮的内容。"""
    agent = Agent(provider=EchoProvider())
    chat = agent.chat()

    r1 = chat.send("第一句")
    assert r1.output == "echo:第一句"

    r2 = chat.send("第二句")
    assert r2.output == "echo:第二句"

    # 历史累积:两轮的 user 消息都在
    user_contents = [m.content for m in chat.session.messages if m.role == "user"]
    assert user_contents == ["第一句", "第二句"]


def test_run在事件循环内报错():
    """已在事件循环中调用同步 run() → 报错引导用 arun()。"""
    agent = Agent(provider=EchoProvider())

    async def inner():
        agent.run("x")  # 此处已在事件循环里

    with pytest.raises(RuntimeError):
        asyncio.run(inner())


def test_chat_是Chat实例():
    agent = Agent(provider=EchoProvider())
    assert isinstance(agent.chat(), Chat)


def test_chat_session_none新建带system():
    agent = Agent(provider=EchoProvider(), system="你是助手")
    chat = agent.chat(session=None)
    assert chat.session.messages[0].role == "system"
    assert chat.session.messages[0].content == "你是助手"


def test_chat_从存盘session继续_企业级模式():
    """企业级无状态模式:存盘 → 重建 Chat(模拟新请求/新进程)→ 历史延续。"""
    from rein.session import Session

    agent = Agent(provider=MockProvider(["第一轮回答", "记得:你叫小明"]))

    # 第一轮
    chat1 = agent.chat()
    chat1.send("我叫小明")
    saved = chat1.session.model_dump_json()  # 存盘(redis/db)

    # 新请求:从存盘 session 重建 Chat(可能在另一进程)
    chat2 = agent.chat(session=Session.model_validate_json(saved))
    r = chat2.send("我叫什么?")
    assert r.output == "记得:你叫小明"

    # 历史跨"请求"延续:两轮 user 消息都在
    users = [m.content for m in chat2.session.messages if m.role == "user"]
    assert users == ["我叫小明", "我叫什么?"]

"""并发安全压测 —— 证明「Agent 无状态蓝图 + Session 隔离」的生产级声称。

同一个 Agent 实例被多请求并发共享(全局单例),每个请求各自独立 Session。
只要 Agent 不持运行态、Session 不互相共享,并发就不会串台。

用无状态的 _EchoProvider(回显输入)做被测对象:不烧 token、不联网、完全确定性,
而且它自己无状态 → 把"是否串台"的责任精确地落在 Agent/Session 上。
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from rein import Agent
from rein.ir import Completion, Message, Usage


class _EchoProvider:
    """无状态测试 provider:回显最后一条 user 消息。无任何内部可变状态 → 可安全并发共享。"""

    async def complete(self, messages, tools=None, **kw):
        user = ""
        for m in reversed(messages):
            if m.role == "user":
                user = m.content or ""
                break
        return Completion(
            message=Message(role="assistant", content=f"echo:{user}"),
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
        )


def test_协程并发100会话不串台():
    """同一个 Agent 并发跑 100 个不同输入,每个输出必须对应自己的输入。"""
    agent = Agent(provider=_EchoProvider())  # 无状态蓝图,全局共享

    async def main():
        return await asyncio.gather(*[agent.arun(f"req-{i}") for i in range(100)])

    results = asyncio.run(main())
    for i, r in enumerate(results):
        assert r.output == f"echo:req-{i}", f"第 {i} 个请求串台了:{r.output!r}"


def test_并发会话状态相互独立():
    """每个并发 run 的 Session 是独立的:messages 只含自己的输入,不混入别人的。"""
    agent = Agent(provider=_EchoProvider())

    async def main():
        return await asyncio.gather(*[agent.arun(f"q{i}") for i in range(50)])

    results = asyncio.run(main())
    for i, r in enumerate(results):
        users = [m.content for m in r.session.messages if m.role == "user"]
        assert users == [f"q{i}"], f"第 {i} 个 session 混入了别的消息:{users}"


def test_多线程并发run不串台():
    """多线程(ThreadPoolExecutor)各自跑同步 agent.run:线程间也不串台。

    agent.run 内部各自 asyncio.run(独立事件循环)+ 独立 Session + 无状态 provider,
    所以线程并发同样安全。
    """
    agent = Agent(provider=_EchoProvider())

    def one(i: int) -> str:
        return agent.run(f"t{i}").output

    with ThreadPoolExecutor(max_workers=8) as ex:
        outs = list(ex.map(one, range(50)))

    for i, o in enumerate(outs):
        assert o == f"echo:t{i}", f"第 {i} 个线程串台了:{o!r}"

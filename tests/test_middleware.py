"""中间件测试(M4 批1):洋葱顺序 / 短路 / 改 messages / 覆盖多轮 / 无中间件等价裸 loop。"""

from rein.agent import Agent
from rein.ir import Completion, Message, ToolCall, Usage
from rein.providers import MockProvider


class _CaptureProvider:
    """记录每次 complete 收到的 messages,固定回一句文本。"""

    def __init__(self):
        self.seen_messages = None
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.seen_messages = list(messages)
        return Completion(
            message=Message(role="assistant", content="ok"),
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def stream(self, *a, **k):
        if False:  # 不被测试用到,只是让它成为合法 async generator
            yield


def _tool_agent(script):
    agent = Agent(provider=MockProvider(script))

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    return agent


def test_洋葱顺序():
    order: list = []
    agent = Agent(provider=MockProvider(["答案"]))

    @agent.middleware
    async def outer(ctx, call_next):
        order.append("outer-before")
        ctx = await call_next(ctx)
        order.append("outer-after")
        return ctx

    @agent.middleware
    async def inner(ctx, call_next):
        order.append("inner-before")
        ctx = await call_next(ctx)
        order.append("inner-after")
        return ctx

    agent.run("hi")  # 纯文本 → 一步结束,中间件过一次
    assert order == ["outer-before", "inner-before", "inner-after", "outer-after"]


def test_中间件可改messages():
    provider = _CaptureProvider()
    agent = Agent(provider=provider)

    @agent.middleware
    async def inject(ctx, call_next):
        ctx.session.messages.insert(0, Message(role="system", content="注入的系统提示"))
        return await call_next(ctx)

    agent.run("问题")
    assert provider.seen_messages[0].content == "注入的系统提示"


def test_短路_不调callnext跳过step():
    provider = _CaptureProvider()
    agent = Agent(provider=provider)

    @agent.middleware
    async def guard(ctx, call_next):
        # 短路:不调 call_next,直接把会话结束 → step 不执行,模型不被调用
        ctx.session.done = True
        ctx.session.stop_reason = "blocked"
        return ctx

    r = agent.run("敏感内容")
    assert provider.calls == 0  # 模型从未被调用(被短路)
    assert r.session.done is True
    assert r.stop_reason == "blocked"


def test_中间件覆盖每一步():
    count = {"n": 0}
    agent = _tool_agent([[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "等于 3"])

    @agent.middleware
    async def counter(ctx, call_next):
        count["n"] += 1
        return await call_next(ctx)

    r = agent.run("加")
    assert r.output == "等于 3"
    assert count["n"] == 3  # model → tool → model,每步都过中间件


def test_after阶段可读ctx_steps():
    kinds: list = []
    agent = _tool_agent([[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "好"])

    @agent.middleware
    async def observe(ctx, call_next):
        ctx = await call_next(ctx)
        kinds.extend(s.kind for s in ctx.steps)
        return ctx

    agent.run("加")
    assert "model" in kinds and "tool" in kinds


def test_无中间件等价裸loop():
    agent = _tool_agent([[ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})], "等于 5"])
    r = agent.run("加")
    assert r.output == "等于 5"
    assert [s.kind for s in r.steps] == ["model", "tool", "model"]


def test_中间件统一try_except():
    """把 call_next 包在 try 里即可统一兜错(此处验证机制可用,不强造异常)。"""
    caught = {"err": None}
    agent = Agent(provider=MockProvider(["答案"]))

    @agent.middleware
    async def safety(ctx, call_next):
        try:
            return await call_next(ctx)
        except Exception as e:  # pragma: no cover
            caught["err"] = e
            ctx.session.done = True
            return ctx

    r = agent.run("hi")
    assert r.output == "答案"
    assert caught["err"] is None

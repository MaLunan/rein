"""钩子 / 事件 / 权限重构测试(M4 批2)。

- 钩子(before/after model/tool)= 中间件语法糖,before 可短路。
- 事件(agent.on)= 只读观测。
- 权限重构:ask 已从 loop.step 移到内置 permission_middleware,行为不变。
"""

from rein.agent import Agent
from rein.config import LoopConfig
from rein.ir import Completion, Message, ToolCall, Usage
from rein.providers import MockProvider
from rein.session import Stage


def _tool_agent(script, *, log=None, **kw):
    agent = Agent(provider=MockProvider(script), **kw)

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        if log is not None:
            log.append((a, b))
        return a + b

    return agent


# ---------- 钩子 ----------


def test_before_tool短路阻止执行():
    log: list = []
    agent = _tool_agent(
        [[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "我换个方式"], log=log
    )

    @agent.before_tool
    async def block(ctx):
        # 短路:不执行工具,自己回填并推进
        for call in ctx.session.pending_tool_calls:
            ctx.session.messages.append(
                Message(role="tool", tool_call_id=call.id, content="被钩子拦截")
            )
        ctx.session.pending_tool_calls = []
        ctx.session.stage = Stage.CALL_MODEL
        return False  # 短路

    r = agent.run("加")
    assert log == []  # 工具没有执行
    assert r.output == "我换个方式"


def test_before_model短路立即结束():
    provider_calls: list = []

    class P:
        async def complete(self, messages, tools=None, **kwargs):
            provider_calls.append(1)
            return Completion(
                message=Message(role="assistant", content="x"),
                finish_reason="stop",
                usage=Usage(),
            )

        async def stream(self, *a, **k):
            if False:
                yield

    agent = Agent(provider=P())

    @agent.before_model
    async def gate(ctx):
        ctx.session.done = True
        ctx.session.stop_reason = "halted"
        return False

    r = agent.run("hi")
    assert provider_calls == []  # 模型从未被调
    assert r.stop_reason == "halted"


def test_after_model观测steps():
    seen: list = []
    agent = Agent(provider=MockProvider(["答案"]))

    @agent.after_model
    async def obs(ctx):
        seen.append([s.kind for s in ctx.steps])

    agent.run("hi")
    assert seen == [["model"]]


def test_after_tool只在工具步触发():
    fired: list = []
    agent = _tool_agent([[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "完成"])

    @agent.after_tool
    async def obs(ctx):
        fired.append([s.kind for s in ctx.steps])

    agent.run("加")
    assert fired == [["tool"]]  # 只在 RUN_TOOLS 那步触发


# ---------- 事件(只读) ----------


def test_on_step事件每步触发():
    events: list = []
    agent = _tool_agent([[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "完成"])
    agent.on("step", lambda ctx: events.append(ctx.stage.value))

    agent.run("加")
    assert events == ["call_model", "run_tools", "call_model"]


def test_on_tool事件只工具步():
    n: list = []
    agent = _tool_agent([[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "完成"])
    agent.on("tool", lambda ctx: n.append(1))

    agent.run("加")
    assert n == [1]


# ---------- 权限重构后的回归 ----------


def test_权限ask重构后仍中断并可resume():
    log: list = []
    agent = _tool_agent(
        [[ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})], "等于 5"],
        log=log,
        config=LoopConfig(permission="ask"),
    )
    r = agent.run("加")
    assert r.status == "interrupted"
    assert r.interrupt.type == "need_approval"
    assert log == []  # 未批准前工具不执行

    r2 = agent.resume(r.session, approve=True)
    assert r2.status == "done"
    assert r2.output == "等于 5"
    assert log == [(2, 3)]

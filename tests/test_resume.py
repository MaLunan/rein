"""恢复机制测试(M2 批1):ask 工具边界中断 → 序列化 → resume(批准/拒绝/多轮)。

全部用 MockProvider,不联网。验证「恢复=喂回 session,不依赖进程内挂起状态」。
"""

from rein.agent import Agent
from rein.config import LoopConfig
from rein.ir import ToolCall
from rein.session import Session, Stage


def _agent(script: list, log: list | None = None) -> Agent:
    """造一个 permission='ask' 的 agent,带一个会记录调用的 add 工具。"""
    from rein.providers import MockProvider

    agent = Agent(provider=MockProvider(script), config=LoopConfig(permission="ask"))

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        if log is not None:
            log.append((a, b))
        return a + b

    return agent


def test_ask在工具边界中断():
    agent = _agent([[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "和是 3"])
    r = agent.run("加一下")

    assert r.status == "interrupted"
    assert r.stop_reason == "interrupted"
    assert r.interrupt is not None
    assert r.interrupt.type == "need_approval"
    assert r.interrupt.tool_call.name == "add"
    # 工具尚未执行,会话停在 RUN_TOOLS、还没结束
    assert r.session.stage == Stage.RUN_TOOLS
    assert r.session.done is False
    assert r.session.pending_tool_calls[0].name == "add"


def test_中断态可序列化往返():
    agent = _agent([[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "ok"])
    r = agent.run("加")

    raw = r.session.model_dump_json()
    back = Session.model_validate_json(raw)
    assert back.stage == Stage.RUN_TOOLS
    assert back.pending_tool_calls[0].arguments == {"a": 1, "b": 2}
    assert back.done is False
    # RunResult 整体也能往返(含 interrupt)
    from rein.result import RunResult

    r_back = RunResult.model_validate_json(r.model_dump_json())
    assert r_back.status == "interrupted"
    assert r_back.interrupt.type == "need_approval"


def test_approve批准后续跑到done():
    log: list = []
    agent = _agent([[ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})], "和是 5"], log)
    r = agent.run("加")
    assert r.status == "interrupted"

    r2 = agent.resume(r.session, approve=True)
    assert r2.status == "done"
    assert r2.output == "和是 5"
    assert log == [(2, 3)]  # 工具确实被执行了
    assert any(m.role == "tool" and m.content == "5" for m in r2.session.messages)


def test_reject拒绝后模型自愈():
    log: list = []
    agent = _agent(
        [[ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})], "好的,我换个方式"], log
    )
    r = agent.run("加")
    r2 = agent.resume(r.session, approve=False)

    assert r2.status == "done"
    assert r2.output == "好的,我换个方式"
    assert log == []  # 工具没有被执行
    assert any(m.role == "tool" and "拒绝" in m.content for m in r2.session.messages)


def test_多轮审批():
    log: list = []
    script = [
        [ToolCall(id="1", name="add", arguments={"a": 1, "b": 1})],
        [ToolCall(id="2", name="add", arguments={"a": 2, "b": 2})],
        "两次都算完了",
    ]
    agent = _agent(script, log)

    r = agent.run("连算两次")
    assert r.status == "interrupted"  # 第一次工具待批

    r = agent.resume(r.session, approve=True)
    assert r.status == "interrupted"  # 第二次工具又待批

    r = agent.resume(r.session, approve=True)
    assert r.status == "done"
    assert r.output == "两次都算完了"
    assert log == [(1, 1), (2, 2)]


def test_存盘读盘后resume等价():
    """端到端往返:run→中断→存盘→读盘→resume→done,证明不依赖进程内状态。"""
    log: list = []
    agent = _agent([[ToolCall(id="1", name="add", arguments={"a": 4, "b": 5})], "等于 9"], log)
    r = agent.run("加")

    # 模拟存盘/读盘(另一处恢复)
    loaded = Session.model_validate_json(r.session.model_dump_json())
    r2 = agent.resume(loaded, approve=True)

    assert r2.status == "done"
    assert r2.output == "等于 9"
    assert log == [(4, 5)]


# ---------- error 中断(可重试错误转中断;致命错误直接抛)----------


class _RateLimit(Exception):
    status_code = 429


class _AuthError(Exception):
    status_code = 401


class _FlakyProvider:
    """前 fail_times 次调用抛指定异常,之后返回一句文本。"""

    def __init__(self, fail_times: int, then_text: str, exc: Exception):
        self.fail_times = fail_times
        self.then_text = then_text
        self.exc = exc
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        from rein.ir import Completion, Message, Usage

        return Completion(
            message=Message(role="assistant", content=self.then_text),
            finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def stream(self, *a, **k):
        raise self.exc
        yield  # pragma: no cover


def test_可重试错误转error中断_可resume重试():
    p = _FlakyProvider(fail_times=1, then_text="终于成功", exc=_RateLimit())
    agent = Agent(provider=p)

    r = agent.run("hi")
    assert r.status == "interrupted"
    assert r.interrupt.type == "error"
    assert r.session.stage == Stage.CALL_MODEL  # 停在问模型,可重试

    r2 = agent.resume(r.session)  # 重跑 complete(这次成功)
    assert r2.status == "done"
    assert r2.output == "终于成功"
    assert p.calls == 2


def test_致命错误直接抛():
    import pytest

    p = _FlakyProvider(fail_times=99, then_text="x", exc=_AuthError())
    agent = Agent(provider=p)
    with pytest.raises(_AuthError):
        agent.run("hi")


# ---------- run_interactive:CLI 审批循环(注入 approver 以可测) ----------


def test_run_interactive_全批准跑到done():
    log: list = []
    agent = _agent([[ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})], "和是 5"], log)
    r = agent.run_interactive("加", approver=lambda itr: True)
    assert r.status == "done"
    assert r.output == "和是 5"
    assert log == [(2, 3)]


def test_run_interactive_拒绝则不执行():
    log: list = []
    agent = _agent([[ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})], "那我不调了"], log)
    r = agent.run_interactive("加", approver=lambda itr: False)
    assert r.status == "done"
    assert r.output == "那我不调了"
    assert log == []


def test_run_interactive_多轮逐个审批():
    log: list = []
    script = [
        [ToolCall(id="1", name="add", arguments={"a": 1, "b": 1})],
        [ToolCall(id="2", name="add", arguments={"a": 2, "b": 2})],
        "都算完了",
    ]
    agent = _agent(script, log)
    # approver 每次都批准
    r = agent.run_interactive("连算两次", approver=lambda itr: True)
    assert r.status == "done"
    assert r.output == "都算完了"
    assert log == [(1, 1), (2, 2)]


# ---------- need_input:用 aresume(answer=) 注入补充信息继续 ----------


def test_aresume_注入answer继续():
    import asyncio

    from rein.providers import MockProvider

    agent = Agent(provider=MockProvider(["第一轮回答", "收到补充,继续"]))
    r1 = agent.run("第一个问题")
    assert r1.status == "done"

    # 把人补充的信息作为新 user 消息注入,继续问模型(need_input 的恢复路径)
    r2 = asyncio.run(agent.aresume(r1.session, answer="这是补充信息"))
    assert r2.status == "done"
    assert r2.output == "收到补充,继续"
    assert any(m.role == "user" and m.content == "这是补充信息" for m in r2.session.messages)

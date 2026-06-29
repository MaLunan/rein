"""loop 测试:多轮编排 / 工具异常自愈 / 直接结束 / 熔断触发 / RunResult 序列化。

全部用 MockProvider 驱动,不联网、不烧 token、结果可复现。
"""

import asyncio

from rein.config import LoopConfig
from rein.ir import Message, ToolCall
from rein.loop import arun, run
from rein.providers import MockProvider
from rein.result import RunResult
from rein.session import Session, Stage
from rein.tools import Tool, ToolRegistry


def _registry() -> ToolRegistry:
    """含一个正常工具 add 和一个会抛错的工具 boom。"""

    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    def boom() -> str:
        "会抛异常"
        raise ValueError("炸了")

    reg = ToolRegistry()
    reg.add(Tool.from_function(add))
    reg.add(Tool.from_function(boom))
    return reg


def _session(prompt: str = "hi") -> Session:
    return Session(messages=[Message(role="user", content=prompt)])


def test_纯文本直接结束():
    """模型直接给文本 → 一步结束,output 是该文本。"""
    p = MockProvider(["你好"])
    r = run(_session(), p, ToolRegistry())
    assert r.status == "done"
    assert r.stop_reason == "done"
    assert r.output == "你好"
    assert str(r) == "你好"  # __str__ 直接给答案
    assert len(r.steps) == 1 and r.steps[0].kind == "model"


def test_多轮_模型到工具到回填到再答():
    """模型先调 add,工具回填后模型再给最终答案。"""
    script = [
        [ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})],
        "结果是 3",
    ]
    r = run(_session(), MockProvider(script), _registry())
    assert r.status == "done"
    assert r.output == "结果是 3"
    # 流水账:model(请求工具) → tool(add) → model(最终答案)
    kinds = [s.kind for s in r.steps]
    assert kinds == ["model", "tool", "model"]
    assert r.steps[1].is_error is False
    # 工具结果已回填进历史(role=tool 的消息存在,内容含 "3")
    tool_msgs = [m for m in r.session.messages if m.role == "tool"]
    assert tool_msgs and tool_msgs[0].content == "3"
    # 会话正常收尾
    assert r.session.done is True and r.session.stage == Stage.DONE


def test_工具异常自愈_不中断loop():
    """工具抛异常被封装成错误结果回填,loop 继续,模型给出最终答案。"""
    script = [
        [ToolCall(id="1", name="boom", arguments={})],
        "我已处理错误",
    ]
    r = run(_session(), MockProvider(script), _registry())
    assert r.status == "done"
    assert r.output == "我已处理错误"
    tool_step = [s for s in r.steps if s.kind == "tool"][0]
    assert tool_step.is_error is True
    assert "炸了" in [m for m in r.session.messages if m.role == "tool"][0].content


def test_熔断_轮数闸触发():
    """max_iterations=2,模型一直要调工具 → 第 2 次模型调用后触顶停止。"""
    script = [[ToolCall(id=str(i), name="add", arguments={"a": 1, "b": 1})] for i in range(10)]
    cfg = LoopConfig(max_iterations=2)
    r = run(_session(), MockProvider(script), _registry(), cfg)
    assert r.stop_reason == "max_iterations"
    assert r.session.done is True
    # 模型只被调了 2 次
    assert r.session.iteration == 2


def test_熔断_重复闸触发():
    """模型连续请求完全相同的工具调用 → detect_loops 在阈值处停止。"""
    same = [ToolCall(id="x", name="add", arguments={"a": 1, "b": 1})]
    script = [list(same) for _ in range(10)]  # 每轮都一样
    cfg = LoopConfig(detect_loops=True, repeat_threshold=3, max_iterations=50)
    r = run(_session(), MockProvider(script), _registry(), cfg)
    assert r.stop_reason == "loop_detected"


def test_usage累计():
    """每次模型调用都累加 token(MockProvider 默认每次 input=1/output=1)。"""
    script = [
        [ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})],
        "好了",
    ]
    r = run(_session(), MockProvider(script), _registry())
    # 两次模型调用 → 输入 2 + 输出 2
    assert r.usage.input_tokens == 2
    assert r.usage.output_tokens == 2


def test_RunResult_序列化往返():
    """RunResult(含内嵌 Session)能 dump 成 JSON 再还原,字段一致。"""
    r = run(_session(), MockProvider(["答案"]), ToolRegistry())
    raw = r.model_dump_json()
    back = RunResult.model_validate_json(raw)
    assert back.output == "答案"
    assert back.stop_reason == "done"
    assert back.session.done is True
    assert len(back.steps) == len(r.steps)


def test_arun_异步入口也能跑():
    """直接用异步内核 arun 跑,结果与同步 run 一致。"""
    comp = asyncio.run(arun(_session(), MockProvider(["异步答案"]), ToolRegistry()))
    assert comp.output == "异步答案"

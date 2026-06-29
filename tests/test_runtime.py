"""runtime 测试:LocalRuntime 执行(同步/异步)、异常封装、未找到、权限、并发保序。"""

import asyncio

from rein.ir import ToolCall
from rein.runtime import LocalRuntime
from rein.tools import Tool, ToolRegistry


def _registry() -> ToolRegistry:
    """构造一个含 同步 / 异步 / 会抛错 三种工具的登记册。"""

    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    async def aecho(s: str) -> str:
        "异步回显"
        return s

    def boom() -> str:
        "会抛异常"
        raise ValueError("炸了")

    reg = ToolRegistry()
    reg.add(Tool.from_function(add))
    reg.add(Tool.from_function(aecho))
    reg.add(Tool.from_function(boom))
    return reg


def test_执行同步工具():
    rt, reg = LocalRuntime(), _registry()
    r = asyncio.run(rt.execute(ToolCall(id="1", name="add", arguments={"a": 2, "b": 3}), reg))
    assert r.is_error is False
    assert r.content == "5"


def test_执行异步工具():
    rt, reg = LocalRuntime(), _registry()
    r = asyncio.run(rt.execute(ToolCall(id="1", name="aecho", arguments={"s": "hi"}), reg))
    assert r.content == "hi"


def test_工具异常被封装为错误结果_不抛出():
    rt, reg = LocalRuntime(), _registry()
    r = asyncio.run(rt.execute(ToolCall(id="1", name="boom", arguments={}), reg))
    assert r.is_error is True
    assert "炸了" in r.content


def test_未找到工具():
    rt, reg = LocalRuntime(), _registry()
    r = asyncio.run(rt.execute(ToolCall(id="1", name="nope", arguments={}), reg))
    assert r.is_error is True


def test_权限deny直接拒绝():
    rt, reg = LocalRuntime(), _registry()
    r = asyncio.run(
        rt.execute(ToolCall(id="1", name="add", arguments={"a": 1, "b": 1}), reg, permission="deny")
    )
    assert r.is_error is True
    assert "拒绝" in r.content


def test_并发执行多工具且结果保序():
    rt, reg = LocalRuntime(), _registry()
    calls = [
        ToolCall(id="1", name="add", arguments={"a": 1, "b": 1}),
        ToolCall(id="2", name="add", arguments={"a": 10, "b": 10}),
    ]
    rs = asyncio.run(rt.execute_all(calls, reg))
    assert rs[0].content == "2" and rs[0].tool_call_id == "1"
    assert rs[1].content == "20" and rs[1].tool_call_id == "2"

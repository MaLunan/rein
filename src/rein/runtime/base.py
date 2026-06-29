"""Runtime 接口:工具执行层。

Runtime 负责「在哪、怎么」执行模型请求的工具调用 —— 把 ToolCall 变成 ToolResult。
M0 只有 LocalRuntime(本进程执行);DockerRuntime(容器沙箱)留到 M4。
"""

from typing import Protocol, runtime_checkable

from rein.ir import ToolCall, ToolResult
from rein.tools import ToolRegistry


@runtime_checkable
class Runtime(Protocol):
    """工具执行层的统一接口。"""

    async def execute(
        self,
        call: ToolCall,
        registry: ToolRegistry,
        permission: str = "allow",
    ) -> ToolResult:
        """执行单个工具调用,返回结果(出错也封装成 ToolResult,不抛异常)。"""
        ...

    async def execute_all(
        self,
        calls: list[ToolCall],
        registry: ToolRegistry,
        permission: str = "allow",
    ) -> list[ToolResult]:
        """并发执行多个工具调用,结果按入参顺序返回(保序)。"""
        ...

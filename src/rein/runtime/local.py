"""LocalRuntime —— 在本进程执行工具(M0 默认)。

职责:
- 按工具名从 registry 找到工具并调用。
- 权限:allow 直接执行;deny 返回错误结果;ask 抛 NotImplementedError(M2 再做)。
- 异常封装:工具内部异常【不抛到 loop】,封装成 ToolResult(is_error=True),让模型读到错误自愈。
- 同步工具丢线程池(不阻塞事件循环),异步工具直接 await。
- 返回值用 serialize_result 转成文本(对象不进 Session,文本才进)。
- execute_all 用 asyncio.gather 并发执行,结果保序。
"""

import asyncio

import anyio

from rein.ir import ToolCall, ToolResult
from rein.tools import ToolRegistry, serialize_result


class LocalRuntime:
    """在当前进程里执行工具调用。"""

    async def execute(
        self,
        call: ToolCall,
        registry: ToolRegistry,
        permission: str = "allow",
    ) -> ToolResult:
        # --- 权限闸 ---
        if permission == "deny":
            return ToolResult(
                tool_call_id=call.id,
                content=f"权限拒绝:工具 '{call.name}' 的调用被策略拒绝。",
                is_error=True,
            )
        if permission == "ask":
            # 需要"暂停 → 等人确认 → 恢复",属于 M2 的可恢复机制
            raise NotImplementedError("ask 权限(人工确认)将在 M2 实现")

        # --- 找工具 ---
        tool = registry.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"错误:未找到名为 '{call.name}' 的工具。",
                is_error=True,
            )

        # --- 执行(异常一律封装成错误结果,绝不抛到 loop) ---
        try:
            if tool.is_async:
                result = await tool.fn(**call.arguments)
            else:
                # 同步函数丢线程池,避免阻塞事件循环
                result = await anyio.to_thread.run_sync(lambda: tool.fn(**call.arguments))
            return ToolResult(
                tool_call_id=call.id,
                content=serialize_result(result),
                is_error=False,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                content=f"工具执行出错:{type(e).__name__}: {e}",
                is_error=True,
            )

    async def execute_all(
        self,
        calls: list[ToolCall],
        registry: ToolRegistry,
        permission: str = "allow",
    ) -> list[ToolResult]:
        """并发执行多个工具调用;gather 默认保序,返回顺序与 calls 一致。"""
        results = await asyncio.gather(*[self.execute(c, registry, permission) for c in calls])
        return list(results)

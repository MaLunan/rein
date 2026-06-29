"""Rein —— 一个极薄但生产级的单-agent harness 框架。

公开 API 一站式导出:5 行示例所需的 `Agent` / `@tool` / `run` 都在这里。

    from rein import Agent

    agent = Agent("anthropic/claude-opus-4-8")

    @agent.tool
    def now() -> str:
        "返回当前时间"
        return "2026-06-27"

    print(agent.run("现在几号?"))
"""

__version__ = "0.0.1"

from rein.a2a import A2AServer, serve_a2a
from rein.agent import Agent, Chat, tool
from rein.compaction import (
    CompactionStrategy,
    SlidingWindow,
    SummarizeCompaction,
    estimate_tokens,
)
from rein.config import LoopConfig
from rein.ir import (
    Completion,
    Message,
    StreamChunk,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from rein.loop import aresume, arun, astream, run, step
from rein.middleware import StepContext
from rein.otel import export_run
from rein.plugins import load_plugins, plugin_names
from rein.providers import FallbackProvider, LiteLLMProvider, MockProvider, Provider
from rein.result import Interrupt, RunResult, Step
from rein.runtime import DockerRuntime, LocalRuntime, Runtime
from rein.scaffold import available_templates, create_project
from rein.session import Session, Stage
from rein.store import FileSessionStore, MemorySessionStore, SessionStore
from rein.tools import Tool, ToolRegistry

__all__ = [
    "__version__",
    # 门面(最常用)
    "Agent",
    "Chat",
    "tool",
    # 统一 IR(内部通用数据类型)
    "Message",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "Completion",
    "StreamChunk",
    # 配置 / 状态 / 结果
    "LoopConfig",
    "Session",
    "Stage",
    "RunResult",
    "Step",
    "Interrupt",
    # 工具系统
    "Tool",
    "ToolRegistry",
    # 持久化(M2)
    "SessionStore",
    "MemorySessionStore",
    "FileSessionStore",
    # 上下文压缩(M3)
    "CompactionStrategy",
    "SlidingWindow",
    "SummarizeCompaction",
    "estimate_tokens",
    # 可观测导出(M3,OTel 走 extras)
    "export_run",
    # 扩展层(M4)
    "StepContext",
    "load_plugins",
    "plugin_names",
    # 脚手架(M5)
    "create_project",
    "available_templates",
    # A2A 服务端(把 agent 暴露给别的 agent 调用)
    "serve_a2a",
    "A2AServer",
    # 模型接入层
    "Provider",
    "MockProvider",
    "LiteLLMProvider",
    "FallbackProvider",
    # 工具执行层
    "Runtime",
    "LocalRuntime",
    "DockerRuntime",
    # Loop(进阶:自行驱动单步状态机)
    "step",
    "arun",
    "run",
    "astream",
    "aresume",
]

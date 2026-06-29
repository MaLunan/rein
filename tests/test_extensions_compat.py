"""扩展层兼容性回归(M4 批3,验收7):中间件 + 中断/恢复 共存;插件发现;DockerRuntime 接口。"""

import asyncio

from rein.agent import Agent
from rein.config import LoopConfig
from rein.ir import ToolCall
from rein.plugins import load_plugins, plugin_names
from rein.providers import MockProvider
from rein.runtime import DockerRuntime
from rein.tools import ToolRegistry

# ---------- 中间件 + 中断/恢复 共存(硬约束:栈每步重建、不跨步持状态)----------


def test_中间件在中断恢复后继续工作():
    seen: list = []
    log: list = []
    agent = Agent(
        provider=MockProvider(
            [[ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})], "等于 5"]
        ),
        config=LoopConfig(permission="ask"),
    )

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        log.append((a, b))
        return a + b

    @agent.middleware
    async def tracker(ctx, call_next):
        seen.append(ctx.stage.value)
        return await call_next(ctx)

    r = agent.run("加")
    assert r.status == "interrupted"
    before = list(seen)
    assert before == ["call_model", "run_tools"]  # 中断前走过这两步

    r2 = agent.resume(r.session, approve=True)
    assert r2.status == "done"
    assert r2.output == "等于 5"
    assert log == [(2, 3)]
    # resume 后中间件栈按需重建、继续记录(恢复后的 call_model)
    assert len(seen) > len(before)
    assert seen[-1] == "call_model"


# ---------- 插件发现 ----------


def test_load_plugins_无插件不报错():
    # 用一个不存在的 group → 空结果,不抛
    assert load_plugins("rein.plugins.__nonexistent_test__") == {}


def test_plugin_names_可调用():
    assert plugin_names("rein.plugins.__nonexistent_test__") == []


# ---------- DockerRuntime 接口(不进容器的分支,无需 docker)----------


def test_docker_runtime_deny不进容器():
    rt = DockerRuntime()
    r = asyncio.run(
        rt.execute(ToolCall(id="1", name="add", arguments={}), ToolRegistry(), permission="deny")
    )
    assert r.is_error is True
    assert "拒绝" in r.content


def test_docker_runtime_未找到工具():
    rt = DockerRuntime()
    r = asyncio.run(rt.execute(ToolCall(id="1", name="nope", arguments={}), ToolRegistry()))
    assert r.is_error is True
    assert "未找到" in r.content

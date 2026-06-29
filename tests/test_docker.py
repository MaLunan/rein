"""DockerRuntime 容器冒烟(M4)——默认 skip:需 docker SDK + 可用守护进程。

开启:`pip install 'rein[docker]'` 且本机 docker 正常,然后 `pytest tests/test_docker.py`。
"""

import asyncio

import pytest

docker = pytest.importorskip("docker")

# 没有可用的 docker 守护进程也跳过(只装 SDK 不够)
try:
    docker.from_env().ping()
except Exception:  # pragma: no cover
    pytest.skip("docker 守护进程不可用", allow_module_level=True)

from rein.ir import ToolCall  # noqa: E402
from rein.runtime import DockerRuntime  # noqa: E402
from rein.tools import Tool, ToolRegistry  # noqa: E402


def _registry() -> ToolRegistry:
    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    reg = ToolRegistry()
    reg.add(Tool.from_function(add))
    return reg


def test_docker_执行纯函数工具():
    rt = DockerRuntime()
    r = asyncio.run(
        rt.execute(ToolCall(id="1", name="add", arguments={"a": 2, "b": 3}), _registry())
    )
    assert r.is_error is False
    assert r.content.strip() == "5"

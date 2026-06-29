"""DockerRuntime —— 在容器沙箱里执行工具(M4,可选 extras)。

定位:沙箱是为「跑 LLM 生成的代码」准备的。它实现与 LocalRuntime 一致的 Runtime
接口,所以 `runtime=DockerRuntime()` 即可替换 LocalRuntime(改配置即切换)。

务实范围(请知悉):工具是宿主进程里的 Python 函数,没法把函数对象塞进容器调用。
本实现用 `inspect.getsource` 取工具函数源码,在容器里 `python -c "源码 + 调用"` 执行,
捕获 stdout 当结果。因此:
- 适合【纯函数 + 标准库】工具;
- 依赖闭包 / 外部状态 / 第三方库的工具需要自定义镜像,否则会得到清晰报错;
- 这不是「万能容器执行」,而是沙箱骨架 + 最常见用例。

安全默认(保守):网络隔离开、内存上限、自动删除容器。`docker` 走延迟 import。
"""

import asyncio
import inspect
import json
import textwrap

import anyio

from rein.ir import ToolCall, ToolResult
from rein.tools import ToolRegistry


def _build_script(source: str, fn_name: str, arguments: dict) -> str:
    """生成在容器里执行的脚本:定义函数 + 用 JSON 参数调用 + 打印结果。"""
    # 去掉源码开头的单行装饰器(如 @agent.tool —— 容器里没有 agent)
    lines = [ln for ln in source.splitlines() if not ln.lstrip().startswith("@")]
    clean_src = "\n".join(lines)
    args_json = json.dumps(arguments, ensure_ascii=False)
    return (
        f"{clean_src}\n"
        "import json as _json\n"
        f"_args = _json.loads({args_json!r})\n"
        f"_result = {fn_name}(**_args)\n"
        "print(_result)\n"
    )


class DockerRuntime:
    """在容器内执行工具的 Runtime(接口对齐 LocalRuntime)。"""

    def __init__(
        self,
        image: str = "python:3.12-slim",
        *,
        network_disabled: bool = True,
        mem_limit: str = "256m",
        **container_kwargs,
    ):
        """
        Args:
            image:            执行用的镜像。
            network_disabled: 是否切断容器网络(默认 True,保守)。
            mem_limit:        内存上限(默认 256m)。
            **container_kwargs: 透传给 docker 的其它容器参数。
        """
        self.image = image
        self.network_disabled = network_disabled
        self.mem_limit = mem_limit
        self.container_kwargs = container_kwargs

    async def execute(
        self,
        call: ToolCall,
        registry: ToolRegistry,
        permission: str = "allow",
    ) -> ToolResult:
        # 权限:deny 在此短路;ask 由 loop 的 permission_middleware 处理,不应直达 runtime
        if permission == "deny":
            return ToolResult(
                tool_call_id=call.id,
                content=f"权限拒绝:工具 '{call.name}' 的调用被策略拒绝。",
                is_error=True,
            )
        if permission == "ask":
            raise NotImplementedError("ask 由 loop 的 permission_middleware 处理")

        tool = registry.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"错误:未找到名为 '{call.name}' 的工具。",
                is_error=True,
            )

        try:
            # 容器执行是阻塞 IO,丢线程池避免阻塞事件循环
            output = await anyio.to_thread.run_sync(
                lambda: self._run_in_container(tool, call.arguments)
            )
            return ToolResult(tool_call_id=call.id, content=output, is_error=False)
        except Exception as e:
            # 沙箱内/外的异常都封装成错误结果,不抛到 loop(与 LocalRuntime 一致)
            return ToolResult(
                tool_call_id=call.id,
                content=f"沙箱执行出错:{type(e).__name__}: {e}",
                is_error=True,
            )

    def _run_in_container(self, tool, arguments: dict) -> str:
        try:
            import docker
        except ImportError as e:
            raise ImportError(
                "DockerRuntime 需要:pip install 'rein[docker]',且本机有可用的 docker 守护进程"
            ) from e

        try:
            source = textwrap.dedent(inspect.getsource(tool.fn))
        except (OSError, TypeError) as e:
            raise RuntimeError(
                f"无法取得工具 '{tool.name}' 的源码(闭包 / 内置 / 动态生成的函数不支持沙箱执行)"
            ) from e

        script = _build_script(source, tool.fn.__name__, arguments)
        client = docker.from_env()
        raw = client.containers.run(
            self.image,
            ["python", "-c", script],
            network_disabled=self.network_disabled,
            mem_limit=self.mem_limit,
            remove=True,
            stdout=True,
            stderr=True,
            **self.container_kwargs,
        )
        return raw.decode("utf-8").strip() if isinstance(raw, bytes) else str(raw).strip()

    async def execute_all(
        self,
        calls: list[ToolCall],
        registry: ToolRegistry,
        permission: str = "allow",
    ) -> list[ToolResult]:
        """并发执行多个工具调用,结果保序(与 LocalRuntime 一致)。"""
        results = await asyncio.gather(*[self.execute(c, registry, permission) for c in calls])
        return list(results)

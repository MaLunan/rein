"""Runtime 子系统:工具执行层。

- Runtime:统一接口(协议)
- LocalRuntime:本进程执行(M0 默认)
- DockerRuntime:容器沙箱执行(M4,docker 走 extras)
"""

from rein.runtime.base import Runtime
from rein.runtime.docker import DockerRuntime
from rein.runtime.local import LocalRuntime

__all__ = ["Runtime", "LocalRuntime", "DockerRuntime"]

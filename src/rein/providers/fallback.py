"""FallbackProvider —— 主/备模型自动切换(M1)。

它本身就是一个 Provider(实现 complete + stream),内部持有一串 provider:
    [主, 备1, 备2, ...]
按顺序尝试,主模型限流/超时/5xx 时自动切到下一个,对 Loop 完全透明。

设计纪律(对应 M1 注意点 3):
- 只对【可重试错误】(限流 429 / 超时 / 5xx / 连接错误)切换;
  鉴权(401/403)、参数错误(400/422)等【直接抛】,切了也没用。
- 每个 provider 可带少量重试 + 指数退避(base_delay * 2**attempt)。
- 不 import litellm:可重试与否纯按「HTTP 状态码 + 异常类名」判定,
  这样核心层零厂商依赖,新厂商的异常只要带 status_code 或常见命名就能识别。
- 流式 fallback 的红线:一旦已经吐出过 chunk,就【不能再切换】(流出去的收不回),
  中途失败直接抛;只有在产出首个 chunk 之前失败才允许 fallback。
"""

import asyncio
from collections.abc import AsyncIterator

from rein.ir import Completion, Message, StreamChunk, ToolSpec
from rein.providers.base import Provider

# 可重试的 HTTP 状态码(限流 / 超时 / 服务端错误)
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
# 明确【不可】重试的状态码(鉴权 / 参数 / 找不到)
_FATAL_STATUS = {400, 401, 403, 404, 405, 422}
# 异常类名里出现这些关键字 → 视为可重试(各厂商命名习惯)
_RETRYABLE_NAMES = (
    "ratelimit",
    "timeout",
    "serviceunavailable",
    "apiconnection",
    "overloaded",
    "internalserver",
    "tryagain",
)
# 异常类名里出现这些关键字 → 明确不可重试
_FATAL_NAMES = (
    "authentication",
    "permission",
    "notfound",
    "badrequest",
    "invalidrequest",
    "unprocessable",
)


def _status_of(e: Exception) -> int | None:
    """尽力从异常里抠出 HTTP 状态码(不同 SDK 字段名不同)。"""
    for attr in ("status_code", "code", "http_status", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    return None


def is_retryable(e: Exception) -> bool:
    """判断一个异常是否「值得换一家/再试一次」。

    优先看状态码(最可靠);否则看类名关键字;都识别不了 → 默认【不重试】(保守:
    未知错误盲目重试只会放大故障,fallback 只针对有明确信号的限流/服务不可用)。
    """
    status = _status_of(e)
    if status is not None:
        if status in _RETRYABLE_STATUS:
            return True
        if status in _FATAL_STATUS:
            return False
    name = type(e).__name__.lower()
    if any(k in name for k in _FATAL_NAMES):
        return False
    if any(k in name for k in _RETRYABLE_NAMES):
        return True
    return False


class FallbackProvider:
    """按 [主, 备...] 顺序尝试,可重试错误时自动切换的 Provider。"""

    def __init__(
        self,
        providers: list[Provider],
        *,
        max_retries_per_provider: int = 1,
        base_delay: float = 0.5,
    ):
        """
        Args:
            providers:                非空的 provider 列表,第 0 个为主,其余为备。
            max_retries_per_provider: 每个 provider 在切换前的额外重试次数(默认 1)。
            base_delay:               指数退避基准秒数(delay = base_delay * 2**attempt);
                                      设 0 可关闭等待(测试用)。
        """
        if not providers:
            raise ValueError("FallbackProvider 至少需要一个 provider。")
        self.providers = providers
        self.max_retries_per_provider = max_retries_per_provider
        self.base_delay = base_delay

    async def _sleep(self, attempt: int) -> None:
        """指数退避;base_delay=0 时直接跳过(不浪费测试时间)。"""
        if self.base_delay > 0:
            await asyncio.sleep(self.base_delay * (2**attempt))

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> Completion:
        last_err: Exception | None = None
        for provider in self.providers:
            for attempt in range(self.max_retries_per_provider + 1):
                try:
                    return await provider.complete(messages, tools, **kwargs)
                except Exception as e:
                    if not is_retryable(e):
                        raise  # 鉴权/参数等致命错误:切了也没用,直接抛
                    last_err = e
                    # 同一 provider 还有重试余额 → 退避后再试;否则换下一个 provider
                    if attempt < self.max_retries_per_provider:
                        await self._sleep(attempt)
        # 所有 provider 都试过仍失败
        assert last_err is not None
        raise last_err

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        last_err: Exception | None = None
        for provider in self.providers:
            for attempt in range(self.max_retries_per_provider + 1):
                produced = False  # 本次尝试是否已经吐过 chunk
                try:
                    async for chunk in provider.stream(messages, tools, **kwargs):
                        produced = True
                        yield chunk
                    return  # 正常流完
                except Exception as e:
                    # 已经吐过内容 → 不能重试/切换(流出去的收不回),直接抛
                    if produced or not is_retryable(e):
                        raise
                    last_err = e
                    if attempt < self.max_retries_per_provider:
                        await self._sleep(attempt)
        assert last_err is not None
        raise last_err

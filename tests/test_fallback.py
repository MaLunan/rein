"""fallback 测试(M1):可重试错误自动切换 / 致命错误直接抛 / 重试退避 / 流式 fallback。

全部本地可跑,不联网(base_delay=0 关掉真实等待)。
"""

import asyncio

import pytest

from rein.agent import Agent
from rein.providers import FallbackProvider, MockProvider
from rein.providers.fallback import is_retryable

# ---------- 自定义异常:模拟不同类型的厂商错误 ----------


class RateLimitError(Exception):
    status_code = 429


class ServerError(Exception):
    status_code = 503


class AuthenticationError(Exception):
    status_code = 401


class WeirdError(Exception):
    """既无状态码、类名也不含已知关键字 → 默认不可重试。"""


class _FailProvider:
    """每次调用都抛指定异常的假 Provider(记录被调次数)。"""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        raise self.exc

    async def stream(self, messages, tools=None, **kwargs):
        self.calls += 1
        raise self.exc
        yield  # pragma: no cover  (让它成为 async generator)


# ---------- is_retryable 判定 ----------


def test_is_retryable_状态码():
    assert is_retryable(RateLimitError()) is True
    assert is_retryable(ServerError()) is True
    assert is_retryable(AuthenticationError()) is False


def test_is_retryable_类名兜底():
    class ConnectionTimeout(Exception):
        pass

    class InvalidRequestError(Exception):
        pass

    assert is_retryable(ConnectionTimeout()) is True
    assert is_retryable(InvalidRequestError()) is False


def test_is_retryable_未知默认不重试():
    assert is_retryable(WeirdError()) is False


# ---------- complete fallback ----------


def test_fallback_限流时切到备用():
    flaky = _FailProvider(RateLimitError())
    backup = MockProvider(["备用答案"])
    fp = FallbackProvider([flaky, backup], max_retries_per_provider=0, base_delay=0)

    comp = asyncio.run(fp.complete([]))
    assert comp.message.content == "备用答案"
    assert flaky.calls == 1  # 主试过一次,失败后切备用


def test_fallback_鉴权错误直接抛不切换():
    flaky = _FailProvider(AuthenticationError())
    backup = MockProvider(["不该用到我"])
    fp = FallbackProvider([flaky, backup], max_retries_per_provider=0, base_delay=0)

    with pytest.raises(AuthenticationError):
        asyncio.run(fp.complete([]))
    assert backup._index == 0  # 备用根本没被调用


def test_fallback_同provider重试次数():
    flaky = _FailProvider(ServerError())
    backup = MockProvider(["最终备用"])
    # 每个 provider 额外重试 2 次 → 主被调 3 次,仍失败后切备用
    fp = FallbackProvider([flaky, backup], max_retries_per_provider=2, base_delay=0)

    comp = asyncio.run(fp.complete([]))
    assert comp.message.content == "最终备用"
    assert flaky.calls == 3


def test_fallback_全部失败抛最后错误():
    f1 = _FailProvider(RateLimitError())
    f2 = _FailProvider(ServerError())
    fp = FallbackProvider([f1, f2], max_retries_per_provider=0, base_delay=0)

    with pytest.raises(ServerError):
        asyncio.run(fp.complete([]))


# ---------- stream fallback ----------


def test_fallback_流式切到备用():
    flaky = _FailProvider(RateLimitError())
    backup = MockProvider(["流式备用"])
    fp = FallbackProvider([flaky, backup], max_retries_per_provider=0, base_delay=0)

    async def go():
        out = []
        async for c in fp.stream([]):
            if c.type == "text":
                out.append(c.text)
        return out

    texts = asyncio.run(go())
    assert "".join(texts) == "流式备用"


# ---------- Agent 集成 ----------


def test_agent_fallback_用Provider对象():
    """Agent(provider=主, fallback=[备]) 主限流时切备用。"""
    flaky = _FailProvider(RateLimitError())
    # 注意:provider 注入会跳过懒建;这里用 fallback 列表 + 一个无 model 的链
    agent = Agent(fallback=[flaky, MockProvider(["代答"])])
    # 关掉退避等待与同 provider 重试,聚焦验证「切换」本身(避免测试变慢)
    prov = agent._resolve_provider()
    prov.base_delay = 0
    prov.max_retries_per_provider = 0
    r = agent.run("hi")
    assert r.output == "代答"
    assert flaky.calls == 1  # 主只试 1 次就切到备用

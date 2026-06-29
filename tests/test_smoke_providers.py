"""真实厂商冒烟测试(M1)—— 默认 skip,需要真实 key + 联网才跑。

如何开启:
    pip install 'rein[litellm]'
    export ANTHROPIC_API_KEY=sk-...        # 或对应厂商的 key
    export REIN_SMOKE=1
    .venv/bin/python -m pytest tests/test_smoke_providers.py -v

设计:这些测试会真的花钱、需要网络,所以默认 skip(没设 REIN_SMOKE 就不跑),
绝不混进日常 `pytest`(那一套必须 0 成本、可离线、可复现)。
"""

import os

import pytest

# 没设开关就整文件跳过
pytestmark = pytest.mark.skipif(
    not os.getenv("REIN_SMOKE"),
    reason="真实厂商冒烟:需设 REIN_SMOKE=1 且配好对应厂商 key、装好 litellm 才跑",
)


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_smoke_anthropic_纯文本():
    """Anthropic:一句纯文本问答能跑通,返回非空。"""
    pytest.importorskip("litellm")
    from rein import Agent

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("缺 ANTHROPIC_API_KEY")

    agent = Agent("anthropic/claude-opus-4-8")
    r = agent.run("用一个词回答:你好吗?")
    assert r.output and len(r.output) > 0
    assert r.stop_reason == "done"


def test_smoke_openai兼容_工具调用():
    """OpenAI 兼容(默认 DeepSeek):带工具的多轮能跑通。"""
    pytest.importorskip("litellm")
    from rein import Agent

    model = os.getenv("REIN_SMOKE_OPENAI_MODEL", "deepseek/deepseek-chat")
    if not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        pytest.skip("缺 DEEPSEEK_API_KEY / OPENAI_API_KEY")

    agent = Agent(model)

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    r = agent.run("用 add 工具算 2 加 3,然后只回答数字。")
    assert r.output is not None
    # 真实成本应被 litellm 计算并填入(若厂商支持)
    assert r.usage.input_tokens > 0


def test_smoke_流式():
    """流式:能实时收到文本片段。"""
    pytest.importorskip("litellm")
    from rein import Agent

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("缺 ANTHROPIC_API_KEY")

    agent = Agent("anthropic/claude-opus-4-8")

    async def go():
        texts = []
        async for c in agent.astream("从 1 数到 5。"):
            if c.type == "text":
                texts.append(c.text)
        return texts

    texts = _run(go())
    assert len("".join(texts)) > 0

"""providers 测试:MockProvider 按预设脚本依次返回(文本 / 工具调用);真实成本提取。"""

import asyncio

from rein.ir import Message, ToolCall
from rein.providers import MockProvider
from rein.providers.litellm import LiteLLMProvider, _extract_cost


def test_mock_返回纯文本():
    p = MockProvider(["你好"])
    comp = asyncio.run(p.complete([Message(role="user", content="hi")]))
    assert comp.finish_reason == "stop"
    assert comp.message.content == "你好"


def test_mock_返回工具调用():
    calls = [ToolCall(id="1", name="search", arguments={"q": "天气"})]
    p = MockProvider([calls])
    comp = asyncio.run(p.complete([Message(role="user", content="hi")]))
    assert comp.finish_reason == "tool_calls"
    assert comp.message.tool_calls[0].name == "search"


def test_mock_按顺序消费():
    p = MockProvider([[ToolCall(id="1", name="f", arguments={})], "完成"])
    c1 = asyncio.run(p.complete([]))
    c2 = asyncio.run(p.complete([]))
    assert c1.finish_reason == "tool_calls"
    assert c2.finish_reason == "stop"
    assert c2.message.content == "完成"


def test_mock_脚本用尽不卡死():
    """脚本耗尽后应返回一个正常结束,避免 loop 无限等待。"""
    p = MockProvider([])
    comp = asyncio.run(p.complete([]))
    assert comp.finish_reason == "stop"


# ---------- 真实成本提取(M1,用假 resp 验证,不依赖 litellm 联网)----------


class _FakeUsage:
    prompt_tokens = 12
    completion_tokens = 8


class _FakeMsg:
    content = "你好"
    tool_calls = None


class _FakeChoice:
    message = _FakeMsg()
    finish_reason = "stop"


class _FakeResp:
    choices = [_FakeChoice()]
    usage = _FakeUsage()
    _hidden_params = {"response_cost": 0.00123}


def test_extract_cost_有成本():
    assert _extract_cost(_FakeResp()) == 0.00123


def test_extract_cost_无成本返回None():
    class NoCost:
        _hidden_params = {}

    assert _extract_cost(NoCost()) is None
    assert _extract_cost(object()) is None  # 连 _hidden_params 都没有也不报错


def test_from_lite_response_带成本与用量():
    comp = LiteLLMProvider._from_lite_response(_FakeResp())
    assert comp.usage.input_tokens == 12
    assert comp.usage.output_tokens == 8
    assert comp.usage.cost_usd == 0.00123
    assert comp.finish_reason == "stop"
    assert comp.message.content == "你好"

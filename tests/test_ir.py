"""ir.py 的测试:验证统一 IR 类型的「序列化往返」、默认值、Usage 相加。

「序列化往返」= 把对象转成 JSON 字符串(model_dump_json),再从字符串还原
(model_validate_json),还原后应与原对象完全相等。这是 Session 能存盘/恢复的根基。
"""

from rein.ir import Completion, Message, ToolCall, Usage


def test_工具调用消息_序列化往返():
    """assistant 发起工具调用的消息,往返后应完全相等。"""
    tc = ToolCall(id="c1", name="search", arguments={"q": "天气"})
    msg = Message(role="assistant", tool_calls=[tc])
    assert Message.model_validate_json(msg.model_dump_json()) == msg


def test_工具结果消息_序列化往返():
    """role=tool 的工具结果消息,往返后应完全相等。"""
    msg = Message(role="tool", tool_call_id="c1", content="晴")
    assert Message.model_validate_json(msg.model_dump_json()) == msg


def test_completion_序列化往返():
    """模型一次返回(含 usage)往返后应完全相等。"""
    comp = Completion(
        message=Message(role="assistant", content="你好"),
        finish_reason="stop",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    assert Completion.model_validate_json(comp.model_dump_json()) == comp


def test_message_默认值():
    """content 默认空串、tool_calls 默认 None。"""
    assert Message(role="assistant").content == ""
    assert Message(role="user", content="hi").tool_calls is None


def test_usage_相加():
    """两个 Usage 相加:token 逐项相加,成本相加。"""
    u = Usage(input_tokens=1, output_tokens=2, cost_usd=0.1) + Usage(
        input_tokens=3, output_tokens=4, cost_usd=0.2
    )
    assert (u.input_tokens, u.output_tokens) == (4, 6)
    assert abs(u.cost_usd - 0.3) < 1e-9


def test_usage_相加_成本可缺失():
    """两边都没有成本时,相加结果的成本应为 None(而不是 0)。"""
    u = Usage(input_tokens=1) + Usage(input_tokens=2)
    assert u.input_tokens == 3
    assert u.cost_usd is None

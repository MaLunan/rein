"""session.py 的测试:Stage 可序列化、Session 默认状态,以及最关键的——
一个「推进到一半」的 Session 能存盘并完整恢复(这是 M2 断点续跑的地基)。
"""

from rein.ir import Message, ToolCall, Usage
from rein.session import Session, Stage


def test_stage_是可序列化字符串():
    """Stage 继承 str,本身就等于对应的字符串,序列化天然无障碍。"""
    assert Stage.CALL_MODEL == "call_model"
    assert Stage.RUN_TOOLS.value == "run_tools"
    assert Stage.DONE == "done"


def test_新会话的默认状态():
    """全新 Session:从 CALL_MODEL 开始,各计数为 0,未结束。"""
    s = Session()
    assert s.stage == Stage.CALL_MODEL
    assert s.messages == []
    assert s.pending_tool_calls == []
    assert s.iteration == 0
    assert s.repeat_count == 0
    assert s.done is False
    assert s.stop_reason is None


def test_带完整状态的会话能存盘并完整恢复():
    """M0 为 M2 留形状的核心验证:
    一个带历史、带待执行工具调用、停在 RUN_TOOLS、带熔断计数的 Session,
    序列化再反序列化后必须完全相等,且关键状态原样还在。"""
    call = ToolCall(id="c1", name="weather", arguments={"city": "北京"})
    s = Session(
        messages=[
            Message(role="user", content="天气?"),
            Message(role="assistant", tool_calls=[call]),
        ],
        stage=Stage.RUN_TOOLS,
        pending_tool_calls=[call],
        iteration=2,
        usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.01),
        repeat_count=1,
        last_signature="weather:{'city':'北京'}",
    )

    restored = Session.model_validate_json(s.model_dump_json())

    # 整体完全相等
    assert restored == s
    # 关键状态:阶段、待执行工具、计数都原样还在(否则恢复后会出错)
    assert restored.stage == Stage.RUN_TOOLS
    assert restored.pending_tool_calls[0].name == "weather"
    assert restored.pending_tool_calls[0].arguments == {"city": "北京"}
    assert restored.iteration == 2
    assert restored.usage.input_tokens == 100

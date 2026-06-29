"""result.py 的测试:RunResult 的 __str__ 行为,以及序列化往返(含内嵌 Session)。"""

from rein.ir import Usage
from rein.result import RunResult, Step
from rein.session import Session


def test_str返回output方便直接打印():
    """print(result) 应直接显示答案文本(渐进式暴露:小白只想看结果)。"""
    r = RunResult(status="done", output="北京今天晴", session=Session())
    assert str(r) == "北京今天晴"


def test_output为空时str是空串():
    r = RunResult(status="done", output=None, session=Session())
    assert str(r) == ""


def test_默认值():
    r = RunResult(status="done", output="x", session=Session())
    assert r.stop_reason == "done"
    assert r.steps == []
    assert r.interrupt is None


def test_runresult序列化往返_含session和steps():
    """整个结果报告(含内嵌的完整会话和流水账)能存盘并完整还原。"""
    r = RunResult(
        status="done",
        output="答案",
        session=Session(iteration=3),
        usage=Usage(input_tokens=50, output_tokens=10),
        stop_reason="done",
        steps=[Step(index=0, kind="model", summary="模型说要查天气")],
    )
    restored = RunResult.model_validate_json(r.model_dump_json())
    assert restored == r
    assert restored.session.iteration == 3
    assert restored.steps[0].summary == "模型说要查天气"

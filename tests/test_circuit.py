"""circuit.py 测试:四道熔断闸各自能触发 + 工具调用签名。"""

import time

from rein.circuit import check_circuit, signature
from rein.config import LoopConfig
from rein.ir import ToolCall, Usage
from rein.session import Session


def test_未触顶返回None():
    """全新会话、阈值充足时,不应触发任何闸。"""
    assert check_circuit(Session(), LoopConfig(), time.monotonic()) is None


def test_轮数闸():
    s = Session(iteration=50)
    assert check_circuit(s, LoopConfig(max_iterations=50), time.monotonic()) == "max_iterations"


def test_token闸():
    s = Session(usage=Usage(input_tokens=150, output_tokens=60))  # 共 210 > 200
    assert check_circuit(s, LoopConfig(max_tokens=200), time.monotonic()) == "max_tokens"


def test_超时闸():
    """start_time 设在 10 秒前 → 已超过 1 秒上限。"""
    assert check_circuit(Session(), LoopConfig(timeout_s=1), time.monotonic() - 10) == "timeout"


def test_重复闸():
    s = Session(repeat_count=3)
    cfg = LoopConfig(detect_loops=True, repeat_threshold=3)
    assert check_circuit(s, cfg, time.monotonic()) == "loop_detected"


def test_signature_同名同参得到相同签名():
    a = [ToolCall(id="1", name="f", arguments={"x": 1})]
    b = [ToolCall(id="2", name="f", arguments={"x": 1})]  # id 不同,但名+参相同
    assert signature(a) == signature(b)


def test_signature_不同参得到不同签名():
    a = [ToolCall(id="1", name="f", arguments={"x": 1})]
    b = [ToolCall(id="1", name="f", arguments={"x": 2})]
    assert signature(a) != signature(b)

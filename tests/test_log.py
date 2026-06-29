"""log.py 测试:默认安静 / enable 输出 / JSON 结构化 / disable 恢复 / loop 埋点 / 脱敏。"""

import io
import json
import logging

from rein import Agent, MockProvider, ToolCall, disable_logging, enable_logging
from rein.log import logger


def test_默认挂NullHandler保持安静():
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_enable后能输出文本():
    buf = io.StringIO()
    enable_logging("INFO", stream=buf)
    logger.info("hello-text")
    disable_logging()
    assert "hello-text" in buf.getvalue()


def test_json模式带结构化字段():
    buf = io.StringIO()
    enable_logging("INFO", json=True, stream=buf)
    logger.info("evt", extra={"trace_id": "ab12", "tool": "search"})
    disable_logging()
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rec["msg"] == "evt"
    assert rec["trace_id"] == "ab12"
    assert rec["tool"] == "search"
    assert rec["level"] == "INFO"


def test_disable后恢复安静():
    buf = io.StringIO()
    enable_logging("INFO", stream=buf)
    disable_logging()
    logger.info("after-disable")
    assert "after-disable" not in buf.getvalue()


def test_loop埋点产出run生命周期日志():
    buf = io.StringIO()
    enable_logging("INFO", json=True, stream=buf)
    agent = Agent(provider=MockProvider([[ToolCall(id="1", name="ping", arguments={})], "完成"]))

    @agent.tool
    def ping() -> str:
        "ping"
        return "pong"

    agent.run("t")
    disable_logging()
    out = buf.getvalue()
    assert "run started" in out
    assert "tool done" in out
    assert "run finished" in out


def test_工具结果脱敏_不进日志():
    """密钥/敏感:工具返回内容绝不出现在日志里,只记工具名。"""
    buf = io.StringIO()
    enable_logging("INFO", json=True, stream=buf)
    agent = Agent(provider=MockProvider([[ToolCall(id="1", name="ping", arguments={})], "done"]))

    @agent.tool
    def ping() -> str:
        "ping"
        return "SUPER-SECRET"

    agent.run("t")
    disable_logging()
    out = buf.getvalue()
    assert "SUPER-SECRET" not in out  # 工具结果不进日志
    assert "ping" in out  # 但工具名要有

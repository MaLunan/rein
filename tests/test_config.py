"""config.py 的测试:LoopConfig 的默认值、可覆盖、非法值校验、序列化往返。"""

import pytest
from pydantic import ValidationError

from rein.config import LoopConfig


def test_默认值是生产安全的保守值():
    """不传任何参数时,四道闸 + 权限都应是预设的安全默认。"""
    c = LoopConfig()
    assert c.max_iterations == 50
    assert c.max_tokens == 200_000
    assert c.timeout_s == 120
    assert c.detect_loops is True
    assert c.repeat_threshold == 3
    assert c.permission == "allow"


def test_可以覆盖默认值():
    """用户可以按需覆盖,比如关掉 token 上限、改成 deny。"""
    c = LoopConfig(max_iterations=10, permission="deny", max_tokens=None)
    assert c.max_iterations == 10
    assert c.permission == "deny"
    assert c.max_tokens is None


def test_非法权限值被拒绝():
    """permission 只能是 allow/ask/deny,传别的应当报校验错误。"""
    with pytest.raises(ValidationError):
        LoopConfig(permission="bogus")


def test_序列化往返():
    c = LoopConfig(max_iterations=7)
    assert LoopConfig.model_validate_json(c.model_dump_json()) == c

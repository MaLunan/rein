"""OTel 导出测试(M3)——默认 skip:没装 opentelemetry 就跳过(可观测核心零依赖)。

要真跑:`pip install 'rein[otel]'`。用内存 exporter 验证 run/step 的 span 结构,不联网。
"""

import pytest

# 没装 opentelemetry-sdk 就整文件跳过
pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rein import Agent, MockProvider, ToolCall, export_run


def _tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("rein-test"), exporter


def test_export_run_产出run与step的span():
    agent = Agent(
        provider=MockProvider(
            [[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})], "等于 3"]
        )
    )

    @agent.tool
    def add(a: int, b: int) -> int:
        "求和"
        return a + b

    result = agent.run("加")
    tracer, exporter = _tracer_and_exporter()
    export_run(result, tracer=tracer)

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "rein.run" in names
    assert any(n.startswith("rein.step.") for n in names)

    run_span = next(s for s in spans if s.name == "rein.run")
    assert run_span.attributes["rein.status"] == "done"
    assert run_span.attributes["rein.stop_reason"] == "done"
    # model → tool → model = 3 个 step span
    step_spans = [s for s in spans if s.name.startswith("rein.step.")]
    assert len(step_spans) == 3


def test_export_run_无tracer用全局():
    """不传 tracer 也不报错(走全局 provider;此处只验证不抛异常)。"""
    agent = Agent(provider=MockProvider(["好"]))
    result = agent.run("hi")
    export_run(result)  # 不应抛

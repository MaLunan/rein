"""OpenTelemetry 导出(M3,可选 extras)。

可观测的【核心】是 RunResult 这份结构化数据(核心层零依赖产出)。本模块只是把它
【导出】到 OpenTelemetry 的 adapter —— `opentelemetry` 只在函数内延迟 import,
没装也不影响 `import rein` / 用核心功能。要用它:`pip install 'rein[otel]'`。

映射关系:一次 run = 一个父 span;每个 Step = 一个子 span(带 kind/耗时/用量/错误)。
这样在 Jaeger / Tempo / Langfuse(兼容 OTLP)里就能看到一次 agent 运行的完整时间线。
"""

from rein.result import RunResult


def export_run(result: RunResult, tracer=None) -> None:
    """把一次 RunResult 导出为 OpenTelemetry trace(run 父 span + 每步子 span)。

    Args:
        result: 要导出的运行结果。
        tracer: 可选的 OTel tracer;不给则用 `trace.get_tracer("rein")`(走全局 provider)。
    """
    try:
        from opentelemetry import trace
    except ImportError as e:
        raise ImportError("OTel 导出需要安装:pip install 'rein[otel]'(opentelemetry-sdk 等)") from e

    tracer = tracer or trace.get_tracer("rein")

    with tracer.start_as_current_span("rein.run") as run_span:
        run_span.set_attribute("rein.status", result.status)
        run_span.set_attribute("rein.stop_reason", result.stop_reason)
        run_span.set_attribute("rein.usage.input_tokens", result.usage.input_tokens)
        run_span.set_attribute("rein.usage.output_tokens", result.usage.output_tokens)
        if result.usage.cost_usd is not None:
            run_span.set_attribute("rein.usage.cost_usd", result.usage.cost_usd)
        if result.elapsed_s is not None:
            run_span.set_attribute("rein.elapsed_s", result.elapsed_s)
        run_span.set_attribute("rein.steps", len(result.steps))

        # 每一步一个子 span
        for step in result.steps:
            with tracer.start_as_current_span(f"rein.step.{step.kind}") as sp:
                sp.set_attribute("rein.step.index", step.index)
                sp.set_attribute("rein.step.kind", step.kind)
                sp.set_attribute("rein.step.summary", step.summary)
                if step.duration_s is not None:
                    sp.set_attribute("rein.step.duration_s", step.duration_s)
                if step.is_error is not None:
                    sp.set_attribute("rein.step.is_error", step.is_error)
                if step.usage is not None:
                    sp.set_attribute("rein.step.output_tokens", step.usage.output_tokens)

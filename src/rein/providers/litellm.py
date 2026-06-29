"""LiteLLMProvider —— 接入真实大模型(经 LiteLLM,wrap 100+ 厂商)。

关键设计:litellm 是可选依赖(extras),所以【只在真正调用时才 import】,
绝不在模块顶层 import —— 这样没装 litellm 的环境照样能 import rein、用 MockProvider 跑测试。
寻址沿用 litellm 的 "provider/model" 格式,如 "anthropic/claude-opus-4-8"、"openai/gpt-4o"。

注:IR ↔ litellm 的格式转换 M0 给出最小可用骨架,M1 再完善(流式、能力差异等)。
"""

import json
from collections.abc import AsyncIterator

from rein.ir import Completion, Message, StreamChunk, ToolCall, ToolSpec, Usage


def _extract_cost(resp) -> float | None:
    """从 litellm 响应里尽力取出「真实美元成本」(M1)。

    litellm 会把它算好的成本塞进 `_hidden_params["response_cost"]`;
    取不到(老版本 / 流式 / 未知模型)就返回 None,绝不报错 —— 成本是「锦上添花」,
    不能因为拿不到成本就让正常调用失败。
    """
    try:
        hidden = getattr(resp, "_hidden_params", None) or {}
        cost = hidden.get("response_cost")
        if isinstance(cost, (int, float)):
            return float(cost)
    except Exception:
        pass
    return None


class LiteLLMProvider:
    """把统一 IR 转成 litellm 调用,再把结果转回 IR。"""

    def __init__(self, model: str, **default_params):
        """
        Args:
            model:           litellm 寻址,如 "anthropic/claude-opus-4-8"。
            **default_params: 透传给 litellm 的默认参数(temperature 等)。
        """
        self.model = model
        self.default_params = default_params

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> Completion:
        # 延迟 import:没装 litellm 的环境照样能 import rein
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "使用 LiteLLMProvider 需要安装 litellm:pip install 'rein[litellm]'"
            ) from e

        lite_messages = [self._to_lite_message(m) for m in messages]
        lite_tools = [self._to_lite_tool(t) for t in tools] if tools else None

        resp = await litellm.acompletion(
            model=self.model,
            messages=lite_messages,
            tools=lite_tools,
            **{**self.default_params, **kwargs},
        )
        return self._from_lite_response(resp)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用:实时吐文本,工具调用分片在内部累积拼装,完整后一次性给出。"""
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "使用 LiteLLMProvider 需要安装 litellm:pip install 'rein[litellm]'"
            ) from e

        lite_messages = [self._to_lite_message(m) for m in messages]
        lite_tools = [self._to_lite_tool(t) for t in tools] if tools else None

        resp = await litellm.acompletion(
            model=self.model,
            messages=lite_messages,
            tools=lite_tools,
            stream=True,
            # 让 litellm 在流末尾附带 usage(OpenAI 兼容);拿不到也不报错。
            stream_options={"include_usage": True},
            **{**self.default_params, **kwargs},
        )

        # 工具调用分片累积:index -> {id, name, arguments(累积的 JSON 字符串片段)}
        tool_acc: dict[int, dict] = {}
        finish_reason: str | None = None
        usage: Usage | None = None

        async for chunk in resp:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                choice = choices[0]
                delta = getattr(choice, "delta", None)

                # 文本增量 → 实时发出
                content = getattr(delta, "content", None) if delta else None
                if content:
                    yield StreamChunk(type="text", text=content)

                # 工具调用分片 → 按 index 累积(name 一次性给、arguments 分片拼)
                for tcd in (getattr(delta, "tool_calls", None) or []) if delta else []:
                    idx = getattr(tcd, "index", 0) or 0
                    slot = tool_acc.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                    if getattr(tcd, "id", None):
                        slot["id"] = tcd.id
                    fn = getattr(tcd, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            slot["arguments"] += fn.arguments

                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason

            # usage 可能挂在 chunk 顶层(流末尾)
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage = Usage(
                    input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(u, "completion_tokens", 0) or 0,
                )

        # 流结束:把累积的分片拼成完整 ToolCall
        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_acc):
            slot = tool_acc[idx]
            raw = slot["arguments"]
            try:
                args = json.loads(raw) if raw else {}
            except Exception:
                args = {}
            tool_calls.append(
                ToolCall(id=slot["id"] or f"call_{idx}", name=slot["name"] or "", arguments=args)
            )

        if tool_calls:
            yield StreamChunk(type="tool_calls", tool_calls=tool_calls)
            yield StreamChunk(type="done", finish_reason="tool_calls", usage=usage or Usage())
        else:
            yield StreamChunk(
                type="done", finish_reason=finish_reason or "stop", usage=usage or Usage()
            )

    # ---- IR → litellm 格式 ----

    @staticmethod
    def _to_lite_message(m: Message) -> dict:
        """IR Message → litellm/OpenAI 消息 dict。"""
        d: dict = {"role": m.role}
        if m.content:
            d["content"] = m.content
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        return d

    @staticmethod
    def _to_lite_tool(t: ToolSpec) -> dict:
        """IR ToolSpec → litellm/OpenAI 工具格式。"""
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }

    # ---- litellm 响应 → IR ----

    @staticmethod
    def _from_lite_response(resp) -> Completion:
        """litellm 响应 → IR Completion。"""
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage_obj = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            cost_usd=_extract_cost(resp),
        )

        finish = "tool_calls" if tool_calls else (choice.finish_reason or "stop")
        if finish not in ("stop", "tool_calls", "length", "error"):
            finish = "stop"

        return Completion(
            message=Message(
                role="assistant",
                content=msg.content or "",
                tool_calls=tool_calls or None,
            ),
            finish_reason=finish,  # type: ignore[arg-type]
            usage=usage,
        )

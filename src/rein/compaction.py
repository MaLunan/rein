"""上下文压缩(Compaction)—— 让长任务不被上下文窗口撑爆(M3)。

核心理念:压缩是对 `session.messages` 的【纯变换】(messages → messages),
进出都是普通 `Message`,所以压缩产物照样可序列化、照样能 resume(不破坏 M2)。

提供两种策略(都实现同一个 CompactionStrategy 协议):
- SlidingWindow:    只保留 system + 最近 N 条,丢更早的(纯裁剪,零成本)。
- SummarizeCompaction:超 token 阈值时,把旧历史折叠成一条摘要,保留近期原文。
                     默认机械摘要(不联网、可测);注入 Provider 可上 LLM 真摘要。

token 计数用【近似估算】(见 estimate_tokens):中文字符≈1 token、ASCII≈0.25 token。
这是保守上界(宁可早压一点,也别撑爆窗口);需要精确计数可后续接 LiteLLM。
"""

import json
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from rein.ir import Message


def estimate_tokens(messages: "list[Message] | str") -> int:
    """粗略估算一段文本 / 一组消息的 token 数(近似,非精确)。

    规则:非 ASCII 字符(含中日韩)按 1 token,ASCII 按 0.25 token 累加。
    这是个保守上界,用于「该不该压缩」的触发判断,不用于计费。
    """
    if isinstance(messages, str):
        texts = [messages]
    else:
        texts = []
        for m in messages:
            if m.content:
                texts.append(m.content)
            for tc in m.tool_calls or []:
                texts.append(tc.name)
                texts.append(json.dumps(tc.arguments, ensure_ascii=False))
    total = 0.0
    for t in texts:
        for ch in t:
            total += 1.0 if ord(ch) > 127 else 0.25
    return int(total) + 1


@runtime_checkable
class CompactionStrategy(Protocol):
    """上下文压缩策略的统一接口:输入消息列表,输出(可能更短的)消息列表。"""

    async def compact(self, messages: list[Message]) -> list[Message]:
        """对消息历史做一次压缩;没超过自身阈值时应原样返回。"""
        ...


def _drop_orphan_tools(messages: list[Message]) -> list[Message]:
    """去掉【开头】的孤儿 tool 消息。

    裁剪/折叠可能把某个 assistant 的 tool_calls 删掉,而它的 tool 结果留了下来 ——
    这种「没有对应调用的 tool 消息」会让厂商 API 报错。开头的孤儿直接丢掉最稳妥。
    """
    out = list(messages)
    while out and out[0].role == "tool":
        out.pop(0)
    return out


class SlidingWindow:
    """滑动窗口:只保留 system 提示 + 最近 max_messages 条,丢弃更早的历史。"""

    def __init__(self, max_messages: int, *, keep_system: bool = True):
        """
        Args:
            max_messages: 保留的「最近非 system 消息」条数上限。
            keep_system:  是否始终保留 system 提示(默认 True)。
        """
        self.max_messages = max_messages
        self.keep_system = keep_system

    async def compact(self, messages: list[Message]) -> list[Message]:
        non_system = [m for m in messages if m.role != "system"]
        if len(non_system) <= self.max_messages:
            return messages  # 没超,原样返回

        system = [m for m in messages if m.role == "system"] if self.keep_system else []
        kept = non_system[-self.max_messages :]
        return system + _drop_orphan_tools(kept)


class SummarizeCompaction:
    """摘要式压缩:超 token 阈值时,把旧历史折叠成一条摘要,保留近期原文。"""

    def __init__(
        self,
        max_tokens: int,
        *,
        keep_recent: int = 4,
        summarizer: "Callable[[list[Message]], Awaitable[str]] | object | None" = None,
    ):
        """
        Args:
            max_tokens:  触发压缩的估算 token 阈值。
            keep_recent: 保留最近多少条原文不折叠(默认 4)。
            summarizer:  摘要器。None→机械摘要(拼接截断,不联网);
                         传入有 .complete 的 Provider→调 LLM 真摘要;
                         也可传入 async callable(messages)->str。
        """
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.summarizer = summarizer

    async def compact(self, messages: list[Message]) -> list[Message]:
        if estimate_tokens(messages) <= self.max_tokens:
            return messages  # 没超,原样返回

        system = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]
        if len(non_system) <= self.keep_recent:
            return messages  # 旧历史不足以折叠,不动

        old = non_system[: -self.keep_recent]
        recent = _drop_orphan_tools(non_system[-self.keep_recent :])

        summary_text = await self._summarize(old)
        summary_msg = Message(role="system", content=f"[历史摘要] {summary_text}")
        return system + [summary_msg] + recent

    async def _summarize(self, old: list[Message]) -> str:
        """把一段旧消息浓缩成一句摘要文本。"""
        if self.summarizer is None:
            # 机械摘要:拼接旧消息要点并截断(不联网、确定性,便于测试)
            parts: list[str] = []
            for m in old:
                if m.content:
                    parts.append(f"{m.role}:{m.content}")
                elif m.tool_calls:
                    names = ", ".join(tc.name for tc in m.tool_calls)
                    parts.append(f"{m.role} 调用了工具 {names}")
            joined = " | ".join(parts)
            return joined[:500] + ("…" if len(joined) > 500 else "")

        # 注入了 Provider → 调 LLM 真摘要
        if hasattr(self.summarizer, "complete"):
            convo = "\n".join(f"{m.role}: {m.content}" for m in old if m.content)
            prompt = [Message(role="user", content=f"请用简洁中文概括以下对话要点:\n\n{convo}")]
            comp = await self.summarizer.complete(prompt)
            return comp.message.content

        # 普通 async callable
        return await self.summarizer(old)  # type: ignore[operator]

"""Agentic Loop —— 框架的「心脏」:可序列化的单步状态机。

设计铁律(对应 DESIGN D4、M0 注意点 1/2):
- 所有跨步状态都在 Session 里,loop 自身【无状态】。
- `step()` 只推进【一格】(一个 Stage),这样「工具执行前」这个边界天然存在,
  M2 的 HITL(人工确认)/ 断点续跑就在这一格开头返回 Interrupt 即可,签名不用改。
- M0 的 `step()` 永远返回 interrupt=None(permission=allow 一路到底)。

状态机流转(见 session.Stage):
    CALL_MODEL --(模型给文本)----> DONE
    CALL_MODEL --(模型要调工具)--> RUN_TOOLS
    RUN_TOOLS  --(工具执行完)----> CALL_MODEL

对外两个入口:
- `arun(...)`:异步内核,驱动 step 循环 + 每步前查熔断,产出 RunResult。
- `run(...)` :同步门面,= asyncio.run(arun(...));已在事件循环里则报错引导用 arun。
"""

import asyncio
import time
from collections.abc import AsyncIterator

from rein.circuit import check_circuit, signature
from rein.compaction import CompactionStrategy
from rein.config import LoopConfig
from rein.ir import Completion, Message, StreamChunk, Usage
from rein.middleware import Middleware, StepContext, dispatch, permission_middleware

# Provider 只用于类型标注;为避免「核心层 import 具体实现」,这里用最小依赖。
from rein.providers.base import Provider
from rein.providers.fallback import is_retryable
from rein.result import Interrupt, RunResult, Step
from rein.runtime.base import Runtime
from rein.runtime.local import LocalRuntime
from rein.session import Session, Stage
from rein.tools import ToolRegistry


def _truncate(s: str | None, n: int = 120) -> str:
    """把一段文本截断到 n 个字符,给流水账(Step.summary)用,避免摘要过长。"""
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def _last_assistant_text(session: Session) -> str | None:
    """从对话历史里倒着找最后一条「有内容的 assistant 文本」,作为最终答案。

    为什么倒着找:loop 结束时,最后一条 assistant 消息通常就是最终回答;
    但若因熔断在工具阶段停下,最后一条可能是「只含 tool_calls 的空文本」,
    那就继续往前找到上一条有内容的回答(没有则 None)。
    """
    for m in reversed(session.messages):
        if m.role == "assistant" and m.content:
            return m.content
    return None


def _advance_after_model(session: Session, completion: Completion) -> Step:
    """模型轮的「推进逻辑」(纯状态推进,被 step 与 astream 复用)。

    无论结果是流式攒出来的、还是一次性 complete 拿到的,只要拼成完整 Completion,
    推进方式就【完全一致】—— 这正是「流式不改主干」的落点。返回本步的流水账 Step。
    """
    # 累计用量与轮数(iteration = 模型调用次数,对齐 max_iterations 语义)
    session.usage = session.usage + completion.usage
    session.iteration += 1

    msg = completion.message
    session.messages.append(msg)  # 把模型这次产出的消息记入历史

    if completion.finish_reason == "tool_calls" and msg.tool_calls:
        # 模型要调工具 → 更新「重复检测」状态,排队待执行,转 RUN_TOOLS
        sig = signature(msg.tool_calls)
        if sig == session.last_signature:
            session.repeat_count += 1  # 与上一轮调用完全相同 → 连击 +1
        else:
            session.repeat_count = 1  # 出现新调用 → 计数归 1(本身算第 1 次)
            session.last_signature = sig
        session.pending_tool_calls = list(msg.tool_calls)
        session.stage = Stage.RUN_TOOLS
        return Step(
            index=0,
            kind="model",
            summary="请求工具:" + ", ".join(tc.name for tc in msg.tool_calls),
            tool_calls=list(msg.tool_calls),
            usage=completion.usage,
        )

    # 模型给出文本回答 → 正常结束
    session.done = True
    session.stage = Stage.DONE
    session.stop_reason = "done"
    return Step(
        index=0,
        kind="model",
        summary=_truncate(msg.content),
        usage=completion.usage,
    )


async def _run_tools(
    session: Session,
    registry: ToolRegistry,
    config: LoopConfig,
    runtime: Runtime,
) -> list[Step]:
    """工具轮的「执行 + 回填」(被 step 与 astream 复用)。"""
    calls = session.pending_tool_calls
    t0 = time.monotonic()
    results = await runtime.execute_all(calls, registry, config.permission)
    dt = time.monotonic() - t0  # 整批并发耗时;并发下每个工具的墙钟≈整批,近似记入

    recs: list[Step] = []
    for call, res in zip(calls, results, strict=True):
        # 工具结果回填为 role="tool" 消息,用 tool_call_id 指回对应调用
        session.messages.append(
            Message(role="tool", tool_call_id=res.tool_call_id, content=res.content)
        )
        recs.append(
            Step(
                index=0,
                kind="tool",
                summary=f"{call.name} → {_truncate(res.content, 80)}",
                is_error=res.is_error,
                duration_s=dt,
            )
        )

    session.pending_tool_calls = []
    session.stage = Stage.CALL_MODEL  # 回到问模型,带着工具结果继续下一轮
    return recs


async def step(
    session: Session,
    provider: Provider,
    registry: ToolRegistry,
    config: LoopConfig,
    runtime: Runtime,
) -> tuple[Session, list[Step], Interrupt | None]:
    """推进状态机【一格】,返回 (推进后的 session, 本格产生的流水账, 中断或 None)。

    - CALL_MODEL:调一次模型 → 累计 usage/iteration → 要调工具则转 RUN_TOOLS,否则结束。
    - RUN_TOOLS :执行待办工具(并发保序)→ 结果回填 → 转回 CALL_MODEL。
    M0 永远返回 interrupt=None;Step 的 index 占位为 0,由 arun 统一重排。
    """
    # ---------- 阶段一:问模型 ----------
    if session.stage == Stage.CALL_MODEL:
        t0 = time.monotonic()
        tools = registry.specs() or None  # 没有工具就不传(空列表 → None)
        try:
            completion = await provider.complete(session.messages, tools=tools)
        except Exception as e:
            # 错误分流(M2):
            # - 可重试错误(限流/超时/5xx)→ 建模成 error 中断态,让调用方 resume 重试,
            #   而不是直接炸掉整个 run(stage 仍是 CALL_MODEL,resume 会重跑这一步)。
            # - 致命错误(鉴权/参数错误)→ 直接抛:重试也没用,抛出来最清晰。
            if is_retryable(e):
                return (
                    session,
                    [],
                    Interrupt(
                        type="error",
                        message=f"调用模型出错(可重试):{type(e).__name__}: {e}",
                    ),
                )
            raise
        dt = time.monotonic() - t0

        rec = _advance_after_model(session, completion)
        rec.duration_s = dt
        return session, [rec], None

    # ---------- 阶段二:干活(执行工具) ----------
    if session.stage == Stage.RUN_TOOLS:
        # M4:权限(ask)已移到内置 permission_middleware 统一处理,loop.step 不再写权限特例。
        # 能走到这里 = 已被放行,直接执行(deny 仍由 Runtime 在执行层回 is_error)。
        recs = await _run_tools(session, registry, config, runtime)
        return session, recs, None

    # DONE 阶段不应再被 step(驱动循环以 session.done 为终止条件);防御性返回。
    return session, [], None


async def _core_step(ctx: StepContext) -> StepContext:
    """中间件洋葱的最内层:真正执行一步 step,把结果填回 ctx。"""
    session, recs, interrupt = await step(
        ctx.session, ctx.provider, ctx.registry, ctx.config, ctx.runtime
    )
    ctx.session = session
    ctx.steps = recs
    ctx.interrupt = interrupt
    return ctx


async def arun(
    session: Session,
    provider: Provider,
    registry: ToolRegistry,
    config: LoopConfig | None = None,
    runtime: Runtime | None = None,
    compaction: CompactionStrategy | None = None,
    middlewares: list[Middleware] | None = None,
) -> RunResult:
    """驱动 step 循环跑完一次,产出 RunResult(异步内核)。

    每步【之前】先查四道熔断闸:任一触顶就安全终止(置 done + stop_reason),
    绝不再多调一次模型 / 多跑一次工具。
    M3:若提供 compaction,则在每次「问模型」之前对历史做一次压缩(超窗才压)。
    M4:每一步 step 都经过中间件洋葱(无中间件时等价裸 step,不改变默认行为)。
    """
    config = config or LoopConfig()
    runtime = runtime or LocalRuntime()
    # M4:用户中间件在外层,内置权限中间件总在最内层(紧贴 step)
    mws = (middlewares or []) + [permission_middleware]
    start = time.monotonic()
    steps: list[Step] = []

    while not session.done:
        reason = check_circuit(session, config, start)
        if reason:
            session.stop_reason = reason
            session.done = True
            break

        # M3:问模型前压缩上下文(纯变换 messages,不破坏可序列化/恢复)
        if compaction is not None and session.stage == Stage.CALL_MODEL:
            session.messages = await compaction.compact(session.messages)

        # M4:单步 step 经过中间件洋葱(环绕单步、栈无状态、每步重建 → 兼容恢复)
        ctx = StepContext(session, session.stage, config, provider, registry, runtime)
        ctx = await dispatch(mws, ctx, _core_step)
        session, recs, interrupt = ctx.session, ctx.steps, ctx.interrupt
        for r in recs:
            r.index = len(steps)  # 统一重排流水账序号(全局连续)
            steps.append(r)

        if interrupt is not None:  # M2:在工具边界产出中断态,返回给调用方等待 resume
            session.stop_reason = "interrupted"  # 不置 done —— 要能从此 session 恢复
            return RunResult(
                status="interrupted",
                output=_last_assistant_text(session),
                session=session,
                usage=session.usage,
                elapsed_s=time.monotonic() - start,
                stop_reason="interrupted",
                steps=steps,
                interrupt=interrupt,
            )

    return RunResult(
        status="done",
        output=_last_assistant_text(session),
        session=session,
        usage=session.usage,
        elapsed_s=time.monotonic() - start,
        stop_reason=session.stop_reason or "done",
        steps=steps,
    )


def run(
    session: Session,
    provider: Provider,
    registry: ToolRegistry,
    config: LoopConfig | None = None,
    runtime: Runtime | None = None,
    compaction: CompactionStrategy | None = None,
    middlewares: list[Middleware] | None = None,
) -> RunResult:
    """同步门面:= asyncio.run(arun(...))。

    若检测到当前已在事件循环中(Jupyter / Web 框架),直接报错并引导改用
    `await arun(...)` —— 绝不偷偷嵌套事件循环(那会抛 RuntimeError 或更隐蔽的 bug)。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 没有正在运行的事件循环 → 安全,自己起一个跑
        return asyncio.run(
            arun(session, provider, registry, config, runtime, compaction, middlewares)
        )
    # 已在事件循环里 → 不能再 asyncio.run,引导用异步接口
    raise RuntimeError("检测到当前已在事件循环中(如 Jupyter / Web 环境),请改用 `await arun(...)`。")


async def astream(
    session: Session,
    provider: Provider,
    registry: ToolRegistry,
    config: LoopConfig | None = None,
    runtime: Runtime | None = None,
    compaction: CompactionStrategy | None = None,
) -> AsyncIterator[StreamChunk]:
    """流式驱动:边跑边把模型吐字实时 yield 出来(M1)。

    设计纪律(M1 注意点 1):流式只是【旁路观测】——
    - CALL_MODEL 阶段改用 provider.stream 实时透传 text/tool_calls 片段,
      但分片收完会拼成【完整 Completion】,再走与非流式一模一样的 `_advance_after_model`。
    - RUN_TOOLS / 熔断 / 状态推进逻辑与 arun 完全一致,主干一行都不为流式改。
    最后 yield 一个 type="done" 片段(累计用量 + 结束原因)。

    注:本函数不返回 RunResult(async 生成器不能带返回值)。需要完整结果报告/
    会话状态时请用 arun;流式场景关注的是「实时文本 + 最终用量」。
    """
    config = config or LoopConfig()
    runtime = runtime or LocalRuntime()
    start = time.monotonic()

    while not session.done:
        reason = check_circuit(session, config, start)
        if reason:
            session.stop_reason = reason
            session.done = True
            break

        if session.stage == Stage.CALL_MODEL:
            # M3:问模型前压缩上下文(与非流式一致)
            if compaction is not None:
                session.messages = await compaction.compact(session.messages)
            tools = registry.specs() or None
            text_parts: list[str] = []
            stream_tool_calls = None
            finish = "stop"
            usage = Usage()

            # 实时消费分片:text 透传给调用方;tool_calls 已是完整的
            async for chunk in provider.stream(session.messages, tools=tools):
                if chunk.type == "text":
                    if chunk.text:
                        text_parts.append(chunk.text)
                        yield chunk  # 旁路:实时把吐字发出去
                elif chunk.type == "tool_calls":
                    stream_tool_calls = chunk.tool_calls
                    yield chunk
                elif chunk.type == "done":
                    finish = chunk.finish_reason or finish
                    if chunk.usage is not None:
                        usage = chunk.usage

            # 把分片拼成完整 Completion,复用非流式推进(主干不变)
            content = "".join(text_parts)
            if stream_tool_calls:
                completion = Completion(
                    message=Message(
                        role="assistant", content=content, tool_calls=stream_tool_calls
                    ),
                    finish_reason="tool_calls",
                    usage=usage,
                )
            else:
                safe_finish = (
                    finish if finish in ("stop", "tool_calls", "length", "error") else "stop"
                )
                completion = Completion(
                    message=Message(role="assistant", content=content),
                    finish_reason=safe_finish,  # type: ignore[arg-type]
                    usage=usage,
                )
            _advance_after_model(session, completion)

        elif session.stage == Stage.RUN_TOOLS:
            # permission="ask" 的审批中断与流式正交:流式吐字在 CALL_MODEL 阶段,
            # 到工具边界若需审批,流式在此优雅停止(标记 interrupted);
            # 审批/恢复请改用非流式 run + resume(两条路不混用)。
            if config.permission == "ask" and session.pending_tool_calls:
                session.stop_reason = "interrupted"
                break
            await _run_tools(session, registry, config, runtime)

    # 收尾:发一个 done 片段(累计用量 + 结束原因)
    yield StreamChunk(
        type="done",
        finish_reason=session.stop_reason or "done",
        usage=session.usage,
    )


async def aresume(
    session: Session,
    provider: Provider,
    registry: ToolRegistry,
    config: LoopConfig | None = None,
    *,
    approve: bool = True,
    answer: str | None = None,
    runtime: Runtime | None = None,
    compaction: CompactionStrategy | None = None,
    middlewares: list[Middleware] | None = None,
) -> RunResult:
    """从一个【中断态 session】恢复执行(M2 的核心)。

    恢复 = 喂回 session 数据,从它的 stage 继续 —— 绝不依赖任何进程内挂起状态。
    本函数先处理掉「当前这批待办」,再交回 arun 继续驱动:

    - need_approval + approve=True  → 用 allow 执行这批工具,回填结果,继续。
    - need_approval + approve=False → 把这批回填成「被拒绝」错误文本,继续(模型读到后自愈)。
    - need_input(M3 细化)           → 把 answer 作为一条 user 消息注入后继续。

    继续过程中若再次遇到 ask 工具,会再次中断 —— 多轮审批天然支持。
    """
    config = config or LoopConfig()
    runtime = runtime or LocalRuntime()

    if session.stage == Stage.RUN_TOOLS and session.pending_tool_calls:
        calls = session.pending_tool_calls
        if approve:
            # 人已批准 → 用 allow 执行(绕过 ask),结果回填
            results = await runtime.execute_all(calls, registry, "allow")
            for res in results:
                session.messages.append(
                    Message(role="tool", tool_call_id=res.tool_call_id, content=res.content)
                )
        else:
            # 人已拒绝 → 回填「被拒绝」文本,让模型据此改走安全做法(自愈)
            for call in calls:
                session.messages.append(
                    Message(
                        role="tool",
                        tool_call_id=call.id,
                        content="该工具调用被用户拒绝执行。请换一种不需要该操作的方式继续。",
                    )
                )
        session.pending_tool_calls = []
        session.stage = Stage.CALL_MODEL
    elif answer is not None:
        # need_input 场景:把人补充的信息作为 user 消息注入,然后继续问模型
        session.messages.append(Message(role="user", content=answer))
        session.stage = Stage.CALL_MODEL

    # 清掉中断标记,继续正常驱动(熔断从这次 resume 重新计时)
    session.done = False
    session.stop_reason = None
    return await arun(session, provider, registry, config, runtime, compaction, middlewares)

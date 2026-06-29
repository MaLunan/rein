"""门面层:Agent(无状态蓝图)、Chat(会话句柄)、tool(工具装饰器)。

这是用户最常碰到的一层,目标是守住「5 行示例」:

    from rein import Agent
    agent = Agent("anthropic/claude-opus-4-8")

    @agent.tool
    def now() -> str:
        "返回当前时间"
        return "2026-06-27"

    print(agent.run("现在几号?"))

核心设计(对应 DESIGN D3「Agent / Session / Chat 三者分离」):
- Agent 是【无状态蓝图】:只存「怎么跑」(model/provider/工具/system/config),
  【绝不存单次运行的状态】。每次 run 都新建一份独立 Session —— 因此同一个 Agent
  可以被并发用于多个会话而互不串台(并发安全)。
- Session 是【可序列化状态】(见 session.py),由 loop 推进。
- Chat 是【会话句柄】:持有一份持续的 Session,支撑多轮对话(历史跨轮保留)。
"""

import asyncio
from collections.abc import AsyncIterator, Callable

from rein.compaction import CompactionStrategy
from rein.config import LoopConfig
from rein.ir import Message, StreamChunk, Usage
from rein.loop import aresume as _aresume
from rein.loop import arun as _arun
from rein.loop import astream as _astream
from rein.loop import run as _run
from rein.providers.base import Provider
from rein.result import RunResult
from rein.session import Session, Stage
from rein.tools import Tool, ToolRegistry


def tool(fn: Callable) -> Tool:
    """模块级工具装饰器:把一个普通函数包成 Tool,供 `Agent(tools=[...])` 使用。

        @tool
        def add(a: int, b: int) -> int:
            "求和"
            return a + b

        agent = Agent("...", tools=[add])

    注意:被它装饰后 `add` 变成 Tool 对象(框架组合用);若想保留函数本体可调用,
    请改用 `@agent.tool`(注册后返回原函数)。
    """
    return Tool.from_function(fn)


class Agent:
    """无状态的 agent 蓝图:定义「怎么跑」,每次运行产出独立 Session。"""

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: Provider | None = None,
        fallback: list | None = None,
        tools: list | None = None,
        system: str | None = None,
        config: LoopConfig | None = None,
        compaction: CompactionStrategy | None = None,
    ):
        """
        Args:
            model:    LiteLLM 寻址(如 "anthropic/claude-opus-4-8");首次 run 时才懒建 provider。
            provider: 直接注入 Provider(测试用 MockProvider);注入后忽略 model/fallback 的懒建。
            fallback: 备用模型列表(M1)。元素可为模型字符串或 Provider 对象;主模型限流/
                      报错时按序自动切换。例:Agent("anthropic/...", fallback=["openai/gpt-4o"])。
            tools:    工具列表,元素可为「裸函数」或「Tool 对象」,自动登记。
            system:   系统提示,放在每次会话的最前面。
            config:   loop 配置(熔断 + 权限);默认 LoopConfig()。
            compaction: 上下文压缩策略(M3)。给定后,每次问模型前自动压缩超窗历史。
                        例:Agent("...", compaction=SummarizeCompaction(max_tokens=8000))。
        """
        self.model = model
        self._provider = provider
        self.fallback = fallback or []
        self.system = system
        self.config = config or LoopConfig()
        self.compaction = compaction
        self.middlewares: list = []  # M4:洋葱中间件,用 @agent.middleware 注册
        self._event_handlers: dict = {}  # M4:只读事件订阅(agent.on)

        # 工具登记册:裸函数自动 Tool.from_function,已是 Tool 的直接收。
        self.registry = ToolRegistry()
        for t in tools or []:
            self.registry.add(t if isinstance(t, Tool) else Tool.from_function(t))

    # ---- 工具注册 ----

    def tool(self, fn: Callable) -> Callable:
        """方法装饰器:注册一个工具,并【返回原函数】(你照常能直接调用它)。

        @agent.tool
        def search(q: str) -> str: ...
        """
        self.registry.add(Tool.from_function(fn))
        return fn

    def middleware(self, fn: Callable) -> Callable:
        """方法装饰器:注册一个洋葱中间件(M4),返回原函数。

            @agent.middleware
            async def timing(ctx, call_next):
                ctx = await call_next(ctx)   # 调下一层;不调即短路
                return ctx

        注册顺序即洋葱顺序(先注册的在最外层)。中间件环绕【单步 step】,
        不得跨步持状态(要持有就放进 ctx.session),以兼容中断/恢复。
        """
        self.middlewares.append(fn)
        return fn

    # ---- 钩子语法糖(M4):本质都是中间件的便捷封装 ----

    def _add_stage_hook(self, stage: Stage, fn: Callable, *, after: bool) -> Callable:
        """把一个 before/after 钩子转成中间件并注册(钩子 = 中间件的糖)。"""

        async def mw(ctx, call_next):
            if not after and ctx.stage == stage:
                if await fn(ctx) is False:
                    return ctx  # before 钩子返回 False → 短路该步
            ctx = await call_next(ctx)
            if after and ctx.stage == stage:
                await fn(ctx)
            return ctx

        self.middlewares.append(mw)
        return fn

    def before_model(self, fn: Callable) -> Callable:
        """钩子:每次「问模型」之前运行;`async def hook(ctx)`,返回 False 可短路该步。"""
        return self._add_stage_hook(Stage.CALL_MODEL, fn, after=False)

    def after_model(self, fn: Callable) -> Callable:
        """钩子:每次「问模型」之后运行(观测 / 修改 ctx)。"""
        return self._add_stage_hook(Stage.CALL_MODEL, fn, after=True)

    def before_tool(self, fn: Callable) -> Callable:
        """钩子:每次「执行工具」之前运行;返回 False 可短路(不执行工具)。"""
        return self._add_stage_hook(Stage.RUN_TOOLS, fn, after=False)

    def after_tool(self, fn: Callable) -> Callable:
        """钩子:每次「执行工具」之后运行(观测 / 修改 ctx)。"""
        return self._add_stage_hook(Stage.RUN_TOOLS, fn, after=True)

    # ---- 只读事件订阅(M4)----

    def on(self, event: str, handler: Callable) -> Callable:
        """订阅只读事件。event:"step"(每步后)/"tool"(工具步后)。

        handler(ctx) 只用于【观测】(打日志 / 上报),不应改变流程 —— 要改流程请用中间件/钩子。
        """
        if not self._event_handlers:  # 第一次订阅:装上事件分发中间件(只装一次)
            self.middlewares.append(self._event_middleware)
        self._event_handlers.setdefault(event, []).append(handler)
        return handler

    async def _event_middleware(self, ctx, call_next):
        """内置事件分发:每步结束后,把 ctx 推给订阅者(只读)。"""
        ctx = await call_next(ctx)
        for h in self._event_handlers.get("step", []):
            h(ctx)
        if ctx.stage == Stage.RUN_TOOLS:
            for h in self._event_handlers.get("tool", []):
                h(ctx)
        return ctx

    # ---- provider 解析(懒建 + 可注入)----

    def _resolve_provider(self) -> Provider:
        """拿到本次运行要用的 Provider:优先注入的;否则按 model(+fallback)懒建。

        - 只有 model:返回单个 LiteLLMProvider。
        - model + fallback:把 [主, 备...] 懒建成链,包进 FallbackProvider(自动切换)。
        - fallback 元素既可是模型字符串,也可是已构造好的 Provider 对象。
        """
        if self._provider is not None:
            return self._provider
        if self.model is None and not self.fallback:
            raise ValueError(
                "Agent 既没有注入 provider,也没有给 model。"
                "请用 Agent(model='厂商/模型') 或 Agent(provider=MockProvider([...]))。"
            )
        # 懒建:此处 import 不会触发 litellm(litellm 只在 complete() 内部才 import)
        from rein.providers.litellm import LiteLLMProvider

        def _to_provider(x) -> Provider:
            # 已经是 Provider(鸭子类型:有 complete 方法)就直接用,否则当模型字符串懒建
            return x if hasattr(x, "complete") else LiteLLMProvider(x)

        chain: list[Provider] = []
        if self.model is not None:
            chain.append(LiteLLMProvider(self.model))
        chain.extend(_to_provider(f) for f in self.fallback)

        if len(chain) == 1:
            self._provider = chain[0]
        else:
            from rein.providers.fallback import FallbackProvider

            self._provider = FallbackProvider(chain)
        return self._provider

    # ---- 运行 ----

    def _new_session(self, prompt: str) -> Session:
        """为「一次新运行」构造独立 Session(system + user),这是并发安全的基石。"""
        messages: list[Message] = []
        if self.system:
            messages.append(Message(role="system", content=self.system))
        messages.append(Message(role="user", content=prompt))
        return Session(messages=messages)

    async def arun(self, prompt: str) -> RunResult:
        """异步运行一次:新建独立 Session,交给 loop 跑完,返回 RunResult。"""
        session = self._new_session(prompt)
        return await _arun(
            session,
            self._resolve_provider(),
            self.registry,
            self.config,
            compaction=self.compaction,
            middlewares=self.middlewares,
        )

    def run(self, prompt: str) -> RunResult:
        """同步运行一次(= asyncio.run(arun));已在事件循环里会报错引导用 arun。"""
        session = self._new_session(prompt)
        return _run(
            session,
            self._resolve_provider(),
            self.registry,
            self.config,
            compaction=self.compaction,
            middlewares=self.middlewares,
        )

    async def astream(self, prompt: str) -> AsyncIterator[StreamChunk]:
        """流式运行一次:实时把模型吐字发出来(M1)。

            async for chunk in agent.astream("讲个笑话"):
                if chunk.type == "text":
                    print(chunk.text, end="", flush=True)

        说明:流式只关注「实时文本 + 最终用量」;需要完整 RunResult / 会话状态请用 run/arun。
        """
        session = self._new_session(prompt)
        async for chunk in _astream(
            session,
            self._resolve_provider(),
            self.registry,
            self.config,
            compaction=self.compaction,
        ):
            yield chunk

    async def aresume(
        self, session: Session, *, approve: bool = True, answer: str | None = None
    ) -> RunResult:
        """从中断态 session 异步恢复执行(M2)。见 loop.aresume。"""
        return await _aresume(
            session,
            self._resolve_provider(),
            self.registry,
            self.config,
            approve=approve,
            answer=answer,
            compaction=self.compaction,
            middlewares=self.middlewares,
        )

    def resume(
        self, session: Session, *, approve: bool = True, answer: str | None = None
    ) -> RunResult:
        """从中断态 session 同步恢复执行(M2)。

            r = agent.run("危险操作")           # permission="ask"
            if r.status == "interrupted":
                r = agent.resume(r.session, approve=False)   # 拒绝 → 模型自愈

        已在事件循环里会报错引导用 aresume。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aresume(session, approve=approve, answer=answer))
        raise RuntimeError(
            "检测到当前已在事件循环中(如 Jupyter / Web 环境),请改用 `await agent.aresume(...)`。"
        )

    def run_interactive(self, prompt: str, approver: Callable | None = None) -> RunResult:
        """CLI 友好的「带审批」运行(M2):permission='ask' 时每遇审批中断就问一次,自动续跑。

        本质 = run → need_approval 中断 → 本地应答 → resume,循环到 done —— 与服务端
        「产出中断态交给上层」共用【同一套】中断/恢复机制,不是另写一套(见 M2 注意点 4)。

        Args:
            prompt:   用户输入。
            approver: 审批回调 `approver(interrupt) -> bool`;默认在终端用 input() 问 y/N。
                      测试 / 自动化时可注入自定义策略。
        """
        if approver is None:

            def approver(interrupt) -> bool:
                ans = input(f"{interrupt.message} [y/N] ").strip().lower()
                return ans in ("y", "yes", "是", "批准")

        result = self.run(prompt)
        while (
            result.status == "interrupted"
            and result.interrupt is not None
            and result.interrupt.type == "need_approval"
        ):
            result = self.resume(result.session, approve=approver(result.interrupt))
        return result

    def chat(self, session: Session | None = None) -> "Chat":
        """开一个多轮会话句柄(持续保留对话历史)。

        Args:
            session: 可选的已有会话。给定时从它继续(企业级无状态服务的关键:
                     每个请求 `agent.chat(session=store.load(conv_id))` 把状态喂回来);
                     不给则新建(带 system 提示)。
        """
        return Chat(self, session=session)


class Chat:
    """会话句柄:持有一份持续的 Session,支撑多轮对话(历史跨轮保留)。

    与 Agent.run 的区别:Agent.run 每次都是「全新独立会话」;Chat 把历史攒在
    自己的 Session 里,一轮接一轮,模型能看到之前说过的话。
    """

    def __init__(self, agent: Agent, session: Session | None = None):
        """
        Args:
            agent:   背后的 Agent 蓝图。
            session: 已有会话(从 store 读回的);不给则新建一个带 system 提示的空会话。
        """
        self.agent = agent
        if session is not None:
            self.session = session  # 企业级:从存盘的会话继续
        else:
            messages: list[Message] = []
            if agent.system:
                messages.append(Message(role="system", content=agent.system))
            self.session = Session(messages=messages)

    def _prepare_turn(self, prompt: str) -> None:
        """开始新一轮:追加用户消息,并把【单轮运行态】复位(历史与系统提示保留)。

        为什么要复位:Session 里既有「跨轮要留的」(messages),也有「每轮独立的」
        (stage/done/iteration/熔断计数/usage)。新一轮必须从 CALL_MODEL 重新开始、
        熔断预算重新计 —— 否则上一轮的 done=True / iteration 会让这一轮立刻停。
        """
        self.session.messages.append(Message(role="user", content=prompt))
        self.session.stage = Stage.CALL_MODEL
        self.session.done = False
        self.session.stop_reason = None
        self.session.iteration = 0
        self.session.repeat_count = 0
        self.session.last_signature = None
        self.session.pending_tool_calls = []
        self.session.usage = Usage()  # 每轮独立的熔断 token 预算

    async def asend(self, prompt: str) -> RunResult:
        """异步发一轮消息,返回本轮结果;会话历史已更新到 self.session。"""
        self._prepare_turn(prompt)
        result = await _arun(
            self.session,
            self.agent._resolve_provider(),
            self.agent.registry,
            self.agent.config,
            compaction=self.agent.compaction,
            middlewares=self.agent.middlewares,
        )
        self.session = result.session  # 把推进后的会话(含本轮新消息)存回
        return result

    def send(self, prompt: str) -> RunResult:
        """同步发一轮消息(= asyncio.run(asend));已在事件循环里会报错引导用 asend。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asend(prompt))
        raise RuntimeError(
            "检测到当前已在事件循环中(如 Jupyter / Web 环境),请改用 `await chat.asend(...)`。"
        )

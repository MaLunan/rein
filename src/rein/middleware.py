"""中间件引擎(M4)—— 与可恢复 loop 兼容的扩展机制(机制=核心,实例=扩展)。

主心智只有一个:【洋葱式中间件】。一个中间件长这样:

    async def my_mw(ctx, call_next):
        # before:可读改 ctx.session.messages、可短路(不调 call_next)
        ctx = await call_next(ctx)      # 调下一层,最内层是真正的 step
        # after:可读 ctx.steps / ctx.interrupt
        return ctx

硬约束(对应 M4 注意点 1,保证兼容 M2 的中断/恢复):
- 中间件环绕的是【单步 step】,不是整个 loop —— 每个 stage 都重新过一遍栈。
- 中间件栈【无状态】:每步重建,要持有状态就放进 Session(否则中断/恢复就丢了)。
- 中断点仍只在工具执行边界,中间件不得在中途「暂停」。

短路 = 不调 call_next 直接 return:跳过这一步的 step(例:权限拒绝时直接回填)。
统一 try/except = 把 call_next 包在 try 里。
"""

from collections.abc import Awaitable, Callable

from rein.config import LoopConfig
from rein.providers.base import Provider
from rein.result import Interrupt, Step
from rein.runtime.base import Runtime
from rein.session import Session, Stage
from rein.tools import ToolRegistry


class StepContext:
    """一次「单步推进」的上下文,在中间件洋葱里层层传递(运行期对象,不序列化)。"""

    def __init__(
        self,
        session: Session,
        stage: Stage,
        config: LoopConfig,
        provider: Provider,
        registry: ToolRegistry,
        runtime: Runtime,
    ):
        # —— 输入(中间件可读;session.messages 可改)——
        self.session = session
        self.stage = stage
        self.config = config
        self.provider = provider
        self.registry = registry
        self.runtime = runtime
        # —— 输出(step 执行后由内核填充,after 阶段中间件可读)——
        self.steps: list[Step] = []
        self.interrupt: Interrupt | None = None


# 中间件类型:接收 (ctx, call_next),返回推进后的 ctx
Middleware = Callable[
    ["StepContext", Callable[["StepContext"], Awaitable["StepContext"]]], Awaitable["StepContext"]
]

# 最内层核心(真正执行 step)的类型
CoreStep = Callable[["StepContext"], Awaitable["StepContext"]]


def _wrap(mw: Middleware, next_handler: CoreStep) -> CoreStep:
    """把一个中间件包在 next_handler 外面,形成洋葱的一层。"""

    async def wrapped(ctx: StepContext) -> StepContext:
        return await mw(ctx, next_handler)

    return wrapped


async def dispatch(
    middlewares: list[Middleware],
    ctx: StepContext,
    core: CoreStep,
) -> StepContext:
    """把 middlewares 组合成洋葱(列表第 0 个在最外层),最内层是 core,执行并返回 ctx。

    无中间件时直接执行 core —— 等价于「裸 step」,所以加中间件机制不改变默认行为。
    """
    handler = core
    for mw in reversed(middlewares):
        handler = _wrap(mw, handler)
    return await handler(ctx)


async def permission_middleware(ctx: StepContext, call_next: CoreStep) -> StepContext:
    """内置权限中间件(M4):工具执行前统一处理 `ask`(产中断态并短路)。

    loop 总把它放在洋葱【最内层】(紧贴 step),于是 `loop.step` 里不再写任何权限特例 ——
    这就是「权限即钩子」。分工:
    - ask  → 在此产出 need_approval 中断并短路(不执行工具),等待 resume。
    - allow→ 放行(call_next → step 执行工具)。
    - deny → 放行到 step,由 Runtime 在执行层把每个工具回填成 is_error(执行层职责)。
    """
    if ctx.stage == Stage.RUN_TOOLS and ctx.session.pending_tool_calls:
        if ctx.config.permission == "ask":
            names = ", ".join(tc.name for tc in ctx.session.pending_tool_calls)
            ctx.interrupt = Interrupt(
                type="need_approval",
                tool_call=ctx.session.pending_tool_calls[0],
                message=f"待批准执行工具:{names}。请用 resume(approve=True) 批准或 resume(approve=False) 拒绝。",
            )
            return ctx  # 短路:不调 call_next,不执行工具
    return await call_next(ctx)

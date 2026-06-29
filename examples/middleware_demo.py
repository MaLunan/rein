"""middleware_demo —— 中间件 / 钩子 / 事件演示,无需 key、不联网(M4)。

展示同一套「洋葱中间件」机制的三种用法:
- @agent.middleware:完整洋葱(可短路、可统一 try/except)
- @agent.before_tool / @agent.after_model:钩子语法糖
- agent.on("step"):只读事件观测

运行:
    python examples/middleware_demo.py
"""

from rein import Agent, MockProvider, ToolCall


def main() -> None:
    agent = Agent(
        provider=MockProvider(
            [[ToolCall(id="1", name="add", arguments={"a": 3, "b": 4})], "3 加 4 等于 7。"]
        ),
        system="你是个会用工具的助手。",
    )

    @agent.tool
    def add(a: int, b: int) -> int:
        "计算两个整数之和"
        return a + b

    # 1) 完整中间件:给每一步计时并打印
    import time

    @agent.middleware
    async def timing(ctx, call_next):
        t0 = time.monotonic()
        ctx = await call_next(ctx)
        dt = (time.monotonic() - t0) * 1000
        print(f"[middleware] {ctx.stage.value} 耗时 {dt:.2f}ms")
        return ctx

    # 2) 钩子语法糖:工具执行前打印将要调用什么
    @agent.before_tool
    async def announce(ctx):
        names = ", ".join(tc.name for tc in ctx.session.pending_tool_calls)
        print(f"[before_tool] 即将执行:{names}")

    # 3) 只读事件:观测每一步
    agent.on("step", lambda ctx: print(f"[event] 走过 {ctx.stage.value}"))

    print("=== 运行 ===")
    result = agent.run("帮我算 3 加 4")
    print("\n=== 答案 ===")
    print(result)


if __name__ == "__main__":
    main()

"""mock_demo —— 不联网、无需 API key 的完整多轮 loop 演示。

用 MockProvider 预设一段「剧本」,演示一个真实 agent 的典型回合:
    模型决定调工具 → 框架执行工具 → 结果回填 → 模型据此给出最终答案。

运行:
    python examples/mock_demo.py
"""

from rein import Agent, MockProvider, ToolCall


def main() -> None:
    # 预设剧本:第 1 次模型「要调 add(3, 4)」,第 2 次模型「给出最终回答」。
    script = [
        [ToolCall(id="call_1", name="add", arguments={"a": 3, "b": 4})],
        "3 加 4 等于 7。",
    ]
    agent = Agent(provider=MockProvider(script), system="你是一个会用工具的助手。")

    @agent.tool
    def add(a: int, b: int) -> int:
        "计算两个整数之和"
        return a + b

    result = agent.run("帮我算一下 3 加 4")

    # RunResult:print 直接看答案(渐进式暴露)
    print("=== 最终答案 ===")
    print(result)  # 触发 __str__ → 输出 output

    # 想看过程就取 .steps / .usage / .stop_reason
    print("\n=== 运行流水账 ===")
    for s in result.steps:
        line = f"[{s.index}] {s.kind}: {s.summary}"
        if s.kind == "tool":
            line += f"  (is_error={s.is_error})"
        print(line)

    print("\n=== 统计 ===")
    print(f"stop_reason = {result.stop_reason}")
    print(f"usage = 输入 {result.usage.input_tokens} / 输出 {result.usage.output_tokens} token")
    print(f"消息条数 = {len(result.session.messages)}")


if __name__ == "__main__":
    main()

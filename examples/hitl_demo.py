"""hitl_demo —— 人工审批(HITL)演示,无需 key、不联网(M2)。

演示 permission="ask" 时:危险工具在执行前暂停 → 产出中断态 → 人决定批准/拒绝 →
resume 续跑。展示「拒绝 → 模型自愈改走安全做法」这条关键路径。

运行:
    python examples/hitl_demo.py
"""

from rein import Agent, LoopConfig, MockProvider, ToolCall


def main() -> None:
    # 预设剧本:模型先想调 delete_file(危险),被拒后改口给安全回答。
    script = [
        [ToolCall(id="1", name="delete_file", arguments={"path": "/etc/passwd"})],
        "好的,我不删除该文件,改为列出它的信息供你确认。",
    ]
    agent = Agent(
        provider=MockProvider(script),
        config=LoopConfig(permission="ask"),  # 关键:工具执行前要人批准
    )

    @agent.tool
    def delete_file(path: str) -> str:
        "删除一个文件(危险操作)"
        return f"已删除 {path}"

    # 第一步:run 到工具边界就暂停
    result = agent.run("帮我清理 /etc/passwd")
    print("=== 第 1 步:run ===")
    print("status     :", result.status)  # interrupted
    print("中断类型   :", result.interrupt.type)  # need_approval
    print("待批准     :", result.interrupt.message)

    # 此刻 result.session 可以存盘(FileSessionStore),改天再 resume —— 这里直接拒绝
    print("\n=== 第 2 步:人工拒绝 → resume(approve=False) ===")
    result = agent.resume(result.session, approve=False)
    print("status     :", result.status)  # done
    print("模型最终答案:", result.output)  # 模型读到拒绝后自愈

    # 流水账
    print("\n=== 运行流水账 ===")
    for s in result.steps:
        print(f"[{s.index}] {s.kind}: {s.summary}")


if __name__ == "__main__":
    main()

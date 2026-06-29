"""把一个 Rein agent 暴露成 A2A 服务,让别的 agent 来发现 + 调用(无 key、不联网)。

先用 A2AServer 的纯逻辑演示 Agent Card / Task / 流式;要起真实 HTTP 服务见文件底部。

运行:
    python examples/a2a_server.py
"""

import asyncio

from rein import A2AServer, Agent, MockProvider

# 真实用:Agent("anthropic/claude-opus-4-8", system="...")
agent = Agent(
    provider=MockProvider(["已收到 A2A 调用,分析完毕。", "流式逐字回答"]), system="数据分析助手"
)
server = A2AServer(agent, name="数据分析助手", description="提供数据分析能力")


def _req(text, method="message/send"):
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": text}]}},
    }


def main() -> None:
    # ① 发现:Agent Card(注意 capabilities.streaming = True)
    card = server.agent_card()
    print("=== ① Agent Card ===")
    print("name:", card["name"], "| streaming:", card["capabilities"]["streaming"])

    # ② 调用:message/send → 返回一个 Task(状态机:submitted→working→completed)
    task = server.handle_rpc(_req("帮我分析销售数据"))["result"]
    print("\n=== ② message/send → Task ===")
    print("task.state:", task["status"]["state"])
    print("结果:", task["status"]["message"]["parts"][0]["text"])

    # ③ 查询:tasks/get
    got = server.handle_rpc(
        {"jsonrpc": "2.0", "id": "2", "method": "tasks/get", "params": {"id": task["id"]}}
    )["result"]
    print("\n=== ③ tasks/get ===")
    print("state:", got["status"]["state"])

    # ④ 流式:message/stream → SSE 事件(working → 逐字 artifact → completed)
    print("\n=== ④ message/stream(SSE 事件)===")

    async def collect():
        return [e async for e in server.stream_events(_req("流式回答", "message/stream"))]

    for e in asyncio.run(collect()):
        if e["kind"] == "status-update":
            print("  [status]", e["status"]["state"])
        else:
            print("  [artifact]", e["artifact"]["parts"][0]["text"])


if __name__ == "__main__":
    main()

    # ── 起真实 HTTP 服务(取消注释)──
    # from rein import serve_a2a
    # serve_a2a(
    #     agent, name="数据分析助手", port=8000,
    #     auth_token="my-secret",                 # 可选:Bearer 认证
    #     # store=FileSessionStore("./sessions"), # 可选:contextId 多轮记忆
    # )
    # 别的 agent:
    #   GET  /.well-known/agent.json                       发现
    #   POST /  message/send | tasks/get | tasks/cancel    JSON-RPC
    #   POST /  message/stream                             SSE 流式

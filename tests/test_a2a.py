"""A2A 服务端测试:Agent Card / Task 状态机 / streaming / 认证 / 多轮(纯协议逻辑)。"""

import asyncio

from rein import A2AServer, Agent, MemorySessionStore, MockProvider


def _send(
    text: str, context_id: str | None = None, rpc_id: str = "1", method: str = "message/send"
) -> dict:
    msg = {
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
        "messageId": "m",
        "kind": "message",
    }
    if context_id:
        msg["contextId"] = context_id
    return {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": {"message": msg}}


# ---- Agent Card ----


def test_agent_card_声明streaming():
    srv = A2AServer(
        Agent(provider=MockProvider(["hi"]), system="你是助手"), name="测试", url="http://x:8000"
    )
    card = srv.agent_card()
    assert card["name"] == "测试"
    assert card["capabilities"]["streaming"] is True
    assert card["protocolVersion"]
    assert "securitySchemes" not in card  # 没设 token 就没有


def test_agent_card_带认证有securitySchemes():
    srv = A2AServer(Agent(provider=MockProvider(["hi"])), auth_token="secret")
    card = srv.agent_card()
    assert "securitySchemes" in card
    assert card["security"] == [{"bearerAuth": []}]


# ---- Task 状态机 ----


def test_message_send_返回completed_task():
    srv = A2AServer(Agent(provider=MockProvider(["分析完毕"])))
    resp = srv.handle_rpc(_send("帮我分析"))
    task = resp["result"]
    assert task["kind"] == "task"
    assert task["status"]["state"] == "completed"
    assert task["status"]["message"]["parts"][0]["text"] == "分析完毕"
    assert task["artifacts"][0]["parts"][0]["text"] == "分析完毕"
    # history:用户消息 + agent 回复
    roles = [m["role"] for m in task["history"]]
    assert roles == ["user", "agent"]


def test_tasks_get():
    srv = A2AServer(Agent(provider=MockProvider(["ok"])))
    task_id = srv.handle_rpc(_send("hi"))["result"]["id"]
    got = srv.handle_rpc(
        {"jsonrpc": "2.0", "id": "2", "method": "tasks/get", "params": {"id": task_id}}
    )
    assert got["result"]["id"] == task_id
    assert got["result"]["status"]["state"] == "completed"


def test_tasks_get_不存在报错():
    srv = A2AServer(Agent(provider=MockProvider(["ok"])))
    resp = srv.handle_rpc(
        {"jsonrpc": "2.0", "id": "2", "method": "tasks/get", "params": {"id": "nope"}}
    )
    assert resp["error"]["code"] == -32001


def test_tasks_cancel():
    srv = A2AServer(Agent(provider=MockProvider(["ok"])))
    task_id = srv.handle_rpc(_send("hi"))["result"]["id"]
    resp = srv.handle_rpc(
        {"jsonrpc": "2.0", "id": "3", "method": "tasks/cancel", "params": {"id": task_id}}
    )
    assert resp["result"]["status"]["state"] == "canceled"


def test_未知method报错():
    srv = A2AServer(Agent(provider=MockProvider(["x"])))
    resp = srv.handle_rpc({"jsonrpc": "2.0", "id": "9", "method": "foo/bar", "params": {}})
    assert resp["error"]["code"] == -32601


# ---- 认证 ----


def test_认证_缺token拒绝():
    srv = A2AServer(Agent(provider=MockProvider(["secret data"])), auth_token="s3cr3t")
    resp = srv.handle_rpc(_send("hi"), headers={})  # 没带 Authorization
    assert resp["error"]["code"] == -32001


def test_认证_带正确token放行():
    srv = A2AServer(Agent(provider=MockProvider(["机密结果"])), auth_token="s3cr3t")
    resp = srv.handle_rpc(_send("hi"), headers={"Authorization": "Bearer s3cr3t"})
    assert resp["result"]["status"]["state"] == "completed"


# ---- 多轮(contextId + store)----


def test_contextId多轮():
    agent = Agent(provider=MockProvider(["第一次", "第二次:记得你叫小明"]))
    store = MemorySessionStore()
    srv = A2AServer(agent, store=store)
    srv.handle_rpc(_send("我叫小明", context_id="ctx-1"))
    r2 = srv.handle_rpc(_send("我叫什么?", context_id="ctx-1"))
    assert r2["result"]["status"]["message"]["parts"][0]["text"] == "第二次:记得你叫小明"
    users = [m.content for m in store.load("ctx-1").messages if m.role == "user"]
    assert users == ["我叫小明", "我叫什么?"]


# ---- Streaming(SSE 事件)----


def test_stream_events_产出working到completed():
    agent = Agent(provider=MockProvider(["流式答案"]))
    srv = A2AServer(agent)

    async def collect():
        return [e async for e in srv.stream_events(_send("讲讲", method="message/stream"))]

    events = asyncio.run(collect())
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "status-update" and events[0]["status"]["state"] == "working"
    assert kinds[-1] == "status-update" and events[-1]["status"]["state"] == "completed"
    assert events[-1]["final"] is True
    # 中间有 artifact-update,拼起来是完整答案
    text = "".join(
        p["text"] for e in events if e["kind"] == "artifact-update" for p in e["artifact"]["parts"]
    )
    assert text == "流式答案"

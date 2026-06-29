"""A2A(Agent2Agent)服务端 —— 把一个 Rein Agent 暴露成 A2A 协议的 HTTP 服务。

一行起服务,让任何支持 A2A 的 agent 都能【发现】并【调用】你的 agent:

    from rein import Agent, serve_a2a
    agent = Agent("anthropic/...", system="你是数据分析助手")
    serve_a2a(agent, name="数据分析助手", port=8000)

端点:
- GET  /.well-known/agent.json   →  Agent Card(发现)
- POST /  (JSON-RPC)             →  message/send · message/stream · tasks/get · tasks/cancel

实现的 A2A 能力:
- **Task 状态机**:message/send 返回 Task(submitted→working→completed);tasks/get 查询;tasks/cancel 取消。
- **Streaming(SSE)**:message/stream 用 Server-Sent Events 流式吐 status-update / artifact-update(基于 agent.astream)。
- **认证**:可选 Bearer token —— Agent Card 声明 securitySchemes,请求校验 Authorization 头。
- **多轮**:可选 store + A2A contextId(非流式 message/send)。

设计:核心逻辑(agent_card / handle_rpc / stream_events)是纯方法,不绑 HTTP 框架,可单测、可嵌 FastAPI;
HTTP 壳用标准库 http.server(零额外依赖)。

未内置(可基于本类扩展):push notifications、artifact 的非文本类型、task 的 input-required 往返、resubscribe。
"""

import asyncio
import json
import uuid

A2A_PROTOCOL_VERSION = "0.2.0"


class A2AServer:
    """把一个 Rein Agent 适配成 A2A 协议(纯协议逻辑,不绑 HTTP 框架)。"""

    def __init__(
        self,
        agent,
        *,
        name: str = "Rein Agent",
        description: str | None = None,
        url: str = "http://localhost:8000",
        version: str = "1.0.0",
        store=None,
        auth_token: str | None = None,
    ):
        """
        Args:
            agent:       要暴露的 Rein Agent。
            name:        对外展示名(进 Agent Card)。
            description: 能力描述;默认取 agent 的 system 提示。
            url:         本服务对外地址(进 Agent Card)。
            version:     agent 版本号。
            store:       可选 SessionStore;给定后用 A2A 的 contextId 做跨调用多轮(非流式)。
            auth_token:  可选 Bearer token;给定后请求需带 `Authorization: Bearer <token>`。
        """
        self.agent = agent
        self.name = name
        self.description = description or (getattr(agent, "system", None) or "A Rein agent")
        self.url = url
        self.version = version
        self.store = store
        self.auth_token = auth_token
        self._tasks: dict[str, dict] = {}  # 内存 task 存储(生产可换持久化)

    # ---- Agent Card(发现)----

    def agent_card(self) -> dict:
        """A2A Agent Card(放 /.well-known/agent.json),描述本 agent。"""
        card = {
            "protocolVersion": A2A_PROTOCOL_VERSION,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True,
            },
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [
                {"id": "chat", "name": "chat", "description": self.description, "tags": ["chat"]}
            ],
        }
        if self.auth_token:
            card["securitySchemes"] = {"bearerAuth": {"type": "http", "scheme": "bearer"}}
            card["security"] = [{"bearerAuth": []}]
        return card

    # ---- 认证 ----

    def authorize(self, headers) -> bool:
        """校验请求是否带了正确的 Bearer token(没设 auth_token 则永远放行)。"""
        if not self.auth_token:
            return True
        got = headers.get("Authorization") or headers.get("authorization") or ""
        return got == f"Bearer {self.auth_token}"

    # ---- JSON-RPC 处理(同步:message/send · tasks/get · tasks/cancel)----

    def handle_rpc(self, request: dict, *, headers=None) -> dict:
        """处理一个 A2A JSON-RPC 请求,返回响应 dict(纯逻辑,可单测)。"""
        if not self.authorize(headers or {}):
            return self._err(request, -32001, "Unauthorized")
        method = request.get("method")
        if method == "message/send":
            return self._message_send(request)
        if method == "tasks/get":
            return self._tasks_get(request)
        if method == "tasks/cancel":
            return self._tasks_cancel(request)
        # message/stream 走 SSE 通道(见 stream_events),不在这里
        return self._err(request, -32601, f"Method not found: {method}")

    def _message_send(self, request: dict) -> dict:
        params = request.get("params") or {}
        message = params.get("message") or {}
        text = self._extract_text(message)
        context_id = message.get("contextId")

        task = self._new_task(context_id, message)
        task["status"] = {"state": "working"}
        output = self._run(text, task["contextId"])
        agent_msg = self._agent_message(output, task["contextId"], task["id"])
        task["history"].append(agent_msg)
        task["artifacts"].append(
            {
                "artifactId": uuid.uuid4().hex,
                "name": "response",
                "parts": [{"kind": "text", "text": output}],
            }
        )
        task["status"] = {"state": "completed", "message": agent_msg}
        self._tasks[task["id"]] = task
        return self._ok(request, task)

    def _tasks_get(self, request: dict) -> dict:
        task_id = (request.get("params") or {}).get("id")
        task = self._tasks.get(task_id)  # type: ignore[arg-type]
        if task is None:
            return self._err(request, -32001, f"Task not found: {task_id}")
        return self._ok(request, task)

    def _tasks_cancel(self, request: dict) -> dict:
        task_id = (request.get("params") or {}).get("id")
        task = self._tasks.get(task_id)  # type: ignore[arg-type]
        if task is None:
            return self._err(request, -32001, f"Task not found: {task_id}")
        task["status"] = {"state": "canceled"}
        return self._ok(request, task)

    # ---- Streaming(message/stream → SSE 事件)----

    async def stream_events(self, request: dict):
        """异步产出 A2A SSE 事件(基于 agent.astream)。

        依次:status-update(working) → 若干 artifact-update(文本增量) → status-update(completed,final)。
        注:流式为单轮(不走 store 多轮);需要多轮+记忆请用非流式 message/send。
        """
        params = request.get("params") or {}
        message = params.get("message") or {}
        text = self._extract_text(message)
        context_id = message.get("contextId")

        task = self._new_task(context_id, message)
        self._tasks[task["id"]] = task
        tid, cid = task["id"], task["contextId"]
        artifact_id = uuid.uuid4().hex

        task["status"] = {"state": "working"}
        yield {
            "kind": "status-update",
            "taskId": tid,
            "contextId": cid,
            "status": {"state": "working"},
            "final": False,
        }

        chunks: list[str] = []
        async for chunk in self.agent.astream(text):
            if chunk.type == "text" and chunk.text:
                chunks.append(chunk.text)
                yield {
                    "kind": "artifact-update",
                    "taskId": tid,
                    "contextId": cid,
                    "artifact": {
                        "artifactId": artifact_id,
                        "parts": [{"kind": "text", "text": chunk.text}],
                    },
                    "append": True,
                    "lastChunk": False,
                }

        output = "".join(chunks)
        task["artifacts"] = [
            {
                "artifactId": artifact_id,
                "name": "response",
                "parts": [{"kind": "text", "text": output}],
            }
        ]
        task["history"].append(self._agent_message(output, cid, tid))
        task["status"] = {"state": "completed"}
        yield {
            "kind": "status-update",
            "taskId": tid,
            "contextId": cid,
            "status": {"state": "completed"},
            "final": True,
        }

    # ---- 内部 helpers ----

    def _run(self, text: str, context_id: str | None) -> str:
        if self.store is not None and context_id:
            chat = self.agent.chat(session=self.store.load(context_id))
            result = chat.send(text)
            self.store.save(context_id, chat.session)
            return result.output or ""
        return self.agent.run(text).output or ""

    def _new_task(self, context_id: str | None, user_message: dict | None) -> dict:
        return {
            "id": uuid.uuid4().hex,
            "contextId": context_id or uuid.uuid4().hex,
            "kind": "task",
            "status": {"state": "submitted"},
            "history": [user_message] if user_message else [],
            "artifacts": [],
        }

    def _agent_message(self, text: str, context_id: str, task_id: str) -> dict:
        return {
            "kind": "message",
            "role": "agent",
            "messageId": uuid.uuid4().hex,
            "parts": [{"kind": "text", "text": text}],
            "contextId": context_id,
            "taskId": task_id,
        }

    @staticmethod
    def _ok(request: dict, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}

    @staticmethod
    def _err(request: dict, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": code, "message": message},
        }

    @staticmethod
    def _extract_text(message: dict) -> str:
        parts = message.get("parts") or []
        return "".join(p.get("text", "") for p in parts if p.get("kind") == "text" or "text" in p)


def serve_a2a(
    agent,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    name: str = "Rein Agent",
    description: str | None = None,
    version: str = "1.0.0",
    store=None,
    auth_token: str | None = None,
) -> None:
    """一行起一个 A2A HTTP 服务(标准库实现,阻塞运行,Ctrl-C 退出)。

    端点:
    - GET  /.well-known/agent.json           Agent Card(发现)
    - POST /  message/send / tasks/get / tasks/cancel   →  JSON-RPC 响应
    - POST /  message/stream                 →  text/event-stream(SSE)
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    url = f"http://{host}:{port}"
    server = A2AServer(
        agent,
        name=name,
        description=description,
        url=url,
        version=version,
        store=store,
        auth_token=auth_token,
    )

    class _Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
                self._json(200, server.agent_card())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._json(400, {"error": "invalid json"})
                return
            headers = {"Authorization": self.headers.get("Authorization", "")}
            if not server.authorize(headers):
                self._json(200, server._err(request, -32001, "Unauthorized"))
                return
            if request.get("method") == "message/stream":
                self._sse(request)
            else:
                self._json(200, server.handle_rpc(request, headers=headers))

        def _sse(self, request: dict) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            rpc_id = request.get("id")

            async def pump():
                async for event in server.stream_events(request):
                    payload = json.dumps(
                        {"jsonrpc": "2.0", "id": rpc_id, "result": event}, ensure_ascii=False
                    )
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()

            asyncio.run(pump())

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"A2A 服务已启动: {url}")
    print(f"  Agent Card: {url}/.well-known/agent.json")
    print(f"  能力: streaming(SSE) · task 状态机 · {'Bearer 认证' if auth_token else '无认证'}")
    print("  Ctrl-C 退出")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        print("\nA2A 服务已停止")

"""enterprise_chat —— 企业级无状态多轮会话模式(无 key、不联网)。

要点:Web 服务是无状态的,会话状态不能放内存。正确姿势 =
  Agent(无状态蓝图,一个实例并发服务所有用户) + Session(可序列化状态) + SessionStore。

每个请求:按 conversation_id 从 store 读回会话 → agent.chat(session=...) 继续一轮 → 存回。
这就是 `agent.chat(session=...)` 这个便捷方法的用武之地。

运行:
    python examples/enterprise_chat.py
"""

from rein import Agent, MemorySessionStore, MockProvider

# 一个 Agent 实例,并发服务所有用户(无状态蓝图)
agent = Agent(
    provider=MockProvider(["你好,小明!", "你叫小明。", "新会话:你好陌生人。"]),
    system="你是企业客服助手。",
)

# 生产换成 RedisSessionStore(见文件底部);这里用内存模拟
store = MemorySessionStore()


def handle_message(conversation_id: str, user_msg: str) -> str:
    """一个无状态的消息处理函数(模拟 Web handler):状态全程从 store 进出。"""
    chat = agent.chat(session=store.load(conversation_id))  # None → 自动新建会话
    result = chat.send(user_msg)
    store.save(conversation_id, chat.session)  # 写回(redis/db)
    return result.output


if __name__ == "__main__":
    # 模拟同一用户的两轮(中间可以是不同请求/不同进程)
    print("u42 轮1:", handle_message("u42", "我叫小明"))
    print("u42 轮2:", handle_message("u42", "我叫什么?"))  # 记得上一轮
    # 另一个用户,独立会话
    print("u99 轮1:", handle_message("u99", "在吗"))


# ─────────────────────────────────────────────────────────────
# 生产用 Redis 的 SessionStore 范例(需 pip install redis)。
# 框架只内置 Memory/File store;别的后端实现 save/load 两个方法即可(鸭子类型)。
#
#   from rein import Session
#
#   class RedisSessionStore:
#       def __init__(self, client, prefix="rein:sess:", ttl=None):
#           self.client, self.prefix, self.ttl = client, prefix, ttl
#       def save(self, id: str, session: Session) -> None:
#           self.client.set(self.prefix + id, session.model_dump_json(), ex=self.ttl)
#       def load(self, id: str):
#           raw = self.client.get(self.prefix + id)
#           return Session.model_validate_json(raw) if raw else None
#
#   # store = RedisSessionStore(redis.Redis())
# ─────────────────────────────────────────────────────────────

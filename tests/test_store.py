"""SessionStore 测试(M2 批2):内存/文件往返 + 与 resume 的端到端续跑。"""

from rein.agent import Agent
from rein.config import LoopConfig
from rein.ir import Message, ToolCall
from rein.providers import MockProvider
from rein.session import Session, Stage
from rein.store import FileSessionStore, MemorySessionStore, SessionStore


def _sample_session() -> Session:
    return Session(
        messages=[Message(role="user", content="你好")],
        stage=Stage.RUN_TOOLS,
        pending_tool_calls=[ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})],
    )


def test_memory_store_往返():
    store = MemorySessionStore()
    assert store.load("x") is None  # 不存在 → None

    s = _sample_session()
    store.save("x", s)
    back = store.load("x")
    assert back is not None
    assert back.stage == Stage.RUN_TOOLS
    assert back.pending_tool_calls[0].name == "add"
    assert "x" in store

    store.delete("x")
    assert store.load("x") is None


def test_file_store_往返(tmp_path):
    store = FileSessionStore(tmp_path / "sessions")
    assert store.load("job1") is None

    store.save("job1", _sample_session())
    assert (tmp_path / "sessions" / "job1.json").exists()

    back = store.load("job1")
    assert back.pending_tool_calls[0].arguments == {"a": 1, "b": 2}

    store.delete("job1")
    assert store.load("job1") is None


def test_file_store_拒绝非法id(tmp_path):
    store = FileSessionStore(tmp_path)
    import pytest

    with pytest.raises(ValueError):
        store.save("../escape", _sample_session())


def test_两种store都满足Protocol():
    assert isinstance(MemorySessionStore(), SessionStore)
    assert isinstance(FileSessionStore.__new__(FileSessionStore), SessionStore)


def test_端到端_存盘读盘后resume到done(tmp_path):
    """run→ask中断→存文件→(模拟新进程)读文件→resume→done。"""
    log: list = []

    def build_agent() -> Agent:
        agent = Agent(
            provider=MockProvider(
                [[ToolCall(id="1", name="add", arguments={"a": 3, "b": 4})], "等于 7"]
            ),
            config=LoopConfig(permission="ask"),
        )

        @agent.tool
        def add(a: int, b: int) -> int:
            "求和"
            log.append((a, b))
            return a + b

        return agent

    agent = build_agent()
    store = FileSessionStore(tmp_path / "s")

    r = agent.run("帮我加")
    assert r.status == "interrupted"
    store.save("task", r.session)  # 存盘

    loaded = store.load("task")  # 读盘
    r2 = agent.resume(loaded, approve=True)  # 续跑
    assert r2.status == "done"
    assert r2.output == "等于 7"
    assert log == [(3, 4)]

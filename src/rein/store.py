"""SessionStore —— 会话状态的持久化(M2)。

配合「可序列化 Session + 单步状态机」,持久化变得很朴素:存的就是 Session 的 JSON,
读出来就能交给 resume 从断点续跑。所以本模块只定义一个最小 Protocol +
两个参考实现(内存 / 文件,都只用标准库,不加依赖)。

redis / 数据库 / 对象存储等后端是用户的事:自己实现 save/load 两个方法即可
(鸭子类型,无需继承)。这就是「持久化不绑后端」。
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from rein.session import Session


@runtime_checkable
class SessionStore(Protocol):
    """会话存储的统一接口:存一个 Session、按 id 取回来。"""

    def save(self, id: str, session: Session) -> None:
        """以 id 为键保存一个会话(覆盖同 id)。"""
        ...

    def load(self, id: str) -> Session | None:
        """按 id 取回会话;不存在返回 None。"""
        ...


class MemorySessionStore:
    """进程内字典存储(测试 / 单进程)。

    刻意存「JSON 串」而非 Session 对象本身 —— 这样存取都经过真正的序列化/反序列化,
    语义和文件/数据库一致,避免「内存里能跑、一落盘就出问题」的偏差。
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def save(self, id: str, session: Session) -> None:
        self._data[id] = session.model_dump_json()

    def load(self, id: str) -> Session | None:
        raw = self._data.get(id)
        return Session.model_validate_json(raw) if raw is not None else None

    def delete(self, id: str) -> None:
        """删除一个会话(不存在则忽略)。"""
        self._data.pop(id, None)

    def __contains__(self, id: object) -> bool:
        return id in self._data


class FileSessionStore:
    """把每个会话存成 `{目录}/{id}.json`。无需额外依赖,适合本地长任务续跑。"""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, id: str) -> Path:
        # 简单防御:不允许 id 里带路径分隔符,避免写到目录外
        if "/" in id or "\\" in id or id in ("", ".", ".."):
            raise ValueError(f"非法的会话 id:{id!r}")
        return self.dir / f"{id}.json"

    def save(self, id: str, session: Session) -> None:
        self._path(id).write_text(session.model_dump_json(), encoding="utf-8")

    def load(self, id: str) -> Session | None:
        p = self._path(id)
        if not p.exists():
            return None
        return Session.model_validate_json(p.read_text(encoding="utf-8"))

    def delete(self, id: str) -> None:
        """删除一个会话文件(不存在则忽略)。"""
        self._path(id).unlink(missing_ok=True)

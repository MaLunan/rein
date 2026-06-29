"""tools.py 的测试:schema 自动生成、结果序列化、Tool / ToolRegistry。"""

from typing import Literal

from pydantic import BaseModel

from rein.tools import Tool, ToolRegistry, build_schema, serialize_result

# ---------- build_schema:把函数注解转成「工具说明书」 ----------


def test_schema_基础类型与必填():
    """有注解的参数转对应类型;无默认值的参数进 required。"""

    def f(query: str, limit: int = 10):
        "搜索"

    s = build_schema(f)
    assert s["properties"]["query"] == {"type": "string"}
    assert s["properties"]["limit"] == {"type": "integer"}
    assert s["required"] == ["query"]  # limit 有默认值,不必填


def test_schema_无注解默认string():
    """没写类型注解的参数,降级当 string,不报错。"""

    def f(x):
        pass

    s = build_schema(f)
    assert s["properties"]["x"] == {"type": "string"}


def test_schema_list与可选():
    """list[str] → array;X | None → 取非 None 类型。"""

    def f(tags: list[str], note: str | None = None):
        pass

    s = build_schema(f)
    assert s["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert s["properties"]["note"] == {"type": "string"}
    assert s["required"] == ["tags"]


def test_schema_pydantic参数():
    """pydantic 模型参数 → object。"""

    class Point(BaseModel):
        x: int
        y: int

    def f(p: Point):
        pass

    s = build_schema(f)
    assert s["properties"]["p"]["type"] == "object"


def test_schema_literal字符串枚举():
    """Literal["asc","desc"] → string + enum。"""

    def f(order: Literal["asc", "desc"]):
        pass

    s = build_schema(f)
    assert s["properties"]["order"] == {"type": "string", "enum": ["asc", "desc"]}
    assert s["required"] == ["order"]


def test_schema_literal整数枚举():
    """Literal[1, 2, 3] → integer + enum(不被误判为 boolean)。"""

    def f(level: Literal[1, 2, 3] = 1):
        pass

    s = build_schema(f)
    assert s["properties"]["level"] == {"type": "integer", "enum": [1, 2, 3]}


def test_schema_dict带值类型():
    """dict[str, int] → object + additionalProperties 描述值类型。"""

    def f(scores: dict[str, int]):
        pass

    s = build_schema(f)
    assert s["properties"]["scores"] == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


def test_schema_dict无参数():
    """裸 dict → object(无 additionalProperties)。"""

    def f(meta: dict):
        pass

    s = build_schema(f)
    assert s["properties"]["meta"] == {"type": "object"}


def test_schema_嵌套list的literal():
    """list[Literal[...]] → array,元素是 enum。"""

    def f(tags: list[Literal["a", "b"]]):
        pass

    s = build_schema(f)
    assert s["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string", "enum": ["a", "b"]},
    }


# ---------- serialize_result:把任意返回值转成文本 ----------


def test_序列化_字符串原样():
    assert serialize_result("你好") == "你好"


def test_序列化_dict中文不转义():
    assert serialize_result({"city": "北京"}) == '{"city": "北京"}'


def test_序列化_pydantic模型():
    class M(BaseModel):
        a: int

    assert serialize_result(M(a=1)) == '{"a":1}'


def test_序列化_其它转str():
    assert serialize_result(123) == "123"


# ---------- Tool / ToolRegistry ----------


def test_tool_从函数构造():
    def read_file(path: str) -> str:
        "读取文件"
        return ""

    t = Tool.from_function(read_file)
    assert t.name == "read_file"
    assert t.description == "读取文件"
    assert t.is_async is False
    assert t.spec().parameters["required"] == ["path"]


def test_tool_识别异步函数():
    async def af(x: int):
        pass

    t = Tool.from_function(af)
    assert t.is_async is True


def test_registry_注册查找与说明书():
    def a(x: str):
        pass

    def b(y: int):
        pass

    reg = ToolRegistry()
    reg.add(Tool.from_function(a))
    reg.add(Tool.from_function(b))
    assert len(reg) == 2
    assert "a" in reg
    assert reg.get("a").name == "a"
    assert reg.get("zzz") is None
    assert {spec.name for spec in reg.specs()} == {"a", "b"}

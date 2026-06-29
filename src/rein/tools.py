"""工具系统:把「普通 Python 函数」变成「模型能调用的工具」。

三块核心能力:
1. build_schema():读函数的参数与类型注解,自动生成一份「工具说明书」(JSON Schema)
   给模型看 —— 模型靠它知道这个工具有哪些参数、各是什么类型。
   (思路类似 FastAPI 从函数注解自动生成 API 文档。)
2. serialize_result():工具返回值可能是字符串 / 字典 / 对象……统一转成文本,
   才能喂给模型,并存进可序列化的 Session。
3. Tool / ToolRegistry:Tool 把「函数 + 说明书」打包;ToolRegistry 管理一组工具。

注意:Tool 持有真正的函数对象,是【运行期对象、不可序列化】,它活在 Agent 蓝图里,
不会进入 Session;但它生成的 ToolSpec(说明书)是可序列化的,那个才会发给模型。
"""

from __future__ import annotations

import inspect
import json
import types
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from rein.ir import ToolSpec

# Python 基础类型 → JSON Schema 类型 的对照表
_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _enum_base_type(values: list) -> str:
    """从 Literal 的一组字面值推断 JSON Schema 的基础 type。

    注意:bool 必须先于 int 判断 —— 因为 isinstance(True, int) 为真,
    否则布尔枚举会被误判成 integer。混合类型则降级成 string(最宽松)。
    """
    if values and all(isinstance(v, bool) for v in values):
        return "boolean"
    if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "integer"
    if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return "number"
    return "string"


def _type_to_schema(tp: Any) -> dict:
    """把一个 Python 类型注解转成 JSON Schema 片段。

    支持:基础类型、Literal(→enum)、list[T]、dict[K,V]、pydantic 模型、X | None;
    其余无法识别的类型一律降级成 "string"(保证不报错)。
    """
    # 基础类型:str / int / float / bool / list / dict
    if tp in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[tp]}

    # pydantic 模型参数 → 用它自带的 JSON Schema
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return tp.model_json_schema()

    origin = get_origin(tp)

    # Literal["a", "b"] → {"type": 推断, "enum": [...]}:把可选值告诉模型
    if origin is Literal:
        values = list(get_args(tp))
        return {"type": _enum_base_type(values), "enum": values}

    # X | None / Optional[X] / Union[...]
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])
        return {"type": "string"}  # 多类型 union,降级处理

    # list[T] → array,递归元素类型
    if origin is list:
        args = get_args(tp)
        return {"type": "array", "items": _type_to_schema(args[0]) if args else {}}

    # dict[K, V] → object,用 additionalProperties 递归描述「值」的类型
    if origin is dict:
        args = get_args(tp)
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _type_to_schema(args[1])}
        return {"type": "object"}

    # 实在不认识 → 降级成 string,绝不报错
    return {"type": "string"}


def build_schema(fn: Callable) -> dict:
    """从函数签名 + 类型注解,生成参数的 JSON Schema(即 ToolSpec.parameters)。

    规则:
    - 每个参数按其类型注解转成 schema(无注解默认当 string)。
    - 没有默认值的参数 = 必填(放进 required)。
    - 跳过 self / cls 和 *args / **kwargs。
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}  # 拿不到注解就当都没注解,后面默认 string

    properties: dict[str, dict] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue  # 跳过 *args / **kwargs
        tp = hints.get(pname, str)  # 无注解 → 当 string
        properties[pname] = _type_to_schema(tp)
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def serialize_result(value: Any) -> str:
    """把工具的返回值转成「文本」,以便回填给模型 + 存进 Session。

    规则(对应 DESIGN D15):
    - 字符串        → 原样
    - pydantic 模型  → model_dump_json()
    - dict / list   → json.dumps(中文不转义)
    - 其它任何东西   → str()
    对象本身不会进 Session,只有这里转出来的文本会进。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class Tool:
    """一个工具 = 函数本体 + 它的元信息(名字 / 说明 / 参数 schema / 是否异步)。

    这是运行期对象(持有真正的函数),不可序列化,活在 Agent 蓝图里。
    """

    def __init__(
        self,
        fn: Callable,
        name: str,
        description: str,
        parameters: dict,
        is_async: bool,
    ):
        self.fn = fn
        self.name = name
        self.description = description
        self.parameters = parameters
        self.is_async = is_async

    @classmethod
    def from_function(
        cls, fn: Callable, *, name: str | None = None, description: str | None = None
    ) -> Tool:
        """从普通函数构造 Tool:名字默认取函数名,说明默认取 docstring,
        参数 schema 由 build_schema 自动生成,是否异步自动判断。"""
        return cls(
            fn=fn,
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or ""),
            parameters=build_schema(fn),
            is_async=inspect.iscoroutinefunction(fn),
        )

    def spec(self) -> ToolSpec:
        """生成「给模型看的说明书」(可序列化的 ToolSpec)。"""
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


class ToolRegistry:
    """一组工具的登记册:注册、按名查找、批量生成说明书给模型。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        """注册一个工具(同名会覆盖)。"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """按名字取工具,不存在返回 None。"""
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        """把所有工具的说明书打包成列表,发给模型。"""
        return [t.spec() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

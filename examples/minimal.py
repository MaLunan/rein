"""minimal —— 「5 行」真实示例(需要 API key 才能真跑)。

核心就这 5 行:
    1. agent = Agent("anthropic/claude-opus-4-8")
    2. @agent.tool
    3. def now(): ...
    4. result = agent.run("现在几号?")
    5. print(result)

运行前先配好对应厂商的 key(LiteLLM 寻址,见各厂商文档),例如:
    export ANTHROPIC_API_KEY=sk-...
然后:
    python examples/minimal.py

没装 litellm / 没配 key 时,会得到一个清晰的报错提示 —— 这正是设计意图:
本框架装完(pip install rein-agent)即接入真实大模型——litellm 已是核心依赖。
不联网的完整演示请看 examples/mock_demo.py。
"""

from rein import Agent

agent = Agent("anthropic/claude-opus-4-8")


@agent.tool
def now() -> str:
    "返回当前日期"
    return "2026-06-27"


if __name__ == "__main__":
    result = agent.run("今天几号?用工具查。")
    print(result)

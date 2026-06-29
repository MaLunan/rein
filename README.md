# Rein

> **5 行代码,把任意大模型变成能调工具、自己循环干活的 agent。**

Rein 是一个**极薄但生产级**的单-agent harness(智能体运行时)框架。反 LangChain 式的重抽象,聚焦把"单 agent 的 loop + 工具 + 可控 + 可观测"做到极致。

## 安装

```bash
pip install rein              # 核心(仅 pydantic + anyio)
pip install "rein[litellm]"   # 接入真实大模型(100+ 厂商)
```

## 快速开始

```python
from rein import Agent

agent = Agent(model="anthropic/claude-opus-4-8")

@agent.tool
def read_file(path: str) -> str:
    "读取文件内容"
    return open(path).read()

print(agent.run("读 README 并总结"))
```

## 核心特性

- **多厂商**:wrap LiteLLM,一行切换 100+ 厂商模型 + 自动 fallback
- **可序列化状态机**:Loop 状态全在可序列化 Session 里 → 天生支持 HITL 人工审批、断点续跑
- **熔断四道闸**:轮数 / token / 超时 / 重复检测,防 agent 失控烧钱
- **并发安全**:Agent(无状态蓝图)/ Session(状态)/ Chat(会话句柄)三者分离
- **可观测**:结构化 `RunResult` + 可选 OpenTelemetry 导出
- **可扩展**:中间件 / 钩子 / 事件 / Docker 沙箱 / 插件
- **脚手架**:`rein new` 一键起项目

## 许可证

[Apache-2.0](LICENSE)

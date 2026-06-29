"""项目脚手架(M5)—— 一键生成「极简可跑」的 agent 项目。

设计纪律(对应 M5 注意点):
- 默认极简:不学 Django 生成层层目录;一个 agent 项目通常就一个 main.py。
- 模板是「能跑通的最小例子」,不是占位骨架。
- 生成逻辑是【纯函数】(不依赖 typer),所以可单测;CLI(rein new)只是它的门面。

模板:
- minimal:5 行示例。
- coder: 带 read_file / run_shell 工具雏形 + permission="ask"(危险操作要人批准)。
"""

from pathlib import Path

# ---------- 模板内容(可直接运行的最小例子)----------

_MINIMAL_MAIN = '''\
"""{name} —— 用 rein 写的最小 agent。填好 .env 里的 key 后:python main.py"""

from rein import Agent

agent = Agent("anthropic/claude-opus-4-8")


@agent.tool
def now() -> str:
    "返回当前日期"
    return "2026-06-27"


if __name__ == "__main__":
    print(agent.run("今天几号?用工具查。"))
'''

_CODER_MAIN = '''\
"""{name} —— 一个会读文件、跑命令的 coder agent(危险操作需人工批准)。

填好 .env 里的 key 后:python main.py
注意:permission="ask",执行 run_shell 前会暂停等你批准(命令行里输入 y/N)。
"""

import subprocess
from pathlib import Path

from rein import Agent, LoopConfig

agent = Agent(
    "anthropic/claude-opus-4-8",
    config=LoopConfig(permission="ask"),  # 危险操作先问人
    system="你是一个严谨的编程助手,会用工具读文件、跑命令。",
)


@agent.tool
def read_file(path: str) -> str:
    "读取一个文本文件的内容"
    return Path(path).read_text(encoding="utf-8")


@agent.tool
def run_shell(command: str) -> str:
    "执行一条 shell 命令并返回输出(危险,需批准)"
    out = subprocess.run(command, shell=True, capture_output=True, text=True)
    return out.stdout + out.stderr


if __name__ == "__main__":
    # run_interactive:遇到需要批准的工具会在命令行问 y/N
    result = agent.run_interactive("列出当前目录下的文件")
    print(result)
'''

_ENV_EXAMPLE = """\
# 复制本文件为 .env 并填入你要用的厂商 key(经 LiteLLM 寻址)
# Anthropic:
ANTHROPIC_API_KEY=sk-ant-...
# 或 OpenAI 兼容(如 DeepSeek):
# DEEPSEEK_API_KEY=sk-...
# OPENAI_API_KEY=sk-...
"""

_README = """\
# {name}

用 [rein](https://example.com/rein) 写的 agent 项目(由 `rein new` 生成)。

## 跑起来
1. 安装依赖:`pip install rein[litellm]`
2. 复制 `.env.example` 为 `.env`,填入你的 key
3. `python main.py`

## 可选配置
`rein.toml`(已生成示例)是可选的;rein 以代码配置为主,toml 仅作备选。
"""

_REIN_TOML = """\
# rein.toml —— 可选配置(rein 以代码配置为主,这里仅作示例/备选)
[loop]
max_iterations = 50
max_tokens = 200000
timeout_s = 120
permission = "allow"   # allow / ask / deny
"""

_TEMPLATES = {
    "minimal": _MINIMAL_MAIN,
    "coder": _CODER_MAIN,
}


def available_templates() -> list[str]:
    """列出可用模板名。"""
    return list(_TEMPLATES)


def create_project(name: str, template: str = "minimal", target_dir: str | Path = ".") -> Path:
    """生成一个极简可跑的 agent 项目,返回项目目录路径。

    Args:
        name:       项目名(同时作为目录名)。
        template:   模板:"minimal" 或 "coder"。
        target_dir: 在哪个目录下创建项目(默认当前目录)。

    Returns:
        新建的项目目录 Path。

    Raises:
        ValueError:    模板不存在,或项目名非法。
        FileExistsError:目标目录已存在(不覆盖,保护用户已有内容)。
    """
    if template not in _TEMPLATES:
        raise ValueError(f"未知模板:{template!r}。可用:{', '.join(available_templates())}")
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"非法的项目名:{name!r}")

    project = Path(target_dir) / name
    if project.exists():
        raise FileExistsError(f"目标已存在,拒绝覆盖:{project}")

    project.mkdir(parents=True)
    # 就这几个文件,不生成空目录
    (project / "main.py").write_text(_TEMPLATES[template].format(name=name), encoding="utf-8")
    (project / ".env.example").write_text(_ENV_EXAMPLE, encoding="utf-8")
    (project / "README.md").write_text(_README.format(name=name), encoding="utf-8")
    (project / "rein.toml").write_text(_REIN_TOML, encoding="utf-8")
    return project

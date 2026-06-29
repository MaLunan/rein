"""rein CLI(M5)—— `rein new` / `rein dev`。typer 走 extras(`rein[cli]`)。

本模块在顶层 import typer:它只会在用户运行 `rein` 命令(entry point)或显式 import
rein.cli 时被加载 —— 所以没装 typer 也不影响 `import rein` / 用核心功能。

刻意【不做 `rein run`】:跑项目就是 `python main.py`,不搞多余命令(M5 注意点 2)。
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import typer

from rein.scaffold import available_templates, create_project

app = typer.Typer(
    help="rein —— 极薄的单-agent harness 框架脚手架",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def new(
    name: str = typer.Argument(..., help="项目名(同时作为目录名)"),
    template: str = typer.Option(
        "minimal", "--template", "-t", help=f"模板:{' / '.join(available_templates())}"
    ),
):
    """生成一个极简可跑的 agent 项目(就一个 main.py + .env.example,不堆目录)。"""
    try:
        project = create_project(name, template=template)
    except (ValueError, FileExistsError) as e:
        typer.secho(f"✗ {e}", fg=typer.colors.RED)
        raise typer.Exit(1) from e

    typer.secho(f"✓ 已生成项目:{project}", fg=typer.colors.GREEN)
    typer.echo("下一步:")
    typer.echo(f"  cd {name}")
    typer.echo("  cp .env.example .env    # 填入你的 key")
    typer.echo("  python main.py")


@app.command()
def dev(script: str = typer.Argument("main.py", help="要运行的脚本(默认 main.py)")):
    """开发模式:监听文件变化自动重启(热重载)。

    设了 REIN_DEV=1 给子进程 —— 你可在 main.py 里据此挂追踪(例:
    `if os.getenv("REIN_DEV"): agent.on("step", lambda c: print("step:", c.stage.value))`)。
    """
    path = Path(script)
    if not path.exists():
        typer.secho(f"✗ 找不到 {script}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(f"▶ dev:监听 {script} 变化,Ctrl-C 退出", fg=typer.colors.CYAN)
    _watch_and_run(path)


def _watch_and_run(path: Path) -> None:
    """标准库轮询 mtime:文件一变就重启子进程(简单热重载,不引入 watchfiles)。"""
    proc: subprocess.Popen | None = None
    last_mtime = -1.0
    child_env = {**os.environ, "REIN_DEV": "1"}
    try:
        while True:
            mtime = path.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                if proc and proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                typer.secho(f"↻ 重启 {path.name} ...", fg=typer.colors.YELLOW)
                proc = subprocess.Popen([sys.executable, str(path)], env=child_env)
            time.sleep(0.5)
    except KeyboardInterrupt:
        if proc and proc.poll() is None:
            proc.terminate()
        typer.secho("\n已退出 dev 模式", fg=typer.colors.CYAN)


def main() -> None:
    """console_scripts 入口(pyproject 的 [project.scripts] rein = rein.cli:main)。"""
    app()


if __name__ == "__main__":
    main()

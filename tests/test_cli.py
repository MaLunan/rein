"""CLI 测试(M5)——默认 skip:没装 typer 就跳过(CLI 走 rein[cli] extras)。

开启:`pip install 'rein[cli]'`。核心生成逻辑已在 test_scaffold 充分覆盖,
这里只验证 typer 门面把参数正确接到 create_project 上。
"""

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from rein.cli import app

runner = CliRunner()


def test_new_生成项目(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["new", "myproj"])
    assert result.exit_code == 0
    assert (tmp_path / "myproj" / "main.py").exists()
    assert "已生成项目" in result.stdout


def test_new_coder模板(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["new", "c", "-t", "coder"])
    assert result.exit_code == 0
    assert "run_shell" in (tmp_path / "c" / "main.py").read_text(encoding="utf-8")


def test_new_已存在报错退出码(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["new", "dup"])
    result = runner.invoke(app, ["new", "dup"])
    assert result.exit_code == 1


def test_dev_找不到脚本报错(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["dev", "nope.py"])
    assert result.exit_code == 1

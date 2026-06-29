"""脚手架测试(M5 批1):生成极简项目 / 文件齐全 / main.py 语法正确 / 两模板 / 防呆。"""

import pytest

from rein.scaffold import available_templates, create_project


def test_minimal_生成文件齐全(tmp_path):
    proj = create_project("myagent", template="minimal", target_dir=tmp_path)
    assert proj.name == "myagent"
    # 极简:就这几个文件,不生成空目录
    names = sorted(p.name for p in proj.iterdir())
    assert names == [".env.example", ".gitignore", "README.md", "main.py", "rein.toml"]
    # 安全:.gitignore 必须忽略 .env(防含 key 的 .env 被提交进 git)
    assert ".env" in (proj / ".gitignore").read_text(encoding="utf-8")


def test_生成的main_语法正确可编译(tmp_path):
    proj = create_project("a1", template="minimal", target_dir=tmp_path)
    src = (proj / "main.py").read_text(encoding="utf-8")
    compile(src, "main.py", "exec")  # 编译不报错 = 语法正确
    assert "from rein import Agent" in src
    assert "@agent.tool" in src


def test_coder模板_含工具与ask(tmp_path):
    proj = create_project("c1", template="coder", target_dir=tmp_path)
    src = (proj / "main.py").read_text(encoding="utf-8")
    compile(src, "main.py", "exec")
    assert "read_file" in src and "run_shell" in src
    assert 'permission="ask"' in src
    assert "run_interactive" in src


def test_项目名注入到文件(tmp_path):
    proj = create_project("酷应用", template="minimal", target_dir=tmp_path)
    assert "酷应用" in (proj / "README.md").read_text(encoding="utf-8")


def test_未知模板报错(tmp_path):
    with pytest.raises(ValueError):
        create_project("x", template="不存在", target_dir=tmp_path)


def test_非法项目名报错(tmp_path):
    with pytest.raises(ValueError):
        create_project("a/b", target_dir=tmp_path)


def test_已存在不覆盖(tmp_path):
    create_project("dup", target_dir=tmp_path)
    with pytest.raises(FileExistsError):
        create_project("dup", target_dir=tmp_path)


def test_available_templates():
    assert set(available_templates()) == {"minimal", "coder"}

# 贡献指南

感谢你考虑为 Rein 贡献!

## 开发环境

    cd framework
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,litellm,cli]"

## 提交前检查

    ruff format .     # 格式化
    ruff check .      # lint
    mypy              # 类型检查
    pytest            # 测试

或安装 pre-commit 自动跑:`pre-commit install`。

## 约定

- 代码带**中文注释**。
- 新功能**配测试**。
- 提交信息用约定式前缀(feat / fix / chore / docs / style / test)。

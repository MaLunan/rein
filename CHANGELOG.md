# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.2.0] - 2026-06

### Added
- **结构化日志**(`rein.log` / `enable_logging`):默认安静(NullHandler),一行开启;支持 JSON 输出接 ELK/Loki;每条带 `trace_id`;loop 生命周期埋点(run started / tool done / run finished)。
- **并发安全压测**:100 协程并发、会话状态独立、多线程并发,均验证不串台。
- **极薄度 benchmark** 脚本(`benchmarks/bench.py`)。
- **生产部署实践指南**(密钥 / 日志 / 熔断 / 错误处理 / 并发 / 部署 + 上线清单)。

### Changed
- 成熟度 `Development Status` 由 `3 - Alpha` 升至 `4 - Beta`。

### Security
- `rein new` 生成的项目带上 `.gitignore`,默认忽略 `.env`,防止含 key 的 `.env` 被误提交进 git。
- 日志全程脱敏:不记录 API key、完整对话、工具参数与结果。

## [0.1.0] - 2026-06

### Changed
- **`litellm` 提升为核心依赖**:`pip install rein-agent` 装完即接入真实大模型(100+ 厂商);原先需 `[litellm]` extra。

### Added
- M0–M5 完整框架:Loop 单步状态机、多厂商(LiteLLM)、HITL / 断点续跑、上下文压缩、可观测、中间件 / 钩子 / 插件、脚手架。
- A2A 服务端;企业级无状态多轮 Chat。
- 工程化:Apache-2.0 许可、ruff + mypy + pytest-cov、GitHub Actions CI、py.typed、三分目录(framework / docs / website)。

## [0.0.1] - 2026-06
- 初始版本。

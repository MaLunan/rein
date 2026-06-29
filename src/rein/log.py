"""结构化日志(基于标准库 logging,零额外依赖)。

为什么不引 structlog:守住「极薄」—— 标准库 logging 足够,结构化字段靠 `extra={...}`
传入,再用可选的 JSON formatter 输出成单行 JSON(直接喂给 ELK / Loki / CloudWatch)。

库的最佳实践 ——【默认闭嘴】:
- 所有框架日志都走 `logging.getLogger("rein")` 这一个 logger;
- 默认只挂 NullHandler → 用户不配置就【没有任何输出】,绝不污染用户自己的日志;
- 用户想看日志时,一行 `enable_logging()` 打开即可(生产环境也可以不用它、自己接管这个 logger)。

安全(呼应密钥审计):本模块只提供管道,框架内部打日志时【绝不记录 API key / 完整 messages】,
只记摘要、长度、计数 —— 见各埋点处。
"""

import json as _json
import logging
from typing import Any

# 框架统一 logger:所有模块都 `from rein.log import logger` 使用它。
logger = logging.getLogger("rein")
# 默认挂 NullHandler:不配置就不输出,符合「库不该擅自往 root logger 喷日志」的最佳实践。
logger.addHandler(logging.NullHandler())


class JsonFormatter(logging.Formatter):
    """把一条日志(含通过 extra= 传入的结构化字段)格式化成【单行 JSON】。

    便于日志采集系统按字段检索,例如:
        {"ts": "...", "level": "INFO", "logger": "rein", "msg": "tool done",
         "trace_id": "ab12", "tool": "search", "ok": true, "dur_ms": 42}
    """

    # LogRecord 自带的标准属性;提取用户的 extra 字段时要把这些排除掉。
    _RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # 把 logger.info(..., extra={...}) 里的结构化字段并进来
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith("_"):
                data[k] = v
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return _json.dumps(data, ensure_ascii=False, default=str)


def enable_logging(
    level: "str | int" = "INFO",
    *,
    json: bool = False,
    stream: Any = None,
) -> None:
    """一行打开 Rein 的日志输出(便捷函数)。

    生产环境也可以【不用】它 —— 直接 `logging.getLogger("rein")` 自己加 handler、
    接进你已有的日志体系即可。这个函数只是给「想快速看日志」的人兜底。

    Args:
        level:  日志级别,如 "INFO" / "DEBUG" / logging.WARNING。
        json:   True → 输出单行 JSON(接日志系统);False → 人类可读文本。
        stream: 输出流,默认 stderr。
    """
    handler = logging.StreamHandler(stream)
    if json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] rein: %(message)s")
        )
    logger.addHandler(handler)
    logger.setLevel(level)
    # 不向 root logger 冒泡,避免和用户的 root handler 重复打印。
    logger.propagate = False


def disable_logging() -> None:
    """关掉 enable_logging() 加的输出(移除非 NullHandler 的 handler),恢复默认安静。"""
    for h in list(logger.handlers):
        if not isinstance(h, logging.NullHandler):
            logger.removeHandler(h)
    logger.propagate = True

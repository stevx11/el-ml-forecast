"""Structured logging.

Console format in development (readable), JSON in production (parsable). A
correlation id is bound to a :class:`~contextvars.ContextVar` so that every log
line emitted while handling a request or running an ingestion job carries the
same identifier without having to thread it through every function signature.

Hand-rolled rather than pulled from a library: the requirement is a formatter and
a context variable, which is about forty lines, and an extra dependency here
would buy nothing.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Attributes present on every LogRecord. Anything not in this set was passed by
# the caller via `extra=` and belongs in the structured payload.
_RESERVED_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


def set_correlation_id(value: str | None = None) -> str:
    """Bind a correlation id to the current context and return it."""
    resolved = value or uuid.uuid4().hex[:16]
    _correlation_id.set(resolved)
    return resolved


def get_correlation_id() -> str | None:
    """Return the correlation id bound to the current context, if any."""
    return _correlation_id.get()


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    """Return caller-supplied ``extra=`` fields from a record."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED_ATTRS and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Used in production."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if (correlation_id := get_correlation_id()) is not None:
            payload["correlation_id"] = correlation_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        payload.update(_extras(record))
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single line. Used in development."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        prefix = f"{timestamp} {record.levelname:<8} {record.name:<38}"

        parts = [f"{prefix} {record.getMessage()}"]
        if (correlation_id := get_correlation_id()) is not None:
            parts.append(f"[{correlation_id}]")
        if extras := _extras(record):
            parts.append(" ".join(f"{k}={v}" for k, v in extras.items()))

        line = " ".join(parts)
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(*, level: str = "INFO", fmt: str = "console") -> None:
    """Install the root logging configuration. Idempotent.

    Args:
        level: Minimum level for application logs.
        fmt: ``"console"`` or ``"json"``.
    """
    formatter: logging.Formatter = JsonFormatter() if fmt == "json" else ConsoleFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # These are chatty at INFO and drown out anything useful. SQLAlchemy echo is
    # controlled separately by DATABASE_ECHO.
    for noisy in ("uvicorn.access", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger. Thin wrapper so call sites do not import ``logging``."""
    return logging.getLogger(name)

"""Enterprise-grade structured logging helper with secret redaction and context binding.

Supports:
1. Dual formatting: Colored human-readable for dev; JSON lines for production.
2. Contextual metadata binding: trace_id, session_id, step_name.
3. Automatic PII and Secret Redactor: masks API keys and tokens from all log outputs.
4. @timed_block decorator/context manager for granular latency tracking.
"""

from contextlib import contextmanager
import contextvars
from datetime import datetime, timezone
import json
import logging
import re
import sys
import time
from typing import Any, Callable, Generator

# Context variables for request/run tracing across async tasks
_trace_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_session_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)
_step_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("step", default=None)

# Regular expressions for sensitive tokens and API keys
SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),          # OpenAI / Langfuse secrets
    re.compile(r"AIzaSy[a-zA-Z0-9_-]{33}", re.IGNORECASE),         # Google API Keys
    re.compile(r"gsk_[a-zA-Z0-9_-]{20,}", re.IGNORECASE),          # Groq API Keys
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]+", re.IGNORECASE),       # Bearer tokens
    re.compile(r"password=[\"']?[^\s\"']+[\"']?", re.IGNORECASE), # Passwords
]


def redact_secrets(text: str) -> str:
    """Mask known secret patterns with [REDACTED]."""
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter producing machine-readable structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
            "line": record.lineno,
            "file": record.filename,
        }

        # Inject context variables if available
        trace_id = _trace_id_ctx.get()
        if trace_id:
            log_entry["trace_id"] = trace_id

        session_id = _session_id_ctx.get()
        if session_id:
            log_entry["session_id"] = session_id

        step = _step_ctx.get()
        if step:
            log_entry["step"] = step

        # Inject extra attributes passed to logger calls
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            for k, v in record.extra_fields.items():
                if isinstance(v, str):
                    log_entry[k] = redact_secrets(v)
                else:
                    log_entry[k] = v

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class StandardDevFormatter(logging.Formatter):
    """Human-readable formatter with ANSI colors and contextual tags for local dev."""

    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[41m", # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = redact_secrets(record.getMessage())

        ctx_parts = []
        trace_id = _trace_id_ctx.get()
        if trace_id:
            ctx_parts.append(f"trc={trace_id[:8]}")
        step = _step_ctx.get()
        if step:
            ctx_parts.append(f"step={step}")

        ctx_str = f"[{' '.join(ctx_parts)}] " if ctx_parts else ""

        formatted = (
            f"{color}{timestamp} | {record.levelname:<7}{self.RESET} | "
            f"\033[2m{record.name}:{record.lineno}\033[0m | {ctx_str}{msg}"
        )

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


class ContextBoundLogger(logging.LoggerAdapter):
    """Logger adapter allowing explicit context binding per module or request."""

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        if self.extra:
            extra["extra_fields"] = {**self.extra, **extra.get("extra_fields", {})}
        return msg, kwargs

    def bind(self, **kwargs: Any) -> "ContextBoundLogger":
        """Return a new logger adapter bound with additional context dictionary."""
        new_extra = {**(self.extra or {}), **kwargs}
        return ContextBoundLogger(self.logger, new_extra)


def get_logger(name: str, format_mode: str = "dev") -> ContextBoundLogger:
    """Factory for structured loggers."""
    raw_logger = logging.getLogger(name)
    if not raw_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if format_mode == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(StandardDevFormatter())
        raw_logger.addHandler(handler)
        raw_logger.setLevel(logging.DEBUG)
        raw_logger.propagate = False
    return ContextBoundLogger(raw_logger, {})


def set_log_context(trace_id: str | None = None, session_id: str | None = None, step: str | None = None) -> None:
    """Set global contextvars for current async task."""
    if trace_id is not None:
        _trace_id_ctx.set(trace_id)
    if session_id is not None:
        _session_id_ctx.set(session_id)
    if step is not None:
        _step_ctx.set(step)


def clear_log_context() -> None:
    """Reset tracing context variables."""
    _trace_id_ctx.set(None)
    _session_id_ctx.set(None)
    _step_ctx.set(None)


@contextmanager
def timed_block(name: str, logger: ContextBoundLogger | None = None) -> Generator[dict[str, Any], None, None]:
    """Context manager for measuring and logging execution latency in ms."""
    active_logger = logger or get_logger("src.core.timer")
    start = time.perf_counter()
    metrics: dict[str, Any] = {"name": name, "start_time": start}
    active_logger.debug(f"▶ Starting: {name}")
    try:
        yield metrics
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics["latency_ms"] = round(elapsed_ms, 2)
        active_logger.info(f"✔ Completed: {name} in {metrics['latency_ms']}ms", extra={"extra_fields": metrics})

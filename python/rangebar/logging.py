"""Centralized NDJSON logging configuration for rangebar-py.

Implements GitHub Issue #43: Structured logging for checksum verification
and other observability events.

Logs are stored in the repository tree under `logs/` directory.

Service logging (sidecar, kintsugi, recency-backfill):
    Call ``setup_service_logging("sidecar")`` once at CLI entry point.
    This creates per-service NDJSON files with auto-rotation, intercepts
    stdlib logging, and keeps human-readable stderr for systemd journal.
"""

from __future__ import annotations

import logging as _stdlib_logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Logger

# Logs in repository tree (not platformdirs).
# When installed as a package, __file__ resolves inside site-packages,
# so fall back to RANGEBAR_PROJECT_ROOT env var or cwd.
_file_based_root = Path(__file__).parent.parent.parent
_env_root = os.environ.get("RANGEBAR_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root)
elif (_file_based_root / "pyproject.toml").exists():
    PROJECT_ROOT = _file_based_root  # Editable install / dev checkout
else:
    PROJECT_ROOT = Path.cwd()  # Fallback: systemd WorkingDirectory
LOG_DIR = PROJECT_ROOT / "logs"
NDJSON_FILE = LOG_DIR / "events.jsonl"
CHECKSUM_REGISTRY_FILE = LOG_DIR / "checksum_registry.jsonl"

# Lazy initialization flag
_logger_initialized = False
_logger: Logger | None = None
_service_logging_initialized = False


def _get_context_extra(service_name: str = "rangebar-py") -> dict:
    """Get context fields added to every log entry."""
    import socket

    return {
        "service": service_name,
        "environment": os.environ.get("RANGEBAR_ENV", "development"),
        "git_sha": os.environ.get("RANGEBAR_GIT_SHA", "unknown"),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }


class _InterceptHandler(_stdlib_logging.Handler):
    """Route stdlib logging messages to loguru.

    This ensures library code using ``logging.getLogger(__name__)``
    emits structured NDJSON alongside direct loguru calls.
    """

    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        # Use the bound logger (with service context) if available
        target = _logger
        if target is None:
            from loguru import logger as target  # type: ignore[assignment]

        # Map stdlib level to loguru level
        try:
            level = target.level(record.levelname).name  # type: ignore[union-attr]
        except ValueError:
            level = str(record.levelno)

        # Find the caller frame (skip intercept handler frames)
        frame, depth = _stdlib_logging.currentframe(), 2
        while frame and frame.f_code.co_filename == _stdlib_logging.__file__:
            frame = frame.f_back
            depth += 1

        # Bind stdlib logger name as component for traceability
        target.bind(stdlib_logger=record.name).opt(  # type: ignore[union-attr]
            depth=depth, exception=record.exc_info
        ).log(level, record.getMessage())


def setup_service_logging(
    service_name: str,
    *,
    verbose: bool = False,
) -> Logger:
    """Configure structured NDJSON logging for a long-running service.

    Creates per-service NDJSON log file with auto-rotation, intercepts
    stdlib logging, and keeps human-readable stderr for systemd journal.

    Args:
        service_name: Service identifier (e.g., "sidecar", "kintsugi",
            "recency-backfill"). Used in log filenames and structured fields.
        verbose: Enable DEBUG level (default: INFO).

    Returns:
        Configured loguru Logger instance.
    """
    global _service_logging_initialized, _logger_initialized, _logger

    from loguru import logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Remove default loguru handler
    logger.remove()

    # Bind service context to every entry
    context = _get_context_extra(service_name)
    logger = logger.bind(**context)

    level = "DEBUG" if verbose else "INFO"

    # 1. Per-service NDJSON file sink (auto-rotating)
    service_log = LOG_DIR / f"{service_name}.jsonl"
    logger.add(
        service_log,
        format="{message}",
        serialize=True,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        enqueue=True,
        level="DEBUG",  # Always capture DEBUG in file
    )

    # 2. Shared events.jsonl sink (cross-service correlation)
    logger.add(
        NDJSON_FILE,
        format="{message}",
        serialize=True,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        enqueue=True,
        level="INFO",  # Only INFO+ in shared log
    )

    # 3. Console stderr (human-readable for systemd journal)
    def _console_format(record: dict) -> str:
        svc = record["extra"].get("service", service_name)
        stdlib = record["extra"].get("stdlib_logger")
        source = f"{svc}:{stdlib}" if stdlib else svc
        return (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            + source
            + " | {message}\n"
        )

    logger.add(
        sys.stderr,
        format=_console_format,
        level=level,
        colorize=False,  # No ANSI in journal
    )

    # 4. Intercept stdlib logging → loguru
    _stdlib_logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    _logger = logger
    _logger_initialized = True
    _service_logging_initialized = True

    logger.info(
        "Service logging initialized",
        service=service_name,
        log_file=str(service_log),
        level=level,
    )
    return logger


def get_logger() -> Logger:
    """Get the configured logger instance.

    Lazy initialization to avoid import-time side effects.
    If ``setup_service_logging()`` was already called, returns that logger.
    """
    global _logger_initialized, _logger

    if _logger_initialized and _logger is not None:
        return _logger

    try:
        from loguru import logger
    except ImportError:
        # Fallback to standard logging if loguru not installed
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        )
        # Return a minimal logger-like object
        return logging.getLogger("rangebar")  # type: ignore[return-value]

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add context to every log entry
    context = _get_context_extra()
    logger = logger.bind(**context)

    # NDJSON file sink (in repo tree)
    logger.add(
        NDJSON_FILE,
        format="{message}",
        serialize=True,  # NDJSON output
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        enqueue=True,
        level="DEBUG",
    )

    # Console (human-readable, INFO+)
    console_format = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{extra[component]}</cyan> - <level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level="INFO",
        colorize=True,
        filter=lambda record: "component" in record["extra"],
    )

    _logger = logger
    _logger_initialized = True
    return logger


def log_checksum_event(
    event_type: str,
    symbol: str,
    date: str,
    trace_id: str,
    **kwargs: object,
) -> None:
    """Log a checksum-related event.

    Args:
        event_type: Type of event (checksum_fetch_start, checksum_verify_success, etc.)
        symbol: Trading symbol (e.g., "BTCUSDT")
        date: Date being processed (YYYY-MM-DD)
        trace_id: Correlation ID for the request chain
        **kwargs: Additional event-specific fields
    """
    logger = get_logger()
    logger.bind(
        component="checksum",
        event_type=event_type,
        symbol=symbol,
        date=date,
        trace_id=trace_id,
        **kwargs,
    ).info(f"{event_type}: {symbol} {date}")


def log_download_event(
    event_type: str,
    symbol: str,
    date: str,
    trace_id: str,
    **kwargs: object,
) -> None:
    """Log a download-related event.

    Args:
        event_type: Type of event (download_start, download_complete, etc.)
        symbol: Trading symbol (e.g., "BTCUSDT")
        date: Date being processed (YYYY-MM-DD)
        trace_id: Correlation ID for the request chain
        **kwargs: Additional event-specific fields
    """
    logger = get_logger()
    logger.bind(
        component="download",
        event_type=event_type,
        symbol=symbol,
        date=date,
        trace_id=trace_id,
        **kwargs,
    ).info(f"{event_type}: {symbol} {date}")


def log_checkpoint_event(
    event_type: str,
    symbol: str,
    trace_id: str,
    **kwargs: object,
) -> None:
    """Log a checkpoint lifecycle event (Issue #97).

    Args:
        event_type: Type of event (checkpoint_save, checkpoint_restore, etc.)
        symbol: Trading symbol (e.g., "BTCUSDT")
        trace_id: Correlation ID for the pipeline run
        **kwargs: Additional event-specific fields
    """
    logger = get_logger()
    logger.bind(
        component="checkpoint",
        event_type=event_type,
        symbol=symbol,
        trace_id=trace_id,
        **kwargs,
    ).info(f"{event_type}: {symbol}")


def generate_trace_id(prefix: str = "rb") -> str:
    """Generate a unique trace ID for request correlation.

    Args:
        prefix: Prefix for the trace ID (default: "rb" for rangebar)

    Returns:
        Trace ID in format "{prefix}-{hex8}" (e.g., "rb-a1b2c3d4")
    """
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:8]}"

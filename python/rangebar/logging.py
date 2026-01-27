"""Centralized NDJSON logging configuration for rangebar-py.

Implements GitHub Issue #43: Structured logging for checksum verification
and other observability events.

Logs are stored in the repository tree under `logs/` directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Logger

# Logs in repository tree (not platformdirs)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # rangebar-py/
LOG_DIR = PROJECT_ROOT / "logs"
NDJSON_FILE = LOG_DIR / "events.jsonl"
CHECKSUM_REGISTRY_FILE = LOG_DIR / "checksum_registry.jsonl"

# Lazy initialization flag
_logger_initialized = False
_logger: Logger | None = None


def _get_context_extra() -> dict:
    """Get context fields added to every log entry."""
    import socket

    return {
        "service": "rangebar-py",
        "environment": os.environ.get("RANGEBAR_ENV", "development"),
        "git_sha": os.environ.get("RANGEBAR_GIT_SHA", "unknown"),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }


def get_logger() -> Logger:
    """Get the configured logger instance.

    Lazy initialization to avoid import-time side effects.
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


def generate_trace_id(prefix: str = "rb") -> str:
    """Generate a unique trace ID for request correlation.

    Args:
        prefix: Prefix for the trace ID (default: "rb" for rangebar)

    Returns:
        Trace ID in format "{prefix}-{hex8}" (e.g., "rb-a1b2c3d4")
    """
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:8]}"

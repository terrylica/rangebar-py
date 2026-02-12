"""Layer 3 live streaming sidecar for range bar construction (Issue #91).

Long-running sidecar that streams live trades from Binance WebSocket,
constructs range bars in real-time with full 58-column microstructure
features, and writes them to ClickHouse.

Architecture:
    Binance WS (per symbol) → Rust LiveBarEngine → Python → ClickHouse

The Rust engine handles WebSocket → trade → bar entirely without GIL.
Python only crosses the boundary per completed bar (~1-6/sec), then
writes to ClickHouse via existing ``store_bars_batch()`` path.

Usage
-----
Start sidecar for specific symbols:

>>> from rangebar.sidecar import run_sidecar, SidecarConfig
>>> config = SidecarConfig(symbols=["BTCUSDT"], thresholds=[250, 500])
>>> run_sidecar(config)  # Blocks until interrupted

Start with gap-fill on startup:

>>> config = SidecarConfig(
...     symbols=["BTCUSDT"], thresholds=[250], gap_fill_on_startup=True,
... )
>>> run_sidecar(config)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (from .mise.toml [env] SSoT)
# =============================================================================

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("1", "true", "yes")


@dataclass
class SidecarConfig:
    """Configuration for the streaming sidecar."""

    symbols: list[str] = field(default_factory=list)
    thresholds: list[int] = field(default_factory=lambda: [250, 500, 750, 1000])
    include_microstructure: bool = True
    gap_fill_on_startup: bool = True
    verbose: bool = False
    timeout_ms: int = 5000

    @classmethod
    def from_env(cls) -> SidecarConfig:
        """Load config from RANGEBAR_STREAMING_* env vars."""
        symbols_str = os.environ.get("RANGEBAR_STREAMING_SYMBOLS", "")
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]

        thresholds_str = os.environ.get(
            "RANGEBAR_STREAMING_THRESHOLDS", "250,500,750,1000",
        )
        thresholds = [int(t.strip()) for t in thresholds_str.split(",") if t.strip()]

        return cls(
            symbols=symbols,
            thresholds=thresholds,
            include_microstructure=_env_bool("RANGEBAR_STREAMING_MICROSTRUCTURE", True),
            gap_fill_on_startup=_env_bool("RANGEBAR_STREAMING_GAP_FILL", True),
            verbose=_env_bool("RANGEBAR_STREAMING_VERBOSE", False),
            timeout_ms=_env_int("RANGEBAR_STREAMING_TIMEOUT_MS", 5000),
        )


# =============================================================================
# Checkpoint persistence (Issue #93)
# =============================================================================

def _checkpoint_dir() -> Path:
    """Checkpoint directory for streaming state."""
    from platformdirs import user_cache_dir
    d = Path(user_cache_dir("rangebar")) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_path(symbol: str, threshold: int) -> Path:
    return _checkpoint_dir() / f"streaming_{symbol}_{threshold}.json"


def _load_checkpoint(symbol: str, threshold: int) -> dict | None:
    path = _checkpoint_path(symbol, threshold)
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_checkpoint(symbol: str, threshold: int, bar_dict: dict) -> None:
    path = _checkpoint_path(symbol, threshold)
    checkpoint = {
        "symbol": symbol,
        "threshold": threshold,
        "last_bar_timestamp_ms": (
            bar_dict.get("timestamp_ms") or bar_dict.get("close_time")
        ),
        "updated_at": time.time(),
    }
    path.write_text(json.dumps(checkpoint))


# =============================================================================
# ClickHouse write path
# =============================================================================

def _write_bar_to_clickhouse(symbol: str, threshold: int, bar_dict: dict) -> None:
    """Write a single completed bar to ClickHouse via store_bars_batch()."""
    from rangebar.clickhouse.cache import RangeBarCache

    # Remove metadata keys added by PyLiveBarEngine
    clean = {k: v for k, v in bar_dict.items() if not k.startswith("_")}
    bars_df = pl.DataFrame([clean])

    with RangeBarCache() as cache:
        cache.store_bars_batch(
            symbol=symbol,
            threshold_decimal_bps=threshold,
            bars=bars_df,
        )


# =============================================================================
# Gap-fill on startup
# =============================================================================

def _gap_fill(symbols: list[str], thresholds: list[int]) -> None:
    """Run Layer 2 recency backfill for all symbol x threshold pairs."""
    try:
        from rangebar.recency import backfill_recent
    except ImportError:
        logger.warning("recency module not available, skipping gap-fill")
        return

    for symbol in symbols:
        for threshold in thresholds:
            try:
                result = backfill_recent(symbol, threshold_decimal_bps=threshold)
                if result and result.bars_written > 0:
                    logger.info(
                        "gap-fill: %s@%d — %d bars, gap was %.0fs",
                        symbol, threshold, result.bars_written, result.gap_seconds,
                    )
            except Exception:
                logger.exception("gap-fill failed for %s@%d", symbol, threshold)


# =============================================================================
# Main sidecar loop
# =============================================================================

def run_sidecar(config: SidecarConfig) -> None:
    """Main sidecar entry point. Blocks until interrupted.

    Parameters
    ----------
    config
        Sidecar configuration. Use ``SidecarConfig.from_env()`` for
        env-var-driven setup, or construct directly.
    """
    from rangebar._core import LiveBarEngine

    if not config.symbols:
        msg = (
            "No symbols configured. "
            "Set RANGEBAR_STREAMING_SYMBOLS or pass symbols list."
        )
        raise ValueError(msg)

    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger.info(
        "starting sidecar: symbols=%s, thresholds=%s, microstructure=%s",
        config.symbols, config.thresholds, config.include_microstructure,
    )

    # Gap-fill (Layer 2) for all symbol x threshold pairs
    if config.gap_fill_on_startup:
        logger.info("running gap-fill before starting live stream")
        _gap_fill(config.symbols, config.thresholds)

    # Start Rust engine
    engine = LiveBarEngine(
        config.symbols,
        config.thresholds,
        config.include_microstructure,
    )
    engine.start()
    logger.info("live bar engine started")

    # Signal handling for graceful shutdown
    running = True

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal running
        logger.info("received signal %d, shutting down", signum)
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    bars_written = 0
    try:
        while running:
            bar = engine.next_bar(timeout_ms=config.timeout_ms)
            if bar is None:
                continue

            symbol = bar.pop("_symbol", "UNKNOWN")
            threshold = bar.pop("_threshold", 0)

            try:
                _write_bar_to_clickhouse(symbol, threshold, bar)
                bars_written += 1
                _save_checkpoint(symbol, threshold, bar)

                if config.verbose or bars_written % 100 == 0:
                    logger.info(
                        "%s@%d: bar written (close=%s, total=%d)",
                        symbol, threshold, bar.get("close"), bars_written,
                    )
            except Exception:
                logger.exception("failed to write bar for %s@%d", symbol, threshold)

    finally:
        engine.stop()
        metrics = engine.get_metrics()
        logger.info(
            "sidecar stopped: bars_written=%d, engine_metrics=%s",
            bars_written, metrics,
        )

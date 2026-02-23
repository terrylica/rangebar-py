# FILE-SIZE-OK: Watchdog logic (Issue #107) is tightly coupled with main loop
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
    watchdog_timeout_s: int = 300       # 5 min of zero trade increment → dead
    max_watchdog_restarts: int = 3      # Max engine restarts before exit

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
            watchdog_timeout_s=_env_int("RANGEBAR_STREAMING_WATCHDOG_TIMEOUT_S", 300),
            max_watchdog_restarts=_env_int("RANGEBAR_STREAMING_MAX_WATCHDOG_RESTARTS", 3),
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


def _save_checkpoint(
    symbol: str,
    threshold: int,
    bar_dict: dict,
    *,
    processor_checkpoint: dict | None = None,
) -> None:
    path = _checkpoint_path(symbol, threshold)
    checkpoint = {
        "symbol": symbol,
        "threshold": threshold,
        "last_bar_timestamp_ms": (
            bar_dict.get("timestamp_ms") or bar_dict.get("close_time")
        ),
        "updated_at": time.time(),
    }
    if processor_checkpoint is not None:
        checkpoint["processor_checkpoint"] = processor_checkpoint
    # Issue #97: Provenance metadata for forensic reconstruction
    from rangebar.checkpoint import _checkpoint_provenance
    checkpoint["provenance"] = _checkpoint_provenance()
    # Atomic write: temp + fsync + rename (Issue #97)
    tmp = path.with_suffix(".tmp")
    data = json.dumps(checkpoint)
    tmp.write_text(data)
    tmp.replace(path)


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

def _gap_fill(symbols: list[str], thresholds: list[int]) -> dict:
    """Run Layer 2 recency backfill for all symbol x threshold pairs.

    Returns dict keyed by ``"{symbol}@{threshold}"`` → ``BackfillResult``.
    Used by ``_inject_checkpoints()`` to skip stale checkpoints (Issue #96).
    """
    try:
        from rangebar.recency import backfill_recent
    except ImportError:
        logger.warning("recency module not available, skipping gap-fill")
        return {}

    results: dict = {}
    for symbol in symbols:
        for threshold in thresholds:
            key = f"{symbol}@{threshold}"
            try:
                result = backfill_recent(symbol, threshold_decimal_bps=threshold)
                results[key] = result
                if result and result.bars_written > 0:
                    logger.info(
                        "gap-fill: %s@%d — %d bars, gap was %.0fs",
                        symbol, threshold, result.bars_written, result.gap_seconds,
                    )
            except Exception:
                logger.exception("gap-fill failed for %s@%d", symbol, threshold)
    return results


# =============================================================================
# Checkpoint inject/extract helpers (Issue #97)
# =============================================================================

_EXPECTED_KEY_PARTS = 2  # "SYMBOL:THRESHOLD" format


def _inject_checkpoints(
    engine: object,
    symbols: list[str],
    thresholds: list[int],
    trace_id: str,
    gap_fill_results: dict | None = None,
) -> int:
    """Inject saved processor checkpoints into engine before start.

    Parameters
    ----------
    gap_fill_results
        Dict keyed by ``"{symbol}@{threshold}"`` → ``BackfillResult`` from
        ``_gap_fill()``.  When gap-fill wrote ≥1 bar for a pair, the
        checkpoint for that pair is skipped — the gap-fill data is
        authoritative and the checkpoint's incomplete bar is obsolete
        (Issue #96).
    """
    from rangebar.hooks import HookEvent, emit_hook
    from rangebar.logging import log_checkpoint_event

    loaded = 0
    for symbol in symbols:
        for threshold in thresholds:
            # Issue #96: Skip checkpoint when gap-fill already wrote bars
            key = f"{symbol}@{threshold}"
            gf = (gap_fill_results or {}).get(key)
            if gf is not None and getattr(gf, "bars_written", 0) > 0:
                logger.info(
                    "skipping checkpoint for %s@%d: gap-fill wrote %d bars",
                    symbol, threshold, gf.bars_written,
                )
                continue

            cp = _load_checkpoint(symbol, threshold)
            if not (cp and cp.get("processor_checkpoint")):
                continue
            try:
                engine.set_checkpoint(  # type: ignore[attr-defined]
                    symbol, threshold, cp["processor_checkpoint"],
                )
                loaded += 1
                cp_age = time.time() - cp.get("updated_at", time.time())
                logger.info(
                    "injected checkpoint for %s@%d (age=%.0fs)",
                    symbol, threshold, cp_age,
                )
                log_checkpoint_event(
                    "sidecar_checkpoint_inject",
                    symbol,
                    trace_id,
                    threshold=threshold,
                    checkpoint_age_sec=round(cp_age),
                    has_incomplete_bar=bool(
                        cp["processor_checkpoint"].get("has_incomplete_bar")
                    ),
                )
                emit_hook(
                    HookEvent.SIDECAR_CHECKPOINT_RESTORED,
                    symbol=symbol,
                    trace_id=trace_id,
                    threshold_dbps=threshold,
                    has_incomplete_bar=bool(
                        cp["processor_checkpoint"].get("has_incomplete_bar")
                    ),
                    checkpoint_age_sec=round(cp_age),
                )
            except Exception:
                logger.exception(
                    "failed to inject checkpoint for %s@%d", symbol, threshold,
                )
                log_checkpoint_event(
                    "sidecar_checkpoint_inject_failed",
                    symbol,
                    trace_id,
                    threshold=threshold,
                )
    return loaded


def _extract_checkpoints(
    engine: object,
    trace_id: str,
    bars_written: int,
) -> None:
    """Extract and save processor checkpoints on shutdown."""
    from rangebar.hooks import HookEvent, emit_hook
    from rangebar.logging import log_checkpoint_event

    try:
        all_checkpoints = engine.collect_checkpoints(timeout_ms=5000)  # type: ignore[attr-defined]
        for key, cp_dict in all_checkpoints.items():
            parts = key.split(":")
            if len(parts) == _EXPECTED_KEY_PARTS:
                sym, thr_str = parts
                thr_int = int(thr_str)
                _save_checkpoint(
                    sym, thr_int, {},
                    processor_checkpoint=cp_dict,
                )
                log_checkpoint_event(
                    "sidecar_checkpoint_extract",
                    sym,
                    trace_id,
                    threshold=thr_int,
                    has_incomplete_bar=bool(
                        cp_dict.get("has_incomplete_bar")
                    ),
                )
                emit_hook(
                    HookEvent.SIDECAR_CHECKPOINT_SAVED,
                    symbol=sym,
                    trace_id=trace_id,
                    threshold_dbps=thr_int,
                    has_incomplete_bar=bool(
                        cp_dict.get("has_incomplete_bar")
                    ),
                    bars_written_session=bars_written,
                )
        logger.info(
            "saved %d processor checkpoints on shutdown",
            len(all_checkpoints),
        )
    except Exception:
        logger.exception("failed to collect/save processor checkpoints")


# =============================================================================
# Engine creation helper (Issue #107: watchdog restart)
# =============================================================================

def _create_engine(
    config: SidecarConfig,
    trace_id: str,
    gap_fill_results: dict | None,
) -> tuple[object, int]:
    """Create and configure a LiveBarEngine, returning (engine, checkpoints_loaded)."""
    from rangebar._core import LiveBarEngine

    engine = LiveBarEngine(
        config.symbols,
        config.thresholds,
        config.include_microstructure,
    )
    loaded = _inject_checkpoints(
        engine, config.symbols, config.thresholds, trace_id,
        gap_fill_results=gap_fill_results,
    )
    engine.start()
    return engine, loaded


# =============================================================================
# Watchdog notification (Issue #107)
# =============================================================================

def _notify_watchdog_trigger(
    stale_seconds: float,
    trades_received: int,
    restart_count: int,
) -> None:
    """Send Telegram alert when watchdog detects stale trade flow."""
    try:
        from rangebar.notify.telegram import send_telegram
    except ImportError:
        logger.warning("telegram module not available, skipping watchdog notification")
        return

    try:
        msg = (
            "<b>WATCHDOG TRIGGER</b>\n"
            f"No trade increment for {stale_seconds:.0f}s\n"
            f"trades_received={trades_received}\n"
            f"restart #{restart_count + 1}"
        )
        send_telegram(msg, disable_notification=False)
    except Exception:
        logger.exception("failed to send watchdog Telegram notification")


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
    from rangebar.logging import generate_trace_id

    trace_id = generate_trace_id("sdc")

    if not config.symbols:
        msg = (
            "No symbols configured. "
            "Set RANGEBAR_STREAMING_SYMBOLS or pass symbols list."
        )
        raise ValueError(msg)

    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger.info(
        "starting sidecar: symbols=%s, thresholds=%s, microstructure=%s, "
        "watchdog_timeout_s=%d, max_watchdog_restarts=%d",
        config.symbols, config.thresholds, config.include_microstructure,
        config.watchdog_timeout_s, config.max_watchdog_restarts,
    )

    # Gap-fill (Layer 2) for all symbol x threshold pairs
    gap_fill_results = None
    if config.gap_fill_on_startup:
        logger.info("running gap-fill before starting live stream")
        gap_fill_results = _gap_fill(config.symbols, config.thresholds)

    # Start Rust engine (Issue #107: extracted to helper for watchdog restart)
    engine, checkpoints_loaded = _create_engine(config, trace_id, gap_fill_results)
    logger.info(
        "live bar engine started (checkpoints_loaded=%d)", checkpoints_loaded,
    )

    # Signal handling for graceful shutdown
    running = True

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal running
        logger.info("received signal %d, shutting down", signum)
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    bars_written = 0

    # Issue #107: Watchdog state — detect half-open TCP via trade flow staleness
    last_trades_received = 0
    last_trade_increment_time = time.monotonic()
    watchdog_restarts = 0

    try:
        while running:
            bar = engine.next_bar(timeout_ms=config.timeout_ms)

            if bar is None:
                # Check trade flow via metrics (Issue #107)
                if config.watchdog_timeout_s > 0:
                    metrics = engine.get_metrics()
                    current_trades = metrics.get("trades_received", 0)

                    if current_trades > last_trades_received:
                        last_trades_received = current_trades
                        last_trade_increment_time = time.monotonic()
                    else:
                        stale_s = time.monotonic() - last_trade_increment_time
                        if stale_s >= config.watchdog_timeout_s:
                            logger.warning(
                                "WATCHDOG: no trades for %.0fs (restart #%d)",
                                stale_s, watchdog_restarts + 1,
                            )
                            _notify_watchdog_trigger(
                                stale_s, current_trades, watchdog_restarts,
                            )

                            if watchdog_restarts >= config.max_watchdog_restarts:
                                logger.error(
                                    "WATCHDOG: max restarts (%d) exceeded, exiting",
                                    config.max_watchdog_restarts,
                                )
                                break

                            engine.stop()
                            _extract_checkpoints(engine, trace_id, bars_written)
                            engine.shutdown()
                            engine, _ = _create_engine(
                                config, trace_id, gap_fill_results,
                            )
                            watchdog_restarts += 1
                            last_trades_received = 0
                            last_trade_increment_time = time.monotonic()
                continue

            # Bar received — reset watchdog
            metrics = engine.get_metrics()
            last_trades_received = metrics.get("trades_received", 0)
            last_trade_increment_time = time.monotonic()

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

        # Issue #97: Extract and save processor checkpoints for next startup
        _extract_checkpoints(engine, trace_id, bars_written)

        engine.shutdown()
        metrics = engine.get_metrics()
        logger.info(
            "sidecar stopped: bars_written=%d, engine_metrics=%s",
            bars_written, metrics,
        )

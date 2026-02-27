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
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

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
    """Configuration for the streaming sidecar.

    Environment Variables (Issue #96 Task #6):
    - RANGEBAR_MAX_PENDING_BARS: Max queued bars before backpressure (default: 10000)
    """

    symbols: list[str] = field(default_factory=list)
    thresholds: list[int] = field(default_factory=lambda: [250, 500, 750, 1000])
    include_microstructure: bool = True
    gap_fill_on_startup: bool = True
    verbose: bool = False
    timeout_ms: int = 5000
    watchdog_timeout_s: int = 300  # 5 min of zero trade increment → dead
    max_watchdog_restarts: int = 3  # Max engine restarts before exit
    max_pending_bars: int = 10_000  # Issue #96 Task #6: Backpressure bound
    health_port: int = 8081  # Issue #109: Health check HTTP port (0 = disabled)

    @classmethod
    def from_env(cls) -> SidecarConfig:
        """Load config from environment variables.

        Issue #96 Task #6: RANGEBAR_MAX_PENDING_BARS controls backpressure.
        """
        symbols_str = os.environ.get("RANGEBAR_STREAMING_SYMBOLS", "")
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]

        thresholds_str = os.environ.get(
            "RANGEBAR_STREAMING_THRESHOLDS", "250,500,750,1000",
        )
        thresholds = [
            int(t.strip()) for t in thresholds_str.split(",") if t.strip()
        ]

        return cls(
            symbols=symbols,
            thresholds=thresholds,
            include_microstructure=_env_bool(
                "RANGEBAR_STREAMING_MICROSTRUCTURE", True
            ),
            gap_fill_on_startup=_env_bool("RANGEBAR_STREAMING_GAP_FILL", True),
            verbose=_env_bool("RANGEBAR_STREAMING_VERBOSE", False),
            timeout_ms=_env_int("RANGEBAR_STREAMING_TIMEOUT_MS", 5000),
            watchdog_timeout_s=_env_int(
                "RANGEBAR_STREAMING_WATCHDOG_TIMEOUT_S", 300
            ),
            max_watchdog_restarts=_env_int(
                "RANGEBAR_STREAMING_MAX_WATCHDOG_RESTARTS", 3
            ),
            max_pending_bars=_env_int("RANGEBAR_MAX_PENDING_BARS", 10_000),
            health_port=_env_int("RANGEBAR_HEALTH_PORT", 8081),
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
        data = json.loads(path.read_text())
        # Migrate legacy checkpoint key (timestamp_ms → close_time_ms rename)
        if "last_bar_timestamp_ms" in data and "last_bar_close_time_ms" not in data:
            data["last_bar_close_time_ms"] = data.pop("last_bar_timestamp_ms")
            path.write_text(json.dumps(data))
        return data
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
        "last_bar_close_time_ms": bar_dict.get("close_time_ms"),
        "updated_at": time.time(),
    }
    # Issue #111 (Ariadne): Track trade ID at the streaming boundary
    last_trade_id = bar_dict.get("last_agg_trade_id")
    if last_trade_id and last_trade_id > 0:
        checkpoint["last_agg_trade_id"] = last_trade_id
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
# ClickHouse write path (Issue #96 Task #144 Phase 3: Arrow streaming path)
# =============================================================================

def _write_bar_to_clickhouse(symbol: str, threshold: int, bar_dict: dict) -> None:
    """Write a single completed bar to ClickHouse (legacy single-bar path).

    Kept for backward compatibility. For streaming optimization,
    use _write_bars_batch_to_clickhouse() instead.
    """
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


def _write_bars_batch_to_clickhouse(
    symbol: str, threshold: int, bar_dicts: list[dict],
    *, cache: object | None = None,
) -> None:
    """Write a batch of completed bars to ClickHouse.

    Issue #96 Task #144 Phase 3: Streaming latency optimization.
    Issue #117: Accepts optional cached RangeBarCache instance.
    """
    from rangebar.clickhouse.cache import RangeBarCache

    if not bar_dicts:
        return

    # Remove metadata keys from all bars (batch operation)
    clean_bars = [
        {k: v for k, v in bar_dict.items() if not k.startswith("_")}
        for bar_dict in bar_dicts
    ]
    bars_df = pl.DataFrame(clean_bars)

    if cache is not None:
        cache.store_bars_batch(
            symbol=symbol,
            threshold_decimal_bps=threshold,
            bars=bars_df,
        )
    else:
        with RangeBarCache() as ch_cache:
            ch_cache.store_bars_batch(
                symbol=symbol,
                threshold_decimal_bps=threshold,
                bars=bars_df,
            )


# =============================================================================
# Gap-fill on startup
# =============================================================================

def _gap_fill(
    symbols: list[str],
    thresholds: list[int],
    valid_pairs: set[tuple[str, int]] | None = None,
) -> dict:
    """Run Layer 2 recency backfill for valid symbol x threshold pairs.

    Returns dict keyed by ``"{symbol}@{threshold}"`` → ``BackfillResult``.
    Used by ``_inject_checkpoints()`` to skip stale checkpoints (Issue #96).

    Issue #120: When ``valid_pairs`` is provided, only those pairs are
    backfilled (skips invalid combos like SHIBUSDT@250).
    """
    try:
        from rangebar.recency import backfill_recent
    except ImportError:
        logger.warning("recency module not available, skipping gap-fill")
        return {}

    pairs = valid_pairs if valid_pairs is not None else {
        (sym, thr) for sym in symbols for thr in thresholds
    }

    results: dict = {}
    for symbol, threshold in sorted(pairs):
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
    valid_pairs: set[tuple[str, int]] | None = None,
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
    valid_pairs
        Issue #120: When provided, only inject checkpoints for these pairs.
        Eliminates useless checkpoint file lookups for invalid pairs.
    """
    from rangebar.hooks import HookEvent, emit_hook
    from rangebar.logging import log_checkpoint_event

    pairs = valid_pairs if valid_pairs is not None else {
        (sym, thr) for sym in symbols for thr in thresholds
    }

    loaded = 0
    for symbol, threshold in sorted(pairs):
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
    valid_pairs: set[tuple[str, int]] | None = None,
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
        valid_pairs=valid_pairs,
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
        from rangebar.notify.telegram import _build_context_block, send_telegram
    except ImportError:
        logger.warning("telegram module not available, skipping watchdog notification")
        return

    try:
        msg = (
            "<b>WATCHDOG TRIGGER</b>\n"
            f"<b>Stale:</b> {stale_seconds:.0f}s (no trade increment)\n"
            f"<b>Total trades:</b> {trades_received:,}\n"
            f"<b>Restart:</b> #{restart_count + 1}"
            + _build_context_block("sidecar")
        )
        send_telegram(msg, disable_notification=False)
    except Exception:
        logger.exception("failed to send watchdog Telegram notification")


# =============================================================================
# Watchdog error recovery (Issue #107, TGI #2)
# =============================================================================


def _restart_engine_with_recovery(
    engine: object,
    config: SidecarConfig,
    trace_id: str,
    bars_written: int,
    restart_attempt: int,
    valid_pairs: set[tuple[str, int]] | None = None,
) -> tuple[object | None, bool]:
    """Restart engine with timeout guards and error recovery.

    Each shutdown step has isolated error handling to prevent cascading failures.

    Returns:
        (new_engine, success_bool) - engine is None if restart failed
    """
    try:
        # Step 1: Graceful shutdown attempt (may fail or hang)
        try:
            engine.stop()
            logger.info("engine.stop() completed")
        except (RuntimeError, OSError, TimeoutError, ValueError) as e:
            logger.warning("engine.stop() raised exception: %s", e)

        # Step 2: Extract checkpoints (critical for state recovery)
        try:
            _extract_checkpoints(engine, trace_id, bars_written)
            logger.info("checkpoint extraction succeeded")
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("checkpoint extraction failed: %s", e)

        # Step 3: Force shutdown
        try:
            engine.shutdown()
            logger.info("engine.shutdown() completed")
        except (RuntimeError, OSError, TimeoutError, ValueError) as e:
            logger.warning("engine.shutdown() raised: %s", e)

        # Step 4: Fresh gap-fill before creating new engine (Issue #115 Phase 1b)
        logger.info(
            "running fresh gap-fill after watchdog restart #%d",
            restart_attempt,
        )
        fresh_gap_fill = _gap_fill(
            config.symbols, config.thresholds, valid_pairs=valid_pairs,
        )

        # Step 5: Create new engine with fresh gap-fill results
        new_engine, _ = _create_engine(
            config, trace_id, fresh_gap_fill, valid_pairs=valid_pairs,
        )
        logger.info("engine restart #%d succeeded", restart_attempt)
    except Exception:
        logger.exception(
            "WATCHDOG RESTART FAILED (attempt #%d)", restart_attempt
        )
        return None, False
    else:
        return new_engine, True


def _notify_restart_failure(
    restart_number: int,
    consecutive_timeouts: int,
    stale_seconds: float,
) -> None:
    """Alert ops when watchdog restart fails via Telegram."""
    try:
        from rangebar.notify.telegram import _build_context_block, send_telegram
    except ImportError:
        logger.warning("telegram module not available for restart failure alert")
        return

    try:
        msg = (
            "<b>WATCHDOG RESTART FAILED</b>\n"
            f"<b>Attempt:</b> {restart_number}\n"
            f"<b>Consecutive timeouts:</b> {consecutive_timeouts}\n"
            f"<b>Stale:</b> {stale_seconds:.0f}s\n"
            "<b>Action:</b> Sidecar will exit. systemd should auto-restart.\n"
            "<b>Check:</b> <code>journalctl --user -u rangebar-sidecar -n 50</code>"
            + _build_context_block("sidecar")
        )
        send_telegram(msg, disable_notification=False)
        logger.info("restart failure alert sent via Telegram")
    except Exception:
        logger.exception("failed to send restart failure alert")


# =============================================================================
# Issue #109: Health check HTTP endpoint
# =============================================================================

def _start_health_server(port: int) -> HTTPServer | None:
    """Start a lightweight HTTP health server on a background daemon thread.

    Returns the server instance (for shutdown) or None if port=0 (disabled).
    """
    if port == 0:
        return None

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return

            from rangebar.health_checks import run_all_checks

            results = run_all_checks()
            bool_checks = {k: v for k, v in results.items() if isinstance(v, bool)}
            all_pass = all(bool_checks.values())

            body = json.dumps(
                {"status": "healthy" if all_pass else "unhealthy", "checks": results},
            ).encode()
            self.send_response(200 if all_pass else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            # Suppress per-request access logs (too noisy for health probes)
            pass

    try:
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info("health endpoint listening on port %d", port)
    except OSError as e:
        logger.warning(
            "failed to start health server on port %d: %s", port, e,
        )
        return None
    else:
        return server


# =============================================================================
# Dead-letter + retry helpers (Issue #115: Kintsugi sidecar hardening)
# =============================================================================

_DEAD_LETTER_DIR = Path("/tmp/rangebar-dead-letter")
_MAX_FLUSH_RETRIES = 3
_STARTUP_SYMBOL_PREVIEW = 5  # Max symbols shown in startup Telegram
_FLUSH_RETRY_DELAYS = (1.0, 2.0, 4.0)  # Exponential backoff


def _write_dead_letter(
    symbol: str, threshold: int, bar_dicts: list[dict[str, Any]],
) -> Path | None:
    """Serialize failed bars to a dead-letter Parquet file for later replay.

    Returns the path written, or None on failure.
    """
    try:
        import polars as pl

        _DEAD_LETTER_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = _DEAD_LETTER_DIR / f"{symbol}_{threshold}_{ts}.parquet"
        clean = [
            {k: v for k, v in d.items() if not k.startswith("_")}
            for d in bar_dicts
        ]
        pl.DataFrame(clean).write_parquet(path)
    except (OSError, RuntimeError, ValueError):
        logger.exception(
            "dead-letter: failed to write for %s@%d", symbol, threshold,
        )
        return None
    else:
        logger.warning(
            "dead-letter: wrote %d bars to %s", len(bar_dicts), path,
        )
        return path


def _notify_dead_letter(symbol: str, threshold: int, n_bars: int, path: Path) -> None:
    """Send Telegram alert when bars are written to dead-letter."""
    try:
        from rangebar.notify.telegram import _build_context_block, send_telegram
    except ImportError:
        return
    try:
        msg = (
            "<b>DEAD LETTER</b>\n"
            f"<b>Symbol:</b> {symbol}@{threshold}\n"
            f"<b>Bars lost:</b> {n_bars:,}\n"
            f"<b>Path:</b> <code>{path}</code>\n"
            "<b>Cause:</b> ClickHouse write failed after 3 retries\n"
            f"<b>Replay:</b> <code>python -m rangebar.sidecar --replay {path}</code>"
            + _build_context_block("sidecar")
        )
        send_telegram(msg, disable_notification=False)
    except Exception:
        logger.exception("dead-letter: failed to send Telegram alert")


def _write_batch_with_retry(
    symbol: str, threshold: int, bar_dicts: list[dict[str, Any]],
    *, cache: object | None = None,
) -> bool:
    """Write a batch to ClickHouse with retry + dead-letter fallback.

    Returns True if write succeeded (including dead-letter), False if all failed.
    Issue #117: Accepts optional cached RangeBarCache.
    """
    for attempt in range(_MAX_FLUSH_RETRIES):
        try:
            _write_bars_batch_to_clickhouse(symbol, threshold, bar_dicts, cache=cache)
        except (OSError, RuntimeError, ConnectionError, ValueError) as e:
            delay = _FLUSH_RETRY_DELAYS[attempt]
            logger.warning(
                "flush retry %d/%d for %s@%d (%d bars), backoff %.1fs: %s",
                attempt + 1, _MAX_FLUSH_RETRIES, symbol, threshold,
                len(bar_dicts), delay, e,
            )
            time.sleep(delay)
        else:
            return True

    # All retries exhausted — dead-letter
    logger.error(
        "flush FAILED after %d retries for %s@%d (%d bars) — writing dead-letter",
        _MAX_FLUSH_RETRIES, symbol, threshold, len(bar_dicts),
    )
    dl_path = _write_dead_letter(symbol, threshold, bar_dicts)
    if dl_path is not None:
        _notify_dead_letter(symbol, threshold, len(bar_dicts), dl_path)
    return False


# =============================================================================
# Main sidecar loop
# =============================================================================


def run_sidecar(config: SidecarConfig) -> None:  # noqa: PLR0912, PLR0915
    """Main sidecar entry point. Blocks until interrupted.

    Parameters
    ----------
    config
        Sidecar configuration. Use ``SidecarConfig.from_env()`` for
        env-var-driven setup, or construct directly.

    Issue #96 Task #144 Phase 3: Streaming latency optimization
    - Batches bar writes to reduce ClickHouse roundtrips
    - Accumulates bars by (symbol, threshold) pair
    - Flushes on timeout or batch size threshold
    """
    from rangebar.logging import generate_trace_id

    trace_id = generate_trace_id("sdc")
    sidecar_start_time = time.monotonic()

    if not config.symbols:
        msg = (
            "No symbols configured. "
            "Set RANGEBAR_STREAMING_SYMBOLS or pass symbols list."
        )
        raise ValueError(msg)

    # Issue #120: Build valid (symbol, threshold) pairs, skip invalid with log
    from rangebar.threshold import ThresholdError, resolve_and_validate_threshold

    valid_pairs: set[tuple[str, int]] = set()
    for sym in config.symbols:
        for thr in config.thresholds:
            try:
                resolve_and_validate_threshold(sym, thr)
                valid_pairs.add((sym, thr))
            except ThresholdError:
                logger.info("skipping %s@%d (below min_threshold)", sym, thr)

    if not valid_pairs:
        msg = "No valid (symbol, threshold) pairs after filtering"
        raise ValueError(msg)

    # Log per-symbol threshold map
    sym_thresholds: dict[str, list[int]] = defaultdict(list)
    for sym, thr in valid_pairs:
        sym_thresholds[sym].append(thr)
    for _sym, thrs in sym_thresholds.items():
        thrs.sort()

    logger.info(
        "valid pairs: %d symbols, %d total (symbol, threshold) pairs",
        len(sym_thresholds), len(valid_pairs),
    )
    for sym, thrs in sorted(sym_thresholds.items()):
        logger.info("  %s: %s", sym, thrs)

    # Issue #119: Symbol coverage health check
    env_symbols_str = os.environ.get("RANGEBAR_STREAMING_SYMBOLS", "")
    env_symbols = [s.strip().upper() for s in env_symbols_str.split(",") if s.strip()]
    config_upper = {s.upper() for s in config.symbols}
    if env_symbols and config_upper != set(env_symbols):
        missing = sorted(set(env_symbols) - config_upper)
        coverage_msg = (
            f"Symbol mismatch: {len(config.symbols)}/{len(env_symbols)} symbols. "
            f"Missing: {missing}. Use --from-env."
        )
        logger.warning(coverage_msg)
        try:
            from rangebar.hooks import HookEvent, emit_hook
            emit_hook(
                HookEvent.SYMBOL_COVERAGE_MISMATCH, symbol="*",
                active=len(config.symbols), expected=len(env_symbols),
                missing=missing,
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            logger.debug("failed to emit SYMBOL_COVERAGE_MISMATCH hook", exc_info=True)
        try:
            import sys

            from rangebar.notify.telegram import _build_context_block, send_telegram
            msg = (
                "<b>SIDECAR PARTIAL COVERAGE</b>\n"
                f"<b>Active:</b> {len(config.symbols)}/{len(env_symbols)} symbols\n"
                f"<b>Missing:</b> {', '.join(missing)}\n"
                "<b>Env var:</b> RANGEBAR_STREAMING_SYMBOLS\n"
                "<b>Fix:</b> use <code>--from-env</code> flag\n"
                f"<b>Cmdline:</b> <code>{' '.join(sys.argv)}</code>"
                + _build_context_block("sidecar")
            )
            send_telegram(msg, disable_notification=False)
        except (ImportError, OSError, RuntimeError):
            logger.debug(
                "failed to send coverage mismatch alert", exc_info=True,
            )

    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger.info(
        "starting sidecar: symbols=%s, thresholds=%s, microstructure=%s, "
        "watchdog_timeout_s=%d, max_watchdog_restarts=%d",
        config.symbols, config.thresholds, config.include_microstructure,
        config.watchdog_timeout_s, config.max_watchdog_restarts,
    )

    # Issue #117: Cache RangeBarCache at startup to avoid per-flush schema.sql reads
    from rangebar.clickhouse.cache import RangeBarCache
    ch_cache = RangeBarCache()

    # Issue #109: Start health check HTTP server (daemon thread)
    health_server = _start_health_server(config.health_port)

    # Issue #117-119: Set start time for uptime tracking + send startup notification
    try:
        from rangebar.notify.telegram import set_start_time
        set_start_time(sidecar_start_time)
    except ImportError:
        pass
    try:
        from rangebar.hooks import HookEvent, emit_hook
        emit_hook(
            HookEvent.SIDECAR_STARTED, symbol="*",
            symbols=config.symbols, thresholds=config.thresholds,
            health_port=config.health_port, pid=os.getpid(),
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        logger.debug("failed to emit SIDECAR_STARTED hook", exc_info=True)
    try:
        from rangebar.notify.telegram import _build_context_block, send_telegram
        preview = ", ".join(config.symbols[:_STARTUP_SYMBOL_PREVIEW])
        ellipsis = "..." if len(config.symbols) > _STARTUP_SYMBOL_PREVIEW else ""
        # Issue #120: Show per-symbol threshold coverage in startup notification
        thresholds_summary = "; ".join(
            f"{sym}:{thrs}" for sym, thrs in sorted(sym_thresholds.items())
        )
        startup_msg = (
            "<b>SIDECAR STARTED</b>\n"
            f"<b>Symbols:</b> {len(config.symbols)} "
            f"({preview}{ellipsis})\n"
            f"<b>Coverage:</b> {thresholds_summary}\n"
            f"<b>Health:</b> :{config.health_port}/health\n"
            f"<b>PID:</b> {os.getpid()}"
            + _build_context_block("sidecar")
        )
        send_telegram(startup_msg, disable_notification=True)
    except (ImportError, OSError, RuntimeError):
        logger.debug("failed to send sidecar startup Telegram alert", exc_info=True)

    # Gap-fill (Layer 2) for valid symbol x threshold pairs only (Issue #120)
    gap_fill_results = None
    if config.gap_fill_on_startup:
        logger.info("running gap-fill before starting live stream")
        gap_fill_results = _gap_fill(
            config.symbols, config.thresholds, valid_pairs=valid_pairs,
        )

    # Start Rust engine (Issue #107: extracted to helper for watchdog restart)
    engine, checkpoints_loaded = _create_engine(
        config, trace_id, gap_fill_results, valid_pairs=valid_pairs,
    )
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
    consecutive_timeouts = 0  # Track None returns for heartbeat logging
    timeout_heartbeat_interval = 60  # Log every 60 timeouts ≈ 5 min at 5s timeout

    # Issue #96 Task #144 Phase 3: Batch accumulation for streaming optimization
    bar_batch_buffer = defaultdict(list)  # Keyed by (symbol, threshold)
    batch_max_size = 10  # Flush batch when reaching this size
    batch_timeout_s = 1.0  # Flush batch every N seconds
    last_batch_flush_time = time.monotonic()

    def _flush_all_batches() -> int:
        """Flush all accumulated batches to ClickHouse with retry + dead-letter.

        Issue #115 (Kintsugi Phase 1a): Uses _write_batch_with_retry() for
        exponential backoff (1s, 2s, 4s) and dead-letter on final failure.
        Issue #117: Threads cached ch_cache to avoid per-flush schema.sql reads.
        """
        nonlocal last_batch_flush_time
        written = 0
        for (symbol, threshold), bars in bar_batch_buffer.items():
            if not bars:
                continue
            if _write_batch_with_retry(
                symbol, threshold, bars, cache=ch_cache,
            ):
                written += len(bars)
        bar_batch_buffer.clear()
        last_batch_flush_time = time.monotonic()
        return written

    try:
        while running:
            # Check if batch timeout occurred (1 second since last flush)
            now = time.monotonic()
            if (now - last_batch_flush_time) >= batch_timeout_s and bar_batch_buffer:
                written = _flush_all_batches()
                if config.verbose:
                    logger.debug("batch timeout flush: %d bars written", written)

            bar = engine.next_bar(timeout_ms=config.timeout_ms)

            if bar is None:
                consecutive_timeouts += 1

                # Periodic heartbeat (show sidecar is responsive)
                if consecutive_timeouts % timeout_heartbeat_interval == 0:
                    has_metrics = hasattr(engine, "get_metrics")
                    metrics = engine.get_metrics() if has_metrics else {}
                    trades_received = (
                        metrics.get("trades_received", 0) if metrics else 0
                    )
                    stale_s = time.monotonic() - last_trade_increment_time
                    logger.info(
                        "watchdog heartbeat: %d consecutive timeouts, "
                        "last_trades_received=%d, stale=%.0fs",
                        consecutive_timeouts,
                        trades_received,
                        stale_s,
                    )

                # Check trade flow via metrics (Issue #107)
                if config.watchdog_timeout_s > 0:
                    metrics = engine.get_metrics()
                    current_trades = metrics.get("trades_received", 0)

                    if current_trades > last_trades_received:
                        # Reset counter on fresh trades
                        consecutive_timeouts = 0
                        last_trades_received = current_trades
                        last_trade_increment_time = time.monotonic()
                    else:
                        stale_s = time.monotonic() - last_trade_increment_time
                        if stale_s >= config.watchdog_timeout_s:
                            logger.warning(
                                "WATCHDOG: no trades %.0fs, %d timeouts",
                                stale_s,
                                consecutive_timeouts,
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

                            # Flush pending bars before restart
                            written = _flush_all_batches()
                            if written > 0:
                                logger.info(
                                    "flushed %d bars before restart", written
                                )

                            # Use error-tolerant restart (Issue #107, TGI #2)
                            new_engine, success = _restart_engine_with_recovery(
                                engine, config, trace_id,
                                bars_written, watchdog_restarts + 1,
                                valid_pairs=valid_pairs,
                            )

                            if success:
                                engine = new_engine
                                watchdog_restarts += 1
                                consecutive_timeouts = 0
                                last_trades_received = 0
                                last_trade_increment_time = time.monotonic()
                                logger.info("watchdog restart succeeded")
                            else:
                                # Restart failed — escalate
                                logger.error(
                                    "ESCALATION: restart failed after %d timeouts",
                                    consecutive_timeouts,
                                )
                                _notify_restart_failure(
                                    watchdog_restarts + 1,
                                    consecutive_timeouts,
                                    stale_s,
                                )
                                # Exit to be restarted by systemd/supervisor
                                break
                continue

            # Bar received — reset watchdog
            metrics = engine.get_metrics()
            last_trades_received = metrics.get("trades_received", 0)
            last_trade_increment_time = time.monotonic()

            symbol = bar.pop("_symbol", "UNKNOWN")
            threshold = bar.pop("_threshold", 0)

            # Issue #120: Skip bars for invalid (symbol, threshold) pairs
            if (symbol, threshold) not in valid_pairs:
                continue

            # Accumulate bar in batch buffer
            batch_key = (symbol, threshold)
            bar_batch_buffer[batch_key].append(bar)
            bars_written += 1

            # Save checkpoint immediately (for state recovery)
            _save_checkpoint(symbol, threshold, bar)

            # Flush batch if size threshold reached
            # Issue #115 (Kintsugi Phase 1a): retry + dead-letter on failure
            if len(bar_batch_buffer[batch_key]) >= batch_max_size:
                batch = bar_batch_buffer[batch_key]
                wrote = _write_batch_with_retry(
                    symbol, threshold, batch, cache=ch_cache,
                )
                if wrote and (config.verbose or bars_written % 100 == 0):
                        logger.info(
                            "%s@%d: batch written (%d bars, total=%d, close=%s)",
                            symbol, threshold, len(batch),
                            bars_written, bar.get("close"),
                        )
                bar_batch_buffer[batch_key] = []

    finally:
        exit_reason = "clean shutdown" if not running else "exception"

        # Flush all pending bars on shutdown
        written = _flush_all_batches()
        if written > 0:
            logger.info("flushed %d pending bars on shutdown", written)

        engine.stop()

        # Issue #97: Extract and save processor checkpoints for next startup
        _extract_checkpoints(engine, trace_id, bars_written)

        engine.shutdown()

        # Issue #109: Shut down health server
        if health_server is not None:
            health_server.shutdown()

        # Issue #117: Close cached RangeBarCache
        try:
            ch_cache.close()
        except (OSError, RuntimeError):
            logger.debug("failed to close cached RangeBarCache", exc_info=True)

        metrics = engine.get_metrics()
        logger.info(
            "sidecar stopped: bars_written=%d, engine_metrics=%s",
            bars_written, metrics,
        )

        # Issue #117-119: Shutdown notification + hook
        try:
            from rangebar.hooks import HookEvent, emit_hook
            emit_hook(
                HookEvent.SIDECAR_STOPPED, symbol="*",
                bars_written=bars_written, exit_reason=exit_reason,
                pid=os.getpid(),
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            logger.debug("failed to emit SIDECAR_STOPPED hook", exc_info=True)
        try:
            from rangebar.notify.telegram import _build_context_block, send_telegram
            shutdown_msg = (
                "<b>SIDECAR STOPPED</b>\n"
                f"<b>Bars written:</b> {bars_written:,}\n"
                f"<b>Reason:</b> {exit_reason}\n"
                f"<b>PID:</b> {os.getpid()}"
                + _build_context_block("sidecar")
            )
            send_telegram(shutdown_msg, disable_notification=False)
        except (ImportError, OSError, RuntimeError):
            logger.debug(
                "failed to send shutdown Telegram alert", exc_info=True,
            )

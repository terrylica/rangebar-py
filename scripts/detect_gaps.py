#!/usr/bin/env python3
# FILE-SIZE-OK: standalone script, splitting would reduce usability
# Issue #102: Gap Regression Prevention — WAVE-1 gap detector deliverable
"""Gap detector for ClickHouse range_bars table.

Detects temporal gaps, price gaps, and coverage anomalies in the
rangebar_cache.range_bars table. Designed to run standalone (no rangebar
package dependency) using clickhouse_connect directly.

Usage:
    python detect_gaps.py                        # Check all TIER1 symbols, all thresholds
    python detect_gaps.py --symbol BTCUSDT       # Check specific symbol
    python detect_gaps.py --threshold 500        # Check specific threshold
    python detect_gaps.py --min-gap-hours 6      # Custom gap threshold (default: 6)
    python detect_gaps.py --price-gap-dbps 5000  # Custom price gap threshold (default: 5000 = 5x)
    python detect_gaps.py --recent-days 30       # Only check last 30 days
    python detect_gaps.py --summary              # Summary only (no individual gaps)
    python detect_gaps.py --json                 # JSON output for programmatic use

Exit codes:
    0 - CLEAN: no gaps detected
    1 - GAPS FOUND: gaps detected (details printed)
    2 - ERROR: connection or query failure

Environment variables:
    RANGEBAR_CH_HOSTS     SSH alias for ClickHouse host (default: bigblack)
    RANGEBAR_MODE         Connection mode: remote, local, auto (default: remote)
    CLICKHOUSE_HOST       Direct host override (default: localhost)
    CLICKHOUSE_PORT       Port override (default: 8123)

Connection:
    In 'remote' mode (default), establishes an SSH tunnel to RANGEBAR_CH_HOSTS
    and queries ClickHouse through the tunnel. In 'local' mode, connects to
    localhost directly. The script manages tunnel lifecycle automatically.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import clickhouse_connect

# =============================================================================
# Constants (inlined to avoid rangebar package dependency)
# =============================================================================

# TIER1_SYMBOLS from python/rangebar/constants.py
# These are the base names (without USDT suffix) — we append USDT for queries
TIER1_BASES: tuple[str, ...] = (
    "AAVE", "ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "ETH",
    "FIL", "LINK", "LTC", "NEAR", "SOL", "SUI", "UNI", "WIF", "WLD", "XRP",
)

# Additional symbols that may be in ClickHouse but not in TIER1
# (DOT, ATOM, ETC, XLM, SHIB, TRX, ZEC, HBAR, PAXG, PEPE, TON)
EXTRA_BASES: tuple[str, ...] = (
    "ATOM", "DOT", "ETC", "HBAR", "PAXG", "PEPE", "SHIB", "TON", "TRX",
    "XLM", "ZEC",
)

ALL_BASES: tuple[str, ...] = (*TIER1_BASES, *EXTRA_BASES)

# Standard cache thresholds (dbps)
THRESHOLDS: tuple[int, ...] = (250, 500, 750, 1000)

# Default gap detection parameters
DEFAULT_MIN_GAP_HOURS = 6.0
DEFAULT_PRICE_GAP_DBPS = 5000  # 5x threshold = suspicious price jump


# =============================================================================
# Issue #126: Resolve ouroboros mode from Settings when not specified via CLI
def _resolve_ouroboros_mode() -> str:
    """Resolve ouroboros mode from RANGEBAR_OUROBOROS_MODE env var."""
    try:
        from rangebar.ouroboros import get_operational_ouroboros_mode
        return get_operational_ouroboros_mode()
    except (ImportError, ValueError, OSError):
        return "month"  # Issue #126: new system-wide default


# SSH Tunnel (minimal standalone implementation)
# =============================================================================

def _find_free_port() -> int:
    """Find a free local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if TCP port is open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


class SSHTunnel:
    """Minimal SSH tunnel manager for ClickHouse access."""

    def __init__(self, ssh_alias: str, remote_port: int = 8123) -> None:
        self.ssh_alias = ssh_alias
        self.remote_port = remote_port
        self.local_port: int | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def start(self, timeout: float = 10.0) -> int:
        """Start tunnel, return local port."""
        self.local_port = _find_free_port()
        self._process = subprocess.Popen(
            [
                "ssh", "-N",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3",
                "-L", f"{self.local_port}:localhost:{self.remote_port}",
                self.ssh_alias,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _is_port_open("localhost", self.local_port):
                return self.local_port
            exit_code = self._process.poll()
            if exit_code is not None:
                if exit_code == 0:
                    time.sleep(0.3)
                    if _is_port_open("localhost", self.local_port):
                        self._process = None
                        return self.local_port
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode()
                self._process = None
                msg = f"SSH tunnel to {self.ssh_alias} failed (exit={exit_code}): {stderr}"
                raise RuntimeError(msg)
            time.sleep(0.2)
        self.stop()
        msg = f"SSH tunnel to {self.ssh_alias} timed out after {timeout}s"
        raise RuntimeError(msg)

    def stop(self) -> None:
        """Stop the tunnel."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._process.kill()
                    self._process.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            finally:
                for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                    if stream is not None:
                        with contextlib.suppress(OSError):
                            stream.close()
                self._process = None


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Gap:
    """A detected gap in range bar data."""
    symbol: str
    threshold_dbps: int
    gap_start_ms: int
    gap_end_ms: int
    gap_hours: float
    close_before: float
    open_after: float
    price_gap_dbps: float
    gap_type: str  # "temporal", "price", "both"

    @property
    def gap_start_dt(self) -> str:
        return datetime.fromtimestamp(self.gap_start_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def gap_end_dt(self) -> str:
        return datetime.fromtimestamp(self.gap_end_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class CoverageSummary:
    """Coverage summary for a (symbol, threshold) pair."""
    symbol: str
    threshold_dbps: int
    total_bars: int
    earliest_ms: int
    latest_ms: int
    days_covered: float
    bars_per_day: float
    gap_count: int
    total_gap_hours: float
    max_gap_hours: float
    hours_since_last_bar: float = 0.0
    is_stale: bool = False

    @property
    def earliest_dt(self) -> str:
        return datetime.fromtimestamp(self.earliest_ms / 1000, tz=UTC).strftime("%Y-%m-%d")

    @property
    def latest_dt(self) -> str:
        return datetime.fromtimestamp(self.latest_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


@dataclass
class DurationAnomaly:
    """A bar with anomalously long duration."""

    symbol: str
    threshold_dbps: int
    close_time_ms: int
    duration_hours: float
    bar_open: float
    bar_close: float
    internal_range_dbps: float

    @property
    def close_dt(self) -> str:
        return datetime.fromtimestamp(
            self.close_time_ms / 1000, tz=UTC,
        ).strftime("%Y-%m-%d %H:%M")


@dataclass
class TradeIdGap:
    """A gap detected via trade ID continuity check."""

    symbol: str
    threshold_dbps: int
    close_time_ms: int
    prev_close_time_ms: int
    first_agg_trade_id: int
    prev_last_agg_trade_id: int
    missing_trades: int
    classification: str  # "registered", "unregistered", "pre_tracking"
    known_gap_reason: str | None = None

    @property
    def close_dt(self) -> str:
        return datetime.fromtimestamp(
            self.close_time_ms / 1000, tz=UTC,
        ).strftime("%Y-%m-%d %H:%M")

    @property
    def gap_date(self) -> str:
        """Date of the gap (from prev bar close time)."""
        return datetime.fromtimestamp(
            self.prev_close_time_ms / 1000, tz=UTC,
        ).strftime("%Y-%m-%d")


@dataclass
class BackfillQueueStatus:
    """Status of the backfill_requests queue."""

    by_status: dict[str, int] = field(default_factory=dict)
    oldest_pending: str | None = None
    max_running_hours: float = 0.0
    is_healthy: bool = True


@dataclass
class DetectionResult:
    """Full detection result."""
    gaps: list[Gap] = field(default_factory=list)
    coverage: list[CoverageSummary] = field(default_factory=list)
    stale_pairs: list[CoverageSummary] = field(default_factory=list)
    duration_anomalies: list[DurationAnomaly] = field(
        default_factory=list,
    )
    trade_id_gaps: list[TradeIdGap] = field(default_factory=list)
    backfill_queue: BackfillQueueStatus | None = None
    checked_pairs: int = 0
    skipped_pairs: int = 0  # pairs with 0 bars


# =============================================================================
# Connection
# =============================================================================

def get_client(
    host: str = "localhost",
    port: int = 8123,
) -> clickhouse_connect.driver.Client:
    """Get a ClickHouse client connection."""
    import clickhouse_connect
    client = clickhouse_connect.get_client(host=host, port=port)
    client.command("SELECT 1")  # verify
    return client


def connect() -> tuple[clickhouse_connect.driver.Client, SSHTunnel | None]:
    """Connect to ClickHouse based on environment configuration.

    Returns (client, tunnel). Caller must stop tunnel when done.
    """
    mode = os.getenv("RANGEBAR_MODE", "remote").lower()
    ch_host = os.getenv("CLICKHOUSE_HOST", "localhost")
    ch_port = int(os.getenv("CLICKHOUSE_PORT", "8123"))

    if mode == "local":
        return get_client(ch_host, ch_port), None

    if mode == "remote":
        ssh_alias = os.getenv("RANGEBAR_CH_HOSTS", "bigblack").split(",")[0].strip()
        # Check if already tunneled (e.g., mise run preflight already opened it)
        if _is_port_open("localhost", 18123, timeout=0.5):
            return get_client("localhost", 18123), None
        # Try direct connection to resolved SSH alias
        try:
            result = subprocess.run(
                ["ssh", "-G", ssh_alias],
                capture_output=True, text=True, timeout=2, check=False,
            )
            for line in result.stdout.splitlines():
                if line.startswith("hostname "):
                    resolved_ip = line.split()[1]
                    if _is_port_open(resolved_ip, 8123, timeout=2.0):
                        return get_client(resolved_ip, 8123), None
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        # Fallback: SSH tunnel
        tunnel = SSHTunnel(ssh_alias)
        local_port = tunnel.start()
        return get_client("localhost", local_port), tunnel

    # auto mode: try localhost first
    if _is_port_open(ch_host, ch_port, timeout=1.0):
        return get_client(ch_host, ch_port), None

    # Fallback to remote
    ssh_alias = os.getenv("RANGEBAR_CH_HOSTS", "bigblack").split(",")[0].strip()
    if ssh_alias:
        tunnel = SSHTunnel(ssh_alias)
        local_port = tunnel.start()
        return get_client("localhost", local_port), tunnel

    print("ERROR: No ClickHouse connection available.", file=sys.stderr)
    print("Set RANGEBAR_MODE=local or RANGEBAR_CH_HOSTS=<ssh-alias>", file=sys.stderr)
    sys.exit(2)


# =============================================================================
# Gap Detection Queries
# =============================================================================

def discover_pairs(client: clickhouse_connect.driver.Client) -> list[tuple[str, int]]:
    """Discover all (symbol, threshold) pairs present in ClickHouse."""
    result = client.query("""
        SELECT DISTINCT symbol, threshold_decimal_bps
        FROM rangebar_cache.range_bars
        ORDER BY symbol, threshold_decimal_bps
    """)
    return [(row[0], row[1]) for row in result.result_rows]


def detect_gaps_for_pair(
    client: clickhouse_connect.driver.Client,
    symbol: str,
    threshold: int,
    min_gap_ms: int,
    price_gap_dbps: float,
    recent_cutoff_ms: int | None = None,
    ouroboros_mode: str = "year",
) -> tuple[list[Gap], CoverageSummary | None]:
    """Detect gaps for a single (symbol, threshold) pair.

    Uses the SSoT SQL pattern from gap_detection.sql adapted for
    clickhouse_connect parameterized queries.
    """
    # First get coverage summary
    where_clause = ""
    params: dict = {"symbol": symbol, "threshold": threshold, "ouroboros": ouroboros_mode}
    if recent_cutoff_ms is not None:
        where_clause = "AND close_time_ms >= {cutoff:Int64}"
        params["cutoff"] = recent_cutoff_ms

    summary_query = f"""
        SELECT
            count() AS total_bars,
            min(close_time_ms) AS earliest_ms,
            max(close_time_ms) AS latest_ms
        FROM rangebar_cache.range_bars FINAL
        WHERE symbol = {{symbol:String}}
          AND threshold_decimal_bps = {{threshold:UInt32}}
          AND ouroboros_mode = {{ouroboros:String}}
          {where_clause}
    """
    result = client.query(summary_query, parameters=params)
    row = result.result_rows[0]
    total_bars, earliest_ms, latest_ms = row[0], row[1], row[2]

    if total_bars == 0:
        return [], None

    days_covered = (latest_ms - earliest_ms) / (86400 * 1000)
    bars_per_day = total_bars / max(days_covered, 0.001)

    # Detect temporal and price gaps using window functions (lagInFrame)
    gap_query = f"""
        SELECT
            prev_close_time_ms,
            close_time_ms,
            delta_ms,
            prev_close,
            curr_open,
            price_gap_dbps
        FROM (
            SELECT
                close_time_ms,
                open AS curr_open,
                close,
                lagInFrame(close_time_ms, 1) OVER (ORDER BY close_time_ms) AS prev_close_time_ms,
                lagInFrame(close, 1) OVER (ORDER BY close_time_ms) AS prev_close,
                close_time_ms - lagInFrame(close_time_ms, 1) OVER (ORDER BY close_time_ms) AS delta_ms,
                abs(open - lagInFrame(close, 1) OVER (ORDER BY close_time_ms))
                    / nullIf(lagInFrame(close, 1) OVER (ORDER BY close_time_ms), 0) * 10000 AS price_gap_dbps,
                row_number() OVER (ORDER BY close_time_ms) AS rn
            FROM rangebar_cache.range_bars FINAL
            WHERE symbol = {{symbol:String}}
              AND threshold_decimal_bps = {{threshold:UInt32}}
              AND ouroboros_mode = {{ouroboros:String}}
              {where_clause}
        )
        WHERE rn > 1
          AND (delta_ms >= {{min_gap_ms:Int64}} OR price_gap_dbps >= {{price_dbps:Float64}})
        ORDER BY delta_ms DESC
    """
    params["min_gap_ms"] = min_gap_ms
    params["price_dbps"] = price_gap_dbps

    result = client.query(gap_query, parameters=params)

    gaps: list[Gap] = []
    for row in result.result_rows:
        prev_ts, curr_ts, delta_ms, prev_close, curr_open, pgap = row
        gap_hours = delta_ms / 3600000.0
        is_temporal = delta_ms >= min_gap_ms
        is_price = pgap is not None and pgap >= price_gap_dbps

        if is_temporal and is_price:
            gap_type = "both"
        elif is_temporal:
            gap_type = "temporal"
        else:
            gap_type = "price"

        gaps.append(Gap(
            symbol=symbol,
            threshold_dbps=threshold,
            gap_start_ms=prev_ts,
            gap_end_ms=curr_ts,
            gap_hours=round(gap_hours, 2),
            close_before=prev_close,
            open_after=curr_open,
            price_gap_dbps=round(pgap, 1) if pgap is not None else 0.0,
            gap_type=gap_type,
        ))

    coverage = CoverageSummary(
        symbol=symbol,
        threshold_dbps=threshold,
        total_bars=total_bars,
        earliest_ms=earliest_ms,
        latest_ms=latest_ms,
        days_covered=round(days_covered, 1),
        bars_per_day=round(bars_per_day, 1),
        gap_count=len([g for g in gaps if g.gap_type in ("temporal", "both")]),
        total_gap_hours=round(sum(g.gap_hours for g in gaps if g.gap_type in ("temporal", "both")), 1),
        max_gap_hours=round(max((g.gap_hours for g in gaps if g.gap_type in ("temporal", "both")), default=0.0), 2),
    )

    return gaps, coverage


# =============================================================================
# Duration Anomaly Detection (Phase 10e — Feb 27 incident fix)
# =============================================================================

def detect_duration_anomalies(
    client: clickhouse_connect.driver.Client,
    symbol: str,
    threshold: int,
    max_duration_hours: float,
    recent_cutoff_ms: int | None = None,
    ouroboros_mode: str = "year",
) -> list[DurationAnomaly]:
    """Detect bars with anomalously long duration.

    A BTCUSDT@1000 bar normally takes minutes to hours. A 39h+ bar
    indicates missed data during that period (Feb 27 incident).
    """
    max_dur_us = int(max_duration_hours * 3600 * 1e6)
    params: dict = {
        "symbol": symbol,
        "threshold": threshold,
        "ouroboros": ouroboros_mode,
        "max_dur_us": max_dur_us,
    }
    where_clause = ""
    if recent_cutoff_ms is not None:
        where_clause = "AND close_time_ms >= {cutoff:Int64}"
        params["cutoff"] = recent_cutoff_ms

    query = f"""
        SELECT
            close_time_ms,
            duration_us / 1e6 / 3600 AS duration_hours,
            open AS bar_open,
            close AS bar_close,
            abs(open - close)
                / nullIf(close, 0) * 10000 AS range_dbps
        FROM rangebar_cache.range_bars FINAL
        WHERE symbol = {{symbol:String}}
          AND threshold_decimal_bps = {{threshold:UInt32}}
          AND ouroboros_mode = {{ouroboros:String}}
          AND duration_us > {{max_dur_us:Int64}}
          {where_clause}
        ORDER BY duration_us DESC
        LIMIT 100
    """
    result = client.query(query, parameters=params)
    anomalies: list[DurationAnomaly] = []
    for row in result.result_rows:
        ts, dur_h, o, c, rng = row
        anomalies.append(DurationAnomaly(
            symbol=symbol,
            threshold_dbps=threshold,
            close_time_ms=ts,
            duration_hours=round(dur_h, 1),
            bar_open=o,
            bar_close=c,
            internal_range_dbps=round(rng, 1) if rng else 0.0,
        ))
    return anomalies


# =============================================================================
# Backfill Queue Health (Phase 10d)
# =============================================================================

def check_backfill_queue(
    client: clickhouse_connect.driver.Client,
) -> BackfillQueueStatus:
    """Check health of the backfill_requests queue."""
    status = BackfillQueueStatus()
    try:
        result = client.query("""
            SELECT
                status,
                count() AS n,
                min(toString(requested_at)) AS oldest,
                max(
                    CASE WHEN status = 'running'
                        THEN dateDiff(
                            'hour', started_at, now()
                        )
                        ELSE 0
                    END
                ) AS max_running_h
            FROM rangebar_cache.backfill_requests FINAL
            GROUP BY status
        """)
        for row in result.result_rows:
            s, n, oldest, max_h = row
            status.by_status[s] = n
            if s == "pending" and oldest:
                status.oldest_pending = oldest
            if s == "running" and max_h:
                status.max_running_hours = float(max_h)
        pending = status.by_status.get("pending", 0)
        if pending > 50 or status.max_running_hours > 4:
            status.is_healthy = False
    except (OSError, RuntimeError):
        # Table may not exist — that's OK
        pass
    return status


# =============================================================================
# Trade ID Continuity (Phase 10f — zero-tolerance, registry-gated)
# =============================================================================

def _load_known_gaps() -> dict[str, list[tuple]]:
    """Load known_gaps from symbols.toml for gap classification.

    Returns dict mapping symbol → list of (start_date, end_date, reason).
    Standalone: reads TOML directly, no rangebar package dependency.
    """
    import tomllib as _tomllib
    from pathlib import Path

    # Try repo-root symlink first, then package data path
    candidates = [
        Path(__file__).parent.parent / "symbols.toml",
        Path(__file__).parent.parent / "python" / "rangebar"
        / "data" / "symbols.toml",
    ]
    for path in candidates:
        if path.exists():
            raw = _tomllib.loads(path.read_bytes().decode())
            result: dict[str, list[tuple]] = {}
            for sym, data in raw.get("symbols", {}).items():
                gaps = data.get("known_gaps", [])
                if gaps:
                    result[sym] = [
                        (g["start_date"], g["end_date"],
                         g.get("reason", ""))
                        for g in gaps
                    ]
            return result
    return {}


def classify_trade_id_gap(
    symbol: str,
    gap_date_str: str,
    known_gaps: dict[str, list[tuple]],
) -> tuple[str, str | None]:
    """Classify a trade ID gap against the symbol registry.

    Returns (classification, reason) where classification is
    "registered" or "unregistered".
    """
    from datetime import date as _date

    gap_d = _date.fromisoformat(gap_date_str)
    sym_gaps = known_gaps.get(symbol, [])
    for start_d, end_d, reason in sym_gaps:
        if start_d <= gap_d <= end_d:
            return "registered", reason
    return "unregistered", None


def detect_trade_id_gaps(
    client: clickhouse_connect.driver.Client,
    symbol: str,
    threshold: int,
    known_gaps: dict[str, list[tuple]],
    recent_cutoff_ms: int | None = None,
    ouroboros_mode: str = "year",
) -> list[TradeIdGap]:
    """Detect trade ID continuity breaks.

    For consecutive bars, first_agg_trade_id[n+1] should equal
    last_agg_trade_id[n] + 1. Any gap means dropped trades.
    """
    params: dict = {
        "symbol": symbol,
        "threshold": threshold,
        "ouroboros": ouroboros_mode,
    }
    where_clause = ""
    if recent_cutoff_ms is not None:
        where_clause = "AND close_time_ms >= {cutoff:Int64}"
        params["cutoff"] = recent_cutoff_ms

    query = f"""
        SELECT
            close_time_ms,
            first_agg_trade_id,
            last_agg_trade_id,
            prev_last_id,
            prev_close_time_ms,
            missing_trades
        FROM (
            SELECT
                close_time_ms,
                first_agg_trade_id,
                last_agg_trade_id,
                lagInFrame(last_agg_trade_id, 1)
                    OVER w AS prev_last_id,
                lagInFrame(close_time_ms, 1)
                    OVER w AS prev_close_time_ms,
                first_agg_trade_id
                    - lagInFrame(last_agg_trade_id, 1)
                    OVER w - 1 AS missing_trades,
                row_number() OVER w AS rn
            FROM rangebar_cache.range_bars FINAL
            WHERE symbol = {{symbol:String}}
              AND threshold_decimal_bps = {{threshold:UInt32}}
              AND ouroboros_mode = {{ouroboros:String}}
              AND first_agg_trade_id > 0
              AND last_agg_trade_id > 0
              {where_clause}
            WINDOW w AS (ORDER BY close_time_ms)
        )
        WHERE rn > 1 AND missing_trades != 0
        ORDER BY close_time_ms
    """
    result = client.query(query, parameters=params)

    gaps: list[TradeIdGap] = []
    for row in result.result_rows:
        (ts, first_id, _last_id,
         prev_last_id, prev_ts, missing) = row
        gap_date_str = datetime.fromtimestamp(
            prev_ts / 1000, tz=UTC,
        ).strftime("%Y-%m-%d")
        classification, reason = classify_trade_id_gap(
            symbol, gap_date_str, known_gaps,
        )
        gaps.append(TradeIdGap(
            symbol=symbol,
            threshold_dbps=threshold,
            close_time_ms=ts,
            prev_close_time_ms=prev_ts,
            first_agg_trade_id=first_id,
            prev_last_agg_trade_id=prev_last_id,
            missing_trades=missing,
            classification=classification,
            known_gap_reason=reason,
        ))
    return gaps


# =============================================================================
# Main Detection Loop
# =============================================================================

def run_detection(
    client: clickhouse_connect.driver.Client,
    symbols: list[str] | None = None,
    thresholds: list[int] | None = None,
    min_gap_hours: float = DEFAULT_MIN_GAP_HOURS,
    price_gap_dbps: float = DEFAULT_PRICE_GAP_DBPS,
    recent_days: int | None = None,
    max_stale_hours: float | None = None,
    max_bar_duration_hours: float | None = None,
    check_backfill: bool = False,
    trade_id_continuity: bool = False,
    ouroboros_mode: str = "year",
) -> DetectionResult:
    """Run gap detection across all specified (symbol, threshold) pairs."""
    min_gap_ms = int(min_gap_hours * 3600 * 1000)
    recent_cutoff_ms = None
    if recent_days is not None:
        recent_cutoff_ms = int(
            (time.time() - recent_days * 86400) * 1000,
        )

    now_ms = int(time.time() * 1000)

    # Determine which pairs to check
    if symbols is None and thresholds is None:
        existing_pairs = discover_pairs(client)
        pairs = existing_pairs
    else:
        syms = (
            symbols
            if symbols is not None
            else [f"{b}USDT" for b in ALL_BASES]
        )
        thds = (
            thresholds
            if thresholds is not None
            else list(THRESHOLDS)
        )
        pairs = [(s, t) for s in syms for t in thds]

    result = DetectionResult()

    # Load known gaps once for trade ID classification
    known_gaps_registry: dict[str, list[tuple]] = {}
    if trade_id_continuity:
        known_gaps_registry = _load_known_gaps()

    for symbol, threshold in pairs:
        gaps, coverage = detect_gaps_for_pair(
            client, symbol, threshold,
            min_gap_ms=min_gap_ms,
            price_gap_dbps=price_gap_dbps,
            recent_cutoff_ms=recent_cutoff_ms,
            ouroboros_mode=ouroboros_mode,
        )
        if coverage is None:
            result.skipped_pairs += 1
            continue
        result.checked_pairs += 1
        result.gaps.extend(gaps)

        # Freshness check
        coverage.hours_since_last_bar = (
            (now_ms - coverage.latest_ms) / 3600000.0
        )
        if (
            max_stale_hours is not None
            and coverage.hours_since_last_bar > max_stale_hours
        ):
            coverage.is_stale = True
            result.stale_pairs.append(coverage)

        result.coverage.append(coverage)

        # Duration anomaly detection (Phase 10e)
        if max_bar_duration_hours is not None:
            anomalies = detect_duration_anomalies(
                client, symbol, threshold,
                max_duration_hours=max_bar_duration_hours,
                recent_cutoff_ms=recent_cutoff_ms,
                ouroboros_mode=ouroboros_mode,
            )
            result.duration_anomalies.extend(anomalies)

        # Trade ID continuity (Phase 10f)
        if trade_id_continuity:
            tid_gaps = detect_trade_id_gaps(
                client, symbol, threshold,
                known_gaps=known_gaps_registry,
                recent_cutoff_ms=recent_cutoff_ms,
                ouroboros_mode=ouroboros_mode,
            )
            result.trade_id_gaps.extend(tid_gaps)

    # Sort gaps by severity (largest first)
    result.gaps.sort(key=lambda g: g.gap_hours, reverse=True)
    result.stale_pairs.sort(
        key=lambda c: c.hours_since_last_bar, reverse=True,
    )
    result.duration_anomalies.sort(
        key=lambda a: a.duration_hours, reverse=True,
    )
    result.trade_id_gaps.sort(
        key=lambda g: abs(g.missing_trades), reverse=True,
    )

    # Backfill queue health (Phase 10d)
    if check_backfill:
        result.backfill_queue = check_backfill_queue(client)

    return result


# =============================================================================
# Output Formatting
# =============================================================================

def print_text_report(result: DetectionResult, summary_only: bool = False) -> None:
    """Print human-readable report."""
    temporal_gaps = [g for g in result.gaps if g.gap_type in ("temporal", "both")]
    price_gaps = [g for g in result.gaps if g.gap_type in ("price", "both")]

    print("=" * 90)
    print("  RANGE BAR GAP DETECTION REPORT")
    print(f"  Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 90)
    print()
    print(f"  Pairs checked: {result.checked_pairs}")
    print(f"  Pairs skipped (no data): {result.skipped_pairs}")
    print(f"  Temporal gaps found: {len(temporal_gaps)}")
    print(f"  Price gaps found: {len(price_gaps)}")
    print(f"  Stale pairs: {len(result.stale_pairs)}")
    print(
        f"  Duration anomalies: "
        f"{len(result.duration_anomalies)}"
    )
    if result.trade_id_gaps:
        unreg = [
            g for g in result.trade_id_gaps
            if g.classification == "unregistered"
        ]
        reg = len(result.trade_id_gaps) - len(unreg)
        print(
            f"  Trade ID gaps: {len(unreg)} unregistered, "
            f"{reg} registered"
        )
    print()

    if not summary_only and temporal_gaps:
        print("-" * 90)
        print("  TEMPORAL GAPS (sorted by duration, largest first)")
        print("-" * 90)
        print(f"  {'Symbol':<12} {'Threshold':>9} {'Gap Start':<20} {'Gap End':<20} {'Hours':>8} {'Type':<8}")
        print(f"  {'------':<12} {'---------':>9} {'---------':<20} {'-------':<20} {'-----':>8} {'----':<8}")
        for gap in temporal_gaps:
            print(
                f"  {gap.symbol:<12} {gap.threshold_dbps:>6} dbps "
                f"{gap.gap_start_dt:<20} {gap.gap_end_dt:<20} "
                f"{gap.gap_hours:>8.1f} {gap.gap_type:<8}"
            )
        print()

    if not summary_only and price_gaps:
        print("-" * 90)
        print("  PRICE GAPS (sorted by price jump magnitude)")
        print("-" * 90)
        price_gaps_sorted = sorted(price_gaps, key=lambda g: g.price_gap_dbps, reverse=True)
        print(f"  {'Symbol':<12} {'Threshold':>9} {'Time':<20} {'Close':>12} {'Open':>12} {'Gap (dbps)':>12}")
        print(f"  {'------':<12} {'---------':>9} {'----':<20} {'-----':>12} {'----':>12} {'----------':>12}")
        for gap in price_gaps_sorted[:50]:  # Cap at 50
            print(
                f"  {gap.symbol:<12} {gap.threshold_dbps:>6} dbps "
                f"{gap.gap_start_dt:<20} "
                f"{gap.close_before:>12.4f} {gap.open_after:>12.4f} "
                f"{gap.price_gap_dbps:>12.1f}"
            )
        if len(price_gaps_sorted) > 50:
            print(f"  ... and {len(price_gaps_sorted) - 50} more price gaps")
        print()

    # Stale data report
    if result.stale_pairs:
        print("-" * 90)
        print("  STALE DATA (hours since last bar exceeds threshold)")
        print("-" * 90)
        print(f"  {'Symbol':<12} {'Threshold':>9} {'Last Bar':<12} {'Hours Stale':>12} {'Bars/Day':>10}")
        print(f"  {'------':<12} {'---------':>9} {'--------':<12} {'-----------':>12} {'--------':>10}")
        for cov in result.stale_pairs:
            print(
                f"  {cov.symbol:<12} {cov.threshold_dbps:>6} dbps "
                f"{cov.latest_dt:<12} {cov.hours_since_last_bar:>12.1f} "
                f"{cov.bars_per_day:>10.1f}"
            )
        print()

    # Coverage summary
    if result.coverage:
        print("-" * 90)
        print("  COVERAGE SUMMARY")
        print("-" * 90)
        print(
            f"  {'Symbol':<12} {'Threshold':>9} {'Bars':>12} {'From':<12} {'To':<12} "
            f"{'Days':>7} {'Bars/Day':>10} {'Gaps':>5} {'Gap Hrs':>9}"
        )
        print(
            f"  {'------':<12} {'---------':>9} {'----':>12} {'----':<12} {'--':<12} "
            f"{'----':>7} {'--------':>10} {'----':>5} {'-------':>9}"
        )
        for cov in sorted(result.coverage, key=lambda c: (c.symbol, c.threshold_dbps)):
            print(
                f"  {cov.symbol:<12} {cov.threshold_dbps:>6} dbps "
                f"{cov.total_bars:>12,} {cov.earliest_dt:<12} {cov.latest_dt:<12} "
                f"{cov.days_covered:>7.0f} {cov.bars_per_day:>10.1f} "
                f"{cov.gap_count:>5} {cov.total_gap_hours:>9.1f}"
            )
        print()

    # Duration anomalies (Phase 10e)
    if result.duration_anomalies:
        print("-" * 90)
        print("  DURATION ANOMALIES (bars exceeding threshold)")
        print("-" * 90)
        print(
            f"  {'Symbol':<12} {'Threshold':>9} "
            f"{'Close Time':<20} {'Duration(h)':>12} "
            f"{'Open':>12} {'Close':>12}"
        )
        print(
            f"  {'------':<12} {'---------':>9} "
            f"{'----------':<20} {'-----------':>12} "
            f"{'----':>12} {'-----':>12}"
        )
        for a in result.duration_anomalies[:50]:
            print(
                f"  {a.symbol:<12} {a.threshold_dbps:>6} dbps "
                f"{a.close_dt:<20} {a.duration_hours:>12.1f} "
                f"{a.bar_open:>12.2f} {a.bar_close:>12.2f}"
            )
        if len(result.duration_anomalies) > 50:
            n_more = len(result.duration_anomalies) - 50
            print(f"  ... and {n_more} more")
        print()

    # Trade ID continuity (Phase 10f)
    if result.trade_id_gaps:
        unreg = [
            g for g in result.trade_id_gaps
            if g.classification == "unregistered"
        ]
        reg = [
            g for g in result.trade_id_gaps
            if g.classification == "registered"
        ]
        if not summary_only and unreg:
            print("-" * 90)
            print(
                "  TRADE ID CONTINUITY — UNREGISTERED "
                "(zero tolerance)"
            )
            print("-" * 90)
            print(
                f"  {'Symbol':<12} {'Threshold':>9} "
                f"{'Time':<20} {'Missing':>14} "
                f"{'Status':<14}"
            )
            print(
                f"  {'------':<12} {'---------':>9} "
                f"{'----':<20} {'-------':>14} "
                f"{'------':<14}"
            )
            for g in unreg[:100]:
                print(
                    f"  {g.symbol:<12} "
                    f"{g.threshold_dbps:>6} dbps "
                    f"{g.close_dt:<20} "
                    f"{g.missing_trades:>14,} "
                    f"UNREGISTERED"
                )
            if len(unreg) > 100:
                print(f"  ... and {len(unreg) - 100} more")
            print()
        if not summary_only and reg:
            print("-" * 90)
            print(
                "  TRADE ID CONTINUITY — REGISTERED "
                "(informational)"
            )
            print("-" * 90)
            for g in reg[:20]:
                print(
                    f"  {g.symbol:<12} "
                    f"{g.threshold_dbps:>6} dbps "
                    f"{g.gap_date} "
                    f"{g.missing_trades:>14,} trades "
                    f"({g.known_gap_reason})"
                )
            if len(reg) > 20:
                print(f"  ... and {len(reg) - 20} more")
            print()

    # Backfill queue health (Phase 10d)
    if result.backfill_queue is not None:
        bq = result.backfill_queue
        print("-" * 90)
        print("  BACKFILL QUEUE STATUS")
        print("-" * 90)
        for status_name, count in sorted(bq.by_status.items()):
            print(f"  {status_name:<12}: {count}")
        if bq.max_running_hours > 0:
            print(
                f"  Max running time: {bq.max_running_hours:.1f}h"
            )
        if not bq.is_healthy:
            print("  WARNING: Queue unhealthy (>50 pending or "
                  "running >4h)")
        print()

    # Final verdict
    print("=" * 90)
    issues = []
    if temporal_gaps:
        issues.append(
            f"{len(temporal_gaps)} temporal gap(s), "
            f"worst: {temporal_gaps[0].gap_hours:.1f}h "
            f"({temporal_gaps[0].symbol}"
            f"@{temporal_gaps[0].threshold_dbps})"
        )
    if result.stale_pairs:
        worst = result.stale_pairs[0]
        issues.append(
            f"{len(result.stale_pairs)} stale pair(s), "
            f"worst: {worst.hours_since_last_bar:.1f}h "
            f"({worst.symbol}@{worst.threshold_dbps})"
        )
    if result.duration_anomalies:
        worst_d = result.duration_anomalies[0]
        issues.append(
            f"{len(result.duration_anomalies)} duration "
            f"anomaly(s), worst: {worst_d.duration_hours:.1f}h "
            f"({worst_d.symbol}@{worst_d.threshold_dbps})"
        )
    unreg_tid = [
        g for g in result.trade_id_gaps
        if g.classification == "unregistered"
    ]
    if unreg_tid:
        issues.append(
            f"{len(unreg_tid)} unregistered trade ID "
            f"gap(s)"
        )
    bq = result.backfill_queue
    if bq is not None and not bq.is_healthy:
        issues.append("backfill queue unhealthy")
    if issues:
        print(f"  ISSUES DETECTED -- {'; '.join(issues)}")
    else:
        print(
            "  CLEAN -- no gaps, stale data, "
            "or duration anomalies detected"
        )
    print("=" * 90)


def print_json_report(result: DetectionResult) -> None:
    """Print JSON report for programmatic consumption."""
    output: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "checked_pairs": result.checked_pairs,
        "skipped_pairs": result.skipped_pairs,
        "temporal_gap_count": len(
            [g for g in result.gaps
             if g.gap_type in ("temporal", "both")]
        ),
        "price_gap_count": len(
            [g for g in result.gaps
             if g.gap_type in ("price", "both")]
        ),
        "stale_pair_count": len(result.stale_pairs),
        "duration_anomaly_count": len(result.duration_anomalies),
        "gaps": [
            {
                **dict(asdict(g).items()),
                "gap_start": g.gap_start_dt,
                "gap_end": g.gap_end_dt,
            }
            for g in result.gaps
        ],
        "stale_pairs": [
            {
                "symbol": c.symbol,
                "threshold_dbps": c.threshold_dbps,
                "latest_bar": c.latest_dt,
                "hours_since_last_bar": round(
                    c.hours_since_last_bar, 1,
                ),
                "bars_per_day": c.bars_per_day,
            }
            for c in result.stale_pairs
        ],
        "duration_anomalies": [
            asdict(a) for a in result.duration_anomalies
        ],
        "trade_id_gaps": [
            asdict(g) for g in result.trade_id_gaps
        ],
        "trade_id_unregistered_count": len([
            g for g in result.trade_id_gaps
            if g.classification == "unregistered"
        ]),
        "coverage": [asdict(c) for c in result.coverage],
    }
    if result.backfill_queue is not None:
        output["backfill_queue"] = asdict(result.backfill_queue)
    print(json.dumps(output, indent=2))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect gaps in ClickHouse range_bars data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_gaps.py                        # All symbols, all thresholds
  python detect_gaps.py --symbol BTCUSDT       # Single symbol
  python detect_gaps.py --threshold 500        # Single threshold, all symbols
  python detect_gaps.py --min-gap-hours 3      # Lower gap threshold to 3 hours
  python detect_gaps.py --recent-days 7        # Only check last 7 days
  python detect_gaps.py --json                 # Machine-readable output
  python detect_gaps.py --summary              # Skip individual gap listing
        """,
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Check specific symbol (e.g., BTCUSDT). Default: all discovered symbols.",
    )
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Check specific threshold in dbps (e.g., 500). Default: all discovered thresholds.",
    )
    parser.add_argument(
        "--min-gap-hours", type=float, default=DEFAULT_MIN_GAP_HOURS,
        help=f"Minimum temporal gap to report in hours (default: {DEFAULT_MIN_GAP_HOURS}).",
    )
    parser.add_argument(
        "--price-gap-dbps", type=float, default=DEFAULT_PRICE_GAP_DBPS,
        help=f"Minimum price gap to report in dbps (default: {DEFAULT_PRICE_GAP_DBPS}).",
    )
    parser.add_argument(
        "--recent-days", type=int, default=None,
        help="Only check bars from the last N days (default: check all data).",
    )
    parser.add_argument(
        "--max-stale-hours", type=float, default=None,
        help="Flag pairs where last bar is older than N hours (freshness check). "
             "Default: disabled. Recommended: 72 for monitoring.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print coverage summary only, skip individual gap listing.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON for programmatic consumption.",
    )
    parser.add_argument(
        "--tier1-only", action="store_true",
        help="Only check TIER1 symbols (18 symbols x standard thresholds). "
             "Ignores non-TIER1 data in ClickHouse.",
    )
    parser.add_argument(
        "--ouroboros-mode", type=str, default=None,
        choices=["year", "month", "week"],
        help="Ouroboros mode filter (default: from RANGEBAR_OUROBOROS_MODE).",  # Issue #126
    )
    # Phase 10b (#122): Exhaustive scan
    parser.add_argument(
        "--exhaustive", action="store_true",
        help="Scan ALL data (no --recent-days window). "
             "Slower but catches aged-out gaps.",
    )
    # Phase 10e: Duration anomaly detection (Feb 27 fix)
    parser.add_argument(
        "--max-bar-duration-hours", type=float,
        default=None,
        help="Flag bars exceeding N hours duration. "
             "Recommended: 24 for TIER1.",
    )
    # Phase 10d: Backfill queue health
    parser.add_argument(
        "--check-backfill-queue", action="store_true",
        help="Check backfill_requests queue health.",
    )
    # Phase 10f (#123): Trade ID continuity
    parser.add_argument(
        "--trade-id-continuity", action="store_true",
        help="Check trade ID continuity (zero-tolerance, "
             "registry-gated). On by default with "
             "--exhaustive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Connect
    tunnel = None
    try:
        print("Connecting to ClickHouse...", file=sys.stderr)
        client, tunnel = connect()
        print("Connected.", file=sys.stderr)
    except (OSError, RuntimeError, ImportError) as e:
        print(f"ERROR: Failed to connect to ClickHouse: {e}", file=sys.stderr)
        return 2

    try:
        # Build symbol/threshold filters
        # --tier1-only: deterministic TIER1 x standard thresholds (ignores EXTRA data)
        if args.tier1_only and not args.symbol and not args.threshold:
            symbols = [f"{b}USDT" for b in TIER1_BASES]
            thresholds = list(THRESHOLDS)
        else:
            symbols = [args.symbol] if args.symbol else None
            thresholds = [args.threshold] if args.threshold else None

        # --exhaustive overrides --recent-days (Phase 10b)
        recent_days = args.recent_days
        if args.exhaustive:
            recent_days = None

        # --exhaustive implies --trade-id-continuity
        trade_id = (
            args.trade_id_continuity or args.exhaustive
        )

        print("Running gap detection...", file=sys.stderr)
        result = run_detection(
            client,
            symbols=symbols,
            thresholds=thresholds,
            min_gap_hours=args.min_gap_hours,
            price_gap_dbps=args.price_gap_dbps,
            recent_days=recent_days,
            max_stale_hours=args.max_stale_hours,
            max_bar_duration_hours=args.max_bar_duration_hours,
            check_backfill=args.check_backfill_queue,
            trade_id_continuity=trade_id,
            ouroboros_mode=args.ouroboros_mode or _resolve_ouroboros_mode(),
        )

        # Output
        if args.json:
            print_json_report(result)
        else:
            print_text_report(result, summary_only=args.summary)

        # Exit code: 1 if any actionable issues found
        temporal_gaps = [
            g for g in result.gaps
            if g.gap_type in ("temporal", "both")
        ]
        unreg_trade_id = [
            g for g in result.trade_id_gaps
            if g.classification == "unregistered"
        ]
        has_issues = bool(
            temporal_gaps
            or result.stale_pairs
            or result.duration_anomalies
            or unreg_trade_id
            or (result.backfill_queue
                and not result.backfill_queue.is_healthy)
        )
        return 1 if has_issues else 0

    except (OSError, RuntimeError, ValueError) as e:
        print(f"ERROR: Detection failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 2
    finally:
        try:
            client.close()
        except OSError as e:
            print(f"WARNING: Failed to close ClickHouse client: {e}", file=sys.stderr)
        if tunnel is not None:
            tunnel.stop()


if __name__ == "__main__":
    sys.exit(main())

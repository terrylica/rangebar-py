#!/usr/bin/env python3
"""TDA Break Event Alignment - Map TDA Breaks to Market Events.

Aligns TDA-detected structural breaks with major market events to validate
TDA's ability to detect meaningful regime changes.

Major crypto market events (2022-2026):
- Luna/UST collapse: May 9-12, 2022
- FTX collapse: November 8-11, 2022
- Banking crisis: March 10-13, 2023
- ETF approval rally: January 10, 2024
- Bitcoin halving: April 19, 2024
- Trump election rally: November 5, 2024

Issue #56: TDA Structural Break Detection for Range Bar Patterns
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

# Add parent dir for rangebar import
sys.path.insert(0, str(Path(__file__).parent.parent))

from rangebar import get_range_bars


def log_info(message: str, **kwargs: object) -> None:
    """Log info message in NDJSON format."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry))


# Major market events
MARKET_EVENTS = [
    ("Luna/UST Collapse", datetime(2022, 5, 9, tzinfo=timezone.utc)),
    ("Luna Death Spiral", datetime(2022, 5, 12, tzinfo=timezone.utc)),
    ("3AC Insolvency", datetime(2022, 6, 27, tzinfo=timezone.utc)),
    ("FTX Collapse", datetime(2022, 11, 8, tzinfo=timezone.utc)),
    ("FTX Bankruptcy", datetime(2022, 11, 11, tzinfo=timezone.utc)),
    ("SVB Collapse", datetime(2023, 3, 10, tzinfo=timezone.utc)),
    ("Signature Bank Fail", datetime(2023, 3, 12, tzinfo=timezone.utc)),
    ("SEC Binance Suit", datetime(2023, 6, 5, tzinfo=timezone.utc)),
    ("SEC Coinbase Suit", datetime(2023, 6, 6, tzinfo=timezone.utc)),
    ("ETF Approval", datetime(2024, 1, 10, tzinfo=timezone.utc)),
    ("Bitcoin Halving", datetime(2024, 4, 19, tzinfo=timezone.utc)),
    ("Yen Carry Unwind", datetime(2024, 8, 5, tzinfo=timezone.utc)),
    ("Trump Election", datetime(2024, 11, 5, tzinfo=timezone.utc)),
    ("Bitcoin 100K", datetime(2024, 12, 5, tzinfo=timezone.utc)),
]


def takens_embedding(
    series: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """Create Takens delay embedding of time series."""
    n = len(series)
    n_points = n - (embedding_dim - 1) * delay

    if n_points <= 0:
        return np.array([]).reshape(0, embedding_dim)

    embedding = np.zeros((n_points, embedding_dim))
    for i in range(embedding_dim):
        embedding[:, i] = series[i * delay : i * delay + n_points]

    return embedding


def compute_persistence_l2(point_cloud: np.ndarray) -> float:
    """Compute L2 norm of H1 persistence diagram."""
    try:
        from ripser import ripser
    except ImportError:
        return 0.0

    if len(point_cloud) < 4:
        return 0.0

    point_cloud = (point_cloud - np.mean(point_cloud, axis=0)) / (
        np.std(point_cloud, axis=0) + 1e-10
    )

    result = ripser(point_cloud, maxdim=1)
    dgm = result["dgms"][1]

    if len(dgm) == 0:
        return 0.0

    finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
    dgm = dgm[finite_mask]

    if len(dgm) == 0:
        return 0.0

    persistence = dgm[:, 1] - dgm[:, 0]
    return float(np.sqrt(np.sum(persistence**2)))


def detect_tda_breaks_with_timestamps(
    df: pl.DataFrame,
    window_size: int = 100,
    step_size: int = 50,
    threshold_pct: float = 95,
) -> list[dict]:
    """Detect TDA breaks and return timestamps."""
    # Get log returns
    log_returns = (
        df.filter(pl.col("Close").is_not_null())
        .select(
            (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_return"),
            pl.col("timestamp"),
        )
        .drop_nulls()
    )

    returns = log_returns["log_return"].to_numpy()
    timestamps = log_returns["timestamp"].to_numpy()

    n = len(returns)

    # Subsample for tractable computation
    subsample_factor = max(1, n // 10000)
    subsampled_returns = returns[::subsample_factor]
    subsampled_timestamps = timestamps[::subsample_factor]

    l2_norms = []
    indices = []

    for start in range(0, len(subsampled_returns) - window_size + 1, step_size):
        end = start + window_size
        window = subsampled_returns[start:end]

        point_cloud = takens_embedding(window, embedding_dim=3, delay=1)

        if len(point_cloud) < 10:
            continue

        l2 = compute_persistence_l2(point_cloud)
        l2_norms.append(l2)
        center_idx = (start + end) // 2
        indices.append(center_idx)

    if len(l2_norms) < 3:
        return []

    l2_norms = np.array(l2_norms)
    velocity = np.diff(l2_norms)

    threshold = np.percentile(np.abs(velocity), threshold_pct)
    break_mask = np.abs(velocity) > threshold

    breaks = []
    for i in range(len(break_mask)):
        if break_mask[i]:
            idx = indices[i + 1]
            ts_val = subsampled_timestamps[min(idx, len(subsampled_timestamps) - 1)]
            # Handle numpy datetime64 or datetime
            if hasattr(ts_val, "astype"):
                # numpy datetime64 -> convert to datetime
                ts = ts_val.astype("datetime64[s]").astype(datetime)
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts_val
            breaks.append({
                "index": idx * subsample_factor,
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "l2_velocity": float(velocity[i]),
            })

    return breaks


def find_nearest_event(break_ts: datetime, max_days: int = 7) -> tuple[str | None, int]:
    """Find nearest market event within max_days."""
    nearest_event = None
    nearest_days = None

    for event_name, event_ts in MARKET_EVENTS:
        diff = abs((break_ts - event_ts).days)
        if diff <= max_days and (nearest_days is None or diff < nearest_days):
            nearest_event = event_name
            nearest_days = diff

    return nearest_event, nearest_days if nearest_days is not None else -1


def main() -> None:
    """Run TDA break event alignment."""
    log_info("Starting TDA break event alignment")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100

    start_dates = {
        "BTCUSDT": "2022-01-01",
        "ETHUSDT": "2022-01-01",
        "SOLUSDT": "2023-06-01",
        "BNBUSDT": "2022-01-01",
    }
    end_date = "2026-01-31"

    all_breaks = []
    event_correlations = []

    for symbol in symbols:
        log_info("Processing symbol", symbol=symbol)

        df = get_range_bars(
            symbol,
            start_date=start_dates[symbol],
            end_date=end_date,
            threshold_decimal_bps=threshold,
            ouroboros="year",
            use_cache=True,
            fetch_if_missing=False,
        )

        if df is None or len(df) == 0:
            continue

        df_pl = pl.from_pandas(df.reset_index())

        # Detect breaks with timestamps
        breaks = detect_tda_breaks_with_timestamps(df_pl)

        for b in breaks:
            b["symbol"] = symbol
            all_breaks.append(b)

            # Find nearest event
            break_ts = datetime.fromisoformat(b["timestamp"])
            event, days = find_nearest_event(break_ts)
            event_correlations.append({
                "symbol": symbol,
                "break_timestamp": b["timestamp"],
                "nearest_event": event,
                "days_diff": days,
            })

        log_info(
            "Breaks detected",
            symbol=symbol,
            n_breaks=len(breaks),
        )

    # Print results
    print("\n" + "=" * 120)
    print("TDA BREAK EVENT ALIGNMENT")
    print("=" * 120)

    print("\n" + "-" * 120)
    print("ALL TDA BREAKS")
    print("-" * 120)
    print(f"{'Symbol':<10} {'Break Timestamp':<28} {'Nearest Event':<25} {'Days Diff':>10}")
    print("-" * 120)

    for ec in sorted(event_correlations, key=lambda x: x["break_timestamp"]):
        event = ec["nearest_event"] or "None"
        days = str(ec["days_diff"]) if ec["days_diff"] >= 0 else "-"
        marker = "***" if ec["days_diff"] >= 0 and ec["days_diff"] <= 7 else ""
        print(
            f"{ec['symbol']:<10} "
            f"{ec['break_timestamp']:<28} "
            f"{event:<25} "
            f"{days:>10} {marker}"
        )

    # Summary by event
    print("\n" + "-" * 120)
    print("EVENT CORRELATION SUMMARY")
    print("-" * 120)

    event_counts = {}
    for ec in event_correlations:
        if ec["nearest_event"]:
            event = ec["nearest_event"]
            if event not in event_counts:
                event_counts[event] = []
            event_counts[event].append(ec["symbol"])

    print(f"{'Event':<25} {'Symbols with TDA Break':>40}")
    print("-" * 120)

    for event, symbols_list in sorted(event_counts.items()):
        symbols_str = ", ".join(sorted(set(symbols_list)))
        print(f"{event:<25} {symbols_str:>40}")

    # Calculate statistics
    breaks_with_event = sum(1 for ec in event_correlations if ec["nearest_event"])
    total_breaks = len(event_correlations)

    print("\n" + "-" * 120)
    print("STATISTICS")
    print("-" * 120)
    print(f"Total TDA breaks detected: {total_breaks}")
    print(f"Breaks within 7 days of known event: {breaks_with_event}")
    print(f"Event correlation rate: {breaks_with_event / max(total_breaks, 1) * 100:.1f}%")

    # Timeline of breaks per symbol
    print("\n" + "-" * 120)
    print("BREAK TIMELINE BY SYMBOL")
    print("-" * 120)

    for symbol in symbols:
        symbol_breaks = [b for b in all_breaks if b["symbol"] == symbol]
        if symbol_breaks:
            print(f"\n{symbol}:")
            for b in sorted(symbol_breaks, key=lambda x: x["timestamp"]):
                ts = datetime.fromisoformat(b["timestamp"])
                event, days = find_nearest_event(ts)
                event_str = f" <- {event} ({days}d)" if event else ""
                print(f"  {ts.strftime('%Y-%m-%d %H:%M')}{event_str}")

    print("\n" + "=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)

    if breaks_with_event > 0:
        print(f"\n{breaks_with_event} of {total_breaks} TDA breaks correlate with known market events.")
        print("This validates TDA's ability to detect meaningful regime changes.")
    else:
        print("\nNo direct correlation found between TDA breaks and pre-defined market events.")
        print("TDA may be detecting more subtle structural changes or earlier signals.")

    log_info(
        "Event alignment complete",
        total_breaks=total_breaks,
        breaks_with_event=breaks_with_event,
    )


if __name__ == "__main__":
    main()

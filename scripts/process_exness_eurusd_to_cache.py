"""Process Exness EURUSD Raw_Spread tick data into range bars and cache in ClickHouse.

# Issue #144: Generate EURUSD range bars and cache in ClickHouse

Reads CSV tick data from local cache, generates range bars at multiple thresholds,
and stores them in bigblack ClickHouse for pattern research.

Usage:
    RANGEBAR_CH_HOSTS=bigblack uv run python scripts/process_exness_eurusd_to_cache.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl


def log(level: str, message: str, **kwargs: object) -> None:
    """Log in NDJSON format."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def load_exness_ticks(cache_dir: Path) -> pl.LazyFrame:
    """Load all Exness EURUSD tick data as a LazyFrame.

    Exness CSV format:
    "Exness","Symbol","Timestamp","Bid","Ask"
    "exness","EURUSD_Raw_Spread","2022-01-02 22:06:28.773Z",1.13734,1.1375899999999999
    """
    csv_files = sorted(cache_dir.glob("Exness_EURUSD_Raw_Spread_*.csv"))
    log("INFO", "Found CSV files", count=len(csv_files))

    if not csv_files:
        msg = f"No CSV files found in {cache_dir}"
        raise FileNotFoundError(msg)

    # Read all CSVs and concatenate
    dfs = []
    for csv_file in csv_files:
        df = pl.scan_csv(csv_file)
        dfs.append(df)

    combined = pl.concat(dfs)

    # Transform to standard tick format for rangebar
    # Use Bid price for range bars (execution price)
    return combined.select([
        pl.col("Timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%.3fZ").alias("timestamp"),
        pl.col("Bid").alias("price"),
        pl.lit(1.0).alias("quantity"),  # Volume not available for forex
    ]).sort("timestamp")


def process_ticks_to_range_bars(
    ticks: pl.LazyFrame,
    threshold_decimal_bps: int,
) -> pl.DataFrame:
    """Process ticks into range bars using process_trades_polars.

    For forex, we use Bid price as execution price since:
    - Exness Raw_Spread has Bid == Ask 97.81% of the time
    - We use Bid as the execution price
    """
    from rangebar import process_trades_polars

    log("INFO", "Starting range bar processing", threshold_dbps=threshold_decimal_bps)

    # Transform to expected format for process_trades_polars
    # Required columns: timestamp (int64 ms epoch), price, quantity
    trades = ticks.select([
        pl.col("timestamp").dt.epoch("ms").alias("timestamp"),  # Convert to epoch ms
        pl.col("price"),
        pl.col("quantity"),
    ])

    # Process with Rust via process_trades_polars
    bars_df = process_trades_polars(
        trades,
        threshold_decimal_bps=threshold_decimal_bps,
    )

    log("INFO", "Generated bars", count=len(bars_df), threshold_dbps=threshold_decimal_bps)

    return bars_df


def main() -> None:
    """Process EURUSD tick data to range bars at 50/100/200 dbps."""
    cache_dir = Path.home() / "Library/Caches/rangebar/forex/EURUSD_Raw_Spread"

    log("INFO", "Starting EURUSD range bar generation")

    # Load ticks
    ticks = load_exness_ticks(cache_dir)

    # Process at each threshold
    thresholds = [50, 100, 200]
    for threshold in thresholds:
        log("INFO", "Processing threshold", threshold_dbps=threshold)

        bars_df = process_ticks_to_range_bars(ticks, threshold)

        if len(bars_df) > 0:
            # Save to parquet for inspection (pandas DataFrame)
            output_path = cache_dir / f"range_bars_{threshold}dbps.parquet"
            bars_df.to_parquet(output_path)
            log("INFO", "Saved range bars",
                threshold_dbps=threshold,
                bars=len(bars_df),
                path=str(output_path))
        else:
            log("WARN", "No bars generated", threshold_dbps=threshold)

    log("INFO", "Processing complete")


if __name__ == "__main__":
    main()

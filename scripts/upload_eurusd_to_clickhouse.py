"""Upload EURUSD range bars to ClickHouse cache.

# Issue #144: Generate EURUSD range bars and cache in ClickHouse

Reads generated parquet files and uploads to bigblack ClickHouse.
Uses Polars for all data processing (no pandas).

Usage:
    RANGEBAR_CH_HOSTS=bigblack uv run python scripts/upload_eurusd_to_clickhouse.py
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


def upload_to_clickhouse(
    parquet_path: Path,
    symbol: str,
    threshold_decimal_bps: int,
) -> int:
    """Upload range bars to ClickHouse using Polars with SSH tunnel.

    Returns number of rows inserted.
    """
    import clickhouse_connect
    from rangebar.clickhouse.tunnel import SSHTunnel

    # Read parquet with Polars
    bars_df = pl.read_parquet(parquet_path)

    # The parquet has Open, High, Low, Close, Volume columns
    # Transform to ClickHouse schema
    insert_df = bars_df.with_columns([
        pl.lit(symbol).alias("symbol"),
        pl.lit(threshold_decimal_bps).cast(pl.UInt32).alias("threshold_decimal_bps"),
        # Timestamp column (datetime to epoch ms)
        (pl.col("timestamp").dt.epoch("ms")).alias("close_time_ms"),
        pl.col("Open").alias("open"),
        pl.col("High").alias("high"),
        pl.col("Low").alias("low"),
        pl.col("Close").alias("close"),
        pl.col("Volume").alias("volume"),
        # Defaults for forex (no microstructure)
        pl.lit(0.0).alias("vwap"),
        pl.lit(0.0).alias("buy_volume"),
        pl.lit(0.0).alias("sell_volume"),
        pl.lit(0).cast(pl.UInt32).alias("individual_trade_count"),
        pl.lit(0).cast(pl.UInt32).alias("agg_record_count"),
        pl.lit(f"{symbol}_{threshold_decimal_bps}_forex").alias("cache_key"),
        pl.lit("11.4.0").alias("rangebar_version"),
        pl.lit(0).cast(pl.Int64).alias("source_start_ts"),
        pl.lit(0).cast(pl.Int64).alias("source_end_ts"),
        pl.lit("none").alias("ouroboros_mode"),
    ]).select([
        "symbol",
        "threshold_decimal_bps",
        "close_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "buy_volume",
        "sell_volume",
        "individual_trade_count",
        "agg_record_count",
        "cache_key",
        "rangebar_version",
        "source_start_ts",
        "source_end_ts",
        "ouroboros_mode",
    ])

    # Connect via SSH tunnel to bigblack
    with SSHTunnel("bigblack") as local_port:
        client = clickhouse_connect.get_client(
            host="localhost",
            port=local_port,
            username="default",
            password="",
        )

        # Insert using arrow format (zero-copy from Polars)
        client.insert_arrow(
            "rangebar_cache.range_bars",
            insert_df.to_arrow(),
        )

    return len(insert_df)


def main() -> None:
    """Upload EURUSD range bars to ClickHouse."""
    cache_dir = Path.home() / "Library/Caches/rangebar/forex/EURUSD_Raw_Spread"

    log("INFO", "Starting ClickHouse upload")

    thresholds = [50, 100, 200]
    total_rows = 0

    for threshold in thresholds:
        parquet_path = cache_dir / f"range_bars_{threshold}dbps.parquet"
        if not parquet_path.exists():
            log("WARN", "Parquet file not found", threshold_dbps=threshold, path=str(parquet_path))
            continue

        log("INFO", "Uploading bars", threshold_dbps=threshold)

        rows = upload_to_clickhouse(parquet_path, "EURUSD", threshold)
        total_rows += rows
        log("INFO", "Uploaded", threshold_dbps=threshold, rows=rows)

    log("INFO", "Upload complete", total_rows=total_rows)


if __name__ == "__main__":
    main()

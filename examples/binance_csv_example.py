#!/usr/bin/env python3
"""Binance CSV loading example.

This example demonstrates how to load Binance aggTrades CSV files
and convert them to range bars for backtesting.

Binance aggTrades CSV format:
    agg_trade_id,price,quantity,first_trade_id,last_trade_id,timestamp,is_buyer_maker,is_best_match

You can download Binance historical data from:
    https://data.binance.vision/?prefix=data/spot/monthly/aggTrades/

Example files:
    BTCUSDT-aggTrades-2024-01.zip
    ETHUSDT-aggTrades-2024-01.zip
"""

import sys
from pathlib import Path

import pandas as pd

from rangebar import process_trades_to_dataframe


def load_binance_csv(csv_path: str, threshold_bps: int = 250) -> pd.DataFrame:
    """Load Binance aggTrades CSV and convert to range bars.

    Parameters
    ----------
    csv_path : str
        Path to Binance aggTrades CSV file
    threshold_bps : int
        Range bar threshold in 0.1 basis point units (250 = 25bps = 0.25%)

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame ready for backtesting

    Raises
    ------
    FileNotFoundError
        If CSV file doesn't exist
    ValueError
        If CSV is missing required columns
    """
    print(f"Loading Binance CSV: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = {"timestamp", "price", "quantity"}
    if not required_cols.issubset(df.columns):
        # Try alternate column names
        if "agg_trade_id" in df.columns:
            # Standard Binance format - already has correct names
            pass
        else:
            raise ValueError(
                f"CSV missing required columns. "
                f"Required: {required_cols}, Found: {set(df.columns)}"
            )

    print(f"  Loaded {len(df):,} trade records")
    print(f"  Date range: {pd.to_datetime(df['timestamp'], unit='ms').min()} → "
          f"{pd.to_datetime(df['timestamp'], unit='ms').max()}")
    print(f"  Price range: ${df['price'].min():.2f} → ${df['price'].max():.2f}")

    # Convert to range bars
    print(f"\nConverting to range bars (threshold = {threshold_bps * 0.1:.1f}bps = {threshold_bps * 0.001:.3f}%)...")

    range_bars = process_trades_to_dataframe(
        df[["timestamp", "price", "quantity"]],
        threshold_bps=threshold_bps
    )

    print(f"  Generated {len(range_bars):,} range bars")
    if len(range_bars) > 0:
        print(f"  Compression ratio: {len(df) / len(range_bars):.1f}x")
    else:
        print("  ⚠️  No range bars generated (price movement below threshold)")

    return range_bars


def create_sample_csv(output_path: str = "sample_binance_aggTrades.csv") -> str:
    """Create a sample Binance-format CSV for demonstration.

    Parameters
    ----------
    output_path : str
        Path to save sample CSV

    Returns
    -------
    str
        Path to created CSV file
    """
    print(f"Creating sample Binance aggTrades CSV: {output_path}")

    # Generate realistic sample data (1 week of minute-level data)
    base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
    base_price = 42000.0

    records = []
    price = base_price
    for i in range(10_000):  # ~1 week at 1 minute intervals
        # Simulate realistic price movement (cumulative random walk)
        # This creates larger price swings suitable for range bars
        price_change = (i % 100 - 50) * 2.0  # ±$100 per step
        price += price_change * 0.1  # Accumulate changes

        records.append({
            "agg_trade_id": 1000000 + i,
            "price": round(price, 2),
            "quantity": round(1.0 + (i % 10) * 0.1, 4),
            "first_trade_id": 2000000 + i * 10,
            "last_trade_id": 2000000 + i * 10 + 9,
            "timestamp": base_time + i * 60000,  # 1 minute intervals
            "is_buyer_maker": i % 2 == 0,
            "is_best_match": True,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"  Created {len(df):,} sample trade records")
    return output_path


def main():
    """Run Binance CSV example."""
    print("=" * 70)
    print("rangebar-py: Binance CSV Loading Example")
    print("=" * 70)
    print()

    # Check if user provided CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

        if not Path(csv_path).exists():
            print(f"❌ Error: File not found: {csv_path}")
            print("\nUsage:")
            print(f"  {sys.argv[0]} <path-to-binance-csv>")
            print(f"  {sys.argv[0]}  # Creates and uses sample CSV")
            sys.exit(1)
    else:
        print("No CSV path provided. Creating sample data...")
        print()
        csv_path = create_sample_csv()
        print()

    # Load and convert
    try:
        df = load_binance_csv(csv_path, threshold_bps=250)

        if len(df) == 0:
            print("\n⚠️  No range bars generated. Price movement may be below threshold.")
            print("   Try: Lower threshold_bps or use more volatile data")
            return

        print("\n" + "=" * 70)
        print("Range Bar Summary")
        print("=" * 70)

        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index type: {type(df.index).__name__}")

        print("\nFirst 10 range bars:")
        print(df.head(10).to_string())

        print("\nStatistics:")
        print(df.describe().to_string())

        # Save output
        output_path = "range_bars_output.csv"
        df.to_csv(output_path)
        print(f"\n✅ Range bars saved to: {output_path}")

        print("\n" + "=" * 70)
        print("✅ Binance CSV example completed successfully!")
        print("=" * 70)

        print("\nNext steps:")
        print("  - Use the output CSV with your backtesting framework")
        print("  - Try different threshold_bps values (100-500 recommended)")
        print("  - Compare range bars vs time-based bars")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

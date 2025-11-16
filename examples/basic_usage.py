#!/usr/bin/env python3
"""Basic rangebar-py usage example.

This example demonstrates the most straightforward way to use rangebar-py:
1. Create synthetic trade data
2. Convert to range bars
3. Display results
"""

from rangebar import process_trades_to_dataframe


def main():
    """Run basic usage example."""
    print("=" * 70)
    print("rangebar-py: Basic Usage Example")
    print("=" * 70)

    # Create synthetic trade data (simulating Binance aggTrades format)
    # Format: timestamp (ms), price, quantity
    print("\n1. Creating synthetic trade data (100 trades)...")

    trades = [
        {
            "timestamp": 1704067200000 + i * 1000,  # 1 second intervals
            "price": 42000.0 + i * 10.0,  # Price increases by $10 each trade
            "quantity": 1.0 + (i % 5) * 0.5,  # Volume varies
        }
        for i in range(100)
    ]

    print(f"   Generated {len(trades)} trades")
    print(f"   Price range: ${trades[0]['price']:.2f} → ${trades[-1]['price']:.2f}")
    print(f"   Time span: {(trades[-1]['timestamp'] - trades[0]['timestamp']) / 1000:.0f} seconds")

    # Convert to range bars with 0.25% threshold (25 basis points)
    print("\n2. Converting to range bars (threshold = 25bps = 0.25%)...")

    df = process_trades_to_dataframe(trades, threshold_bps=250)

    print(f"   Generated {len(df)} range bars from {len(trades)} trades")
    print(f"   Compression ratio: {len(trades) / len(df):.1f}x")

    # Display results
    print("\n3. Range Bar DataFrame Structure:")
    print(f"   Index: {type(df.index).__name__}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Shape: {df.shape}")

    print("\n4. First 5 range bars:")
    print(df.head().to_string())

    print("\n5. DataFrame Statistics:")
    print(df.describe().to_string())

    # Verify OHLCV invariants
    print("\n6. OHLCV Invariants Check:")
    high_check = (df["High"] >= df["Open"]).all() and (df["High"] >= df["Close"]).all()
    low_check = (df["Low"] <= df["Open"]).all() and (df["Low"] <= df["Close"]).all()
    volume_check = (df["Volume"] > 0).all()

    print(f"   High >= max(Open, Close): {'✅ PASS' if high_check else '❌ FAIL'}")
    print(f"   Low <= min(Open, Close): {'✅ PASS' if low_check else '❌ FAIL'}")
    print(f"   Volume > 0: {'✅ PASS' if volume_check else '❌ FAIL'}")

    # Show temporal distribution
    print("\n7. Temporal Distribution:")
    df_with_duration = df.copy()
    df_with_duration["Duration"] = df_with_duration.index.to_series().diff()
    avg_duration = df_with_duration["Duration"].mean()
    print(f"   Average bar duration: {avg_duration}")
    print(f"   Min bar duration: {df_with_duration['Duration'].min()}")
    print(f"   Max bar duration: {df_with_duration['Duration'].max()}")

    print("\n" + "=" * 70)
    print("✅ Basic usage example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

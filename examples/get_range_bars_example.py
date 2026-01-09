#!/usr/bin/env python3
"""get_range_bars() example - the recommended primary API.

This example demonstrates the single entry point for downstream users:
1. Fetch data and generate range bars in one call
2. Use threshold presets
3. Access microstructure data
4. Try different market types

This is the recommended approach for most users.
"""

from rangebar import (
    THRESHOLD_DECIMAL_MAX,
    THRESHOLD_DECIMAL_MIN,
    THRESHOLD_PRESETS,
    TIER1_SYMBOLS,
    get_range_bars,
)


def main() -> None:
    """Run get_range_bars() examples."""
    print("=" * 70)
    print("rangebar-py: get_range_bars() Example (Primary API)")
    print("=" * 70)

    # Show available constants
    print("\n1. Available Configuration Constants:")
    print(f"   TIER1_SYMBOLS: {len(TIER1_SYMBOLS)} symbols")
    print(f"     {TIER1_SYMBOLS[:6]}...")
    print(f"   THRESHOLD_PRESETS: {THRESHOLD_PRESETS}")
    print(f"   Valid range: {THRESHOLD_DECIMAL_MIN} to {THRESHOLD_DECIMAL_MAX}")

    # Basic usage - fetch and process in one call
    print("\n2. Basic Usage (Binance Spot):")
    print("   Fetching BTCUSDT data for 2024-01-01...")

    df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-01")

    print(f"   Generated {len(df)} range bars")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Index type: {type(df.index).__name__}")

    # Using threshold presets
    print("\n3. Using Threshold Presets:")
    for preset_name in ["tight", "medium", "wide"]:
        df = get_range_bars(
            "BTCUSDT", "2024-01-01", "2024-01-01", threshold_decimal_bps=preset_name
        )
        print(f"   {preset_name} ({THRESHOLD_PRESETS[preset_name]}): {len(df)} bars")

    # With microstructure data
    print("\n4. Microstructure Data:")
    df_micro = get_range_bars(
        "BTCUSDT", "2024-01-01", "2024-01-01", include_microstructure=True
    )
    print(f"   Columns: {list(df_micro.columns)}")
    print(f"   Sample VWAP: {df_micro['vwap'].iloc[0]:.2f}")
    print(f"   Sample buy_volume: {df_micro['buy_volume'].iloc[0]:.4f}")
    print(f"   Sample sell_volume: {df_micro['sell_volume'].iloc[0]:.4f}")

    # Different market types
    print("\n5. Different Market Types:")
    markets = ["spot", "futures-um"]
    for market in markets:
        df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-01", market=market)
        print(f"   {market}: {len(df)} bars")

    # Show first few bars
    print("\n6. Sample Range Bars (first 5):")
    df = get_range_bars("BTCUSDT", "2024-01-01", "2024-01-01")
    print(df.head().to_string())

    # Integration with backtesting.py
    print("\n7. backtesting.py Integration:")
    print("   DataFrame is ready for direct use with backtesting.py:")
    print(
        """
   from backtesting import Backtest, Strategy
   from rangebar import get_range_bars

   df = get_range_bars("BTCUSDT", "2024-01-01", "2024-06-30")
   bt = Backtest(df, MyStrategy, cash=10000, commission=0.0002)
   stats = bt.run()
"""
    )

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

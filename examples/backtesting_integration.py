#!/usr/bin/env python3
"""backtesting.py integration example.

This example demonstrates complete integration with backtesting.py:
1. Generate/load tick data
2. Convert to range bars
3. Define trading strategy
4. Run backtest
5. Analyze results

Requires: pip install backtesting.py

Note: This example uses synthetic data. For real backtesting, replace
with actual Binance CSV data using binance_csv_example.py.
"""

import pandas as pd
import numpy as np

from rangebar import process_trades_to_dataframe

# Check if backtesting.py is installed
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    from backtesting.test import SMA
except ImportError:
    print("❌ Error: backtesting.py is not installed")
    print("\nInstall with:")
    print("  pip install backtesting.py")
    print("\nOr:")
    print("  uv pip install backtesting.py")
    exit(1)


def generate_synthetic_trades(
    n_trades: int = 10_000,
    base_price: float = 42000.0,
    volatility: float = 0.01,
) -> pd.DataFrame:
    """Generate synthetic trade data with realistic price movement.

    Parameters
    ----------
    n_trades : int
        Number of trades to generate
    base_price : float
        Starting price
    volatility : float
        Price volatility (std dev of returns)

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamp, price, quantity columns
    """
    print(f"Generating {n_trades:,} synthetic trades...")

    # Generate timestamps (1 minute intervals)
    timestamps = pd.date_range("2024-01-01", periods=n_trades, freq="min")

    # Generate price using geometric brownian motion
    np.random.seed(42)  # Reproducible results
    returns = np.random.normal(0, volatility, n_trades)
    price_multipliers = np.exp(returns)
    prices = base_price * np.cumprod(price_multipliers)

    # Generate volume (correlated with volatility)
    volumes = 1.0 + np.abs(returns) * 100

    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "quantity": volumes,
    })

    print(f"  Price range: ${df['price'].min():.2f} → ${df['price'].max():.2f}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    return df


class RangeBarMACrossover(Strategy):
    """Moving Average Crossover Strategy on Range Bars.

    Strategy Logic:
    - Buy when fast MA crosses above slow MA (golden cross)
    - Sell when fast MA crosses below slow MA (death cross)

    Parameters:
    - fast: Fast MA period (default: 20 bars)
    - slow: Slow MA period (default: 50 bars)
    """

    fast = 20
    slow = 50

    def init(self):
        """Initialize strategy indicators."""
        # Calculate moving averages on range bars
        self.sma_fast = self.I(SMA, self.data.Close, self.fast)
        self.sma_slow = self.I(SMA, self.data.Close, self.slow)

    def next(self):
        """Execute on each new bar."""
        # Buy signal: Fast MA crosses above Slow MA
        if crossover(self.sma_fast, self.sma_slow):
            if not self.position:
                self.buy()

        # Sell signal: Fast MA crosses below Slow MA
        elif crossover(self.sma_slow, self.sma_fast):
            if self.position:
                self.position.close()


def main():
    """Run backtesting.py integration example."""
    print("=" * 70)
    print("rangebar-py: backtesting.py Integration Example")
    print("=" * 70)
    print()

    # Step 1: Generate synthetic trade data
    trades_df = generate_synthetic_trades(n_trades=10_000)

    # Step 2: Convert to range bars
    print("\nConverting to range bars (threshold = 25bps = 0.25%)...")
    range_bars = process_trades_to_dataframe(trades_df, threshold_bps=250)

    print(f"  Generated {len(range_bars):,} range bars from {len(trades_df):,} trades")
    print(f"  Compression ratio: {len(trades_df) / len(range_bars):.1f}x")

    # Validate format
    print("\nValidating OHLCV format for backtesting.py...")
    assert isinstance(range_bars.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert list(range_bars.columns) == ["Open", "High", "Low", "Close", "Volume"], \
        "Columns must be [Open, High, Low, Close, Volume]"
    assert not range_bars.isnull().any().any(), "DataFrame must have no NaN values"
    print("  ✅ Format validation passed")

    # Step 3: Run backtest
    print("\n" + "=" * 70)
    print("Running Backtest: MA Crossover Strategy on Range Bars")
    print("=" * 70)

    bt = Backtest(
        range_bars,
        RangeBarMACrossover,
        cash=10_000,
        commission=0.002,  # 0.2% commission
        exclusive_orders=True,
    )

    stats = bt.run()

    # Step 4: Display results
    print("\n" + "=" * 70)
    print("Backtest Results")
    print("=" * 70)
    print(stats)

    # Highlight key metrics
    print("\n" + "=" * 70)
    print("Key Performance Metrics")
    print("=" * 70)
    print(f"  Initial Capital:     ${stats['Start']:.2f}")
    print(f"  Final Value:         ${stats['End']:.2f}")
    print(f"  Total Return:        {stats['Return [%]']:.2f}%")
    print(f"  Buy & Hold Return:   {stats['Buy & Hold Return [%]']:.2f}%")
    print(f"  Sharpe Ratio:        {stats['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown:        {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  Number of Trades:    {stats['# Trades']}")
    print(f"  Win Rate:            {stats['Win Rate [%]']:.2f}%")

    # Step 5: Save and plot (optional)
    print("\n" + "=" * 70)
    print("Visualization")
    print("=" * 70)

    # Save stats to CSV
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv("backtest_stats.csv")
    print("  Saved backtest statistics to: backtest_stats.csv")

    # Plot (opens in browser)
    try:
        print("  Opening interactive plot in browser...")
        bt.plot(resample=False)  # Don't resample - we want to see range bars as-is
    except Exception as e:
        print(f"  ⚠️  Could not open plot: {e}")
        print("     (This is normal in headless environments)")

    print("\n" + "=" * 70)
    print("✅ backtesting.py integration completed successfully!")
    print("=" * 70)

    print("\nNext steps:")
    print("  1. Try different threshold_bps values (100-500)")
    print("  2. Compare performance vs time-based bars")
    print("  3. Test with real Binance CSV data")
    print("  4. Experiment with different strategies")


if __name__ == "__main__":
    main()

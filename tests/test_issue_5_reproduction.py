"""
Reproduction test for Issue #5: Range bar with 5.58% range when threshold is 0.25%
https://github.com/terrylica/rangebar/issues/5

Problem:
- Open: 85,372.00
- High: 90,025.82 (+5.45% from open)
- Threshold: 250 dbps = 0.25%
- Expected max high: 85,585.43 (0.25% above open)
- Actual: 21.8x the threshold
"""

from pathlib import Path

import pandas as pd
import pytest
from rangebar import get_range_bars, process_trades_to_dataframe


class TestIssue5ThresholdExceeded:
    """Test cases to reproduce and investigate Issue #5."""

    @pytest.fixture
    def dec12_data_path(self) -> Path:
        """Path to Dec 12, 2025 test data."""
        return (
            Path(__file__).parent.parent
            / "test_data"
            / "issue_5"
            / "BTCUSDT-aggTrades-2025-12-12.csv"
        )

    def test_find_threshold_violations(self, dec12_data_path: Path) -> None:
        """Find all bars where high-low range exceeds threshold."""
        if not dec12_data_path.exists():
            pytest.skip(f"Test data not found: {dec12_data_path}")

        # Load raw trades
        df_trades = pd.read_csv(
            dec12_data_path,
            names=[
                "agg_trade_id",
                "price",
                "quantity",
                "first_trade_id",
                "last_trade_id",
                "timestamp",
                "is_buyer_maker",
                "is_best_match",
            ],
        )

        # Binance 2025 data uses microseconds (16 digits), but API expects milliseconds (13 digits)
        # Check if timestamp is in microseconds (>= 1e15) and convert
        if df_trades["timestamp"].iloc[0] >= 1e15:
            df_trades["timestamp"] = df_trades["timestamp"] // 1000  # Convert us -> ms

        print(f"\nLoaded {len(df_trades):,} aggTrades from Dec 12, 2025")
        print(
            f"Price range: {df_trades['price'].min():.2f} - {df_trades['price'].max():.2f}"
        )

        # Process to range bars
        df = process_trades_to_dataframe(df_trades, threshold_decimal_bps=250)

        print(f"Generated {len(df)} range bars")

        # Calculate actual range percentage for each bar
        df["range_pct"] = (df["High"] - df["Low"]) / df["Low"] * 100

        # Threshold is 0.25%, find bars that exceed it significantly
        threshold_pct = 0.25
        violating_bars = df[
            df["range_pct"] > threshold_pct * 1.5
        ]  # Allow 50% tolerance

        print(
            f"\nBars exceeding 1.5x threshold (>{threshold_pct * 1.5:.3f}%): {len(violating_bars)}"
        )

        if len(violating_bars) > 0:
            print("\n=== Worst Violations ===")
            worst = violating_bars.nlargest(10, "range_pct")
            for idx, row in worst.iterrows():
                print(
                    f"  Open={row['Open']:.2f}, High={row['High']:.2f}, "
                    f"Low={row['Low']:.2f}, Close={row['Close']:.2f}, "
                    f"Range={row['range_pct']:.4f}%"
                )

            # Look for the specific bar from issue
            target_bar = df[
                (df["Open"].between(85370, 85375)) & (df["High"].between(90020, 90030))
            ]
            if not target_bar.empty:
                print("\n=== Target Bar from Issue #5 ===")
                print(
                    target_bar[["Open", "High", "Low", "Close", "Volume", "range_pct"]]
                )

        # Soft assertion: report but don't fail yet (investigation phase)
        max_range = df["range_pct"].max()
        print(f"\nMax range observed: {max_range:.4f}%")
        print(f"Expected max (2x threshold): {threshold_pct * 2:.4f}%")

        if max_range > threshold_pct * 2:
            print(
                f"\n*** BUG CONFIRMED: Max range {max_range:.4f}% exceeds 2x threshold ***"
            )

    def test_trace_specific_violation(self, dec12_data_path: Path) -> None:
        """Trace through trades to find where threshold breach was missed."""
        if not dec12_data_path.exists():
            pytest.skip(f"Test data not found: {dec12_data_path}")

        # Load raw trades
        df_trades = pd.read_csv(
            dec12_data_path,
            names=[
                "agg_trade_id",
                "price",
                "quantity",
                "first_trade_id",
                "last_trade_id",
                "timestamp",
                "is_buyer_maker",
                "is_best_match",
            ],
        )

        # Convert microseconds to milliseconds if needed
        if df_trades["timestamp"].iloc[0] >= 1e15:
            df_trades["timestamp"] = df_trades["timestamp"] // 1000

        # Find trades around 85372 (the problematic open price)
        target_open = 85372.0
        tolerance = 50.0  # Look for trades within $50

        nearby_trades = df_trades[
            df_trades["price"].between(target_open - tolerance, target_open + tolerance)
        ]

        if len(nearby_trades) > 0:
            print(f"\nFound {len(nearby_trades)} trades near price {target_open}")

            # Look for a sequence that could form the problematic bar
            # If bar opened at 85372, threshold is 0.25%, so max should be 85585
            upper_threshold = target_open * 1.0025  # 85,585.43
            lower_threshold = target_open * 0.9975  # 85,158.57

            print(
                f"Expected thresholds: [{lower_threshold:.2f}, {upper_threshold:.2f}]"
            )

            # Find trades that would breach the threshold
            breach_trades = df_trades[
                (df_trades["price"] >= upper_threshold)
                | (df_trades["price"] <= lower_threshold)
            ].head(20)

            if len(breach_trades) > 0:
                print(f"\nFirst trades that would breach from open {target_open}:")
                for _, row in breach_trades.head(10).iterrows():
                    pct_from_open = (row["price"] - target_open) / target_open * 100
                    print(
                        f"  Price={row['price']:.2f} ({pct_from_open:+.4f}% from open)"
                    )

    def test_via_get_range_bars_api(self) -> None:
        """Test using the high-level API that fetches from Binance."""
        # This test requires network access and may hit cache
        try:
            df = get_range_bars(
                symbol="BTCUSDT",
                start_date="2025-12-12",
                end_date="2025-12-12",
                threshold_decimal_bps=250,
            )
        except Exception as e:
            pytest.skip(f"Could not fetch data: {e}")

        df["range_pct"] = (df["High"] - df["Low"]) / df["Low"] * 100

        threshold_pct = 0.25
        max_range = df["range_pct"].max()

        print("\nVia get_range_bars API:")
        print(f"  Total bars: {len(df)}")
        print(f"  Max range: {max_range:.4f}%")
        print(f"  Threshold: {threshold_pct}%")

        # Find violations
        violations = df[df["range_pct"] > threshold_pct * 2]
        if len(violations) > 0:
            print(f"\n  Violations (range > 2x threshold): {len(violations)}")
            print(
                violations.nlargest(5, "range_pct")[
                    ["Open", "High", "Low", "Close", "range_pct"]
                ]
            )

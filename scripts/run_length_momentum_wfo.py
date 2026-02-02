"""Run Length Momentum WFO - Regime Detection via Walk-Forward Optimization.

Tests whether in-sample momentum effect (Δ = P(K=1) - P(K=7)) predicts
out-of-sample momentum persistence.

Hypothesis: If we detect momentum in training window, it will persist in test window.

WFO Design:
- Train: Compute Δ (momentum effect) on N bars
- Decision: If Δ > threshold, predict "momentum regime"
- Test: Measure if momentum actually held in next M bars

No look-ahead bias: threshold is set BEFORE looking at test data.

Usage:
    uv run --python 3.13 python scripts/run_length_momentum_wfo.py

GitHub Issue: https://github.com/terrylica/rangebar-py/issues/58
SRED-Type: applied-research
SRED-Claim: PATTERN-RESEARCH
"""

import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, "/Users/terryli/eon/rangebar-py")

from rangebar import get_range_bars


@dataclass
class WFOResult:
    """Result of a single WFO fold."""

    fold_id: int
    train_delta: float  # Δ = P(K=1) - P(K=7) in training
    test_delta: float  # Δ in test
    predicted_momentum: bool  # True if train_delta > threshold
    actual_momentum: bool  # True if test_delta > 0
    correct: bool  # predicted == actual


def compute_run_lengths(directions: list[str]) -> list[tuple[str, int]]:
    """Extract runs from direction sequence."""
    if not directions:
        return []

    runs = []
    current_dir = directions[0]
    current_len = 1

    for d in directions[1:]:
        if d == current_dir:
            current_len += 1
        else:
            runs.append((current_dir, current_len))
            current_dir = d
            current_len = 1
    runs.append((current_dir, current_len))

    return runs


def compute_momentum_delta(
    directions: list[str], min_samples: int = 20, k_end: int = 5
) -> float | None:
    """Compute Δ = P(reverse|K=1) - P(reverse|K=k_end).

    Using K=5 instead of K=7 for more stable estimates with smaller windows.
    Returns None if insufficient samples.
    """
    runs = compute_run_lengths(directions)
    up_runs = [r[1] for r in runs if r[0] == "U"]

    # Need enough samples at both positions
    at_k1 = sum(1 for r in up_runs if r >= 1)
    at_k_end = sum(1 for r in up_runs if r >= k_end)

    if at_k1 < min_samples or at_k_end < min_samples:
        return None

    # P(reverse at K) = 1 - P(continue to K+1)
    at_k2 = sum(1 for r in up_runs if r >= 2)
    at_k_end_plus_1 = sum(1 for r in up_runs if r >= k_end + 1)

    p_reverse_k1 = 1 - (at_k2 / at_k1)
    p_reverse_k_end = 1 - (at_k_end_plus_1 / at_k_end)

    return p_reverse_k1 - p_reverse_k_end


def run_wfo(
    symbol: str,
    start: str,
    end: str,
    threshold_decimal_bps: int = 250,
    train_bars: int = 20000,
    test_bars: int = 5000,
    delta_threshold: float = 0.05,
) -> list[WFOResult]:
    """Run Walk-Forward Optimization for momentum regime detection.

    Args:
        symbol: Trading pair
        start: Start date
        end: End date
        threshold_decimal_bps: Range bar threshold
        train_bars: Bars in training window
        test_bars: Bars in test window
        delta_threshold: Minimum Δ to predict "momentum regime"

    Returns:
        List of WFOResult for each fold
    """
    print(f"Fetching {symbol} @ {threshold_decimal_bps} dbps ({start} to {end})...")
    df = get_range_bars(symbol, start, end, threshold_decimal_bps=threshold_decimal_bps)
    print(f"  Got {len(df):,} bars")

    directions = ["U" if c > o else "D" for c, o in zip(df["Close"], df["Open"], strict=True)]

    results = []
    fold_id = 0
    step = test_bars  # Non-overlapping test windows

    i = 0
    while i + train_bars + test_bars <= len(directions):
        # Train window
        train_dirs = directions[i : i + train_bars]
        train_delta = compute_momentum_delta(train_dirs)

        # Test window
        test_dirs = directions[i + train_bars : i + train_bars + test_bars]
        test_delta = compute_momentum_delta(test_dirs)

        if train_delta is not None and test_delta is not None:
            predicted_momentum = train_delta > delta_threshold
            actual_momentum = test_delta > 0

            results.append(
                WFOResult(
                    fold_id=fold_id,
                    train_delta=train_delta,
                    test_delta=test_delta,
                    predicted_momentum=predicted_momentum,
                    actual_momentum=actual_momentum,
                    correct=predicted_momentum == actual_momentum,
                )
            )
            fold_id += 1

        i += step

    return results


def main() -> None:
    """Run WFO analysis for momentum regime detection."""
    print("=" * 70)
    print("RUN LENGTH MOMENTUM - WALK-FORWARD REGIME DETECTION")
    print("=" * 70)
    print("\nHypothesis: In-sample momentum effect predicts out-of-sample momentum")
    print()

    # Test on SOLUSDT which showed regime shift
    configs = [
        ("SOLUSDT", 250, "2022-01-01", "2025-12-31"),
        ("BTCUSDT", 250, "2024-01-01", "2025-12-31"),
        ("ETHUSDT", 250, "2022-01-01", "2024-12-31"),
    ]

    for symbol, threshold, start, end in configs:
        print(f"\n{'='*70}")
        print(f"SYMBOL: {symbol} @ {threshold} dbps")
        print("=" * 70)

        try:
            results = run_wfo(symbol, start, end, threshold)

            if not results:
                print("  Insufficient data for WFO")
                continue

            # Analyze predictions
            correct = sum(1 for r in results if r.correct)
            total = len(results)
            accuracy = correct / total

            # When we predict momentum, how often are we right?
            predicted_momentum = [r for r in results if r.predicted_momentum]
            if predicted_momentum:
                tp = sum(1 for r in predicted_momentum if r.actual_momentum)
                precision = tp / len(predicted_momentum)
            else:
                precision = 0.0

            # When momentum actually happened, did we predict it?
            actual_momentum = [r for r in results if r.actual_momentum]
            if actual_momentum:
                recalled = sum(1 for r in actual_momentum if r.predicted_momentum)
                recall = recalled / len(actual_momentum)
            else:
                recall = 0.0

            print(f"\n  WFO Results ({total} folds):")
            print(f"    Overall accuracy: {accuracy:.2%}")
            print(f"    Precision (predict momentum → actual momentum): {precision:.2%}")
            print(f"    Recall (actual momentum → predicted momentum): {recall:.2%}")

            # Analyze train vs test delta correlation
            train_deltas = [r.train_delta for r in results]
            test_deltas = [r.test_delta for r in results]
            correlation = np.corrcoef(train_deltas, test_deltas)[0, 1]
            print(f"    Train-Test Δ correlation: {correlation:.3f}")

            # Show sample predictions
            print("\n  Sample folds:")
            for r in results[:5]:
                pred = "MOMENTUM" if r.predicted_momentum else "NO-MOM"
                actual = "MOMENTUM" if r.actual_momentum else "NO-MOM"
                status = "✓" if r.correct else "✗"
                print(f"    Fold {r.fold_id}: train_Δ={r.train_delta:+.3f} → {pred}, actual={actual} {status}")

        except (ValueError, RuntimeError, ConnectionError) as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Key Metrics:
- Accuracy: How often our regime prediction is correct
- Precision: When we predict momentum, how often it actually happens
- Recall: When momentum happens, how often we predicted it
- Correlation: Does training Δ predict testing Δ?

If correlation > 0.3 and precision > 60%:
  → Momentum regime is PREDICTABLE from in-sample data
  → Can use this for selective trading (trade only in predicted momentum)

If correlation < 0.1:
  → Momentum is not regime-stable
  → Need different approach (maybe volatility-conditioned)
""")


if __name__ == "__main__":
    main()

"""Run Length Momentum Analysis - Multi-Symbol Cross-Validation.

Extends run_length_momentum_analysis.py to test momentum effect across
multiple symbols using existing cached 250 dbps data.

Available Data on littleblack (as of 2026-02-01):
- BTCUSDT @ 250 dbps: 25K bars (2024-01 to 2026-01)
- SOLUSDT @ 250 dbps: 700K bars (2021-12 to 2026-01)

Usage:
    uv run --python 3.13 python scripts/run_length_momentum_multi_symbol.py

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
class RunLengthStats:
    """Statistics for run length analysis."""

    n_bars: int
    n_runs: int
    mean_run_length: float
    max_run_length: int
    reversal_probs: dict[int, float]
    sample_sizes: dict[int, int]


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


def compute_reversal_probabilities(
    run_lengths: list[int], max_k: int = 10, min_samples: int = 30
) -> tuple[dict[int, float], dict[int, int]]:
    """Compute P(reverse | position K) for each position."""
    probs = {}
    samples = {}

    for k in range(1, max_k + 1):
        at_k = sum(1 for r in run_lengths if r >= k)
        at_k_plus_1 = sum(1 for r in run_lengths if r >= k + 1)
        samples[k] = at_k
        if at_k >= min_samples:
            probs[k] = 1 - (at_k_plus_1 / at_k)

    return probs, samples


def analyze_period(
    symbol: str, start: str, end: str, threshold_decimal_bps: int = 250
) -> RunLengthStats:
    """Analyze run length patterns for a specific period."""
    df = get_range_bars(symbol, start, end, threshold_decimal_bps=threshold_decimal_bps)

    directions = ["U" if c > o else "D" for c, o in zip(df["Close"], df["Open"], strict=True)]
    runs = compute_run_lengths(directions)
    up_runs = [r[1] for r in runs if r[0] == "U"]
    probs, samples = compute_reversal_probabilities(up_runs)

    return RunLengthStats(
        n_bars=len(df),
        n_runs=len(up_runs),
        mean_run_length=np.mean(up_runs) if up_runs else 0.0,
        max_run_length=max(up_runs) if up_runs else 0,
        reversal_probs=probs,
        sample_sizes=samples,
    )


def test_momentum_effect(probs: dict[int, float], k_start: int = 1, k_end: int = 7) -> tuple[float, bool]:
    """Test if P(reverse|K=1) > P(reverse|K=7)."""
    if k_start not in probs or k_end not in probs:
        return 0.0, False
    diff = probs[k_start] - probs[k_end]
    return diff, diff > 0


def analyze_symbol(symbol: str, threshold: int, periods: list[tuple[str, str, str]]) -> dict:
    """Analyze momentum effect for a single symbol across periods."""
    print(f"\n{'='*70}")
    print(f"SYMBOL: {symbol} @ {threshold} dbps")
    print("=" * 70)

    results: dict[str, RunLengthStats] = {}
    for start, end, label in periods:
        try:
            print(f"  Fetching {label}...", end=" ")
            stats = analyze_period(symbol, start, end, threshold)
            results[label] = stats
            print(f"{stats.n_bars:,} bars, {stats.n_runs:,} up-runs")
        except (ValueError, RuntimeError, ConnectionError) as e:
            print(f"ERROR: {e}")

    if not results:
        return {"symbol": symbol, "threshold": threshold, "momentum_holds": 0, "total_tests": 0}

    # Test momentum effect
    momentum_holds = 0
    total_tests = 0
    print("\n  Momentum test (P(K=1) > P(K=7)):")
    for label, stats in results.items():
        diff, holds = test_momentum_effect(stats.reversal_probs)
        if 1 in stats.reversal_probs and 7 in stats.reversal_probs:
            total_tests += 1
            if holds:
                momentum_holds += 1
            status = "YES" if holds else "NO"
            print(f"    {label}: P(1)={stats.reversal_probs[1]:.4f}, P(7)={stats.reversal_probs[7]:.4f}, Δ={diff:+.4f} → {status}")

    return {
        "symbol": symbol,
        "threshold": threshold,
        "momentum_holds": momentum_holds,
        "total_tests": total_tests,
        "results": results,
    }


def main() -> None:
    """Run multi-symbol OOD robustness analysis."""
    print("=" * 70)
    print("RUN LENGTH MOMENTUM - MULTI-SYMBOL CROSS-VALIDATION")
    print("=" * 70)
    print("\nHypothesis: Momentum effect (P(reverse) decreasing with K)")
    print("Test: Does this hold across different symbols?")
    print()

    # Symbols with cached 250 dbps data on littleblack
    symbols = [
        ("BTCUSDT", 250, [
            ("2024-01-01", "2024-06-30", "2024-H1"),
            ("2024-07-01", "2024-12-31", "2024-H2"),
            ("2025-01-01", "2025-12-31", "2025"),
        ]),
        ("SOLUSDT", 250, [
            ("2022-01-01", "2022-12-31", "2022"),
            ("2023-01-01", "2023-12-31", "2023"),
            ("2024-01-01", "2024-12-31", "2024"),
            ("2025-01-01", "2025-12-31", "2025"),
        ]),
    ]

    all_results = []
    for symbol, threshold, periods in symbols:
        result = analyze_symbol(symbol, threshold, periods)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("CROSS-SYMBOL SUMMARY")
    print("=" * 70)
    print(f"\n{'Symbol':<12} {'Threshold':<10} {'Momentum Holds':<18} {'OOD Robust?':<12}")
    print("-" * 52)

    total_symbols_robust = 0
    for result in all_results:
        holds = result["momentum_holds"]
        total = result["total_tests"]
        robust = "YES" if holds == total and total > 0 else "PARTIAL" if holds > 0 else "NO"
        if holds == total and total > 0:
            total_symbols_robust += 1
        print(f"{result['symbol']:<12} {result['threshold']:<10} {holds}/{total:<16} {robust:<12}")

    print("\n" + "=" * 70)
    if total_symbols_robust == len(all_results):
        print("*** MOMENTUM EFFECT IS CROSS-SYMBOL ROBUST ***")
    elif total_symbols_robust > 0:
        print(f"*** MOMENTUM EFFECT PARTIALLY ROBUST ({total_symbols_robust}/{len(all_results)} symbols) ***")
    else:
        print("*** MOMENTUM EFFECT NOT ROBUST ACROSS SYMBOLS ***")
    print("=" * 70)


if __name__ == "__main__":
    main()

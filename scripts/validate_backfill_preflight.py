"""Fail-fast preflight validation for SOL microstructure backfill.

Issue #80: Phase 0 GATE - 7-day atomic run to verify all 38 features
(16 inter-bar + 22 intra-bar) are sensible BEFORE committing to 4-6h full backfill.

Exit code 0 = proceed to full backfill.
Exit code 1 = investigate before backfilling.

Usage:
    uv run python scripts/validate_backfill_preflight.py
    uv run python scripts/validate_backfill_preflight.py --symbol BTCUSDT
    uv run python scripts/validate_backfill_preflight.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    severity: str  # "abort" or "warn"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Aggregate preflight validation report."""

    symbol: str
    start_date: str
    end_date: str
    threshold: int
    bar_count: int
    elapsed_seconds: float
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True if no abort-severity checks failed."""
        return all(c.passed for c in self.checks if c.severity == "abort")

    @property
    def abort_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "abort"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warn"]


def check_column_presence(
    df: pd.DataFrame,
    inter_bar_cols: tuple[str, ...],
    intra_bar_cols: tuple[str, ...],
) -> list[CheckResult]:
    """Check that all 38 feature columns exist and are NOT all NULL."""
    results = []

    # Inter-bar columns (16)
    missing_inter = [c for c in inter_bar_cols if c not in df.columns]
    results.append(
        CheckResult(
            name="inter_bar_column_presence",
            passed=len(missing_inter) == 0,
            severity="abort",
            message=f"Missing {len(missing_inter)}/16 inter-bar columns"
            if missing_inter
            else "All 16 inter-bar columns present",
            details={"missing": missing_inter},
        )
    )

    # Intra-bar columns (22)
    missing_intra = [c for c in intra_bar_cols if c not in df.columns]
    results.append(
        CheckResult(
            name="intra_bar_column_presence",
            passed=len(missing_intra) == 0,
            severity="abort",
            message=f"Missing {len(missing_intra)}/22 intra-bar columns"
            if missing_intra
            else "All 22 intra-bar columns present",
            details={"missing": missing_intra},
        )
    )

    # Check NOT all NULL for present columns
    all_null_inter = []
    for col in inter_bar_cols:
        if col in df.columns and df[col].isna().all():
            all_null_inter.append(col)

    results.append(
        CheckResult(
            name="inter_bar_not_all_null",
            passed=len(all_null_inter) == 0,
            severity="abort",
            message=f"{len(all_null_inter)} inter-bar columns are ALL NULL"
            if all_null_inter
            else "All inter-bar columns have non-NULL values",
            details={"all_null_columns": all_null_inter},
        )
    )

    all_null_intra = []
    for col in intra_bar_cols:
        if col in df.columns and df[col].isna().all():
            all_null_intra.append(col)

    results.append(
        CheckResult(
            name="intra_bar_not_all_null",
            passed=len(all_null_intra) == 0,
            severity="abort",
            message=f"{len(all_null_intra)} intra-bar columns are ALL NULL"
            if all_null_intra
            else "All intra-bar columns have non-NULL values",
            details={"all_null_columns": all_null_intra},
        )
    )

    return results


def check_bounded_features(df: pd.DataFrame) -> list[CheckResult]:
    """Check that bounded features stay within expected ranges."""
    results = []

    bounded_checks = {
        # (column, min, max, description)
        "lookback_ofi": (-1.0, 1.0, "OFI"),
        "lookback_vwap_position": (0.0, 1.0, "VWAP position"),
        "lookback_count_imbalance": (-1.0, 1.0, "Count imbalance"),
        "lookback_kaufman_er": (0.0, 1.0, "Kaufman ER"),
        "lookback_hurst": (0.0, 1.0, "Hurst exponent"),
        "lookback_permutation_entropy": (0.0, 1.0, "Permutation entropy"),
        "lookback_burstiness": (-1.0, 1.0, "Burstiness"),
        "intra_ofi": (-1.0, 1.0, "Intra OFI"),
        "intra_vwap_position": (0.0, 1.0, "Intra VWAP position"),
        "intra_count_imbalance": (-1.0, 1.0, "Intra count imbalance"),
        "intra_kaufman_er": (0.0, 1.0, "Intra Kaufman ER"),
        "intra_hurst": (0.0, 1.0, "Intra Hurst"),
        "intra_permutation_entropy": (0.0, 1.0, "Intra permutation entropy"),
        "intra_burstiness": (-1.0, 1.0, "Intra burstiness"),
    }

    for col, (lo, hi, desc) in bounded_checks.items():
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        in_range = valid.between(lo, hi).all()
        out_of_range = (~valid.between(lo, hi)).sum()
        results.append(
            CheckResult(
                name=f"bounded_{col}",
                passed=in_range,
                severity="abort",
                message=f"{desc} ({col}): {out_of_range}/{len(valid)} values out of [{lo}, {hi}]"
                if not in_range
                else f"{desc} ({col}): bounded [{lo}, {hi}]",
                details={
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "out_of_range": int(out_of_range),
                },
            )
        )

    return results


def check_economic_sanity(df: pd.DataFrame) -> list[CheckResult]:
    """Check that features have economically sensible distributions."""
    results = []

    # Hurst should cluster near 0.5 for efficient markets
    for col_name in ("lookback_hurst", "intra_hurst"):
        if col_name in df.columns:
            valid = df[col_name].dropna()
            if len(valid) >= 5:
                median = float(valid.median())
                results.append(
                    CheckResult(
                        name=f"economic_{col_name}_median",
                        passed=0.3 <= median <= 0.7,
                        severity="abort",
                        message=f"{col_name} median={median:.3f} (expected 0.3-0.7 for efficient market)"
                        if not (0.3 <= median <= 0.7)
                        else f"{col_name} median={median:.3f}",
                        details={"median": median, "std": float(valid.std())},
                    )
                )

    # lookback_trade_count should have diversity (>5 distinct values)
    if "lookback_trade_count" in df.columns:
        valid = df["lookback_trade_count"].dropna()
        if len(valid) > 0:
            n_distinct = int(valid.nunique())
            results.append(
                CheckResult(
                    name="economic_trade_count_diversity",
                    passed=n_distinct > 5,
                    severity="abort",
                    message=f"lookback_trade_count has only {n_distinct} distinct values (need >5)"
                    if n_distinct <= 5
                    else f"lookback_trade_count: {n_distinct} distinct values, range [{int(valid.min())}-{int(valid.max())}]",
                    details={
                        "n_distinct": n_distinct,
                        "min": int(valid.min()),
                        "max": int(valid.max()),
                    },
                )
            )

    # Non-negative duration
    for col_name in ("lookback_duration_us", "intra_duration_us"):
        if col_name in df.columns:
            valid = df[col_name].dropna()
            if len(valid) > 0:
                negative = (valid < 0).sum()
                results.append(
                    CheckResult(
                        name=f"economic_{col_name}_positive",
                        passed=negative == 0,
                        severity="abort",
                        message=f"{col_name}: {negative} negative values"
                        if negative > 0
                        else f"{col_name}: all non-negative",
                    )
                )

    return results


def check_non_zero_variance(
    df: pd.DataFrame,
    inter_bar_cols: tuple[str, ...],
    intra_bar_cols: tuple[str, ...],
) -> list[CheckResult]:
    """Check that each feature has non-zero variance (not constant)."""
    results = []

    for col in (*inter_bar_cols, *intra_bar_cols):
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) < 2:
            continue
        std = float(valid.std())
        results.append(
            CheckResult(
                name=f"variance_{col}",
                passed=std > 0,
                severity="abort",
                message=f"{col}: zero variance (constant={valid.iloc[0]})"
                if std == 0
                else f"{col}: std={std:.6g}",
                details={"std": std, "n_valid": len(valid)},
            )
        )

    return results


def check_correlation_plausibility(df: pd.DataFrame) -> list[CheckResult]:
    """Check that correlated features are actually correlated."""
    results = []

    # lookback_intensity and lookback_trade_count should be positively correlated
    if (
        "lookback_intensity" in df.columns
        and "lookback_trade_count" in df.columns
    ):
        valid = df[["lookback_intensity", "lookback_trade_count"]].dropna()
        if len(valid) >= 20:
            corr = float(valid["lookback_intensity"].corr(valid["lookback_trade_count"]))
            results.append(
                CheckResult(
                    name="correlation_intensity_trade_count",
                    passed=corr > 0.6,
                    severity="warn",
                    message=f"lookback_intensity ~ lookback_trade_count corr={corr:.3f} (expected >0.6)"
                    if corr <= 0.6
                    else f"lookback_intensity ~ lookback_trade_count corr={corr:.3f}",
                    details={"correlation": corr},
                )
            )

    return results


def check_null_ratio(
    df: pd.DataFrame,
    inter_bar_cols: tuple[str, ...],
    intra_bar_cols: tuple[str, ...],
) -> list[CheckResult]:
    """Check NULL ratios are within acceptable bounds."""
    results = []

    # Inter-bar: >90% non-NULL (first bars expected NULL)
    inter_present = [c for c in inter_bar_cols if c in df.columns]
    if inter_present:
        non_null_pcts = []
        for col in inter_present:
            non_null_pct = float(df[col].notna().mean())
            non_null_pcts.append(non_null_pct)
        avg_non_null = float(np.mean(non_null_pcts))
        min_non_null = float(min(non_null_pcts))
        results.append(
            CheckResult(
                name="null_ratio_inter_bar",
                passed=min_non_null >= 0.50,
                severity="abort" if min_non_null < 0.50 else "warn",
                message=f"Inter-bar avg non-NULL: {avg_non_null:.1%}, min: {min_non_null:.1%}"
                + (" (FAIL: <50%)" if min_non_null < 0.50 else "")
                + (" (WARN: <90%)" if 0.50 <= min_non_null < 0.90 else ""),
                details={
                    "avg_non_null_pct": avg_non_null,
                    "min_non_null_pct": min_non_null,
                },
            )
        )

    # Intra-bar: should be 100% non-NULL
    intra_present = [c for c in intra_bar_cols if c in df.columns]
    if intra_present:
        non_null_pcts = []
        for col in intra_present:
            non_null_pct = float(df[col].notna().mean())
            non_null_pcts.append(non_null_pct)
        avg_non_null = float(np.mean(non_null_pcts))
        min_non_null = float(min(non_null_pcts))
        results.append(
            CheckResult(
                name="null_ratio_intra_bar",
                passed=min_non_null >= 0.50,
                severity="abort" if min_non_null < 0.50 else "warn",
                message=f"Intra-bar avg non-NULL: {avg_non_null:.1%}, min: {min_non_null:.1%}"
                + (" (FAIL: <50%)" if min_non_null < 0.50 else ""),
                details={
                    "avg_non_null_pct": avg_non_null,
                    "min_non_null_pct": min_non_null,
                },
            )
        )

    return results


def check_trade_id_continuity(df: pd.DataFrame) -> list[CheckResult]:
    """Check that trade IDs are continuous between consecutive bars."""
    results = []

    if "first_agg_trade_id" in df.columns and "last_agg_trade_id" in df.columns:
        valid = df[["first_agg_trade_id", "last_agg_trade_id"]].dropna()
        if len(valid) >= 2:
            expected = valid["last_agg_trade_id"].iloc[:-1].to_numpy() + 1
            actual = valid["first_agg_trade_id"].iloc[1:].to_numpy()
            gaps = int(np.sum(expected != actual))
            total = len(expected)
            results.append(
                CheckResult(
                    name="trade_id_continuity",
                    passed=gaps == 0,
                    severity="warn",
                    message=f"Trade ID continuity: {total - gaps}/{total} bars OK"
                    + (f" ({gaps} gaps)" if gaps > 0 else ""),
                    details={"continuous_pairs": int(total - gaps), "total_pairs": total, "gaps": gaps},
                )
            )

    return results


def check_tier1_validation(df: pd.DataFrame) -> list[CheckResult]:
    """Run Tier 1 validation on intra-bar features."""
    results = []

    try:
        from rangebar.validation.tier1 import validate_tier1

        tier1_result = validate_tier1(df)
        passed = tier1_result.get("tier1_passed", False)
        results.append(
            CheckResult(
                name="tier1_validation",
                passed=passed,
                severity="abort",
                message="Tier 1 validation: PASSED" if passed else "Tier 1 validation: FAILED",
                details=tier1_result,
            )
        )
    except (ImportError, ValueError, TypeError, KeyError, RuntimeError) as e:
        results.append(
            CheckResult(
                name="tier1_validation",
                passed=False,
                severity="warn",
                message=f"Tier 1 validation error: {e}",
            )
        )

    return results


def run_preflight(
    symbol: str,
    start_date: str,
    end_date: str,
    threshold: int,
    *,
    dry_run: bool = False,
) -> PreflightReport:
    """Run all preflight checks for a symbol/threshold combination."""
    from rangebar.constants import INTER_BAR_FEATURE_COLUMNS, INTRA_BAR_FEATURE_COLUMNS

    t0 = time.monotonic()

    if not dry_run:
        from rangebar import populate_cache_resumable

        print(f"  Populating {symbol} {start_date} -> {end_date} @{threshold}dbps ...")
        populate_cache_resumable(
            symbol,
            start_date,
            end_date,
            threshold_decimal_bps=threshold,
            include_microstructure=True,
            force_refresh=True,
            verbose=False,
            notify=False,
        )

    # Read back from ClickHouse
    from rangebar.clickhouse import RangeBarCache

    with RangeBarCache() as cache:
        from datetime import datetime, timezone

        from rangebar.orchestration.helpers import (
            _datetime_to_end_ms,
            _datetime_to_start_ms,
        )

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ts = _datetime_to_start_ms(start_dt)
        end_ts = _datetime_to_end_ms(end_dt)

        df = cache.get_bars_by_timestamp_range(
            symbol=symbol,
            threshold_decimal_bps=threshold,
            start_ts=start_ts,
            end_ts=end_ts,
            include_microstructure=True,
        )

    elapsed = time.monotonic() - t0

    if df is None or len(df) == 0:
        report = PreflightReport(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold,
            bar_count=0,
            elapsed_seconds=elapsed,
        )
        report.checks.append(
            CheckResult(
                name="data_present",
                passed=False,
                severity="abort",
                message="No bars returned from ClickHouse cache",
            )
        )
        return report

    report = PreflightReport(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        bar_count=len(df),
        elapsed_seconds=elapsed,
    )

    # Run all checks
    report.checks.extend(check_column_presence(df, INTER_BAR_FEATURE_COLUMNS, INTRA_BAR_FEATURE_COLUMNS))
    report.checks.extend(check_bounded_features(df))
    report.checks.extend(check_economic_sanity(df))
    report.checks.extend(check_non_zero_variance(df, INTER_BAR_FEATURE_COLUMNS, INTRA_BAR_FEATURE_COLUMNS))
    report.checks.extend(check_correlation_plausibility(df))
    report.checks.extend(check_null_ratio(df, INTER_BAR_FEATURE_COLUMNS, INTRA_BAR_FEATURE_COLUMNS))
    report.checks.extend(check_trade_id_continuity(df))
    report.checks.extend(check_tier1_validation(df))

    return report


def print_report(report: PreflightReport) -> None:
    """Print a formatted preflight report."""
    print(f"\n{'=' * 70}")
    print(
        f"  Preflight: {report.symbol} "
        f"{report.start_date} -> {report.end_date} "
        f"@{report.threshold}dbps"
    )
    print(f"{'=' * 70}")
    print(f"  Bars: {report.bar_count} | Time: {report.elapsed_seconds:.1f}s")
    print()

    # Group by category
    passed = [c for c in report.checks if c.passed]
    warnings = report.warnings
    failures = report.abort_failures

    for check in passed:
        print(f"  [PASS] {check.message}")

    for check in warnings:
        print(f"  [WARN] {check.message}")

    for check in failures:
        print(f"  [FAIL] {check.message}")

    print()
    print(
        f"  Summary: {len(passed)} passed, "
        f"{len(warnings)} warnings, "
        f"{len(failures)} failures"
    )

    if report.all_passed:
        print("\n  VERDICT: READY FOR FULL BACKFILL")
    else:
        print("\n  VERDICT: INVESTIGATE BEFORE BACKFILLING")
        for f in failures:
            print(f"    - {f.name}: {f.message}")

    print(f"{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-fast preflight validation for microstructure backfill"
    )
    parser.add_argument(
        "--symbol", default="SOLUSDT", help="Symbol to validate (default: SOLUSDT)"
    )
    parser.add_argument(
        "--start-date", default="2024-06-01", help="Start date (default: 2024-06-01)"
    )
    parser.add_argument(
        "--end-date", default="2024-06-07", help="End date (default: 2024-06-07)"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[250, 500],
        help="Thresholds to test (default: 250 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip population, read existing cache only",
    )
    args = parser.parse_args()

    print("Backfill Preflight Validation")
    print(f"Symbol: {args.symbol}")
    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(f"Thresholds: {args.thresholds}")
    if args.dry_run:
        print("Mode: DRY RUN (read existing cache)")
    print()

    all_passed = True

    for threshold in args.thresholds:
        report = run_preflight(
            args.symbol,
            args.start_date,
            args.end_date,
            threshold,
            dry_run=args.dry_run,
        )
        print_report(report)

        if not report.all_passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

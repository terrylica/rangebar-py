#!/usr/bin/env python3
"""OHLCV output validation script.

This script validates that rangebar-py output meets all requirements
for backtesting.py compatibility and OHLCV data integrity.

Validation checks:
1. DataFrame structure (DatetimeIndex, column names)
2. Data types (float for OHLCV)
3. OHLCV invariants (High >= Open/Close, Low <= Open/Close)
4. Data completeness (no NaN, no missing values)
5. Temporal ordering (chronological)
6. Volume positivity
7. Precision preservation
"""

from typing import List, Tuple

import pandas as pd
from rangebar import process_trades_to_dataframe


class OHLCVValidator:
    """Validator for OHLCV DataFrame format."""

    def __init__(self, strict: bool = True):
        """Initialize validator.

        Parameters
        ----------
        strict : bool
            If True, raise errors on validation failures.
            If False, collect and report all failures.
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """Validate OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate

        Returns
        -------
        Tuple[bool, List[str], List[str]]
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Run all validation checks
        self._check_index(df)
        self._check_columns(df)
        self._check_data_types(df)
        self._check_completeness(df)
        self._check_ohlc_invariants(df)
        self._check_temporal_ordering(df)
        self._check_volume_positivity(df)
        self._check_precision(df)

        is_valid = len(self.errors) == 0

        if self.strict and not is_valid:
            error_msg = "\n".join(self.errors)
            raise ValueError(f"OHLCV validation failed:\n{error_msg}")

        return is_valid, self.errors, self.warnings

    def _check_index(self, df: pd.DataFrame):
        """Check index is DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            self.errors.append(
                f"Index must be DatetimeIndex, got {type(df.index).__name__}"
            )

    def _check_columns(self, df: pd.DataFrame):
        """Check column names are correct."""
        expected = ["Open", "High", "Low", "Close", "Volume"]
        actual = list(df.columns)

        if actual != expected:
            self.errors.append(
                f"Columns must be {expected}, got {actual}"
            )

    def _check_data_types(self, df: pd.DataFrame):
        """Check data types are numeric."""
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                self.errors.append(
                    f"Column '{col}' must be numeric, got {df[col].dtype}"
                )

    def _check_completeness(self, df: pd.DataFrame):
        """Check for missing values."""
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            self.errors.append(
                f"DataFrame contains NaN values: {null_cols.to_dict()}"
            )

        if len(df) == 0:
            self.warnings.append("DataFrame is empty")

    def _check_ohlc_invariants(self, df: pd.DataFrame):
        """Check OHLC invariants."""
        if "Open" not in df.columns or "High" not in df.columns:
            return

        # High must be >= Open
        if not (df["High"] >= df["Open"]).all():
            invalid = (~(df["High"] >= df["Open"])).sum()
            self.errors.append(
                f"High must be >= Open (violated in {invalid} rows)"
            )

        # High must be >= Close
        if "Close" in df.columns:
            if not (df["High"] >= df["Close"]).all():
                invalid = (~(df["High"] >= df["Close"])).sum()
                self.errors.append(
                    f"High must be >= Close (violated in {invalid} rows)"
                )

        # Low must be <= Open
        if "Low" in df.columns:
            if not (df["Low"] <= df["Open"]).all():
                invalid = (~(df["Low"] <= df["Open"])).sum()
                self.errors.append(
                    f"Low must be <= Open (violated in {invalid} rows)"
                )

        # Low must be <= Close
        if "Low" in df.columns and "Close" in df.columns:
            if not (df["Low"] <= df["Close"]).all():
                invalid = (~(df["Low"] <= df["Close"])).sum()
                self.errors.append(
                    f"Low must be <= Close (violated in {invalid} rows)"
                )

    def _check_temporal_ordering(self, df: pd.DataFrame):
        """Check temporal ordering."""
        if not df.index.is_monotonic_increasing:
            self.errors.append("Index must be monotonically increasing (chronological)")

        if df.index.has_duplicates:
            dup_count = df.index.duplicated().sum()
            self.errors.append(f"Index has {dup_count} duplicate timestamps")

    def _check_volume_positivity(self, df: pd.DataFrame):
        """Check volume is positive."""
        if "Volume" not in df.columns:
            return

        if not (df["Volume"] > 0).all():
            invalid = (~(df["Volume"] > 0)).sum()
            self.errors.append(
                f"Volume must be > 0 (violated in {invalid} rows)"
            )

    def _check_precision(self, df: pd.DataFrame):
        """Check precision preservation."""
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                continue

            # Check for excessive precision loss
            # Range bar should preserve ~8 decimal places from FixedPoint
            sample = df[col].head(100)
            decimals = sample.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
            avg_decimals = decimals.mean()

            if avg_decimals < 2:
                self.warnings.append(
                    f"Column '{col}' may have precision loss (avg {avg_decimals:.1f} decimals)"
                )


def main():
    """Run validation example."""
    print("=" * 70)
    print("rangebar-py: OHLCV Output Validation")
    print("=" * 70)
    print()

    # Generate test data
    print("1. Generating test data...")
    trades = [
        {
            "timestamp": 1704067200000 + i * 1000,
            "price": 42000.0 + i * 10.0,
            "quantity": 1.0 + (i % 5) * 0.5,
        }
        for i in range(1000)
    ]
    print(f"   Created {len(trades)} trades")

    # Convert to range bars
    print("\n2. Converting to range bars...")
    df = process_trades_to_dataframe(trades, threshold_bps=250)
    print(f"   Generated {len(df)} range bars")

    # Validate
    print("\n3. Running validation checks...")
    print("   " + "-" * 66)

    validator = OHLCVValidator(strict=False)
    is_valid, errors, warnings = validator.validate(df)

    # Report results
    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)

    if is_valid:
        print("\n✅ All validation checks passed!")
    else:
        print(f"\n❌ Validation failed with {len(errors)} error(s):")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")

    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")

    # Display DataFrame summary
    print("\n" + "=" * 70)
    print("DataFrame Summary")
    print("=" * 70)
    print(f"\nShape: {df.shape}")
    print(f"Index: {type(df.index).__name__}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    print("\n" + "=" * 70)
    if is_valid:
        print("✅ Output is valid for backtesting.py")
    else:
        print("❌ Output has validation errors")
    print("=" * 70)

    # Test with invalid data
    print("\n" + "=" * 70)
    print("Testing with Invalid Data (Expected Failures)")
    print("=" * 70)

    print("\n4. Creating intentionally invalid DataFrame...")
    invalid_df = df.copy()
    invalid_df.iloc[0, invalid_df.columns.get_loc("High")] = -100  # Invalid: negative
    invalid_df.iloc[1, invalid_df.columns.get_loc("Low")] = 999999  # Invalid: Low > High
    invalid_df.iloc[2, invalid_df.columns.get_loc("Volume")] = 0  # Invalid: zero volume

    print("\n5. Validating invalid data...")
    validator2 = OHLCVValidator(strict=False)
    is_valid2, errors2, warnings2 = validator2.validate(invalid_df)

    print(f"\n   Detected {len(errors2)} error(s) (as expected):")
    for i, error in enumerate(errors2, 1):
        print(f"   {i}. {error}")

    print("\n" + "=" * 70)
    print("✅ Validation script completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

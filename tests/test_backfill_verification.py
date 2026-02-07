"""Tests for backfill column verification (Issue #80, Phase 1).

Validates constants consistency and schema readiness for SOL backfill.
These tests run WITHOUT real data (unit tests only).
Post-backfill verification with real data is done via scripts/validate_backfill_preflight.py.
"""

from __future__ import annotations


class TestColumnConstants:
    """Verify inter-bar and intra-bar column constant counts and contents."""

    def test_inter_bar_feature_count(self):
        """INTER_BAR_FEATURE_COLUMNS should have exactly 16 columns."""
        from rangebar.constants import INTER_BAR_FEATURE_COLUMNS

        assert len(INTER_BAR_FEATURE_COLUMNS) == 16

    def test_intra_bar_feature_count(self):
        """INTRA_BAR_FEATURE_COLUMNS should have exactly 22 columns."""
        from rangebar.constants import INTRA_BAR_FEATURE_COLUMNS

        assert len(INTRA_BAR_FEATURE_COLUMNS) == 22

    def test_total_feature_count(self):
        """Total feature count should be 38 (16 + 22)."""
        from rangebar.constants import (
            INTER_BAR_FEATURE_COLUMNS,
            INTRA_BAR_FEATURE_COLUMNS,
        )

        total = len(INTER_BAR_FEATURE_COLUMNS) + len(INTRA_BAR_FEATURE_COLUMNS)
        assert total == 38

    def test_inter_bar_all_start_with_lookback(self):
        """All inter-bar columns should start with 'lookback_'."""
        from rangebar.constants import INTER_BAR_FEATURE_COLUMNS

        for col in INTER_BAR_FEATURE_COLUMNS:
            assert col.startswith("lookback_"), f"Column {col} doesn't start with lookback_"

    def test_intra_bar_all_start_with_intra(self):
        """All intra-bar columns should start with 'intra_'."""
        from rangebar.constants import INTRA_BAR_FEATURE_COLUMNS

        for col in INTRA_BAR_FEATURE_COLUMNS:
            assert col.startswith("intra_"), f"Column {col} doesn't start with intra_"

    def test_no_duplicate_columns(self):
        """No column name should appear in both inter and intra lists."""
        from rangebar.constants import (
            INTER_BAR_FEATURE_COLUMNS,
            INTRA_BAR_FEATURE_COLUMNS,
        )

        inter_set = set(INTER_BAR_FEATURE_COLUMNS)
        intra_set = set(INTRA_BAR_FEATURE_COLUMNS)
        overlap = inter_set & intra_set
        assert len(overlap) == 0, f"Overlapping columns: {overlap}"

    def test_all_optional_columns_includes_features(self):
        """ALL_OPTIONAL_COLUMNS should contain both inter and intra features."""
        from rangebar.constants import (
            ALL_OPTIONAL_COLUMNS,
            INTER_BAR_FEATURE_COLUMNS,
            INTRA_BAR_FEATURE_COLUMNS,
        )

        all_set = set(ALL_OPTIONAL_COLUMNS)
        for col in INTER_BAR_FEATURE_COLUMNS:
            assert col in all_set, f"Missing inter-bar column: {col}"
        for col in INTRA_BAR_FEATURE_COLUMNS:
            assert col in all_set, f"Missing intra-bar column: {col}"


class TestCacheLayerWiring:
    """Verify inter/intra columns are wired through cache layer."""

    def test_bulk_operations_imports_columns(self):
        """bulk_operations.py should import INTER_BAR and INTRA_BAR columns."""
        import inspect

        from rangebar.clickhouse.bulk_operations import BulkStoreMixin

        source = inspect.getsource(BulkStoreMixin)
        assert "INTER_BAR_FEATURE_COLUMNS" in source
        assert "INTRA_BAR_FEATURE_COLUMNS" in source

    def test_store_bars_bulk_handles_features(self):
        """store_bars_bulk should iterate over feature columns."""
        import inspect

        from rangebar.clickhouse.bulk_operations import BulkStoreMixin

        source = inspect.getsource(BulkStoreMixin.store_bars_bulk)
        assert "INTER_BAR_FEATURE_COLUMNS" in source
        assert "INTRA_BAR_FEATURE_COLUMNS" in source

    def test_store_bars_batch_handles_features(self):
        """store_bars_batch should iterate over feature columns."""
        import inspect

        from rangebar.clickhouse.bulk_operations import BulkStoreMixin

        source = inspect.getsource(BulkStoreMixin.store_bars_batch)
        assert "INTER_BAR_FEATURE_COLUMNS" in source
        assert "INTRA_BAR_FEATURE_COLUMNS" in source

    def test_query_operations_includes_features(self):
        """query_operations.py should include feature columns in SELECT."""
        import inspect

        from rangebar.clickhouse import query_operations

        source = inspect.getsource(query_operations)
        assert "INTER_BAR_FEATURE_COLUMNS" in source
        assert "INTRA_BAR_FEATURE_COLUMNS" in source


class TestFeatureRanges:
    """Verify expected ranges for key features (used by preflight script)."""

    def test_bounded_feature_names(self):
        """Key features that should be bounded [-1,1] or [0,1]."""
        from rangebar.constants import INTER_BAR_FEATURE_COLUMNS

        bounded_01 = {"lookback_vwap_position", "lookback_hurst", "lookback_kaufman_er"}
        bounded_neg1_1 = {"lookback_ofi", "lookback_count_imbalance"}

        inter_set = set(INTER_BAR_FEATURE_COLUMNS)
        for col in bounded_01:
            assert col in inter_set, f"Missing bounded [0,1] feature: {col}"
        for col in bounded_neg1_1:
            assert col in inter_set, f"Missing bounded [-1,1] feature: {col}"

    def test_trade_id_range_columns(self):
        """TRADE_ID_RANGE_COLUMNS should have 2 columns."""
        from rangebar.constants import TRADE_ID_RANGE_COLUMNS

        assert len(TRADE_ID_RANGE_COLUMNS) == 2
        assert "first_agg_trade_id" in TRADE_ID_RANGE_COLUMNS
        assert "last_agg_trade_id" in TRADE_ID_RANGE_COLUMNS

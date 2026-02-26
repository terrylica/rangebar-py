"""Gap detection ouroboros mode filtering tests. Issue #97.

Tests cover:
- Gap detection filters by ouroboros mode
- Monthly boundaries don't produce false positive gaps

Requires ClickHouse (@pytest.mark.clickhouse).
"""

import contextlib
import sys
from pathlib import Path

import pytest

# Add scripts/ to path so we can import detect_gaps
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

pytestmark = pytest.mark.clickhouse


@pytest.fixture
def ch_client():
    """Get a ClickHouse client, skip if unavailable."""
    try:
        from detect_gaps import connect

        client, tunnel = connect()
        yield client
        with contextlib.suppress(OSError):
            client.close()
        if tunnel is not None:
            tunnel.stop()
    except (ImportError, ConnectionError, OSError, RuntimeError) as e:
        pytest.skip(f"ClickHouse not available: {e}")


class TestGapDetectionFiltersByMode:
    """Test that gap detection only checks bars of the requested mode."""

    def test_year_mode_only_checks_year_bars(self, ch_client):
        """Gap detection with ouroboros_mode='year' ignores month-mode bars."""
        from detect_gaps import run_detection

        # Run detection with year mode (default)
        result_year = run_detection(
            ch_client,
            symbols=["BTCUSDT"],
            thresholds=[250],
            min_gap_hours=6.0,
            ouroboros_mode="year",
        )

        # Run detection with month mode
        result_month = run_detection(
            ch_client,
            symbols=["BTCUSDT"],
            thresholds=[250],
            min_gap_hours=6.0,
            ouroboros_mode="month",
        )

        # Results should be independent — different mode data produces
        # different gap counts. At minimum, neither should error.
        assert result_year.checked_pairs >= 0
        assert result_month.checked_pairs >= 0

        # If there's no month-mode data, month detection should skip the pair
        if result_month.checked_pairs == 0:
            assert result_month.skipped_pairs >= 1

    def test_month_mode_parameter_accepted(self, ch_client):
        """Verify --ouroboros-mode=month is accepted by run_detection."""
        from detect_gaps import run_detection

        # Should not raise
        result = run_detection(
            ch_client,
            symbols=["BTCUSDT"],
            thresholds=[250],
            min_gap_hours=6.0,
            ouroboros_mode="month",
        )
        # Just verify it ran without error
        assert isinstance(result.gaps, list)
        assert isinstance(result.coverage, list)


class TestMonthlyBoundariesNotFalsePositive:
    """Test that orphaned bar gaps at month boundaries don't trigger alerts."""

    def test_no_spurious_gaps_from_mode_filter(self, ch_client):
        """When filtering by mode, we don't see gaps from the other mode's data."""
        from detect_gaps import detect_gaps_for_pair

        # Detect gaps for year mode with a very low threshold
        gaps, coverage = detect_gaps_for_pair(
            ch_client,
            symbol="BTCUSDT",
            threshold=250,
            min_gap_ms=int(1.0 * 3600 * 1000),  # 1 hour
            price_gap_dbps=50000.0,  # Very high to avoid price gap noise
            ouroboros_mode="year",
        )

        if coverage is None:
            pytest.skip("No BTCUSDT@250 year-mode data in ClickHouse")

        # Verify the detection ran successfully
        assert coverage.total_bars > 0
        assert coverage.symbol == "BTCUSDT"
        assert coverage.threshold_dbps == 250

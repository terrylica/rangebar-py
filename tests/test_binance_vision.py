"""Tests for Binance Vision end-date availability probe (Issue #88).

Covers:
- probe_latest_available_date() behavior with mock HTTP responses
- Constants (URL pattern, defaults)
- Package exports (__all__ and rangebar re-exports)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import rangebar
from rangebar.binance_vision import (
    _DEFAULT_MAX_LOOKBACK,
    _DEFAULT_PROBE_SYMBOL,
    BINANCE_VISION_AGGTRADES_URL,
    probe_latest_available_date,
)

# =============================================================================
# Helpers
# =============================================================================


def _date_n_days_ago(n: int) -> str:
    """Return YYYY-MM-DD for n days ago (UTC)."""
    today = datetime.now(tz=UTC).date()
    return (today - timedelta(days=n)).isoformat()


def _make_response(status: int = 200) -> MagicMock:
    """Create mock HTTP response with context manager support."""
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# =============================================================================
# Probe tests
# =============================================================================


class TestProbeLatestAvailableDate:
    """Test probe_latest_available_date() with mocked HTTP."""

    @patch("rangebar.binance_vision.urlopen")
    def test_yesterday_available(self, mock_urlopen: MagicMock) -> None:
        """When yesterday's archive exists, return yesterday immediately."""
        mock_urlopen.return_value = _make_response(200)

        result = probe_latest_available_date()

        assert result == _date_n_days_ago(1)
        assert mock_urlopen.call_count == 1

    @patch("rangebar.binance_vision.urlopen")
    def test_yesterday_404_day_before_available(
        self, mock_urlopen: MagicMock
    ) -> None:
        """When yesterday 404s, walk back and return day-before-yesterday."""
        mock_urlopen.side_effect = [
            HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None),
            _make_response(200),
        ]

        result = probe_latest_available_date()

        assert result == _date_n_days_ago(2)
        assert mock_urlopen.call_count == 2

    @patch("rangebar.binance_vision.urlopen")
    def test_all_days_404_returns_fallback(self, mock_urlopen: MagicMock) -> None:
        """When all days 404, return max_lookback days ago as fallback."""
        mock_urlopen.side_effect = HTTPError(
            url="", code=404, msg="Not Found", hdrs=None, fp=None
        )

        result = probe_latest_available_date()

        assert result == _date_n_days_ago(10)
        assert mock_urlopen.call_count == 10

    @patch("rangebar.binance_vision.urlopen")
    def test_network_timeout_falls_back(self, mock_urlopen: MagicMock) -> None:
        """TimeoutError on all days falls back to max_lookback."""
        mock_urlopen.side_effect = TimeoutError("Connection timed out")

        result = probe_latest_available_date(max_lookback=3)

        assert result == _date_n_days_ago(3)
        assert mock_urlopen.call_count == 3

    @patch("rangebar.binance_vision.urlopen")
    def test_os_error_falls_back(self, mock_urlopen: MagicMock) -> None:
        """OSError on all days falls back to max_lookback."""
        mock_urlopen.side_effect = OSError("Network unreachable")

        result = probe_latest_available_date(max_lookback=5)

        assert result == _date_n_days_ago(5)

    @patch("rangebar.binance_vision.urlopen")
    def test_url_error_falls_back(self, mock_urlopen: MagicMock) -> None:
        """URLError on all days falls back to max_lookback."""
        mock_urlopen.side_effect = URLError("Name resolution failed")

        result = probe_latest_available_date(max_lookback=3)

        assert result == _date_n_days_ago(3)
        assert mock_urlopen.call_count == 3

    @patch("rangebar.binance_vision.urlopen")
    def test_custom_symbol_in_url(self, mock_urlopen: MagicMock) -> None:
        """Custom symbol appears in the request URL."""
        mock_urlopen.return_value = _make_response(200)

        probe_latest_available_date(symbol="ETHUSDT")

        # Extract the Request object passed to urlopen
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        url = req.full_url
        assert "ETHUSDT" in url
        assert "BTCUSDT" not in url

    @patch("rangebar.binance_vision.urlopen")
    def test_custom_max_lookback(self, mock_urlopen: MagicMock) -> None:
        """Custom max_lookback limits the number of probes."""
        mock_urlopen.side_effect = HTTPError(
            url="", code=404, msg="Not Found", hdrs=None, fp=None
        )

        result = probe_latest_available_date(max_lookback=2)

        assert mock_urlopen.call_count == 2
        assert result == _date_n_days_ago(2)

    @patch("rangebar.binance_vision.urlopen")
    def test_mixed_errors_then_success(self, mock_urlopen: MagicMock) -> None:
        """Mixed error types followed by success returns correct date."""
        mock_urlopen.side_effect = [
            HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None),
            TimeoutError("Connection timed out"),
            _make_response(200),
        ]

        result = probe_latest_available_date()

        assert result == _date_n_days_ago(3)
        assert mock_urlopen.call_count == 3


# =============================================================================
# Constants tests
# =============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_default_max_lookback_is_10(self) -> None:
        assert _DEFAULT_MAX_LOOKBACK == 10

    def test_default_probe_symbol_is_btcusdt(self) -> None:
        assert _DEFAULT_PROBE_SYMBOL == "BTCUSDT"

    def test_url_has_placeholders(self) -> None:
        assert "{symbol}" in BINANCE_VISION_AGGTRADES_URL
        assert "{date}" in BINANCE_VISION_AGGTRADES_URL

    def test_url_is_https(self) -> None:
        assert BINANCE_VISION_AGGTRADES_URL.startswith("https://")

    def test_url_formats_correctly(self) -> None:
        url = BINANCE_VISION_AGGTRADES_URL.format(
            symbol="BTCUSDT", date="2026-02-08"
        )
        expected = (
            "https://data.binance.vision/data/futures/um/daily/aggTrades"
            "/BTCUSDT/BTCUSDT-aggTrades-2026-02-08.zip"
        )
        assert url == expected


# =============================================================================
# Export tests
# =============================================================================


class TestExports:
    """Test that probe is exported from the rangebar package."""

    def test_probe_exported_from_package(self) -> None:
        assert callable(rangebar.probe_latest_available_date)

    def test_url_constant_exported_from_package(self) -> None:
        assert isinstance(rangebar.BINANCE_VISION_AGGTRADES_URL, str)

    def test_probe_in_all(self) -> None:
        assert "probe_latest_available_date" in rangebar.__all__

    def test_url_constant_in_all(self) -> None:
        assert "BINANCE_VISION_AGGTRADES_URL" in rangebar.__all__

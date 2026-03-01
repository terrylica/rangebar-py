"""Tests for progress logging (Issue #70).

Tests the ResumableObserver class and its integration with
populate_cache_resumable()'s verbose parameter.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from rangebar.checkpoint import populate_cache_resumable
from rangebar.progress import ResumableObserver


class TestResumableObserver:
    """Test ResumableObserver progress tracking."""

    def test_basic_initialization(self) -> None:
        """Observer initializes with correct defaults."""
        observer = ResumableObserver(
            total=100,
            desc="BTCUSDT",
            verbose=False,  # Disable tqdm for unit tests
        )
        assert observer.total == 100
        assert observer.desc == "BTCUSDT"
        assert observer.count == 0
        assert observer.initial == 0
        assert observer.unit == "days"

    def test_resumed_initialization(self) -> None:
        """Observer handles resumed state with initial parameter."""
        observer = ResumableObserver(
            total=100,
            initial=45,
            desc="BTCUSDT",
            verbose=False,
        )
        assert observer.count == 45
        assert observer.initial == 45

    def test_update_increments_count(self) -> None:
        """update() increments the count correctly."""
        observer = ResumableObserver(total=10, verbose=False)
        assert observer.count == 0

        observer.update(1)
        assert observer.count == 1

        observer.update(3)
        assert observer.count == 4

    def test_wrap_iterator(self) -> None:
        """wrap() yields items and updates count."""
        observer = ResumableObserver(total=5, verbose=False)
        items = ["a", "b", "c", "d", "e"]

        result = list(observer.wrap(iter(items)))

        assert result == items
        assert observer.count == 5

    def test_context_manager(self) -> None:
        """Observer works as context manager."""
        with ResumableObserver(total=10, verbose=False) as observer:
            observer.update(5)
            assert observer.count == 5

    def test_log_event_throttling(self) -> None:
        """log_event() respects throttling interval."""
        observer = ResumableObserver(
            total=100,
            verbose=False,
            log_interval_sec=5.0,
        )
        # Logger won't be initialized when verbose=False
        # Just verify it doesn't raise
        observer.log_event("test_event", foo="bar")

    def test_set_postfix_without_tqdm(self) -> None:
        """set_postfix() handles missing tqdm gracefully."""
        observer = ResumableObserver(total=10, verbose=False)
        # Should not raise even without tqdm initialized
        observer.set_postfix(bars=100, rate="2.3/s")

    def test_close_without_tqdm(self) -> None:
        """close() handles missing tqdm gracefully."""
        observer = ResumableObserver(total=10, verbose=False)
        observer.update(5)
        # Should not raise
        observer.close()


class TestResumableObserverWithTqdm:
    """Test ResumableObserver with tqdm enabled (mocked)."""

    def test_tqdm_initialization(self) -> None:
        """Observer initializes tqdm when verbose=True."""
        # Patch tqdm.tqdm where it's imported in the progress module
        with patch("tqdm.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar

            observer = ResumableObserver(
                total=100,
                initial=10,
                desc="BTCUSDT",
                unit="days",
                verbose=True,
            )

            mock_tqdm.assert_called_once()
            call_kwargs = mock_tqdm.call_args.kwargs
            assert call_kwargs["total"] == 100
            assert call_kwargs["initial"] == 10
            assert call_kwargs["desc"] == "BTCUSDT"
            assert call_kwargs["unit"] == "days"

            observer.close()

    def test_tqdm_update_called(self) -> None:
        """update() calls tqdm.update()."""
        with patch("tqdm.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar

            observer = ResumableObserver(total=10, verbose=True)
            observer.update(3)

            mock_pbar.update.assert_called_with(3)
            observer.close()

    def test_tqdm_close_called(self) -> None:
        """close() calls tqdm.close()."""
        with patch("tqdm.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar

            observer = ResumableObserver(total=10, verbose=True)
            observer.close()

            mock_pbar.close.assert_called_once()


class TestPopulateCacheResumableVerbose:
    """Test verbose parameter integration in populate_cache_resumable."""

    def test_verbose_parameter_exists(self) -> None:
        """populate_cache_resumable accepts verbose parameter."""
        sig = inspect.signature(populate_cache_resumable)
        params = sig.parameters

        assert "verbose" in params
        assert params["verbose"].default is True

    def test_verbose_false_no_progress_bar(self) -> None:
        """verbose=False should not initialize progress observer."""
        observer = ResumableObserver(total=10, verbose=False)
        # When verbose=False, tqdm should not be initialized
        assert observer.pbar is None
        observer.update(1)
        observer.close()

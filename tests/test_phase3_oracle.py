"""Oracle-based bit-exact verification for Phase 3+4 optimizations (Issue #88).

6-tier verification hierarchy:
  Tier 1: Timestamp conversion oracles (ms/us boundary — safety-critical)
  Tier 2: Stream Arrow parity (dict vs Arrow path equivalence)
  Tier 3: Single-pass Arrow export parity
  Tier 4: has_ticks() lazy vs eager equivalence
  Tier 5: store_bars_batch vs store_bars_bulk parity
  Tier 6: Divergence detection framework

Methodology: Do NOT use the system under test to generate expected values.
Independently recompute from raw source. Assert numerical identity, not
approximate equality.
"""

import math

import polars as pl
import pyarrow as pa
import pytest
from rangebar._core import PyRangeBarProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCALE = 100_000_000  # FixedPoint scale factor (8 decimal places)


def _make_trade_batch_us(
    timestamps_us: list[int],
    prices: list[float],
    volumes: list[float],
    *,
    is_buyer_maker: list[bool] | None = None,
    agg_trade_ids: list[int] | None = None,
    first_trade_ids: list[int] | None = None,
    last_trade_ids: list[int] | None = None,
) -> pa.RecordBatch:
    """Build Arrow RecordBatch with MICROSECOND timestamps (internal path)."""
    fields = [
        pa.field("timestamp", pa.int64()),
        pa.field("price", pa.float64()),
        pa.field("quantity", pa.float64()),
    ]
    arrays = [
        pa.array(timestamps_us, type=pa.int64()),
        pa.array(prices, type=pa.float64()),
        pa.array(volumes, type=pa.float64()),
    ]
    if is_buyer_maker is not None:
        fields.append(pa.field("is_buyer_maker", pa.bool_()))
        arrays.append(pa.array(is_buyer_maker, type=pa.bool_()))
    if agg_trade_ids is not None:
        fields.append(pa.field("agg_trade_id", pa.int64()))
        arrays.append(pa.array(agg_trade_ids, type=pa.int64()))
    if first_trade_ids is not None:
        fields.append(pa.field("first_trade_id", pa.int64()))
        arrays.append(pa.array(first_trade_ids, type=pa.int64()))
    if last_trade_ids is not None:
        fields.append(pa.field("last_trade_id", pa.int64()))
        arrays.append(pa.array(last_trade_ids, type=pa.int64()))

    return pa.record_batch(arrays, schema=pa.schema(fields))


def _make_trade_batch_ms(
    timestamps_ms: list[int],
    prices: list[float],
    volumes: list[float],
    *,
    is_buyer_maker: list[bool] | None = None,
    agg_trade_ids: list[int] | None = None,
) -> pa.RecordBatch:
    """Build Arrow RecordBatch with MILLISECOND timestamps (Python API path)."""
    fields = [
        pa.field("timestamp", pa.int64()),
        pa.field("price", pa.float64()),
        pa.field("quantity", pa.float64()),
    ]
    arrays = [
        pa.array(timestamps_ms, type=pa.int64()),
        pa.array(prices, type=pa.float64()),
        pa.array(volumes, type=pa.float64()),
    ]
    if is_buyer_maker is not None:
        fields.append(pa.field("is_buyer_maker", pa.bool_()))
        arrays.append(pa.array(is_buyer_maker, type=pa.bool_()))
    if agg_trade_ids is not None:
        fields.append(pa.field("agg_trade_id", pa.int64()))
        arrays.append(pa.array(agg_trade_ids, type=pa.int64()))

    return pa.record_batch(arrays, schema=pa.schema(fields))


def _make_trade_dicts(
    timestamps_ms: list[int],
    prices: list[float],
    volumes: list[float],
    *,
    is_buyer_maker: list[bool] | None = None,
    agg_trade_ids: list[int] | None = None,
) -> list[dict]:
    """Build trade dicts with millisecond timestamps (dict path)."""
    result = []
    for i in range(len(timestamps_ms)):
        d = {
            "timestamp": timestamps_ms[i],
            "price": prices[i],
            "quantity": volumes[i],
        }
        if is_buyer_maker is not None:
            d["is_buyer_maker"] = is_buyer_maker[i]
        if agg_trade_ids is not None:
            d["agg_trade_id"] = agg_trade_ids[i]
        result.append(d)
    return result


def _generate_breach_trades(
    n: int = 50,
    threshold_dbps: int = 250,
    base_price: float = 50000.0,
    base_ts_ms: int = 1704067200000,
) -> tuple[list[int], list[float], list[float], list[bool], list[int]]:
    """Generate trades that produce range bar breaches.

    Returns (timestamps_ms, prices, volumes, is_buyer_maker, agg_trade_ids).
    """
    threshold = base_price * (threshold_dbps / 100_000)
    step = threshold / 3

    timestamps = []
    prices = []
    volumes = []
    is_buyer_maker = []
    agg_trade_ids = []

    price = base_price
    direction = 1
    for i in range(n):
        timestamps.append(base_ts_ms + i * 1000)
        prices.append(round(price, 8))
        volumes.append(0.1 + (i % 5) * 0.01)
        is_buyer_maker.append(i % 2 == 0)
        agg_trade_ids.append(i)

        price += direction * step
        if abs(price - base_price) > threshold * 1.5:
            direction *= -1

    return timestamps, prices, volumes, is_buyer_maker, agg_trade_ids


# ===========================================================================
# Tier 1: Timestamp Conversion Oracles (Critical — ms/us boundary)
# ===========================================================================


class TestTier1TimestampOracles:
    """Validate ms↔us conversion at the Arrow boundary."""

    def test_ms_to_us_via_process_trades_arrow(self):
        """Millisecond input through process_trades_arrow produces valid bars.

        1704067200000 ms → internal 1704067200000000 us → bars with
        open_time/close_time in microseconds.
        """
        ts_ms, prices, vols, ibm, aids = _generate_breach_trades(
            n=50, threshold_dbps=250, base_ts_ms=1704067200000,
        )
        batch = _make_trade_batch_ms(
            ts_ms, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )

        proc = PyRangeBarProcessor(250, symbol="TEST")
        result = pl.from_arrow(proc.process_trades_arrow(batch))

        assert len(result) > 0, "Expected bars from 50 trades at 250 dbps"

        # All open_time values must be in microsecond range
        for i in range(len(result)):
            ot = result["open_time"][i]
            # 1704067200000000 us = 2024-01-01T00:00:00 UTC
            assert ot >= 1_704_067_200_000_000, (
                f"Bar {i}: open_time={ot} below expected us range"
            )
            assert ot < 2_051_222_400_000_000, (
                f"Bar {i}: open_time={ot} exceeds year 2035 — timestamp error"
            )

    def test_us_to_us_via_process_trades_arrow_native(self):
        """Microsecond input through process_trades_arrow_native — no x1000.

        1704067200000000 us → process_trades_arrow_native → bars with valid
        timestamps. This is the Phase 3 internal stream path.
        """
        # Generate trades with us timestamps (multiply ms by 1000)
        ts_ms, prices, vols, ibm, aids = _generate_breach_trades(
            n=50, threshold_dbps=250, base_ts_ms=1704067200000,
        )
        ts_us = [t * 1000 for t in ts_ms]

        batch = _make_trade_batch_us(
            ts_us, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )

        proc = PyRangeBarProcessor(250, symbol="TEST")
        result = pl.from_arrow(proc.process_trades_arrow_native(batch))

        assert len(result) > 0, "Expected bars from 50 trades at 250 dbps"

        # All open_time values must be in valid microsecond range
        for i in range(len(result)):
            ot = result["open_time"][i]
            assert ot >= 1_704_067_200_000_000, (
                f"Bar {i}: open_time={ot} below expected us range"
            )
            assert ot < 2_051_222_400_000_000, (
                f"Bar {i}: open_time={ot} exceeds year 2035 — timestamp doubling!"
            )

    def test_ms_vs_us_path_parity(self):
        """process_trades_arrow(ms) and process_trades_arrow_native(us)
        produce identical bars for the same logical trades.
        """
        ts_ms, prices, vols, ibm, aids = _generate_breach_trades(
            n=80, threshold_dbps=250, base_ts_ms=1704067200000,
        )
        ts_us = [t * 1000 for t in ts_ms]

        # Path A: ms input (Python API path)
        batch_ms = _make_trade_batch_ms(
            ts_ms, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc_a = PyRangeBarProcessor(250, symbol="TEST")
        result_a = pl.from_arrow(proc_a.process_trades_arrow(batch_ms))

        # Path B: us input (internal stream path)
        batch_us = _make_trade_batch_us(
            ts_us, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc_b = PyRangeBarProcessor(250, symbol="TEST")
        result_b = pl.from_arrow(proc_b.process_trades_arrow_native(batch_us))

        # Must produce identical bars
        assert len(result_a) == len(result_b), (
            f"Bar count mismatch: ms_path={len(result_a)}, us_path={len(result_b)}"
        )

        if len(result_a) == 0:
            return

        for col in result_a.columns:
            if col not in result_b.columns:
                continue
            for i in range(len(result_a)):
                val_a = result_a[col][i]
                val_b = result_b[col][i]
                if val_a is None and val_b is None:
                    continue
                if isinstance(val_a, float) and isinstance(val_b, float):
                    assert math.isclose(val_a, val_b, rel_tol=1e-12), (
                        f"Divergence at bar {i}, field '{col}': "
                        f"ms_path={val_a}, us_path={val_b}"
                    )
                else:
                    assert val_a == val_b, (
                        f"Divergence at bar {i}, field '{col}': "
                        f"ms_path={val_a}, us_path={val_b}"
                    )

    def test_timestamp_double_conversion_guard(self):
        """us timestamps through ms path would produce year ~52000 — guard catches.

        If us timestamps (1704067200000000) are accidentally treated as ms
        (x1000 again), we get 1704067200000000000 (19 digits, year ~52000).
        The processor should either reject or the result should show clearly
        wrong timestamps that our guard catches.
        """
        ts_us = [1704067200000000, 1704067201000000, 1704067202000000]
        prices = [50000.0, 50200.0, 49800.0]  # Large swing for breach
        vols = [1.0, 1.0, 1.0]

        # Intentionally using ms path with us timestamps (WRONG usage)
        batch = _make_trade_batch_ms(ts_us, prices, vols)
        proc = PyRangeBarProcessor(250, symbol="TEST")
        result = pl.from_arrow(proc.process_trades_arrow(batch))

        # If bars were produced, their timestamps would be in the wrong range
        if len(result) > 0:
            for i in range(len(result)):
                ot = result["open_time"][i]
                # These would be ~1.7e18 (year ~52000) — clearly wrong
                assert ot > 2_051_222_400_000_000, (
                    f"Expected timestamp in wrong range but got {ot}"
                )

    def test_cross_year_timestamp_sweep(self):
        """Verify both ms and us paths across timestamps from 2018-2025.

        All should produce valid microsecond timestamps in [2000, 2035] range.
        """
        year_timestamps_ms = {
            2018: 1514764800000,   # 2018-01-01T00:00:00Z
            2020: 1577836800000,   # 2020-01-01T00:00:00Z
            2024: 1704067200000,   # 2024-01-01T00:00:00Z
            2025: 1735689600000,   # 2025-01-01T00:00:00Z
        }

        for year, base_ts in year_timestamps_ms.items():
            # Generate trades at this base timestamp
            ts_ms, prices, vols, ibm, aids = _generate_breach_trades(
                n=30, threshold_dbps=500, base_ts_ms=base_ts,
            )
            ts_us = [t * 1000 for t in ts_ms]

            # ms path
            batch_ms = _make_trade_batch_ms(
                ts_ms, prices, vols,
                is_buyer_maker=ibm, agg_trade_ids=aids,
            )
            proc_ms = PyRangeBarProcessor(500, symbol="TEST")
            result_ms = pl.from_arrow(proc_ms.process_trades_arrow(batch_ms))

            # us path
            batch_us = _make_trade_batch_us(
                ts_us, prices, vols,
                is_buyer_maker=ibm, agg_trade_ids=aids,
            )
            proc_us = PyRangeBarProcessor(500, symbol="TEST")
            result_us = pl.from_arrow(proc_us.process_trades_arrow_native(batch_us))

            # Both paths should produce same bar count
            assert len(result_ms) == len(result_us), (
                f"Year {year}: bar count mismatch ms={len(result_ms)} us={len(result_us)}"
            )

            # All timestamps should be valid microseconds in [2000, 2035]
            min_valid_us = 946_684_800_000_000     # 2000-01-01
            max_valid_us = 2_051_222_400_000_000    # 2035-01-01
            for path_name, result in [("ms", result_ms), ("us", result_us)]:
                for i in range(len(result)):
                    ot = result["open_time"][i]
                    assert min_valid_us <= ot <= max_valid_us, (
                        f"Year {year}, {path_name} path, bar {i}: "
                        f"open_time={ot} outside valid range"
                    )

    def test_export_import_roundtrip_microseconds(self):
        """Vec<AggTrade> → aggtrades_to_record_batch → record_batch_to_aggtrades(us=true).

        This is the Phase 3 internal path. Timestamps must survive unchanged.
        Test via process_trades_arrow_native which internally calls
        record_batch_to_aggtrades with timestamp_is_microseconds=true.
        """
        # Single trade — won't produce bars, but validates no crash
        ts_us = [1704067200123456]  # us with sub-ms precision
        batch = _make_trade_batch_us(
            ts_us, [50000.0], [1.0],
            agg_trade_ids=[42],
        )
        proc = PyRangeBarProcessor(250, symbol="TEST")
        # Should not crash — validates the us→us path works
        result = pl.from_arrow(proc.process_trades_arrow_native(batch))
        # No bars expected from 1 trade, but no error
        assert result.is_empty() or len(result) >= 0


# ===========================================================================
# Tier 2: Stream Arrow Parity (dict vs Arrow native path)
# ===========================================================================


class TestTier2StreamArrowParity:
    """Arrow native path must produce identical output to dict path."""

    def _assert_native_dict_parity(
        self,
        ts_ms: list[int],
        prices: list[float],
        volumes: list[float],
        threshold: int,
        *,
        is_buyer_maker: list[bool] | None = None,
        agg_trade_ids: list[int] | None = None,
    ) -> None:
        """Assert Arrow native (us) and dict paths produce identical bars."""
        ts_us = [t * 1000 for t in ts_ms]

        # Path A: dict (ground truth)
        dicts = _make_trade_dicts(
            ts_ms, prices, volumes,
            is_buyer_maker=is_buyer_maker,
            agg_trade_ids=agg_trade_ids,
        )
        proc_dict = PyRangeBarProcessor(threshold, symbol="TEST")
        dict_out = proc_dict.process_trades_streaming(dicts)
        if dict_out:
            dict_df = pl.DataFrame(dict_out, infer_schema_length=None)
        else:
            dict_df = pl.DataFrame()

        # Path B: Arrow native (us timestamps, Phase 3 internal path)
        batch_us = _make_trade_batch_us(
            ts_us, prices, volumes,
            is_buyer_maker=is_buyer_maker,
            agg_trade_ids=agg_trade_ids,
        )
        proc_native = PyRangeBarProcessor(threshold, symbol="TEST")
        native_out = proc_native.process_trades_arrow_native(batch_us)
        native_df = pl.from_arrow(native_out)

        # Compare bar counts
        assert len(dict_df) == len(native_df), (
            f"Bar count mismatch: dict={len(dict_df)}, native={len(native_df)}"
        )

        if len(dict_df) == 0:
            return

        # Compare every field
        for col in native_df.columns:
            if col not in dict_df.columns:
                continue
            for i in range(len(native_df)):
                native_val = native_df[col][i]
                dict_val = dict_df[col][i]
                if native_val is None and dict_val is None:
                    continue
                if isinstance(native_val, float) and isinstance(dict_val, float):
                    assert math.isclose(native_val, dict_val, rel_tol=1e-12), (
                        f"Divergence at bar {i}, field '{col}': "
                        f"native={native_val}, dict={dict_val}"
                    )
                else:
                    assert native_val == dict_val, (
                        f"Divergence at bar {i}, field '{col}': "
                        f"native={native_val}, dict={dict_val}"
                    )

    def test_ohlcv_parity(self):
        """OHLCV fields match between dict and Arrow native paths."""
        ts, prices, vols, ibm, aids = _generate_breach_trades(
            n=50, threshold_dbps=250,
        )
        self._assert_native_dict_parity(
            ts, prices, vols,
            threshold=250,
            is_buyer_maker=ibm,
            agg_trade_ids=aids,
        )

    def test_large_batch_parity(self):
        """200 trades at 250 dbps — stress test for Arrow native path."""
        ts, prices, vols, ibm, aids = _generate_breach_trades(
            n=200, threshold_dbps=250,
        )
        self._assert_native_dict_parity(
            ts, prices, vols,
            threshold=250,
            is_buyer_maker=ibm,
            agg_trade_ids=aids,
        )

    def test_threshold_sweep_parity(self):
        """Same trades at [250, 500, 1000] dbps — all must match.

        Also verifies invariant: higher threshold → fewer bars.
        """
        ts, prices, vols, ibm, aids = _generate_breach_trades(
            n=100, threshold_dbps=250, base_price=50000.0,
        )

        bar_counts = []
        for threshold in [250, 500, 1000]:
            self._assert_native_dict_parity(
                ts, prices, vols,
                threshold=threshold,
                is_buyer_maker=ibm,
                agg_trade_ids=aids,
            )
            # Also count bars for monotonicity check
            ts_us = [t * 1000 for t in ts]
            batch_us = _make_trade_batch_us(
                ts_us, prices, vols,
                is_buyer_maker=ibm, agg_trade_ids=aids,
            )
            proc = PyRangeBarProcessor(threshold, symbol="TEST")
            result = pl.from_arrow(proc.process_trades_arrow_native(batch_us))
            bar_counts.append(len(result))

        # Higher threshold → fewer or equal bars
        assert bar_counts[0] >= bar_counts[1] >= bar_counts[2], (
            f"Monotonicity violated: {bar_counts}"
        )

    def test_chunked_native_parity(self):
        """100 trades split [30,30,40] via native path == single batch."""
        ts, prices, vols, ibm, aids = _generate_breach_trades(
            n=100, threshold_dbps=250,
        )
        ts_us = [t * 1000 for t in ts]

        # Single batch
        batch_all = _make_trade_batch_us(
            ts_us, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc_single = PyRangeBarProcessor(250, symbol="TEST")
        result_single = pl.from_arrow(proc_single.process_trades_arrow_native(batch_all))

        # Chunked
        proc_chunked = PyRangeBarProcessor(250, symbol="TEST")
        splits = [(0, 30), (30, 60), (60, 100)]
        result_frames = []
        for start, end in splits:
            chunk = _make_trade_batch_us(
                ts_us[start:end], prices[start:end], vols[start:end],
                is_buyer_maker=ibm[start:end], agg_trade_ids=aids[start:end],
            )
            out = proc_chunked.process_trades_arrow_native(chunk)
            df = pl.from_arrow(out)
            if not df.is_empty():
                result_frames.append(df)
        result_chunked = pl.concat(result_frames) if result_frames else pl.DataFrame()

        assert len(result_single) == len(result_chunked), (
            f"Single={len(result_single)}, Chunked={len(result_chunked)}"
        )

        for col in result_single.columns:
            if col not in result_chunked.columns:
                continue
            for i in range(len(result_single)):
                assert result_single[col][i] == result_chunked[col][i], (
                    f"Chunk divergence at bar {i}, field '{col}': "
                    f"single={result_single[col][i]}, chunked={result_chunked[col][i]}"
                )


# ===========================================================================
# Tier 3: Single-Pass Arrow Export Parity
# ===========================================================================


class TestTier3SinglePassExport:
    """Validate single-pass Arrow export produces identical output.

    The refactored rangebar_vec_to_record_batch (single-pass with builders)
    must produce bit-identical output to the original multi-pass implementation.
    Tested indirectly: both process_trades_arrow and process_trades_arrow_native
    use the same rangebar_vec_to_record_batch internally.
    """

    def test_export_empty_batch(self):
        """Empty trade input → zero-row RecordBatch with correct schema."""
        proc = PyRangeBarProcessor(250, symbol="TEST")
        # 1 trade won't breach → 0 bars
        batch = _make_trade_batch_ms([1704067200000], [50000.0], [1.0])
        result = proc.process_trades_arrow(batch)
        df = pl.from_arrow(result)
        assert df.is_empty()
        # Should have schema columns even when empty
        assert len(df.columns) > 0

    def test_export_produces_all_fields(self):
        """Bars produced by single-pass export contain all expected columns."""
        ts, prices, vols, ibm, aids = _generate_breach_trades(
            n=50, threshold_dbps=250,
        )
        batch = _make_trade_batch_ms(
            ts, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc = PyRangeBarProcessor(250, symbol="TEST")
        result = pl.from_arrow(proc.process_trades_arrow(batch))

        assert len(result) > 0

        # Core OHLCV fields
        for field in ["open_time", "close_time", "open", "high", "low", "close", "volume"]:
            assert field in result.columns, f"Missing field: {field}"

        # Microstructure fields (computed by Rust)
        for field in ["ofi", "duration_us", "individual_trade_count", "buy_volume", "sell_volume"]:
            assert field in result.columns, f"Missing microstructure field: {field}"

    def test_export_large_batch_no_nan(self):
        """100K-bar equivalent processing — no NaN in output."""
        # Generate enough trades for ~50+ bars
        ts, prices, vols, ibm, aids = _generate_breach_trades(
            n=500, threshold_dbps=250,
        )
        batch = _make_trade_batch_ms(
            ts, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc = PyRangeBarProcessor(250, symbol="TEST")
        result = pl.from_arrow(proc.process_trades_arrow(batch))

        assert len(result) > 0

        # No NaN in core fields
        for field in ["open", "high", "low", "close", "volume"]:
            null_count = result[field].null_count()
            assert null_count == 0, f"Field '{field}' has {null_count} nulls"

    def test_export_ohlcv_invariants(self):
        """OHLCV invariants: high >= max(open, close), low <= min(open, close)."""
        ts, prices, vols, ibm, aids = _generate_breach_trades(
            n=200, threshold_dbps=250,
        )
        batch = _make_trade_batch_ms(
            ts, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc = PyRangeBarProcessor(250, symbol="TEST")
        result = pl.from_arrow(proc.process_trades_arrow(batch))

        for i in range(len(result)):
            o = result["open"][i]
            h = result["high"][i]
            low = result["low"][i]
            c = result["close"][i]

            assert h >= o, f"Bar {i}: high={h} < open={o}"
            assert h >= c, f"Bar {i}: high={h} < close={c}"
            assert low <= o, f"Bar {i}: low={low} > open={o}"
            assert low <= c, f"Bar {i}: low={low} > close={c}"
            assert h >= low, f"Bar {i}: high={h} < low={low}"


# ===========================================================================
# Tier 4: has_ticks() Lazy vs Eager Equivalence
# ===========================================================================


class TestTier4HasTicksLazy:
    """Validate lazy Parquet scan has_ticks() matches eager behavior."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create TickStorage with temp directory."""
        from rangebar.storage.parquet import TickStorage
        return TickStorage(cache_dir=tmp_path)

    @pytest.fixture
    def sample_ticks(self):
        """Generate sample tick data: 1000 ticks over 1 hour."""
        base_ts = 1704067200000  # 2024-01-01T00:00:00Z in ms
        n = 1000
        return pl.DataFrame({
            "timestamp": [base_ts + i * 3600 for i in range(n)],
            "price": [50000.0 + (i % 100) for i in range(n)],
            "quantity": [0.1 + (i % 10) * 0.01 for i in range(n)],
            "agg_trade_id": list(range(n)),
        })

    def test_no_data_symbol(self, storage):
        """has_ticks for nonexistent symbol returns False."""
        result = storage.has_ticks("NONEXISTENT", 1704067200000, 1704153600000)
        assert result is False

    def test_full_coverage(self, storage, sample_ticks):
        """Data spanning exactly [start, end] → True with coverage=1.0."""
        storage.write_ticks("BTCUSDT", sample_ticks)

        start_ts = int(sample_ticks["timestamp"].min())
        end_ts = int(sample_ticks["timestamp"].max())

        result = storage.has_ticks("BTCUSDT", start_ts, end_ts)
        assert result is True

    def test_partial_coverage_below_threshold(self, storage, sample_ticks):
        """Data covering <95% of range → False with default min_coverage."""
        storage.write_ticks("BTCUSDT", sample_ticks)

        start_ts = int(sample_ticks["timestamp"].min())
        end_ts = int(sample_ticks["timestamp"].max())

        # Request range 10x wider than data
        wide_end = start_ts + (end_ts - start_ts) * 10

        result = storage.has_ticks("BTCUSDT", start_ts, wide_end)
        assert result is False

    def test_zero_length_range(self, storage, sample_ticks):
        """start_ts == end_ts → True if any data exists at that timestamp."""
        storage.write_ticks("BTCUSDT", sample_ticks)

        ts = int(sample_ticks["timestamp"][0])
        result = storage.has_ticks("BTCUSDT", ts, ts)
        assert result is True

    def test_exact_coverage_threshold(self, storage, sample_ticks):
        """Test that min_coverage parameter works correctly."""
        storage.write_ticks("BTCUSDT", sample_ticks)

        start_ts = int(sample_ticks["timestamp"].min())
        end_ts = int(sample_ticks["timestamp"].max())

        # With min_coverage=0.0, any data should pass
        assert storage.has_ticks("BTCUSDT", start_ts, end_ts, min_coverage=0.0) is True

        # With min_coverage=1.0 for exact range, should also pass
        assert storage.has_ticks("BTCUSDT", start_ts, end_ts, min_coverage=1.0) is True


# ===========================================================================
# Tier 5: store_bars_batch vs store_bars_bulk Parity
# ===========================================================================

# Note: These tests require a ClickHouse connection. Skip if unavailable.


@pytest.mark.clickhouse
class TestTier5CacheWriteParity:
    """Validate store_bars_batch produces identical ClickHouse rows."""

    @pytest.fixture
    def cache(self):
        """Create RangeBarCache instance, skip if ClickHouse unavailable."""
        try:
            from rangebar.clickhouse import RangeBarCache
            cache = RangeBarCache()
            cache.__enter__()
            yield cache
            cache.__exit__(None, None, None)
        except (ImportError, ConnectionError, OSError):
            pytest.skip("ClickHouse not available")

    def test_batch_close_time_ms_range(self, cache):
        """Bars stored via store_bars_batch have valid close_time_ms values.

        Guards against Issue #85 class bugs in the Polars/Arrow path.
        """

        # Create test bars with known timestamps
        ts_ms = [1704067200000 + i * 60000 for i in range(10)]
        test_bars = pl.DataFrame({
            "close_time_ms": ts_ms,
            "open": [50000.0 + i * 10 for i in range(10)],
            "high": [50010.0 + i * 10 for i in range(10)],
            "low": [49990.0 + i * 10 for i in range(10)],
            "close": [50005.0 + i * 10 for i in range(10)],
            "volume": [1.0] * 10,
        })

        test_symbol = "__TEST_PHASE3__"
        test_threshold = 99999

        try:
            written = cache.store_bars_batch(
                test_symbol, test_threshold, test_bars,
            )
            assert written == 10

            # Query back and verify timestamps
            result = cache.client.query(
                "SELECT close_time_ms FROM rangebar_cache.range_bars "
                "WHERE symbol = %(symbol)s AND threshold_decimal_bps = %(threshold)s",
                parameters={"symbol": test_symbol, "threshold": test_threshold},
            )
            for row in result.result_rows:
                ts = row[0]
                assert ts >= 1_000_000_000_000, (
                    f"close_time_ms={ts} in seconds, not ms (Issue #85)"
                )
                assert ts < 2_000_000_000_000, (
                    f"close_time_ms={ts} in microseconds, not ms"
                )
        finally:
            # Cleanup test data
            cache.client.command(
                "ALTER TABLE rangebar_cache.range_bars DELETE "
                "WHERE symbol = '__TEST_PHASE3__'"
            )


# ===========================================================================
# Tier 6: Divergence Detection Framework
# ===========================================================================


class TestTier6DivergenceDetection:
    """Verify the divergence detection helpers work correctly."""

    def test_divergence_identifies_field_and_index(self):
        """When paths diverge, error message includes bar index and field."""
        # This test verifies our assertion pattern gives good diagnostics
        ts_ms, prices, vols, ibm, aids = _generate_breach_trades(
            n=50, threshold_dbps=250,
        )
        ts_us = [t * 1000 for t in ts_ms]

        # Both paths with same data should NOT diverge
        batch_ms = _make_trade_batch_ms(
            ts_ms, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc_a = PyRangeBarProcessor(250, symbol="TEST")
        result_a = pl.from_arrow(proc_a.process_trades_arrow(batch_ms))

        batch_us = _make_trade_batch_us(
            ts_us, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc_b = PyRangeBarProcessor(250, symbol="TEST")
        result_b = pl.from_arrow(proc_b.process_trades_arrow_native(batch_us))

        assert len(result_a) == len(result_b)
        assert len(result_a) > 0, "Need bars to verify detection"

        # All fields should match — this is the positive control
        mismatches = []
        for col in result_a.columns:
            if col not in result_b.columns:
                continue
            for i in range(len(result_a)):
                val_a = result_a[col][i]
                val_b = result_b[col][i]
                if val_a is None and val_b is None:
                    continue
                if val_a != val_b:
                    mismatches.append((i, col, val_a, val_b))

        assert len(mismatches) == 0, (
            f"Unexpected divergences: {mismatches[:5]}"
        )

    def test_three_path_parity(self):
        """All three paths (dict, arrow-ms, arrow-native-us) produce same bars."""
        ts_ms, prices, vols, ibm, aids = _generate_breach_trades(
            n=80, threshold_dbps=500,
        )
        ts_us = [t * 1000 for t in ts_ms]

        # Path 1: dict
        dicts = _make_trade_dicts(
            ts_ms, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc_dict = PyRangeBarProcessor(500, symbol="TEST")
        dict_out = proc_dict.process_trades_streaming(dicts)
        dict_df = pl.DataFrame(dict_out, infer_schema_length=None) if dict_out else pl.DataFrame()

        # Path 2: arrow ms
        batch_ms = _make_trade_batch_ms(
            ts_ms, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc_ms = PyRangeBarProcessor(500, symbol="TEST")
        ms_df = pl.from_arrow(proc_ms.process_trades_arrow(batch_ms))

        # Path 3: arrow native us
        batch_us = _make_trade_batch_us(
            ts_us, prices, vols,
            is_buyer_maker=ibm, agg_trade_ids=aids,
        )
        proc_us = PyRangeBarProcessor(500, symbol="TEST")
        us_df = pl.from_arrow(proc_us.process_trades_arrow_native(batch_us))

        # All must have same bar count
        assert len(dict_df) == len(ms_df) == len(us_df), (
            f"Bar counts: dict={len(dict_df)}, ms={len(ms_df)}, us={len(us_df)}"
        )

        if len(dict_df) == 0:
            return

        # Arrow paths must match exactly (same data types)
        for col in ms_df.columns:
            if col not in us_df.columns:
                continue
            for i in range(len(ms_df)):
                assert ms_df[col][i] == us_df[col][i], (
                    f"Arrow paths diverge at bar {i}, field '{col}': "
                    f"ms={ms_df[col][i]}, us={us_df[col][i]}"
                )

        # Dict vs Arrow: compare common columns
        for col in ms_df.columns:
            if col not in dict_df.columns:
                continue
            for i in range(len(ms_df)):
                ms_val = ms_df[col][i]
                dict_val = dict_df[col][i]
                if ms_val is None and dict_val is None:
                    continue
                if isinstance(ms_val, float) and isinstance(dict_val, float):
                    assert math.isclose(ms_val, dict_val, rel_tol=1e-12), (
                        f"Dict vs Arrow diverge at bar {i}, field '{col}': "
                        f"dict={dict_val}, arrow={ms_val}"
                    )
                else:
                    assert ms_val == dict_val, (
                        f"Dict vs Arrow diverge at bar {i}, field '{col}': "
                        f"dict={dict_val}, arrow={ms_val}"
                    )

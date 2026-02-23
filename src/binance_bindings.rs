use super::*;
use chrono::NaiveDate;

/// Issue #96 Task #83: Error message constants to reduce rodata duplication
const ERR_INVALID_START_DATE: &str = "Invalid start_date format: ";
const ERR_INVALID_END_DATE: &str = "Invalid end_date format: ";
const ERR_DATE_ORDER: &str = "start_date must be <= end_date";

/// Binance market type enum
#[pyclass(name = "MarketType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyMarketType {
    /// Spot market (BTCUSDT on spot.binance.com)
    Spot = 0,
    /// USD-M Futures (perpetual, USDT-margined)
    FuturesUM = 1,
    /// COIN-M Futures (perpetual, coin-margined)
    FuturesCM = 2,
}

impl PyMarketType {
    const fn to_market_str(self) -> &'static str {
        match self {
            Self::Spot => "spot",
            Self::FuturesUM => "um",
            Self::FuturesCM => "cm",
        }
    }
}

/// Fetch Binance aggTrades data for a symbol and date range.
///
/// Downloads from data.binance.vision and returns as list of trade dicts.
/// This is an internal function - use `get_range_bars()` in Python for
/// the high-level API that handles caching and processing automatically.
///
/// Args:
///     symbol: Trading pair (e.g., "BTCUSDT")
///     start_date: Start date as "YYYY-MM-DD"
///     end_date: End date as "YYYY-MM-DD"
///     market_type: Market type (Spot, FuturesUM, FuturesCM)
///     verify_checksum: Verify SHA-256 checksum of downloaded data (Issue #43).
///         Default: True. Set to False for faster downloads when data integrity
///         is verified elsewhere.
///
/// Returns:
///     List of trade dicts with keys: timestamp, price, quantity, agg_trade_id,
///     first_trade_id, last_trade_id, is_buyer_maker
///
/// Raises:
///     RuntimeError: If checksum verification fails (data corruption detected)
#[pyfunction]
#[pyo3(signature = (symbol, start_date, end_date, market_type = PyMarketType::Spot, verify_checksum = true))]
pub fn fetch_binance_aggtrades(
    py: Python,
    symbol: &str,
    start_date: &str,
    end_date: &str,
    market_type: PyMarketType,
    verify_checksum: bool,
) -> PyResult<Vec<PyObject>> {
    // Parse dates
    let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
        .map_err(|e| PyValueError::new_err(format!("{}{e}", ERR_INVALID_START_DATE)))?;
    let end = NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
        .map_err(|e| PyValueError::new_err(format!("{}{e}", ERR_INVALID_END_DATE)))?;

    if start > end {
        return Err(PyValueError::new_err(ERR_DATE_ORDER));
    }

    // Create loader with market type
    let loader = HistoricalDataLoader::new_with_market(symbol, market_type.to_market_str());

    // Create tokio runtime and fetch data
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

    // Load each day in the range (with checksum verification per Issue #43)
    let mut all_trades = Vec::new();
    let mut current_date = start;

    while current_date <= end {
        let day_result = rt.block_on(
            loader.load_single_day_trades_with_checksum(current_date, verify_checksum),
        );
        match day_result {
            Ok(mut day_trades) => all_trades.append(&mut day_trades),
            Err(rangebar_providers::binance::HistoricalError::ChecksumMismatch {
                date,
                message,
            }) => {
                // Checksum mismatch is a hard error - data corruption detected
                return Err(PyRuntimeError::new_err(format!(
                    "Checksum verification failed for {date}: {message}"
                )));
            }
            Err(e) => {
                // Skip days with no data (weekends, holidays, etc.)
                eprintln!("Warning: No data for {current_date}: {e}");
            }
        }
        current_date += chrono::Duration::days(1);
    }

    if all_trades.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "No data available for {symbol} from {start_date} to {end_date}"
        )));
    }

    // Sort by timestamp
    all_trades.sort_by_key(|trade| trade.timestamp);

    // Convert AggTrades to Python dicts
    let mut results = Vec::with_capacity(all_trades.len());
    for trade in &all_trades {
        let dict = PyDict::new_bound(py);
        // Timestamp is already in microseconds from rangebar-core
        // Convert to milliseconds for Python API consistency
        dict.set_item("timestamp", trade.timestamp / 1000)?;
        dict.set_item("price", trade.price.to_f64())?;
        dict.set_item("quantity", trade.volume.to_f64())?;
        dict.set_item("agg_trade_id", trade.agg_trade_id)?;
        dict.set_item("first_trade_id", trade.first_trade_id)?;
        dict.set_item("last_trade_id", trade.last_trade_id)?;
        dict.set_item("is_buyer_maker", trade.is_buyer_maker)?;
        results.push(dict.into());
    }

    Ok(results)
}

/// Fetch recent aggregated trades from Binance REST API (Issue #92).
///
/// Paginates through `/api/v3/aggTrades` (max 1000 per request) to fill
/// the gap between cached historical data and the current time.
///
/// Args:
///     symbol: Trading symbol (e.g., "BTCUSDT")
///     start_time_ms: Start time in milliseconds (inclusive)
///     end_time_ms: End time in milliseconds (inclusive)
///
/// Returns:
///     List of trade dicts with keys: timestamp (ms), price, quantity,
///     agg_trade_id, first_trade_id, last_trade_id, is_buyer_maker
///
/// Raises:
///     RuntimeError: If REST API request fails or times out
#[pyfunction]
#[pyo3(signature = (symbol, start_time_ms, end_time_ms))]
pub fn fetch_aggtrades_rest(
    py: Python,
    symbol: &str,
    start_time_ms: i64,
    end_time_ms: i64,
) -> PyResult<Vec<PyObject>> {
    let loader = HistoricalDataLoader::new(symbol);

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

    let trades = rt
        .block_on(loader.fetch_aggtrades_rest(start_time_ms, end_time_ms))
        .map_err(|e| PyRuntimeError::new_err(format!("REST API error: {e}")))?;

    // Convert AggTrades to Python dicts (same format as fetch_binance_aggtrades)
    let mut results = Vec::with_capacity(trades.len());
    for trade in &trades {
        let dict = PyDict::new_bound(py);
        // Convert us → ms for Python API consistency
        dict.set_item("timestamp", trade.timestamp / 1000)?;
        dict.set_item("price", trade.price.to_f64())?;
        dict.set_item("quantity", trade.volume.to_f64())?;
        dict.set_item("agg_trade_id", trade.agg_trade_id)?;
        dict.set_item("first_trade_id", trade.first_trade_id)?;
        dict.set_item("last_trade_id", trade.last_trade_id)?;
        dict.set_item("is_buyer_maker", trade.is_buyer_maker)?;
        results.push(dict.into());
    }

    Ok(results)
}

// ========================================================================
// Streaming Trade Iterator (Phase 2: Memory-Efficient Architecture)
// ========================================================================

/// Iterator over Binance aggTrades in hour-based chunks.
///
/// Memory-efficient streaming that yields trade batches instead of loading
/// entire date ranges. Each iteration returns trades for a configurable
/// hour window (default: 6 hours).
///
/// # Memory Efficiency
///
/// | Chunk Size | Peak Memory | Reduction |
/// |------------|-------------|-----------|
/// | 24 hours   | ~213 MB     | 1x        |
/// | 6 hours    | ~46 MB      | 4.6x      |
/// | 1 hour     | ~15 MB      | 14x       |
///
/// # Checksum Verification (Issue #43)
///
/// By default, SHA-256 checksums are verified for each downloaded file.
/// Set `verify_checksum=False` to disable for faster downloads when
/// data integrity is verified elsewhere.
///
/// # Example
///
/// ```python
/// from rangebar._core import stream_binance_trades
///
/// for trades_batch in stream_binance_trades("BTCUSDT", "2024-01-01", "2024-01-07"):
///     # Each batch is ~46 MB (6 hours of trades)
///     print(f"Processing {len(trades_batch)} trades")
///     bars = processor.process_trades_streaming(trades_batch)
/// ```
#[pyclass(name = "BinanceTradeStream")]
pub struct PyBinanceTradeStream {
    inner: IntraDayChunkIterator,
    symbol: String,
}

#[pymethods]
impl PyBinanceTradeStream {
    /// Create a new trade stream.
    ///
    /// Args:
    ///     symbol: Trading pair (e.g., "BTCUSDT")
    ///     start_date: Start date as "YYYY-MM-DD"
    ///     end_date: End date as "YYYY-MM-DD"
    ///     chunk_hours: Hours per chunk (1, 6, 12, or 24). Default: 6.
    ///     market_type: Market type (Spot, FuturesUM, FuturesCM). Default: Spot.
    ///     verify_checksum: Verify SHA-256 checksum of downloaded data (Issue #43).
    ///         Default: True. Set to False for faster downloads when data integrity
    ///         is verified elsewhere.
    #[new]
    #[pyo3(signature = (symbol, start_date, end_date, chunk_hours = 6, market_type = PyMarketType::Spot, verify_checksum = true))]
    fn new(
        symbol: &str,
        start_date: &str,
        end_date: &str,
        chunk_hours: u32,
        market_type: PyMarketType,
        verify_checksum: bool,
    ) -> PyResult<Self> {
        // Parse dates
        let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
            .map_err(|e| PyValueError::new_err(format!("{}{e}", ERR_INVALID_START_DATE)))?;
        let end = NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
            .map_err(|e| PyValueError::new_err(format!("{}{e}", ERR_INVALID_END_DATE)))?;

        if start > end {
            return Err(PyValueError::new_err(ERR_DATE_ORDER));
        }

        if chunk_hours == 0 || chunk_hours > 24 {
            return Err(PyValueError::new_err("chunk_hours must be 1-24"));
        }

        let loader = HistoricalDataLoader::new_with_market(symbol, market_type.to_market_str());
        let inner = IntraDayChunkIterator::with_checksum(
            loader,
            start,
            end,
            chunk_hours,
            verify_checksum,
        );

        Ok(Self {
            inner,
            symbol: symbol.to_uppercase(),
        })
    }

    /// Get the symbol being streamed.
    #[getter]
    fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get the current date being processed.
    #[getter]
    fn current_date(&self) -> String {
        self.inner.current_date().format("%Y-%m-%d").to_string()
    }

    /// Get the current hour within the day.
    #[getter]
    fn current_hour(&self) -> u32 {
        self.inner.current_hour()
    }

    #[allow(clippy::missing_const_for_fn)]
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Get next chunk of trades.
    ///
    /// Returns:
    ///     List of trade dicts, or None if exhausted.
    ///
    /// Raises:
    ///     RuntimeError: If data fetching fails.
    fn __next__(&mut self, py: Python) -> PyResult<Option<Vec<PyObject>>> {
        match self.inner.next() {
            Some(Ok(trades)) => {
                // Convert AggTrades to Python dicts
                let mut results = Vec::with_capacity(trades.len());
                for trade in &trades {
                    let dict = PyDict::new_bound(py);
                    // Convert microseconds to milliseconds for Python API
                    dict.set_item("timestamp", trade.timestamp / 1000)?;
                    dict.set_item("price", trade.price.to_f64())?;
                    dict.set_item("quantity", trade.volume.to_f64())?;
                    dict.set_item("agg_trade_id", trade.agg_trade_id)?;
                    dict.set_item("first_trade_id", trade.first_trade_id)?;
                    dict.set_item("last_trade_id", trade.last_trade_id)?;
                    dict.set_item("is_buyer_maker", trade.is_buyer_maker)?;
                    results.push(dict.into());
                }
                Ok(Some(results))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(format!("Data fetch error: {e}"))),
            None => Ok(None),
        }
    }
}

/// Stream Binance aggTrades data in memory-efficient chunks.
///
/// Returns an iterator that yields trade batches instead of loading
/// the entire date range into memory. This is the recommended way to
/// process large date ranges.
///
/// Args:
///     symbol: Trading pair (e.g., "BTCUSDT")
///     start_date: Start date as "YYYY-MM-DD"
///     end_date: End date as "YYYY-MM-DD"
///     chunk_hours: Hours per chunk (1, 6, 12, or 24). Default: 6.
///     market_type: Market type (Spot, FuturesUM, FuturesCM). Default: Spot.
///     verify_checksum: Verify SHA-256 checksum of downloaded data (Issue #43).
///         Default: True. Set to False for faster downloads when data integrity
///         is verified elsewhere.
///
/// Returns:
///     Iterator yielding lists of trade dicts.
///
/// Example:
///     ```python
///     from rangebar._core import stream_binance_trades, PyRangeBarProcessor
///
///     processor = PyRangeBarProcessor(250, symbol="BTCUSDT")
///
///     for trades_batch in stream_binance_trades("BTCUSDT", "2024-01-01", "2024-01-31"):
///         bars = processor.process_trades_streaming(trades_batch)
///         # Process bars...
///
///     # Get final incomplete bar
///     final_bar = processor.get_incomplete_bar()
///     ```
#[pyfunction]
#[pyo3(signature = (symbol, start_date, end_date, chunk_hours = 6, market_type = PyMarketType::Spot, verify_checksum = true))]
pub fn stream_binance_trades(
    symbol: &str,
    start_date: &str,
    end_date: &str,
    chunk_hours: u32,
    market_type: PyMarketType,
    verify_checksum: bool,
) -> PyResult<PyBinanceTradeStream> {
    PyBinanceTradeStream::new(
        symbol,
        start_date,
        end_date,
        chunk_hours,
        market_type,
        verify_checksum,
    )
}

// ========================================================================
// Phase 3: Arrow-native stream (eliminates Rust→Python→Rust round-trip)
// Requires both data-providers and arrow-export features
// ========================================================================

/// Stream Binance trades as Arrow RecordBatches (zero-copy, no Python dicts).
///
/// Unlike `BinanceTradeStream` which yields `List[Dict]` (7 PyDict.set_item
/// calls per trade through GIL), this yields `PyRecordBatch` directly from
/// Rust `Vec<AggTrade>`. Timestamps are in **microseconds** (internal format).
///
/// Example:
///     ```python
///     from rangebar._core import stream_binance_trades_arrow
///
///     processor = PyRangeBarProcessor(250, symbol="BTCUSDT")
///     for trade_batch in stream_binance_trades_arrow("BTCUSDT", "2024-01-01", "2024-01-07"):
///         # trade_batch is a PyRecordBatch with μs timestamps
///         bars = processor.process_trades_arrow_native(trade_batch)
///     ```
#[cfg(feature = "arrow-export")]
#[pyclass(name = "BinanceTradeStreamArrow")]
pub struct PyBinanceTradeStreamArrow {
    inner: IntraDayChunkIterator,
    symbol: String,
}

#[cfg(feature = "arrow-export")]
#[pymethods]
impl PyBinanceTradeStreamArrow {
    #[new]
    #[pyo3(signature = (symbol, start_date, end_date, chunk_hours = 6, market_type = PyMarketType::Spot, verify_checksum = true))]
    fn new(
        symbol: &str,
        start_date: &str,
        end_date: &str,
        chunk_hours: u32,
        market_type: PyMarketType,
        verify_checksum: bool,
    ) -> PyResult<Self> {
        let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
            .map_err(|e| PyValueError::new_err(format!("{}{e}", ERR_INVALID_START_DATE)))?;
        let end = NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
            .map_err(|e| PyValueError::new_err(format!("{}{e}", ERR_INVALID_END_DATE)))?;

        if start > end {
            return Err(PyValueError::new_err(ERR_DATE_ORDER));
        }

        if chunk_hours == 0 || chunk_hours > 24 {
            return Err(PyValueError::new_err("chunk_hours must be 1-24"));
        }

        let loader = HistoricalDataLoader::new_with_market(symbol, market_type.to_market_str());
        let inner = IntraDayChunkIterator::with_checksum(
            loader,
            start,
            end,
            chunk_hours,
            verify_checksum,
        );

        Ok(Self {
            inner,
            symbol: symbol.to_uppercase(),
        })
    }

    #[getter]
    fn symbol(&self) -> &str {
        &self.symbol
    }

    #[getter]
    fn current_date(&self) -> String {
        self.inner.current_date().format("%Y-%m-%d").to_string()
    }

    #[getter]
    fn current_hour(&self) -> u32 {
        self.inner.current_hour()
    }

    #[allow(clippy::missing_const_for_fn)]
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Get next chunk of trades as an Arrow RecordBatch.
    ///
    /// Returns:
    ///     PyRecordBatch with microsecond timestamps, or None if exhausted.
    ///
    /// Raises:
    ///     RuntimeError: If data fetching fails.
    fn __next__(&mut self) -> PyResult<Option<PyRecordBatch>> {
        match self.inner.next() {
            Some(Ok(trades)) => {
                // Convert Vec<AggTrade> directly to Arrow RecordBatch
                // Timestamps are ALREADY in microseconds (normalized during CSV parse)
                let batch = aggtrades_to_record_batch(&trades);
                Ok(Some(PyRecordBatch::new(batch)))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(format!("Data fetch error: {e}"))),
            None => Ok(None),
        }
    }
}

/// Stream Binance aggTrades as Arrow RecordBatches (Phase 3: zero-copy).
///
/// This is the Arrow-native counterpart of `stream_binance_trades()`.
/// Instead of yielding `List[Dict]`, yields `PyRecordBatch` with microsecond
/// timestamps. Use with `process_trades_arrow_native()` for the full
/// zero-copy pipeline.
///
/// Args:
///     symbol: Trading pair (e.g., "BTCUSDT")
///     start_date: Start date as "YYYY-MM-DD"
///     end_date: End date as "YYYY-MM-DD"
///     chunk_hours: Hours per chunk (1, 6, 12, or 24). Default: 6.
///     market_type: Market type (Spot, FuturesUM, FuturesCM). Default: Spot.
///     verify_checksum: Verify SHA-256 checksum. Default: True.
///
/// Returns:
///     Iterator yielding Arrow RecordBatches.
#[cfg(feature = "arrow-export")]
#[pyfunction]
#[pyo3(signature = (symbol, start_date, end_date, chunk_hours = 6, market_type = PyMarketType::Spot, verify_checksum = true))]
pub fn stream_binance_trades_arrow(
    symbol: &str,
    start_date: &str,
    end_date: &str,
    chunk_hours: u32,
    market_type: PyMarketType,
    verify_checksum: bool,
) -> PyResult<PyBinanceTradeStreamArrow> {
    PyBinanceTradeStreamArrow::new(
        symbol,
        start_date,
        end_date,
        chunk_hours,
        market_type,
        verify_checksum,
    )
}

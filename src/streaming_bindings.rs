use super::*;
use pyo3::types::PyList;
use rangebar_providers::binance::websocket::{
    BinanceWebSocketStream, ReconnectionPolicy, WebSocketError,
};
use rangebar_streaming::processor::{MetricsSummary, StreamingProcessorConfig};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use parking_lot::Mutex;

/// Streaming metrics snapshot for Python
#[pyclass(name = "StreamingMetrics")]
#[derive(Clone)]
pub struct PyStreamingMetrics {
    summary: MetricsSummary,
}

#[pymethods]
impl PyStreamingMetrics {
    /// Number of trades processed
    #[getter]
    const fn trades_processed(&self) -> u64 {
        self.summary.trades_processed
    }

    /// Number of bars generated
    #[getter]
    const fn bars_generated(&self) -> u64 {
        self.summary.bars_generated
    }

    /// Total errors encountered
    #[getter]
    const fn errors_total(&self) -> u64 {
        self.summary.errors_total
    }

    /// Number of backpressure events
    #[getter]
    const fn backpressure_events(&self) -> u64 {
        self.summary.backpressure_events
    }

    /// Memory usage in bytes
    #[getter]
    const fn memory_usage_bytes(&self) -> u64 {
        self.summary.memory_usage_bytes
    }

    /// Bars per aggTrade ratio
    fn bars_per_aggtrade(&self) -> f64 {
        self.summary.bars_per_aggtrade()
    }

    /// Error rate (errors / trades)
    fn error_rate(&self) -> f64 {
        self.summary.error_rate()
    }

    /// Memory usage in MB
    fn memory_usage_mb(&self) -> f64 {
        self.summary.memory_usage_mb()
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingMetrics(trades={}, bars={}, errors={}, backpressure={})",
            self.summary.trades_processed,
            self.summary.bars_generated,
            self.summary.errors_total,
            self.summary.backpressure_events
        )
    }
}

/// Python wrapper for Binance WebSocket live stream
///
/// Connects to Binance WebSocket and processes trades into range bars in real-time.
///
/// Example:
///     >>> stream = PyBinanceLiveStream("BTCUSDT", 250)
///     >>> stream.connect()
///     >>> while stream.is_connected():
///     ...     bar = stream.next_bar(timeout_ms=5000)
///     ...     if bar:
///     ...         print(f"New bar: {bar['close']}")
#[pyclass(name = "BinanceLiveStream")]
pub struct PyBinanceLiveStream {
    symbol: String,
    threshold_decimal_bps: u32,
    /// Bars received from streaming processor
    bars: Arc<Mutex<Vec<rangebar_core::RangeBar>>>,
    /// Flag to track connection state
    connected: Arc<std::sync::atomic::AtomicBool>,
    /// Tokio runtime handle for async operations
    runtime: Option<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyBinanceLiveStream {
    /// Create a new Binance live stream
    ///
    /// Args:
    ///     symbol: Trading pair (e.g., "BTCUSDT")
    ///     threshold_decimal_bps: Range bar threshold in decimal basis points (250 = 0.25%)
    #[new]
    #[pyo3(signature = (symbol, threshold_decimal_bps))]
    pub fn new(symbol: &str, threshold_decimal_bps: u32) -> PyResult<Self> {
        // Validate threshold
        if threshold_decimal_bps == 0 || threshold_decimal_bps > 100_000 {
            return Err(PyValueError::new_err(format!(
                "threshold_decimal_bps must be between 1 and 100000, got {threshold_decimal_bps}"
            )));
        }

        Ok(Self {
            symbol: symbol.to_uppercase(),
            threshold_decimal_bps,
            bars: Arc::new(Mutex::new(Vec::new())),
            connected: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            runtime: None,
        })
    }

    /// Connect to Binance WebSocket and start processing
    ///
    /// This is a blocking call that runs the WebSocket connection in a background thread.
    /// Use `next_bar()` to retrieve completed bars.
    pub fn connect(&mut self, py: Python) -> PyResult<()> {
        if self.connected.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("Already connected"));
        }

        let symbol = self.symbol.clone();
        let threshold = self.threshold_decimal_bps;
        let bars = Arc::clone(&self.bars);
        let connected = Arc::clone(&self.connected);

        // Create tokio runtime for async operations
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

        let handle = runtime.handle().clone();

        // Spawn the WebSocket connection in the runtime
        handle.spawn(async move {
            match Self::run_stream(symbol, threshold, bars, connected).await {
                Ok(()) => println!("Stream ended normally"),
                Err(e) => println!("Stream error: {e:?}"),
            }
        });

        self.runtime = Some(runtime);
        self.connected.store(true, Ordering::Relaxed);

        // Allow threads to release the GIL during long operations
        py.allow_threads(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
        });

        Ok(())
    }

    /// Get the next completed bar (blocking with timeout)
    ///
    /// Args:
    ///     timeout_ms: Maximum time to wait in milliseconds
    ///
    /// Returns:
    ///     Bar dict or None if timeout
    #[pyo3(signature = (timeout_ms = 5000))]
    pub fn next_bar(&self, py: Python, timeout_ms: u64) -> PyResult<Option<PyObject>> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_millis(timeout_ms);

        // Poll for bars with GIL release
        loop {
            // Check for available bars
            {
                let mut bars = self.bars.lock();

                if !bars.is_empty() {
                    let bar = bars.remove(0);
                    return Ok(Some(rangebar_to_dict(py, &bar)?));
                }
            }

            // Check timeout
            if start.elapsed() >= timeout {
                return Ok(None);
            }

            // Release GIL while sleeping
            py.allow_threads(|| {
                std::thread::sleep(std::time::Duration::from_millis(10));
            });

            // Check if still connected
            if !self.connected.load(Ordering::Relaxed) {
                return Ok(None);
            }
        }
    }

    /// Get all pending bars (non-blocking)
    ///
    /// Returns:
    ///     List of bar dicts
    pub fn get_pending_bars(&self, py: Python) -> PyResult<PyObject> {
        let mut bars = self
            .bars
            .lock();

        let result = PyList::empty_bound(py);
        for bar in bars.drain(..) {
            result.append(rangebar_to_dict(py, &bar)?)?;
        }

        Ok(result.into())
    }

    /// Check if the stream is connected
    #[getter]
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Get the symbol this stream is connected to
    #[getter]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get the threshold in decimal basis points
    #[getter]
    pub const fn threshold_decimal_bps(&self) -> u32 {
        self.threshold_decimal_bps
    }

    /// Disconnect from the WebSocket
    pub fn disconnect(&mut self) -> PyResult<()> {
        self.connected.store(false, Ordering::Relaxed);

        // Drop the runtime to stop all tasks
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "BinanceLiveStream(symbol={}, threshold_bps={}, connected={})",
            self.symbol,
            self.threshold_decimal_bps,
            self.is_connected()
        )
    }
}

impl PyBinanceLiveStream {
    async fn run_stream(
        symbol: String,
        threshold_decimal_bps: u32,
        bars: Arc<Mutex<Vec<rangebar_core::RangeBar>>>,
        connected: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), WebSocketError> {
        // Create processor using ExportRangeBarProcessor for streaming
        // NOTE: PyBinanceLiveStream uses ExportRangeBarProcessor (10 columns).
        // For full 58-column bars, use PyLiveBarEngine instead (Issue #91).
        let mut processor =
            rangebar_core::processor::ExportRangeBarProcessor::new(threshold_decimal_bps)
                .map_err(|e| {
                    WebSocketError::InvalidSymbol(format!("Processor creation failed: {e}"))
                })?;

        // Create trade channel for WebSocket → processor pipeline
        let (trade_tx, mut trade_rx) = tokio::sync::mpsc::channel(1000);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let shutdown_clone = shutdown.clone();

        // Spawn reconnecting WebSocket in background
        let sym = symbol.clone();
        tokio::spawn(async move {
            BinanceWebSocketStream::run_with_reconnect(
                &sym,
                trade_tx,
                ReconnectionPolicy::default(),
                shutdown_clone,
            )
            .await;
        });

        // Process trades from channel
        while connected.load(Ordering::Relaxed) {
            match trade_rx.recv().await {
                Some(trade) => {
                    processor.process_trades_continuously(&[trade]);
                    let completed = processor.get_all_completed_bars();
                    if !completed.is_empty() {
                        let mut bar_buffer = bars.lock();
                        bar_buffer.extend(completed);
                    }
                }
                None => {
                    // WebSocket channel closed (all senders dropped)
                    break;
                }
            }
        }

        shutdown.cancel();
        connected.store(false, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for PyBinanceLiveStream {
    fn drop(&mut self) {
        self.connected.store(false, Ordering::Relaxed);
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// Configuration for streaming processor
#[pyclass(name = "StreamingConfig")]
#[derive(Clone)]
pub struct PyStreamingConfig {
    /// Channel capacity for trade input (default: 5000)
    #[pyo3(get, set)]
    pub trade_channel_capacity: usize,
    /// Channel capacity for completed bars (default: 100)
    #[pyo3(get, set)]
    pub bar_channel_capacity: usize,
    /// Memory usage threshold in bytes (default: 100MB)
    #[pyo3(get, set)]
    pub memory_threshold_bytes: usize,
    /// Backpressure timeout in milliseconds (default: 100)
    #[pyo3(get, set)]
    pub backpressure_timeout_ms: u64,
    /// Circuit breaker error rate threshold 0.0-1.0 (default: 0.5)
    #[pyo3(get, set)]
    pub circuit_breaker_threshold: f64,
    /// Circuit breaker timeout in seconds (default: 30)
    #[pyo3(get, set)]
    pub circuit_breaker_timeout_secs: u64,
}

#[pymethods]
impl PyStreamingConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingConfig(trade_cap={}, bar_cap={}, mem_threshold={}MB)",
            self.trade_channel_capacity,
            self.bar_channel_capacity,
            self.memory_threshold_bytes / 1_000_000
        )
    }
}

impl Default for PyStreamingConfig {
    fn default() -> Self {
        let rust_config = StreamingProcessorConfig::default();
        Self {
            trade_channel_capacity: rust_config.trade_channel_capacity,
            bar_channel_capacity: rust_config.bar_channel_capacity,
            memory_threshold_bytes: rust_config.memory_threshold_bytes,
            backpressure_timeout_ms: rust_config.backpressure_timeout.as_millis() as u64,
            circuit_breaker_threshold: rust_config.circuit_breaker_threshold,
            circuit_breaker_timeout_secs: rust_config.circuit_breaker_timeout.as_secs(),
        }
    }
}

impl From<PyStreamingConfig> for StreamingProcessorConfig {
    fn from(config: PyStreamingConfig) -> Self {
        Self {
            trade_channel_capacity: config.trade_channel_capacity,
            bar_channel_capacity: config.bar_channel_capacity,
            memory_threshold_bytes: config.memory_threshold_bytes,
            backpressure_timeout: std::time::Duration::from_millis(
                config.backpressure_timeout_ms,
            ),
            circuit_breaker_threshold: config.circuit_breaker_threshold,
            circuit_breaker_timeout: std::time::Duration::from_secs(
                config.circuit_breaker_timeout_secs,
            ),
        }
    }
}

/// Streaming-oriented range bar processor for custom data sources
///
/// Unlike `PyRangeBarProcessor` which is optimized for batch processing,
/// this processor is designed for real-time streaming where trades arrive
/// one at a time from any data source (WebSocket, message queue, etc.).
///
/// Key features:
/// - Immediate bar extraction (no accumulation)
/// - Memory-bounded operation
/// - State preservation across calls
///
/// Example:
///     >>> processor = StreamingRangeBarProcessor(250)  # 0.25% threshold
///     >>> for trade in live_trade_stream():
///     ...     bars = processor.process_trade(trade)
///     ...     for bar in bars:
///     ...         print(f"Completed bar: {bar['close']}")
#[pyclass(name = "StreamingRangeBarProcessor")]
pub struct PyStreamingRangeBarProcessor {
    processor: rangebar_core::processor::ExportRangeBarProcessor,
    threshold_decimal_bps: u32,
    trades_processed: u64,
    bars_generated: u64,
}

#[pymethods]
impl PyStreamingRangeBarProcessor {
    /// Create a new streaming processor
    ///
    /// Args:
    ///     threshold_decimal_bps: Threshold in decimal basis points (250 = 0.25%)
    #[new]
    #[pyo3(signature = (threshold_decimal_bps))]
    pub fn new(threshold_decimal_bps: u32) -> PyResult<Self> {
        let processor =
            rangebar_core::processor::ExportRangeBarProcessor::new(threshold_decimal_bps)
                .map_err(|e| PyValueError::new_err(format!("Failed to create processor: {e}")))?;

        Ok(Self {
            processor,
            threshold_decimal_bps,
            trades_processed: 0,
            bars_generated: 0,
        })
    }

    /// Process a single trade and return any completed bars
    ///
    /// Args:
    ///     trade: Trade dict with timestamp, price, quantity/volume, is_buyer_maker
    ///
    /// Returns:
    ///     List of completed bar dicts (usually 0 or 1, rarely more)
    pub fn process_trade(&mut self, py: Python, trade: &Bound<PyDict>) -> PyResult<PyObject> {
        let agg_trade = dict_to_agg_trade(py, trade, 0)?;

        self.processor.process_trades_continuously(&[agg_trade]);
        self.trades_processed += 1;

        // Extract completed bars immediately
        let completed = self.processor.get_all_completed_bars();
        self.bars_generated += completed.len() as u64;

        let result = PyList::empty_bound(py);
        for bar in completed {
            result.append(rangebar_to_dict(py, &bar)?)?;
        }

        Ok(result.into())
    }

    /// Process multiple trades and return any completed bars
    ///
    /// Args:
    ///     trades: List of trade dicts
    ///
    /// Returns:
    ///     List of completed bar dicts
    pub fn process_trades(&mut self, py: Python, trades: &Bound<PyList>) -> PyResult<PyObject> {
        let mut agg_trades = Vec::with_capacity(trades.len());
        for (i, trade) in trades.iter().enumerate() {
            let trade_dict: &Bound<PyDict> = trade.downcast()?;
            agg_trades.push(dict_to_agg_trade(py, trade_dict, i)?);
        }

        self.processor.process_trades_continuously(&agg_trades);
        self.trades_processed += agg_trades.len() as u64;

        // Extract completed bars immediately
        let completed = self.processor.get_all_completed_bars();
        self.bars_generated += completed.len() as u64;

        let result = PyList::empty_bound(py);
        for bar in completed {
            result.append(rangebar_to_dict(py, &bar)?)?;
        }

        Ok(result.into())
    }

    /// Get the current incomplete bar (if any)
    ///
    /// Returns:
    ///     Bar dict or None
    pub fn get_incomplete_bar(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        match self.processor.get_incomplete_bar() {
            Some(bar) => Ok(Some(rangebar_to_dict(py, &bar)?)),
            None => Ok(None),
        }
    }

    /// Get streaming metrics
    ///
    /// Returns:
    ///     StreamingMetrics object
    pub const fn get_metrics(&self) -> PyStreamingMetrics {
        PyStreamingMetrics {
            summary: MetricsSummary {
                trades_processed: self.trades_processed,
                bars_generated: self.bars_generated,
                errors_total: 0,
                backpressure_events: 0,
                circuit_breaker_trips: 0,
                memory_usage_bytes: 0,
            },
        }
    }

    /// Get the threshold in decimal basis points
    #[getter]
    pub const fn threshold_decimal_bps(&self) -> u32 {
        self.threshold_decimal_bps
    }

    /// Get number of trades processed
    #[getter]
    pub const fn trades_processed(&self) -> u64 {
        self.trades_processed
    }

    /// Get number of bars generated
    #[getter]
    pub const fn bars_generated(&self) -> u64 {
        self.bars_generated
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingRangeBarProcessor(threshold_bps={}, trades={}, bars={})",
            self.threshold_decimal_bps, self.trades_processed, self.bars_generated
        )
    }
}

// =========================================================================
// Issue #91: LiveBarEngine — full 58-column streaming with canonical processor
// =========================================================================

/// Live bar engine for real-time range bar construction with full microstructure.
///
/// Unlike `BinanceLiveStream` (10 columns via `ExportRangeBarProcessor`), this
/// engine uses the canonical `RangeBarProcessor` with 3-step feature finalization
/// to produce all 58 columns including inter-bar and intra-bar features.
///
/// Architecture: Rust handles WebSocket→trade→bar (no GIL per trade).
/// Python only crosses the boundary per completed bar (~1-6/sec).
///
/// Example:
///     >>> engine = LiveBarEngine(["BTCUSDT", "ETHUSDT"], [250, 500])
///     >>> engine.start()
///     >>> while True:
///     ...     bar = engine.next_bar(timeout_ms=5000)
///     ...     if bar:
///     ...         print(f"{bar['_symbol']} @ {bar['_threshold']}: close={bar['close']}")
#[pyclass(name = "LiveBarEngine")]
pub struct PyLiveBarEngine {
    engine: Option<rangebar_streaming::LiveBarEngine>,
    runtime: Option<tokio::runtime::Runtime>,
    started: bool,
}

#[pymethods]
impl PyLiveBarEngine {
    /// Create a new live bar engine.
    ///
    /// Args:
    ///     symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
    ///     thresholds: List of thresholds in decimal basis points (e.g., [250, 500])
    ///     include_microstructure: Whether to compute all 58 columns (default: True)
    #[new]
    #[pyo3(signature = (symbols, thresholds, include_microstructure=true))]
    pub fn new(
        symbols: Vec<String>,
        thresholds: Vec<u32>,
        include_microstructure: bool,
    ) -> PyResult<Self> {
        if symbols.is_empty() {
            return Err(PyValueError::new_err("symbols must not be empty"));
        }
        if thresholds.is_empty() {
            return Err(PyValueError::new_err("thresholds must not be empty"));
        }
        for &t in &thresholds {
            if t == 0 || t > 100_000 {
                return Err(PyValueError::new_err(format!(
                    "threshold must be between 1 and 100000, got {t}"
                )));
            }
        }

        let mut config = rangebar_streaming::LiveEngineConfig::new(symbols, thresholds);
        config.include_microstructure = include_microstructure;

        let engine = rangebar_streaming::LiveBarEngine::new(config);

        Ok(Self {
            engine: Some(engine),
            runtime: None,
            started: false,
        })
    }

    /// Start all WebSocket connections and processing loops.
    ///
    /// This creates a tokio runtime and spawns one task per symbol.
    /// Non-blocking — use `next_bar()` to consume completed bars.
    pub fn start(&mut self, py: Python) -> PyResult<()> {
        if self.started {
            return Ok(());
        }

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

        // Enter runtime context so engine.start() can spawn tasks
        let _guard = runtime.enter();

        if let Some(ref mut engine) = self.engine {
            engine
                .start()
                .map_err(|e| PyValueError::new_err(format!("Failed to start engine: {e}")))?;
        }

        self.runtime = Some(runtime);
        self.started = true;

        // Brief GIL release to let WS connections initialize
        py.allow_threads(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
        });

        Ok(())
    }

    /// Get the next completed bar. Blocks until available or timeout.
    ///
    /// Returns a dict with all columns (up to 58 with microstructure),
    /// plus `_symbol` and `_threshold` metadata keys.
    /// Returns None on timeout.
    ///
    /// Args:
    ///     timeout_ms: Maximum wait time in milliseconds (default: 5000)
    #[pyo3(signature = (timeout_ms=5000))]
    pub fn next_bar(&mut self, py: Python, timeout_ms: u64) -> PyResult<Option<PyObject>> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Engine not started. Call start() first.")
        })?;

        let engine = self.engine.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("Engine not available")
        })?;

        let timeout = std::time::Duration::from_millis(timeout_ms);

        // Release GIL while waiting for bars from Rust
        let completed = py.allow_threads(|| {
            runtime.block_on(engine.next_bar(tokio::time::Duration::from_millis(
                timeout.as_millis() as u64,
            )))
        });

        match completed {
            Some(completed_bar) => {
                let dict = rangebar_to_dict(py, &completed_bar.bar)?;
                // Add metadata keys
                let py_dict: &Bound<PyDict> = dict.downcast_bound(py)?;
                py_dict.set_item("_symbol", &completed_bar.symbol)?;
                py_dict.set_item("_threshold", completed_bar.threshold_decimal_bps)?;
                Ok(Some(dict))
            }
            None => Ok(None),
        }
    }

    /// Inject a checkpoint for a specific (symbol, threshold) pair.
    /// Must be called before `start()`.
    ///
    /// Args:
    ///     symbol: Trading pair (e.g., "BTCUSDT")
    ///     threshold: Threshold in decimal basis points (e.g., 250)
    ///     checkpoint_dict: Checkpoint dict from `RangeBarProcessor.create_checkpoint()`
    #[pyo3(signature = (symbol, threshold, checkpoint_dict))]
    pub fn set_checkpoint(
        &mut self,
        py: Python,
        symbol: &str,
        threshold: u32,
        checkpoint_dict: &Bound<PyDict>,
    ) -> PyResult<()> {
        if self.started {
            return Err(PyRuntimeError::new_err(
                "Cannot set checkpoint after engine has started",
            ));
        }
        let checkpoint = dict_to_checkpoint(py, checkpoint_dict)?;
        if let Some(ref mut engine) = self.engine {
            engine.set_initial_checkpoint(symbol, threshold, checkpoint);
        }
        Ok(())
    }

    /// Collect processor checkpoints after shutdown.
    ///
    /// Call after `stop()`. Returns a dict mapping "SYMBOL:THRESHOLD" keys
    /// to checkpoint dicts that can be saved and restored on next startup.
    ///
    /// Args:
    ///     timeout_ms: Maximum wait time in milliseconds (default: 5000)
    ///
    /// Returns:
    ///     dict of {str: dict} — checkpoint dicts keyed by "SYMBOL:THRESHOLD"
    #[pyo3(signature = (timeout_ms=5000))]
    pub fn collect_checkpoints(&mut self, py: Python, timeout_ms: u64) -> PyResult<PyObject> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Engine runtime not available")
        })?;

        let engine = self.engine.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("Engine not available")
        })?;

        let timeout = tokio::time::Duration::from_millis(timeout_ms);

        // Release GIL while waiting for checkpoints
        let checkpoints = py.allow_threads(|| {
            runtime.block_on(engine.collect_checkpoints(timeout))
        });

        // Convert to Python dict
        let result = PyDict::new_bound(py);
        for (key, cp) in &checkpoints {
            let cp_dict = checkpoint_to_dict(py, cp)?;
            result.set_item(key, cp_dict)?;
        }

        Ok(result.into())
    }

    /// Graceful shutdown — cancels all WebSocket connections.
    pub fn stop(&mut self) -> PyResult<()> {
        if let Some(ref engine) = self.engine {
            engine.stop();
        }
        self.started = false;
        // Note: don't take() runtime here — collect_checkpoints() may still need it
        Ok(())
    }

    /// Fully release engine resources. Call after collect_checkpoints().
    pub fn shutdown(&mut self) -> PyResult<()> {
        if let Some(ref engine) = self.engine {
            engine.stop();
        }
        self.started = false;
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
        Ok(())
    }

    /// Get engine metrics.
    ///
    /// Returns:
    ///     dict with trades_received, bars_emitted, reconnections
    pub fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        let engine = self.engine.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Engine not available")
        })?;
        let snap = engine.metrics().snapshot();
        let dict = PyDict::new_bound(py);
        dict.set_item("trades_received", snap.trades_received)?;
        dict.set_item("bars_emitted", snap.bars_emitted)?;
        dict.set_item("reconnections", snap.reconnections)?;
        Ok(dict.into())
    }

    /// Whether the engine has been started.
    #[getter]
    pub const fn is_started(&self) -> bool {
        self.started
    }

    fn __repr__(&self) -> String {
        format!("LiveBarEngine(started={})", self.started)
    }
}

impl Drop for PyLiveBarEngine {
    fn drop(&mut self) {
        if let Some(ref engine) = self.engine {
            engine.stop();
        }
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

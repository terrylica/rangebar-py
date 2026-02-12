use super::*;

/// Python-exposed ExnessInstrument enum
///
/// Supported forex instruments from Exness Raw_Spread data.
#[pyclass(name = "ExnessInstrument", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyExnessInstrument {
    EURUSD = 0,
    GBPUSD = 1,
    USDJPY = 2,
    AUDUSD = 3,
    USDCAD = 4,
    NZDUSD = 5,
    EURGBP = 6,
    EURJPY = 7,
    GBPJPY = 8,
    XAUUSD = 9,
}

#[pymethods]
impl PyExnessInstrument {
    /// Get instrument symbol string
    #[getter]
    fn symbol(&self) -> &'static str {
        self.to_rust().symbol()
    }

    /// Get Raw_Spread symbol string (e.g., "EURUSD_Raw_Spread")
    #[getter]
    fn raw_spread_symbol(&self) -> String {
        self.to_rust().raw_spread_symbol()
    }

    /// Check if this is a JPY pair (different pip value)
    #[getter]
    fn is_jpy_pair(&self) -> bool {
        self.to_rust().is_jpy_pair()
    }

    /// Get all supported instruments
    #[staticmethod]
    fn all() -> Vec<PyExnessInstrument> {
        vec![
            Self::EURUSD,
            Self::GBPUSD,
            Self::USDJPY,
            Self::AUDUSD,
            Self::USDCAD,
            Self::NZDUSD,
            Self::EURGBP,
            Self::EURJPY,
            Self::GBPJPY,
            Self::XAUUSD,
        ]
    }
}

impl PyExnessInstrument {
    /// Convert to Rust ExnessInstrument
    const fn to_rust(self) -> ExnessInstrument {
        match self {
            Self::EURUSD => ExnessInstrument::EURUSD,
            Self::GBPUSD => ExnessInstrument::GBPUSD,
            Self::USDJPY => ExnessInstrument::USDJPY,
            Self::AUDUSD => ExnessInstrument::AUDUSD,
            Self::USDCAD => ExnessInstrument::USDCAD,
            Self::NZDUSD => ExnessInstrument::NZDUSD,
            Self::EURGBP => ExnessInstrument::EURGBP,
            Self::EURJPY => ExnessInstrument::EURJPY,
            Self::GBPJPY => ExnessInstrument::GBPJPY,
            Self::XAUUSD => ExnessInstrument::XAUUSD,
        }
    }
}

/// Python-exposed ValidationStrictness enum
#[pyclass(name = "ValidationStrictness", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyValidationStrictness {
    /// Basic checks only (bid > 0, ask > 0, bid < ask)
    Permissive = 0,
    /// + Spread < 10% (catches obvious errors) [DEFAULT]
    Strict = 1,
    /// + Spread < 1% (flags suspicious data)
    Paranoid = 2,
}

impl PyValidationStrictness {
    const fn to_rust(self) -> ValidationStrictness {
        match self {
            Self::Permissive => ValidationStrictness::Permissive,
            Self::Strict => ValidationStrictness::Strict,
            Self::Paranoid => ValidationStrictness::Paranoid,
        }
    }
}

/// Convert SpreadStats to Python dict
fn spread_stats_to_dict(py: Python, stats: &SpreadStats) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("min_spread", stats.min_spread.to_f64())?;
    dict.set_item("max_spread", stats.max_spread.to_f64())?;
    dict.set_item("avg_spread", stats.avg_spread().to_f64())?;
    dict.set_item("tick_count", stats.tick_count)?;
    Ok(dict.into())
}

/// Convert ExnessRangeBar to Python dict
fn exness_rangebar_to_dict(py: Python, bar: &ExnessRangeBar) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);

    // Convert timestamp from microseconds to RFC3339 string
    let timestamp_seconds = bar.base.close_time as f64 / 1_000_000.0;
    let datetime = chrono::DateTime::from_timestamp(
        timestamp_seconds as i64,
        (timestamp_seconds.fract() * 1_000_000_000.0) as u32,
    )
    .ok_or_else(|| PyValueError::new_err("Invalid timestamp"))?;

    dict.set_item("timestamp", datetime.to_rfc3339())?;

    // OHLCV from base bar
    dict.set_item("open", bar.base.open.to_f64())?;
    dict.set_item("high", bar.base.high.to_f64())?;
    dict.set_item("low", bar.base.low.to_f64())?;
    dict.set_item("close", bar.base.close.to_f64())?;
    dict.set_item("volume", bar.base.volume as f64 / 100_000_000.0)?; // Issue #88: i128; Always 0 for Exness

    // Spread statistics
    dict.set_item("spread_stats", spread_stats_to_dict(py, &bar.spread_stats)?)?;

    // Market microstructure (mostly zeros for Exness)
    dict.set_item("vwap", bar.base.vwap.to_f64())?;
    dict.set_item("tick_count", bar.spread_stats.tick_count)?;

    Ok(dict.into())
}

/// Python-exposed ExnessRangeBarBuilder
#[pyclass(name = "ExnessRangeBarBuilder")]
pub struct PyExnessRangeBarBuilder {
    builder: ExnessRangeBarBuilder,
    threshold_decimal_bps: u32,
    instrument: PyExnessInstrument,
}

#[pymethods]
impl PyExnessRangeBarBuilder {
    /// Create new builder for instrument
    ///
    /// Args:
    ///     instrument: ExnessInstrument enum value
    ///     threshold_decimal_bps: Threshold in decimal basis points (250 = 25bps = 0.25%)
    ///     strictness: ValidationStrictness enum (default: Strict)
    #[new]
    #[pyo3(signature = (instrument, threshold_decimal_bps, strictness = PyValidationStrictness::Strict))]
    fn new(
        instrument: PyExnessInstrument,
        threshold_decimal_bps: u32,
        strictness: PyValidationStrictness,
    ) -> PyResult<Self> {
        let builder = ExnessRangeBarBuilder::for_instrument(
            instrument.to_rust(),
            threshold_decimal_bps,
            strictness.to_rust(),
        )
        .map_err(|e| PyValueError::new_err(format!("Failed to create builder: {e}")))?;

        Ok(Self {
            builder,
            threshold_decimal_bps,
            instrument,
        })
    }

    /// Process a single tick
    ///
    /// Args:
    ///     tick: Dict with keys: bid, ask, `timestamp_ms`
    ///
    /// Returns:
    ///     None if bar still accumulating, Dict if bar completed
    fn process_tick(&mut self, py: Python, tick: &Bound<PyDict>) -> PyResult<Option<PyObject>> {
        // Extract tick fields
        let bid: f64 = tick
            .get_item("bid")?
            .ok_or_else(|| PyKeyError::new_err("Missing 'bid'"))?
            .extract()?;
        let ask: f64 = tick
            .get_item("ask")?
            .ok_or_else(|| PyKeyError::new_err("Missing 'ask'"))?
            .extract()?;
        let timestamp_ms: i64 = tick
            .get_item("timestamp_ms")?
            .ok_or_else(|| PyKeyError::new_err("Missing 'timestamp_ms'"))?
            .extract()?;

        let exness_tick = ExnessTick {
            bid,
            ask,
            timestamp_ms,
        };

        match self.builder.process_tick(&exness_tick) {
            Ok(Some(bar)) => Ok(Some(exness_rangebar_to_dict(py, &bar)?)),
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!("Processing failed: {e}"))),
        }
    }

    /// Process multiple ticks at once
    ///
    /// Args:
    ///     ticks: List of tick dicts with keys: bid, ask, `timestamp_ms`
    ///
    /// Returns:
    ///     List of completed bar dicts
    fn process_ticks(
        &mut self,
        py: Python,
        ticks: Vec<Bound<PyDict>>,
    ) -> PyResult<Vec<PyObject>> {
        let mut bars = Vec::new();

        for tick in ticks {
            if let Some(bar) = self.process_tick(py, &tick)? {
                bars.push(bar);
            }
        }

        Ok(bars)
    }

    /// Get incomplete bar if exists
    fn get_incomplete_bar(&self, py: Python) -> PyResult<Option<PyObject>> {
        match self.builder.get_incomplete_bar() {
            Some(bar) => Ok(Some(exness_rangebar_to_dict(py, &bar)?)),
            None => Ok(None),
        }
    }

    /// Get threshold value
    #[getter]
    const fn threshold_decimal_bps(&self) -> u32 {
        self.threshold_decimal_bps
    }

    /// Get instrument
    #[getter]
    const fn instrument(&self) -> PyExnessInstrument {
        self.instrument
    }
}

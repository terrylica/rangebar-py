//! Technical indicators for real-time streaming
//!
//! High-performance streaming indicators using rolling window algorithms
//! for live trading and dashboard applications.

use rangebar_core::RangeBar;
use serde::{Deserialize, Serialize};

/// Simple Moving Average with fixed window size
#[derive(Debug, Clone)]
pub struct SimpleMovingAverage {
    window_size: usize,
    values: Vec<f64>,
    current_index: usize,
    filled: bool,
}

impl SimpleMovingAverage {
    /// Create new SMA with specified window size
    pub fn new(window_size: usize) -> Result<Self, IndicatorError> {
        if window_size == 0 {
            return Err(IndicatorError::InvalidWindowSize);
        }

        Ok(Self {
            window_size,
            values: vec![0.0; window_size],
            current_index: 0,
            filled: false,
        })
    }

    /// Update with new price value and return current SMA
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.values[self.current_index] = value;
        self.current_index = (self.current_index + 1) % self.window_size;

        if !self.filled && self.current_index == 0 {
            self.filled = true;
        }

        if self.filled {
            Some(self.values.iter().sum::<f64>() / self.window_size as f64)
        } else {
            None
        }
    }

    /// Update with RangeBar close price
    pub fn update_from_bar(&mut self, bar: &RangeBar) -> Option<f64> {
        self.update(bar.close.to_f64())
    }
}

/// Exponential Moving Average
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    alpha: f64,
    current_value: Option<f64>,
}

impl ExponentialMovingAverage {
    /// Create new EMA with specified period
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidWindowSize);
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        Ok(Self {
            alpha,
            current_value: None,
        })
    }

    /// Update with new value and return current EMA
    pub fn update(&mut self, value: f64) -> f64 {
        match self.current_value {
            None => {
                self.current_value = Some(value);
                value
            }
            Some(prev) => {
                let new_value = self.alpha * value + (1.0 - self.alpha) * prev;
                self.current_value = Some(new_value);
                new_value
            }
        }
    }

    /// Update with RangeBar close price
    pub fn update_from_bar(&mut self, bar: &RangeBar) -> f64 {
        self.update(bar.close.to_f64())
    }
}

/// MACD (Moving Average Convergence Divergence) indicator
#[derive(Debug, Clone)]
pub struct MACD {
    fast_ema: ExponentialMovingAverage,
    slow_ema: ExponentialMovingAverage,
    signal_ema: ExponentialMovingAverage,
}

impl MACD {
    /// Create new MACD with standard periods (12, 26, 9)
    pub fn new() -> Result<Self, IndicatorError> {
        Self::with_periods(12, 26, 9)
    }

    /// Create new MACD with custom periods
    pub fn with_periods(
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Result<Self, IndicatorError> {
        Ok(Self {
            fast_ema: ExponentialMovingAverage::new(fast_period)?,
            slow_ema: ExponentialMovingAverage::new(slow_period)?,
            signal_ema: ExponentialMovingAverage::new(signal_period)?,
        })
    }

    /// Update with new price and return MACD values
    pub fn update(&mut self, price: f64) -> MACDValue {
        let fast = self.fast_ema.update(price);
        let slow = self.slow_ema.update(price);
        let macd_line = fast - slow;
        let signal_line = self.signal_ema.update(macd_line);
        let histogram = macd_line - signal_line;

        MACDValue {
            macd_line,
            signal_line,
            histogram,
        }
    }

    /// Update with RangeBar close price
    pub fn update_from_bar(&mut self, bar: &RangeBar) -> MACDValue {
        self.update(bar.close.to_f64())
    }
}

impl Default for MACD {
    fn default() -> Self {
        Self::new().expect("Default MACD configuration should be valid")
    }
}

/// MACD indicator output values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACDValue {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
}

/// RSI (Relative Strength Index) indicator
#[derive(Debug, Clone)]
pub struct RSI {
    period: usize,
    gains: Vec<f64>,
    losses: Vec<f64>,
    current_index: usize,
    filled: bool,
    previous_price: Option<f64>,
}

impl RSI {
    /// Create new RSI with specified period (typically 14)
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidWindowSize);
        }

        Ok(Self {
            period,
            gains: vec![0.0; period],
            losses: vec![0.0; period],
            current_index: 0,
            filled: false,
            previous_price: None,
        })
    }

    /// Update with new price and return RSI value
    pub fn update(&mut self, price: f64) -> Option<f64> {
        if let Some(prev_price) = self.previous_price {
            let change = price - prev_price;
            let (gain, loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, -change)
            };

            self.gains[self.current_index] = gain;
            self.losses[self.current_index] = loss;
            self.current_index = (self.current_index + 1) % self.period;

            if !self.filled && self.current_index == 0 {
                self.filled = true;
            }

            if self.filled {
                let avg_gain = self.gains.iter().sum::<f64>() / self.period as f64;
                let avg_loss = self.losses.iter().sum::<f64>() / self.period as f64;

                if avg_loss == 0.0 {
                    Some(100.0)
                } else {
                    let rs = avg_gain / avg_loss;
                    Some(100.0 - (100.0 / (1.0 + rs)))
                }
            } else {
                None
            }
        } else {
            self.previous_price = Some(price);
            None
        }
    }

    /// Update with RangeBar close price
    pub fn update_from_bar(&mut self, bar: &RangeBar) -> Option<f64> {
        self.update(bar.close.to_f64())
    }
}

/// Commodity Channel Index (CCI) indicator
#[derive(Debug, Clone)]
pub struct CCI {
    period: usize,
    typical_prices: Vec<f64>,
    current_index: usize,
    filled: bool,
}

impl CCI {
    /// Create new CCI with specified period (typically 20)
    pub fn new(period: usize) -> Result<Self, IndicatorError> {
        if period == 0 {
            return Err(IndicatorError::InvalidWindowSize);
        }

        Ok(Self {
            period,
            typical_prices: vec![0.0; period],
            current_index: 0,
            filled: false,
        })
    }

    /// Update with new OHLC values and return CCI value
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let typical_price = (high + low + close) / 3.0;

        self.typical_prices[self.current_index] = typical_price;
        self.current_index = (self.current_index + 1) % self.period;

        if !self.filled && self.current_index == 0 {
            self.filled = true;
        }

        if self.filled {
            let sma = self.typical_prices.iter().sum::<f64>() / self.period as f64;
            let mean_deviation = self
                .typical_prices
                .iter()
                .map(|&tp| (tp - sma).abs())
                .sum::<f64>()
                / self.period as f64;

            if mean_deviation == 0.0 {
                Some(0.0)
            } else {
                Some((typical_price - sma) / (0.015 * mean_deviation))
            }
        } else {
            None
        }
    }

    /// Update with RangeBar OHLC values
    pub fn update_from_bar(&mut self, bar: &RangeBar) -> Option<f64> {
        self.update(bar.high.to_f64(), bar.low.to_f64(), bar.close.to_f64())
    }
}

/// Indicator computation errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum IndicatorError {
    #[error("Invalid window size: must be greater than 0")]
    InvalidWindowSize,
    #[error("Insufficient data points for calculation")]
    InsufficientData,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_basic() {
        let mut sma = SimpleMovingAverage::new(3).unwrap();

        assert_eq!(sma.update(1.0), None);
        assert_eq!(sma.update(2.0), None);
        assert_eq!(sma.update(3.0), Some(2.0));
        assert_eq!(sma.update(4.0), Some(3.0));
    }

    #[test]
    fn test_ema_basic() {
        let mut ema = ExponentialMovingAverage::new(2).unwrap();

        let result1 = ema.update(10.0);
        assert_eq!(result1, 10.0);

        let result2 = ema.update(20.0);
        // Alpha = 2/(2+1) = 0.667, EMA = 0.667*20 + 0.333*10 = 16.67
        assert!((result2 - 16.666666666666668).abs() < 1e-10);
    }

    #[test]
    fn test_macd_basic() {
        let mut macd = MACD::new().unwrap();

        let result = macd.update(100.0);
        assert_eq!(result.macd_line, 0.0);
        assert_eq!(result.signal_line, 0.0);
        assert_eq!(result.histogram, 0.0);
    }

    #[test]
    fn test_rsi_basic() {
        let mut rsi = RSI::new(2).unwrap();

        assert_eq!(rsi.update(10.0), None);
        assert_eq!(rsi.update(12.0), None);

        // Should have RSI value after enough data points
        let result = rsi.update(11.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_cci_basic() {
        let mut cci = CCI::new(2).unwrap();

        assert_eq!(cci.update(10.0, 8.0, 9.0), None);

        let result = cci.update(12.0, 10.0, 11.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_indicator_errors() {
        assert!(SimpleMovingAverage::new(0).is_err());
        assert!(ExponentialMovingAverage::new(0).is_err());
        assert!(RSI::new(0).is_err());
        assert!(CCI::new(0).is_err());
    }
}

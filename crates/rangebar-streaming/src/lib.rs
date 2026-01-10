//! Real-time streaming engine for range bar processing
//!
//! This module provides real-time streaming capabilities for processing
//! range bars from live data sources with support for replay, statistics,
//! and indicators.

pub mod processor;
pub mod replay_buffer;

#[cfg(feature = "stats")]
pub mod stats;

#[cfg(feature = "indicators")]
pub mod indicators;

#[cfg(feature = "binance-integration")]
pub mod universal;

// Re-export commonly used types
pub use processor::StreamingProcessor;
pub use replay_buffer::{ReplayBuffer, ReplayBufferStats};

#[cfg(feature = "stats")]
pub use stats::{StatisticsSnapshot, StreamingConfig, StreamingStatsEngine};

#[cfg(feature = "indicators")]
pub use indicators::{
    CCI, ExponentialMovingAverage, IndicatorError, MACD, MACDValue, RSI, SimpleMovingAverage,
};

#[cfg(feature = "binance-integration")]
pub use universal::{StreamError, StreamMode, TradeStream, UniversalStream};

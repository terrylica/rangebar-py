//! Exness range bar construction from Raw_Spread tick data
//!
//! Converts Exness market maker quotes (Bid/Ask only) to range bars
//! using mid-price as synthetic trade price. Wrapper pattern preserves standard
//! RangeBar compatibility while adding spread dynamics as market stress signal.
//!
//! ## Why Exness Raw_Spread?
//!
//! - **CV=8.17**: 8× higher spread variability than Standard variant
//! - **Bimodal distribution**: 98% at 0.0 pips, 2% at 1-9 pips (stress events)
//! - **Zero rate limiting**: 100% reliability
//! - **Simple format**: Monthly ZIP → CSV
//!
//! ## Architecture
//!
//! - **Zero core changes**: Wraps RangeBarProcessor, no algorithm modifications
//! - **Adapter pattern**: ExnessRangeBar { base, spread_stats }
//! - **Fail-fast errors**: All errors propagated immediately to caller
//! - **Out-of-box dependencies**: Standard crates (zip, csv, chrono)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rangebar_providers::exness::{
//!     ExnessRangeBarBuilder,
//!     ExnessTick,
//!     ValidationStrictness,
//! };
//!
//! let mut builder = ExnessRangeBarBuilder::new(
//!     250,                          // 25bps threshold (v3.0.0: 0.1bps units)
//!     "EURUSD_Raw_Spread",          // Exness Raw_Spread variant
//!     ValidationStrictness::Strict  // Default validation
//! ).expect("valid threshold");
//!
//! // Process tick stream
//! # let tick_stream: Vec<ExnessTick> = vec![];
//! for tick in tick_stream {
//!     match builder.process_tick(&tick) {
//!         Ok(Some(bar)) => {
//!             // Bar completed (threshold breached)
//!             println!("Bar: O={} H={} L={} C={}",
//!                      bar.base.open, bar.base.high, bar.base.low, bar.base.close);
//!             println!("Spread: avg={} min={} max={}",
//!                      bar.spread_stats.avg_spread(),
//!                      bar.spread_stats.min_spread,
//!                      bar.spread_stats.max_spread);
//!         }
//!         Ok(None) => {
//!             // Tick processed, bar accumulating
//!         }
//!         Err(e) => {
//!             // Error: log and skip (example only)
//!             eprintln!("Error processing tick: {}", e);
//!         }
//!     }
//! }
//!
//! // Get final incomplete bar
//! if let Some(partial) = builder.get_incomplete_bar() {
//!     println!("Partial bar at stream end");
//! }
//! ```
//!
//! ## Data Format
//!
//! **API**: `https://ticks.ex2archive.com/ticks/{SYMBOL}_Raw_Spread/{year}/{month}/...`
//!
//! **CSV Schema**:
//! ```csv
//! "Exness","Symbol","Timestamp","Bid","Ask"
//! "exness","EURUSD_Raw_Spread","2024-01-15 00:00:00.032Z",1.0945,1.09456
//! ```
//!
//! **Characteristics**:
//! - Monthly granularity (~60K ticks/day for EURUSD)
//! - ZIP compression (~10:1 ratio, ~9MB/month)
//! - ISO 8601 UTC timestamps with millisecond precision
//! - No volume data (Bid/Ask prices only)
//!
//! ## Data Structure Differences
//!
//! | Aspect | Binance aggTrades | Exness Raw_Spread |
//! |--------|------------------|-------------------|
//! | Type | Actual trades | Market maker quotes |
//! | Price | Single execution | Bid + Ask |
//! | Volume | Quantity traded | **None** (no volume data) |
//! | Direction | is_buyer_maker | Unknown (quotes) |
//! | Granularity | Tick-by-tick | Tick-by-tick |
//!
//! ## Volume Semantics
//!
//! - `RangeBar.volume` = 0 (Exness Raw_Spread has no volume data)
//! - `buy_volume` = 0 (direction unknown)
//! - `sell_volume` = 0 (direction unknown)
//! - `SpreadStats` captures market stress via spread dynamics
//!
//! ## Error Handling
//!
//! **Policy**: Fail-fast, all errors propagated immediately
//!
//! - **HTTP errors**: 404, 503, timeout → raise
//! - **ZIP errors**: Extraction failure → raise
//! - **CSV errors**: Parsing failure → raise
//! - **Validation errors**: CrossedMarket, ExcessiveSpread → raise
//! - **Processing errors**: Algorithm failure → raise
//!
//! **No fallbacks, no defaults, no error rate thresholds**.
//!
//! ## SLOs (Service Level Objectives)
//!
//! - **Availability**: 100% fetch success (zero rate limiting)
//! - **Correctness**: 100% validation pass (fail-fast on any error)
//! - **Observability**: 100% error traceability (thiserror with full context)
//! - **Maintainability**: Out-of-box dependencies only (zip, csv, chrono)

pub mod builder;
pub mod client;
pub mod conversion;
pub mod types;

// Re-export main types for convenience
pub use builder::ExnessRangeBarBuilder;
pub use client::ExnessFetcher;
pub use types::{
    ConversionError, ExnessError, ExnessInstrument, ExnessRangeBar, ExnessTick, SpreadStats,
    ValidationStrictness,
};

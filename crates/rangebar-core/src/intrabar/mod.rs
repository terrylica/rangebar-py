//! Intra-bar features computed from constituent aggTrades.
//!
//! Issue #59: Intra-bar microstructure features for large range bars.
//!
//! This module computes 22 features from trades WITHIN each range bar:
//! - 8 ITH features (Investment Time Horizon from trading-fitness algorithms)
//! - 12 statistical features (OFI, intensity, burstiness, Kyle lambda, etc.)
//! - 2 complexity features (Hurst exponent, permutation entropy)
//!
//! ## Key Design: Intra-Bar, Not Lookback
//!
//! Features are computed from trades **WITHIN** each bar (from `open_time` to
//! `close_time`), NOT from a rolling lookback window before the bar opens.
//! This ensures:
//! - Temporal integrity: Features describe the bar itself
//! - No extra memory: Trades already accumulated during bar construction
//! - Consistent semantics: Features = bar characteristics
//!
//! ## ITH Algorithm Origin
//!
//! The ITH (Investment Time Horizon) algorithms are copied from:
//! `trading-fitness/packages/metrics-rust/src/ith.rs`
//!
//! Key implementation notes:
//! 1. **Exact algorithm alignment** with trading-fitness
//! 2. **TMAEG = max_drawdown/max_runup**: No magic numbers, derived from window extremes
//! 3. **Normalization**: Exact functions from `ith_normalize.rs`
//! 4. **CV calculation**: Numba-aligned method with epoch_indices starting at 0

pub mod drawdown;
pub mod features;
pub mod ith;
pub mod normalize;
pub mod types;

// Re-export main types and functions
pub use features::{IntraBarFeatures, compute_intra_bar_features, compute_intra_bar_features_with_scratch};
pub use types::{BearIthResult, BullIthResult};

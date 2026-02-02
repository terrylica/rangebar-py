//! Type definitions for intra-bar ITH analysis.
//!
//! Issue #59: Intra-bar microstructure features for large range bars.
//!
//! ORIGIN: trading-fitness/packages/metrics-rust/src/types.rs
//! COPIED: 2026-02-02
//! MODIFICATIONS: Removed serde derives, adapted for rangebar-core

/// Result of Bull ITH (long position) analysis within a bar.
///
/// Bull ITH tracks excess gains (upside) and excess losses (drawdowns).
/// Epochs trigger when gains exceed losses AND exceed the TMAEG threshold.
#[derive(Debug, Clone, Default)]
pub struct BullIthResult {
    /// Excess gains at each time point.
    pub excess_gains: Vec<f64>,
    /// Excess losses at each time point (drawdowns).
    pub excess_losses: Vec<f64>,
    /// Number of bull epochs detected.
    pub num_of_epochs: usize,
    /// Boolean array marking epoch points.
    pub epochs: Vec<bool>,
    /// Coefficient of variation of epoch intervals.
    pub intervals_cv: f64,
    /// Maximum drawdown observed.
    pub max_drawdown: f64,
}

/// Result of Bear ITH (short position) analysis within a bar.
///
/// Bear ITH is the INVERSE of Bull ITH for short positions.
/// Epochs trigger when gains (from price drops) exceed losses AND exceed TMAEG.
#[derive(Debug, Clone, Default)]
pub struct BearIthResult {
    /// Excess gains at each time point (from price drops).
    pub excess_gains: Vec<f64>,
    /// Excess losses at each time point (runups).
    pub excess_losses: Vec<f64>,
    /// Number of bear epochs detected.
    pub num_of_epochs: usize,
    /// Boolean array marking epoch points.
    pub epochs: Vec<bool>,
    /// Coefficient of variation of epoch intervals.
    pub intervals_cv: f64,
    /// Maximum runup observed (adverse for shorts).
    pub max_runup: f64,
}

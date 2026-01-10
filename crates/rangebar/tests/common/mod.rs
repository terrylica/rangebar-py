//! Shared Exness validation utilities with fail-fast JSON reporting
//!
//! All failures emit machine-readable JSON for Claude Code hook integration.
//! Pattern: Status gates with ✅/❌, fail loudly with detailed context.

use rangebar::providers::exness::{ExnessRangeBar, ExnessTick};
use serde::Serialize;
use std::fs;
use std::path::Path;

// ============================================================================
// Status Gate Types (Fail-Fast Pattern)
// ============================================================================

/// Status gate with fail-fast behavior
///
/// Emits machine-readable JSON on failure for Claude Code lifecycle compliance.
#[derive(Debug, Serialize)]
pub struct ValidationGate {
    pub gate_name: &'static str,
    pub passed: bool,
    pub details: String,
    pub timestamp: String,
    pub metrics: Option<GateMetrics>,
}

#[derive(Debug, Serialize)]
pub struct GateMetrics {
    pub expected: String,
    pub actual: String,
}

impl ValidationGate {
    pub fn pass(gate_name: &'static str, details: impl Into<String>) -> Self {
        Self {
            gate_name,
            passed: true,
            details: details.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            metrics: None,
        }
    }

    pub fn fail(
        gate_name: &'static str,
        details: impl Into<String>,
        expected: &str,
        actual: &str,
    ) -> Self {
        Self {
            gate_name,
            passed: false,
            details: details.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            metrics: Some(GateMetrics {
                expected: expected.to_string(),
                actual: actual.to_string(),
            }),
        }
    }

    /// Panics with JSON-formatted error if gate failed
    #[allow(dead_code)]
    pub fn assert_or_fail(&self) {
        if !self.passed {
            let json = serde_json::to_string_pretty(self).unwrap();
            panic!(
                "\n\n=== VALIDATION GATE FAILED ===\n\
                Gate: {}\n\
                Details: {}\n\
                \n--- Machine-Readable JSON ---\n{}\n\n",
                self.gate_name, self.details, json
            );
        }
    }
}

/// Validation report with all gates
#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub instrument: String,
    pub gates: Vec<ValidationGate>,
    pub all_passed: bool,
    pub generated_at: String,
}

impl ValidationReport {
    pub fn new(instrument: &str) -> Self {
        Self {
            instrument: instrument.to_string(),
            gates: Vec::new(),
            all_passed: true,
            generated_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn add_gate(&mut self, gate: ValidationGate) {
        if !gate.passed {
            self.all_passed = false;
        }
        self.gates.push(gate);
    }

    /// Write JSON report and fail fast if any gate failed
    pub fn finalize(&self, output_dir: &Path) {
        fs::create_dir_all(output_dir).expect("Create output dir");

        // Write JSON report
        let json_path = output_dir.join(format!(
            "{}_validation.json",
            self.instrument.to_lowercase()
        ));
        let json = serde_json::to_string_pretty(self).unwrap();
        fs::write(&json_path, &json).expect("Write JSON report");

        // Write Markdown report
        let md_path = output_dir.join(format!(
            "{}_validation_report.md",
            self.instrument.to_lowercase()
        ));
        let md = self.to_markdown();
        fs::write(&md_path, &md).expect("Write Markdown report");

        println!("Artifacts written to:");
        println!("  - {}", json_path.display());
        println!("  - {}", md_path.display());

        // Fail fast with full report if any gate failed
        if !self.all_passed {
            panic!(
                "\n\n=== VALIDATION FAILED ===\n\
                Instrument: {}\n\
                Failed gates: {:?}\n\
                \n--- Full Report ---\n{}\n\n",
                self.instrument,
                self.gates
                    .iter()
                    .filter(|g| !g.passed)
                    .map(|g| g.gate_name)
                    .collect::<Vec<_>>(),
                json
            );
        }
    }

    fn to_markdown(&self) -> String {
        let mut md = format!("# {} Validation Report\n\n", self.instrument);
        md.push_str(&format!("**Generated**: {}\n\n", self.generated_at));
        md.push_str("## Status Gates\n\n");
        md.push_str("| Gate | Status | Details |\n");
        md.push_str("|------|--------|--------|\n");
        for gate in &self.gates {
            let status = if gate.passed { "✅ PASS" } else { "❌ FAIL" };
            md.push_str(&format!(
                "| {} | {} | {} |\n",
                gate.gate_name, status, gate.details
            ));
        }
        md
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validate temporal integrity with fail-fast
pub fn validate_temporal_integrity(ticks: &[ExnessTick], report: &mut ValidationReport) {
    for i in 1..ticks.len() {
        if ticks[i].timestamp_ms < ticks[i - 1].timestamp_ms {
            report.add_gate(ValidationGate::fail(
                "Temporal Integrity",
                format!(
                    "Violation at tick {}: {} < {}",
                    i,
                    ticks[i].timestamp_ms,
                    ticks[i - 1].timestamp_ms
                ),
                "monotonic increasing",
                &format!("{} < {}", ticks[i].timestamp_ms, ticks[i - 1].timestamp_ms),
            ));
            return;
        }
    }
    report.add_gate(ValidationGate::pass(
        "Temporal Integrity",
        "Monotonic timestamps verified",
    ));
}

/// Validate price range with fail-fast
pub fn validate_price_range(
    ticks: &[ExnessTick],
    min: f64,
    max: f64,
    symbol: &str,
    report: &mut ValidationReport,
) {
    for tick in ticks {
        if tick.bid < min || tick.bid > max {
            report.add_gate(ValidationGate::fail(
                "Price Range",
                format!("{} price {} outside valid range", symbol, tick.bid),
                &format!("[{}, {}]", min, max),
                &format!("{}", tick.bid),
            ));
            return;
        }
    }
    report.add_gate(ValidationGate::pass(
        "Price Range",
        format!("All prices in [{}, {}]", min, max),
    ));
}

/// Validate spread distribution with fail-fast
/// Returns the tight spread percentage for downstream use
///
/// # Arguments
/// * `spread_tolerance` - Maximum spread to consider as "tight spread"
///   - Forex pairs (EURUSD, etc.): use `0.000001` (true zero spread)
///   - XAUUSD (Gold): use `0.10` (spreads are ~$0.06, not zero)
pub fn validate_spread_distribution(
    ticks: &[ExnessTick],
    min_zero_pct: f64,
    spread_tolerance: f64,
    report: &mut ValidationReport,
) -> f64 {
    let zero_spreads = ticks
        .iter()
        .filter(|t| (t.ask - t.bid).abs() < spread_tolerance)
        .count();
    let zero_pct = (zero_spreads as f64 / ticks.len() as f64) * 100.0;

    if zero_pct < min_zero_pct {
        report.add_gate(ValidationGate::fail(
            "Spread Distribution",
            format!(
                "Only {:.1}% zero spread (expected >{:.0}%)",
                zero_pct, min_zero_pct
            ),
            &format!(">{:.0}%", min_zero_pct),
            &format!("{:.1}%", zero_pct),
        ));
    } else {
        report.add_gate(ValidationGate::pass(
            "Spread Distribution",
            format!(
                "{:.1}% zero spread (>{:.0}% required)",
                zero_pct, min_zero_pct
            ),
        ));
    }
    zero_pct
}

/// Validate bar OHLCV integrity with fail-fast
pub fn validate_bar_integrity(bars: &[ExnessRangeBar], report: &mut ValidationReport) {
    for (i, bar) in bars.iter().enumerate() {
        let b = &bar.base;

        // OHLCV integrity checks
        if b.high.0 < b.open.0 || b.high.0 < b.close.0 || b.low.0 > b.open.0 || b.low.0 > b.close.0
        {
            report.add_gate(ValidationGate::fail(
                "OHLCV Integrity",
                format!("Bar {} violates OHLCV invariants", i),
                "high >= max(open,close), low <= min(open,close)",
                &format!(
                    "O={} H={} L={} C={}",
                    b.open.0, b.high.0, b.low.0, b.close.0
                ),
            ));
            return;
        }

        // Exness has no volume data
        if b.volume.0 != 0 {
            report.add_gate(ValidationGate::fail(
                "Volume Semantics",
                format!("Bar {} has non-zero volume", i),
                "0",
                &format!("{}", b.volume.0),
            ));
            return;
        }

        // Spread stats should have data
        if bar.spread_stats.tick_count == 0 {
            report.add_gate(ValidationGate::fail(
                "Spread Stats",
                format!("Bar {} has zero tick count", i),
                ">0",
                "0",
            ));
            return;
        }
    }
    report.add_gate(ValidationGate::pass(
        "OHLCV Integrity",
        format!("All {} bars valid", bars.len()),
    ));
}

/// Validate bar generation produces bars
pub fn validate_bar_generation(bars: &[ExnessRangeBar], report: &mut ValidationReport) {
    if bars.is_empty() {
        report.add_gate(ValidationGate::fail(
            "Bar Generation",
            "No bars generated",
            ">0 bars",
            "0 bars",
        ));
    } else {
        report.add_gate(ValidationGate::pass(
            "Bar Generation",
            format!("{} bars generated", bars.len()),
        ));
    }
}

// ============================================================================
// NDJSON Export Utilities
// ============================================================================

/// Bar record for NDJSON export
#[derive(Serialize)]
pub struct BarRecord {
    pub bar_num: usize,
    pub open_time: i64,
    pub close_time: i64,
    pub open: String,
    pub high: String,
    pub low: String,
    pub close: String,
    pub tick_count: u32,
    pub avg_spread_bps: String,
}

impl BarRecord {
    pub fn from_bar(i: usize, bar: &ExnessRangeBar) -> Self {
        Self {
            bar_num: i,
            open_time: bar.base.open_time,
            close_time: bar.base.close_time,
            open: format!("{:.5}", bar.base.open.to_f64()),
            high: format!("{:.5}", bar.base.high.to_f64()),
            low: format!("{:.5}", bar.base.low.to_f64()),
            close: format!("{:.5}", bar.base.close.to_f64()),
            tick_count: bar.spread_stats.tick_count,
            avg_spread_bps: format!("{:.4}", bar.spread_stats.avg_spread().to_f64() * 10000.0),
        }
    }
}

/// Validation summary for insta snapshots
#[derive(Serialize)]
pub struct ValidationSummary {
    pub instrument: String,
    pub ticks: usize,
    pub bars: usize,
    pub zero_spread_pct: f64,
    pub all_gates_passed: bool,
}

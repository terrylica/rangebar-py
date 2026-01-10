//! Range bar algorithm configuration

use rangebar_core::fixed_point::BASIS_POINTS_SCALE;
use serde::{Deserialize, Serialize};

/// Range bar algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    /// Default threshold in decimal basis points (e.g., 250 = 25bps = 0.25%)
    pub default_threshold_decimal_bps: u32,

    /// Minimum allowed threshold in decimal basis points
    pub min_threshold_decimal_bps: u32,

    /// Maximum allowed threshold in decimal basis points
    pub max_threshold_decimal_bps: u32,

    /// Enable fixed-point arithmetic precision validation
    pub validate_precision: bool,

    /// Number of decimal places for fixed-point arithmetic
    pub fixed_point_decimals: u8,

    /// Enable non-lookahead bias validation
    pub validate_non_lookahead: bool,

    /// Enable zero-duration bar validation (NOTABUG checks)
    pub validate_zero_duration: bool,

    /// Maximum acceptable zero-duration bar percentage
    pub max_zero_duration_percentage: f64,

    /// Enable temporal integrity checks
    pub validate_temporal_integrity: bool,

    /// Enable OHLC consistency validation
    pub validate_ohlc_consistency: bool,

    /// Enable volume consistency validation
    pub validate_volume_consistency: bool,

    /// Batch size for processing large datasets
    pub processing_batch_size: usize,

    /// Enable memory optimization for large datasets
    pub enable_memory_optimization: bool,

    /// Enable performance metrics collection
    pub collect_performance_metrics: bool,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            default_threshold_decimal_bps: 250, // 250 decimal bps = 25bps = 0.25% (v3.0.0)
            min_threshold_decimal_bps: 1,       // 1 decimal bps = 0.1bps = 0.001% (v3.0.0 minimum)
            max_threshold_decimal_bps: 100000,  // 100000 decimal bps = 10000bps = 100% (v3.0.0)
            validate_precision: true,
            fixed_point_decimals: 8,
            validate_non_lookahead: true,
            validate_zero_duration: true,
            max_zero_duration_percentage: 0.1, // 0.1% of total bars
            validate_temporal_integrity: true,
            validate_ohlc_consistency: true,
            validate_volume_consistency: true,
            processing_batch_size: 100_000,
            enable_memory_optimization: true,
            collect_performance_metrics: false,
        }
    }
}

impl AlgorithmConfig {
    /// Convert decimal basis points to decimal threshold
    pub fn threshold_as_decimal(&self, threshold_decimal_bps: Option<u32>) -> f64 {
        let bps = threshold_decimal_bps.unwrap_or(self.default_threshold_decimal_bps);
        bps as f64 / BASIS_POINTS_SCALE as f64
    }

    /// Validate threshold is within acceptable bounds
    pub fn validate_threshold(&self, threshold_decimal_bps: u32) -> Result<(), String> {
        if threshold_decimal_bps < self.min_threshold_decimal_bps {
            return Err(format!(
                "Threshold {} decimal bps is below minimum {} decimal bps",
                threshold_decimal_bps, self.min_threshold_decimal_bps
            ));
        }

        if threshold_decimal_bps > self.max_threshold_decimal_bps {
            return Err(format!(
                "Threshold {} decimal bps exceeds maximum {} decimal bps",
                threshold_decimal_bps, self.max_threshold_decimal_bps
            ));
        }

        Ok(())
    }

    /// Get the upper breach threshold for a given price
    pub fn upper_threshold(&self, price: f64, threshold_decimal_bps: Option<u32>) -> f64 {
        let threshold = self.threshold_as_decimal(threshold_decimal_bps);
        price * (1.0 + threshold)
    }

    /// Get the lower breach threshold for a given price
    pub fn lower_threshold(&self, price: f64, threshold_decimal_bps: Option<u32>) -> f64 {
        let threshold = self.threshold_as_decimal(threshold_decimal_bps);
        price * (1.0 - threshold)
    }

    /// Check if all validations are enabled
    pub fn all_validations_enabled(&self) -> bool {
        self.validate_precision
            && self.validate_non_lookahead
            && self.validate_zero_duration
            && self.validate_temporal_integrity
            && self.validate_ohlc_consistency
            && self.validate_volume_consistency
    }

    /// Get memory optimization level
    pub fn memory_optimization_level(&self) -> MemoryOptimizationLevel {
        if self.enable_memory_optimization {
            if self.processing_batch_size <= 50_000 {
                MemoryOptimizationLevel::High
            } else if self.processing_batch_size <= 100_000 {
                MemoryOptimizationLevel::Medium
            } else {
                MemoryOptimizationLevel::Low
            }
        } else {
            MemoryOptimizationLevel::None
        }
    }
}

/// Memory optimization levels for algorithm processing
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryOptimizationLevel {
    None,   // No optimization
    Low,    // Basic optimization
    Medium, // Moderate optimization
    High,   // Aggressive optimization
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_conversion() {
        let config = AlgorithmConfig::default();

        // v3.0.0: thresholds now in decimal bps
        // Test default threshold (250 decimal bps = 25bps = 0.25%)
        assert_eq!(config.threshold_as_decimal(None), 0.0025);

        // Test custom threshold (800 decimal bps = 80bps = 0.8%)
        assert_eq!(config.threshold_as_decimal(Some(800)), 0.008);

        // Test 1% threshold (1000 decimal bps = 100bps = 1%)
        assert_eq!(config.threshold_as_decimal(Some(1000)), 0.01);
    }

    #[test]
    fn test_threshold_validation() {
        let config = AlgorithmConfig::default();

        // Valid threshold (250 decimal bps = 25bps)
        assert!(config.validate_threshold(250).is_ok());

        // Too low (0 is below minimum of 1)
        assert!(config.validate_threshold(0).is_err());

        // Too high (150000 decimal bps = 15000bps = 150% exceeds max of 100%)
        assert!(config.validate_threshold(150000).is_err());
    }

    #[test]
    fn test_breach_thresholds() {
        let config = AlgorithmConfig::default();
        let price = 50000.0;

        // v3.0.0: Default threshold is 250 decimal bps = 25bps = 0.25%
        let upper = config.upper_threshold(price, None);
        let lower = config.lower_threshold(price, None);

        assert_eq!(upper, 50125.0); // +0.25%
        assert_eq!(lower, 49875.0); // -0.25%

        // 0.8% threshold (800 decimal bps = 80bps = 0.8%)
        let upper_8 = config.upper_threshold(price, Some(800));
        let lower_8 = config.lower_threshold(price, Some(800));

        assert_eq!(upper_8, 50400.0); // +0.8%
        assert_eq!(lower_8, 49600.0); // -0.8%
    }

    #[test]
    fn test_memory_optimization_levels() {
        // Test high memory optimization (small batch size)
        let config1 = AlgorithmConfig {
            processing_batch_size: 25_000,
            ..Default::default()
        };
        assert_eq!(
            config1.memory_optimization_level(),
            MemoryOptimizationLevel::High
        );

        // Test medium memory optimization
        let config2 = AlgorithmConfig {
            processing_batch_size: 75_000,
            ..Default::default()
        };
        assert_eq!(
            config2.memory_optimization_level(),
            MemoryOptimizationLevel::Medium
        );

        // Test low memory optimization (large batch size)
        let config3 = AlgorithmConfig {
            processing_batch_size: 150_000,
            ..Default::default()
        };
        assert_eq!(
            config3.memory_optimization_level(),
            MemoryOptimizationLevel::Low
        );

        // Test disabled memory optimization
        let config4 = AlgorithmConfig {
            enable_memory_optimization: false,
            ..Default::default()
        };
        assert_eq!(
            config4.memory_optimization_level(),
            MemoryOptimizationLevel::None
        );
    }

    #[test]
    fn test_all_validations_enabled() {
        let config = AlgorithmConfig::default();
        assert!(config.all_validations_enabled());

        let mut disabled_config = config.clone();
        disabled_config.validate_precision = false;
        assert!(!disabled_config.all_validations_enabled());
    }
}

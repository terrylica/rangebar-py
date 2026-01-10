//! Configuration management for rangebar
//!
//! Centralized configuration handling with support for:
//! - Default values
//! - Configuration files (TOML)
//! - Environment variables
//! - Command-line arguments
//!
//! Configuration precedence (highest to lowest):
//! 1. Command-line arguments
//! 2. Environment variables
//! 3. Configuration file
//! 4. Default values

mod algorithm;
mod app;
mod data;
mod export;

// Re-export main types
pub use algorithm::AlgorithmConfig;
pub use app::AppConfig;
pub use data::DataConfig;
pub use export::ExportConfig;

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Root configuration structure containing all configuration categories
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Settings {
    /// Application-wide settings
    pub app: AppConfig,

    /// Data source and processing configuration
    pub data: DataConfig,

    /// Range bar algorithm configuration
    pub algorithm: AlgorithmConfig,

    /// Export and output configuration
    pub export: ExportConfig,
}

impl Settings {
    /// Load configuration from multiple sources with proper precedence
    pub fn load() -> Result<Self, config::ConfigError> {
        let builder = config::Config::builder()
            // Start with defaults
            .add_source(config::Config::try_from(&Settings::default())?)
            // Add configuration file if it exists
            .add_source(
                config::File::with_name("rangebar")
                    .format(config::FileFormat::Toml)
                    .required(false),
            )
            // Add environment variables with RANGEBAR_ prefix
            .add_source(
                config::Environment::with_prefix("RANGEBAR")
                    .prefix_separator("_")
                    .separator("_"),
            );

        // Build and deserialize
        let config = builder.build()?;
        config.try_deserialize()
    }

    /// Load configuration from a specific file path
    pub fn load_from_file(path: &Path) -> Result<Self, config::ConfigError> {
        let builder = config::Config::builder()
            .add_source(config::Config::try_from(&Settings::default())?)
            .add_source(config::File::from(path).format(config::FileFormat::Toml));

        let config = builder.build()?;
        config.try_deserialize()
    }

    /// Merge command-line arguments into the loaded configuration
    pub fn merge_cli_args(mut self, cli_args: &dyn CliConfigMerge) -> Self {
        cli_args.merge_into_config(&mut self);
        self
    }
}

/// Trait for merging CLI arguments into configuration
pub trait CliConfigMerge {
    fn merge_into_config(&self, config: &mut Settings);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();

        // Verify all sections are present
        assert_eq!(settings.data.base_url, "https://data.binance.vision/data/");
        // v3.0.0: threshold now 250 decimal bps = 25bps
        assert_eq!(settings.algorithm.default_threshold_decimal_bps, 250);
        assert_eq!(
            settings.export.default_output_dir,
            PathBuf::from("./output")
        );
    }

    #[test]
    fn test_settings_serialization() {
        let settings = Settings::default();

        // Test that settings can be serialized and deserialized
        let toml_str = toml::to_string(&settings).expect("Failed to serialize to TOML");
        let _: Settings = toml::from_str(&toml_str).expect("Failed to deserialize from TOML");
    }
}

//! Application-wide configuration settings

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Application-wide configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Application name for logging and identification
    pub name: String,

    /// Application version
    pub version: String,

    /// Global log level
    pub log_level: LogLevel,

    /// Number of worker threads for parallel processing
    pub worker_threads: Option<usize>,

    /// Maximum concurrent operations
    pub max_concurrent_operations: usize,

    /// Default temporary directory for intermediate files
    pub temp_dir: PathBuf,

    /// Enable performance metrics collection
    pub enable_metrics: bool,

    /// Enable debug mode with additional logging
    pub debug_mode: bool,
}

/// Log level configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            name: "rangebar".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            log_level: LogLevel::Info,
            worker_threads: None, // Auto-detect from system
            max_concurrent_operations: 10,
            temp_dir: PathBuf::from("/tmp"),
            enable_metrics: false,
            debug_mode: false,
        }
    }
}

impl AppConfig {
    /// Get the number of worker threads, auto-detecting if not specified
    pub fn worker_threads(&self) -> usize {
        self.worker_threads.unwrap_or_else(num_cpus::get)
    }

    /// Check if running in debug mode
    pub fn is_debug(&self) -> bool {
        self.debug_mode || matches!(self.log_level, LogLevel::Debug | LogLevel::Trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_threads_auto_detect() {
        let config = AppConfig::default();
        let threads = config.worker_threads();

        // Should auto-detect from system
        assert!(threads > 0);
        assert!(threads <= 1024); // Reasonable upper bound
    }

    #[test]
    fn test_debug_mode_detection() {
        let mut config = AppConfig::default();
        assert!(!config.is_debug());

        config.debug_mode = true;
        assert!(config.is_debug());

        config.debug_mode = false;
        config.log_level = LogLevel::Debug;
        assert!(config.is_debug());
    }
}

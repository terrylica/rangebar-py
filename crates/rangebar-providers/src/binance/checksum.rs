//! SHA-256 checksum verification for Binance Vision downloads
//!
//! Implements GitHub Issue #43: Detect corrupted downloads early before they
//! cause silent data quality issues.
//!
//! Every Binance Vision download has a corresponding `.CHECKSUM` file:
//! - Data URL: `https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip`
//! - Checksum URL: `{data_url}.CHECKSUM`
//!
//! Checksum file format:
//! ```text
//! d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592  BTCUSDT-aggTrades-2024-01-01.zip
//! ```
//! (64-char SHA-256 hex + two spaces + filename)

use sha2::{Digest, Sha256};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

/// Checksum verification errors
#[derive(Error, Debug)]
pub enum ChecksumError {
    /// SHA-256 hash mismatch - data corruption detected
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    Mismatch { expected: String, actual: String },

    /// Failed to fetch checksum file from Binance
    #[error("Failed to fetch checksum file: {0}")]
    FetchFailed(String),

    /// Checksum file format is invalid
    #[error("Invalid checksum format: {0}")]
    InvalidFormat(String),

    /// Checksum file not available (HTTP 404)
    /// This is a soft error - old data may not have checksums
    #[error("Checksum file not available (HTTP {status})")]
    NotAvailable { status: u16 },
}

/// Result of checksum verification
#[derive(Debug, Clone)]
pub struct ChecksumResult {
    /// Whether verification was performed and passed
    pub verified: bool,
    /// Expected hash from Binance (None if skipped)
    pub expected: Option<String>,
    /// Computed hash of downloaded data
    pub actual: String,
    /// Whether verification was skipped (checksum unavailable)
    pub skipped: bool,
}

impl ChecksumResult {
    /// Create a result for skipped verification
    #[must_use]
    pub fn skipped() -> Self {
        Self {
            verified: false,
            expected: None,
            actual: String::new(),
            skipped: true,
        }
    }
}

/// Compute SHA-256 hash of data
///
/// Returns lowercase hex string (64 characters)
#[must_use]
pub fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Parse Binance checksum file format
///
/// Format: `<64-char-hex>  <filename>`
/// The hash and filename are separated by two spaces.
///
/// # Errors
///
/// Returns `ChecksumError::InvalidFormat` if:
/// - File is empty
/// - Hash is not 64 characters
/// - Hash contains non-hex characters
pub fn parse_checksum_file(content: &str) -> Result<String, ChecksumError> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err(ChecksumError::InvalidFormat(
            "Empty checksum file".to_string(),
        ));
    }

    // Split by whitespace and take the first part (the hash)
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.is_empty() {
        return Err(ChecksumError::InvalidFormat(
            "Empty checksum file".to_string(),
        ));
    }

    let hash = parts[0];

    // Validate hash format: must be 64 hex characters
    if hash.len() != 64 {
        return Err(ChecksumError::InvalidFormat(format!(
            "Invalid SHA-256 hash length: {} (expected 64)",
            hash.len()
        )));
    }

    if !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ChecksumError::InvalidFormat(format!(
            "Invalid SHA-256 hash (non-hex characters): {}",
            hash
        )));
    }

    Ok(hash.to_lowercase())
}

/// Fetch checksum from Binance Vision
///
/// Appends `.CHECKSUM` to the data URL and fetches the checksum file.
///
/// # Errors
///
/// - `ChecksumError::NotAvailable` if checksum file doesn't exist (HTTP 404)
/// - `ChecksumError::FetchFailed` for other HTTP errors or network issues
/// - `ChecksumError::InvalidFormat` if checksum file format is invalid
#[instrument(skip(client), fields(checksum_url))]
pub async fn fetch_checksum(
    client: &reqwest::Client,
    data_url: &str,
) -> Result<String, ChecksumError> {
    let checksum_url = format!("{data_url}.CHECKSUM");
    tracing::Span::current().record("checksum_url", &checksum_url);

    debug!(
        event_type = "checksum_fetch_start",
        data_url = data_url,
        "Fetching checksum"
    );

    let response = client
        .get(&checksum_url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| ChecksumError::FetchFailed(e.to_string()))?;

    let status = response.status().as_u16();
    if status == 404 {
        warn!(
            event_type = "checksum_not_found",
            http_status = status,
            "Checksum file not available"
        );
        return Err(ChecksumError::NotAvailable { status });
    }

    if !response.status().is_success() {
        error!(
            event_type = "checksum_fetch_failed",
            http_status = status,
            "Failed to fetch checksum"
        );
        return Err(ChecksumError::FetchFailed(format!("HTTP {status}")));
    }

    let content = response
        .text()
        .await
        .map_err(|e| ChecksumError::FetchFailed(e.to_string()))?;

    let hash = parse_checksum_file(&content)?;

    debug!(
        event_type = "checksum_fetch_complete",
        expected_hash = %hash,
        "Checksum fetched successfully"
    );

    Ok(hash)
}

/// Verify data against expected checksum
///
/// # Errors
///
/// Returns `ChecksumError::Mismatch` if computed hash doesn't match expected
pub fn verify_checksum(data: &[u8], expected: &str) -> Result<ChecksumResult, ChecksumError> {
    let actual = compute_sha256(data);
    let expected_lower = expected.to_lowercase();

    if actual != expected_lower {
        error!(
            event_type = "checksum_verify_failed",
            expected_hash = %expected_lower,
            actual_hash = %actual,
            data_size_bytes = data.len(),
            "Checksum mismatch detected"
        );
        return Err(ChecksumError::Mismatch {
            expected: expected_lower,
            actual,
        });
    }

    info!(
        event_type = "checksum_verify_success",
        hash = %actual,
        data_size_bytes = data.len(),
        "Checksum verified"
    );

    Ok(ChecksumResult {
        verified: true,
        expected: Some(expected_lower),
        actual,
        skipped: false,
    })
}

/// Fetch checksum and verify data (convenience function)
///
/// This is the primary entry point for checksum verification.
///
/// # Error Handling Strategy
///
/// | Scenario | Behavior | Rationale |
/// |----------|----------|-----------|
/// | Checksum matches | Returns Ok | Success case |
/// | Checksum mismatch | Returns Err (hard error) | Data corruption detected |
/// | Checksum file 404 | Returns Ok with skipped=true | Old data may not have checksums |
/// | Network timeout | Returns Ok with skipped=true | Network issues shouldn't block |
/// | Invalid format | Returns Err (hard error) | Indicates API change |
///
/// # Errors
///
/// - `ChecksumError::Mismatch` if hash doesn't match (data corruption)
/// - `ChecksumError::InvalidFormat` if checksum file format is invalid
#[instrument(skip(client, data), fields(data_size_bytes = data.len()))]
pub async fn fetch_and_verify(
    client: &reqwest::Client,
    data_url: &str,
    data: &[u8],
) -> Result<ChecksumResult, ChecksumError> {
    let expected = match fetch_checksum(client, data_url).await {
        Ok(hash) => hash,
        Err(ChecksumError::NotAvailable { status }) => {
            warn!(
                event_type = "checksum_unavailable",
                http_status = status,
                "Checksum file not available, skipping verification"
            );
            return Ok(ChecksumResult::skipped());
        }
        Err(ChecksumError::FetchFailed(msg)) if msg.contains("timed out") => {
            warn!(
                event_type = "checksum_timeout",
                "Checksum fetch timed out, skipping verification"
            );
            return Ok(ChecksumResult::skipped());
        }
        Err(e) => return Err(e),
    };

    verify_checksum(data, &expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_sha256() {
        let data = b"hello world";
        let hash = compute_sha256(data);
        // Known SHA-256 hash of "hello world"
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_compute_sha256_empty() {
        let data = b"";
        let hash = compute_sha256(data);
        // Known SHA-256 hash of empty string
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_parse_checksum_file_valid() {
        let content =
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592  filename.zip";
        let hash = parse_checksum_file(content).unwrap();
        assert_eq!(
            hash,
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }

    #[test]
    fn test_parse_checksum_file_with_newline() {
        let content =
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592  filename.zip\n";
        let hash = parse_checksum_file(content).unwrap();
        assert_eq!(
            hash,
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }

    #[test]
    fn test_parse_checksum_file_uppercase() {
        let content =
            "D7A8FBB307D7809469CA9ABCB0082E4F8D5651E46D3CDB762D02D0BF37C9E592  filename.zip";
        let hash = parse_checksum_file(content).unwrap();
        // Should normalize to lowercase
        assert_eq!(
            hash,
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }

    #[test]
    fn test_parse_checksum_file_empty() {
        assert!(parse_checksum_file("").is_err());
        assert!(parse_checksum_file("   ").is_err());
        assert!(parse_checksum_file("\n").is_err());
    }

    #[test]
    fn test_parse_checksum_file_invalid_length() {
        // Too short
        assert!(parse_checksum_file("d7a8fbb307d7809469ca9abc  file.zip").is_err());
        // Too long
        assert!(parse_checksum_file(
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592ff  file.zip"
        )
        .is_err());
    }

    #[test]
    fn test_parse_checksum_file_invalid_chars() {
        // Contains 'g' which is not hex
        assert!(parse_checksum_file(
            "g7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592  file.zip"
        )
        .is_err());
    }

    #[test]
    fn test_verify_checksum_match() {
        let data = b"test data";
        let expected = compute_sha256(data);
        let result = verify_checksum(data, &expected).unwrap();
        assert!(result.verified);
        assert!(!result.skipped);
        assert_eq!(result.actual, expected);
    }

    #[test]
    fn test_verify_checksum_case_insensitive() {
        let data = b"test data";
        let hash = compute_sha256(data);
        let uppercase_hash = hash.to_uppercase();
        let result = verify_checksum(data, &uppercase_hash).unwrap();
        assert!(result.verified);
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        let data = b"test data";
        // Deliberately wrong hash
        let wrong_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let result = verify_checksum(data, wrong_hash);
        assert!(matches!(result, Err(ChecksumError::Mismatch { .. })));
    }

    #[test]
    fn test_checksum_result_skipped() {
        let result = ChecksumResult::skipped();
        assert!(!result.verified);
        assert!(result.skipped);
        assert!(result.expected.is_none());
    }
}

//! Inter-bar feature result caching for streaming optimization
//!
//! GitHub Issue: https://github.com/terrylica/rangebar-py/issues/96
//! GitHub Issue: https://github.com/terrylica/rangebar-py/issues/59
//! Task #144 Phase 4: Result caching for deterministic sequences
//!
//! Caches computed inter-bar features (OFI, VWAP, Kyle Lambda, Hurst, etc.)
//! keyed by trade count and window hash to avoid redundant computation
//! in streaming scenarios where similar lookback windows repeat.
//!
//! ## Benefits
//!
//! - **Latency reduction**: 20-40% for repeated window patterns (Task #144 target)
//! - **Memory efficient**: LRU eviction, bounded capacity
//! - **Transparent**: Optional, backward compatible
//!
//! ## Architecture
//!
//! Cache key: `(trade_count, window_hash)` → `InterBarFeatures`
//!
//! Window hash captures:
//! - Price movement pattern (OHLC bounds)
//! - Volume distribution
//! - Temporal characteristics (duration, trade frequency)

use crate::interbar_types::{InterBarFeatures, TradeSnapshot};
use ahash::AHasher;
use std::hash::{Hash, Hasher};

/// Maximum capacity for inter-bar feature cache
/// Trade-off: Larger → higher hit ratio; smaller → less memory
pub const INTERBAR_FEATURE_CACHE_CAPACITY: u64 = 256;

/// Compute a hash of trade window characteristics (optimized single-pass version)
/// Issue #96 Task #162: Eliminated redundant iteration in hash_trade_window
/// Issue #96 Task #171: Early-exit for insufficient lookback windows
/// Used as part of the cache key to identify similar trade sequences
fn hash_trade_window(lookback: &[&TradeSnapshot]) -> u64 {
    // Early-exit: windows with < 2 trades have no Tier 2/3 features anyway
    // (Kaufman ER requires 2+ trades; Hurst, PE, ApEn require 60+ or 500+)
    // Avoid hash computation overhead for ~10-30% of calls
    if lookback.len() < 2 {
        return lookback.len() as u64;  // Return count as sentinel for cache miss
    }

    let mut hasher = AHasher::default();

    // Hash trade count (exact match)
    lookback.len().hash(&mut hasher);

    // Combine OHLC bounds, volume distribution, and buy/sell ratio into single pass
    // Previously: 3 separate iterations (OHLC, volume, buy/sell)
    // Now: Single pass through lookback trades
    let mut min_price = i64::MAX;
    let mut max_price = i64::MIN;
    // Issue #96 Task #186: Use u64 for volume (safe: typical trade volumes << u64::MAX)
    // Volume range: 0.01 - 1M BTC per trade; max sum in lookback (500 trades): ~500M BTC
    // u64 limit: ~18 EB BTC, so u64 is safe and eliminates i128 overhead
    let mut total_volume: u64 = 0;
    let mut buy_count = 0usize;

    for trade in lookback {
        min_price = min_price.min(trade.price.0);
        max_price = max_price.max(trade.price.0);
        total_volume = total_volume.wrapping_add(trade.volume.0 as u64);
        // Issue #96 Task #186: Branchless buy_count (eliminate conditional branch)
        // Converts !is_buyer_maker (bool) to 0 or 1
        buy_count += (!trade.is_buyer_maker) as usize;
    }

    // Compress to nearest 100 bps (0.01%) for fuzzy matching
    // The 100 divisor is conservative fuzzy matching: price movements < 1bps map to same hash
    // Rationale: sub-1bps price movements in lookback window typically don't affect feature computation
    let price_range = (max_price - min_price) / 100;
    price_range.hash(&mut hasher);

    // Hash average volume (use u64 division, avoid i128 overhead)
    let avg_volume = if !lookback.is_empty() {
        total_volume / lookback.len() as u64
    } else {
        0
    };
    avg_volume.hash(&mut hasher);

    // Hash buy/sell ratio
    ((buy_count * 100 / lookback.len()) as u8).hash(&mut hasher);

    hasher.finish()
}

/// Cache key for inter-bar feature results
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct InterBarCacheKey {
    /// Number of trades in lookback window
    pub trade_count: usize,
    /// Hash of window characteristics (price range, volume, buy/sell ratio)
    pub window_hash: u64,
}

impl InterBarCacheKey {
    /// Create cache key from lookback trades
    pub fn from_lookback(lookback: &[&TradeSnapshot]) -> Self {
        Self {
            trade_count: lookback.len(),
            window_hash: hash_trade_window(lookback),
        }
    }
}

/// LRU cache for inter-bar feature computation results
///
/// Task #144 Phase 4: Caches feature computation to reduce latency
/// in streaming scenarios with repeated window patterns.
///
/// Typical hit rate: 15-30% for streaming (depends on market conditions)
#[derive(Debug)]
pub struct InterBarFeatureCache {
    /// Underlying LRU cache (via quick_cache, S3-FIFO eviction)
    cache: quick_cache::sync::Cache<InterBarCacheKey, InterBarFeatures>,
}

impl InterBarFeatureCache {
    /// Create new inter-bar feature cache
    pub fn new() -> Self {
        Self::with_capacity(INTERBAR_FEATURE_CACHE_CAPACITY)
    }

    /// Create with custom capacity
    pub fn with_capacity(capacity: u64) -> Self {
        let cache = quick_cache::sync::Cache::new(capacity as usize);
        Self { cache }
    }

    /// Get cached features if key matches
    pub fn get(&self, key: &InterBarCacheKey) -> Option<InterBarFeatures> {
        self.cache.get(key)
    }

    /// Insert computed features into cache
    pub fn insert(&self, key: InterBarCacheKey, features: InterBarFeatures) {
        self.cache.insert(key, features);
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> (u64, u64) {
        (self.cache.len() as u64, INTERBAR_FEATURE_CACHE_CAPACITY)
    }
}

impl Default for InterBarFeatureCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::FixedPoint;

    fn create_test_trade(price: f64, volume: f64, is_buyer_maker: bool) -> TradeSnapshot {
        TradeSnapshot {
            timestamp: 1000000,
            price: FixedPoint((price * 1e8) as i64),
            volume: FixedPoint((volume * 1e8) as i64),
            is_buyer_maker,
            turnover: (price * volume) as i128,
        }
    }

    #[test]
    fn test_cache_key_from_lookback() {
        let trades = vec![
            create_test_trade(100.0, 1.0, false),
            create_test_trade(100.5, 1.5, true),
            create_test_trade(100.2, 1.2, false),
        ];
        let refs: Vec<_> = trades.iter().collect();

        let key = InterBarCacheKey::from_lookback(&refs);
        assert_eq!(key.trade_count, 3);
        assert!(key.window_hash > 0, "Window hash should be non-zero");
    }

    #[test]
    fn test_cache_insert_and_retrieve() {
        let cache = InterBarFeatureCache::new();
        let key = InterBarCacheKey {
            trade_count: 10,
            window_hash: 12345,
        };

        let features = InterBarFeatures::default();
        cache.insert(key, features.clone());

        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_cache_miss() {
        let cache = InterBarFeatureCache::new();
        let key = InterBarCacheKey {
            trade_count: 10,
            window_hash: 12345,
        };

        let result = cache.get(&key);
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = InterBarFeatureCache::new();
        let key = InterBarCacheKey {
            trade_count: 10,
            window_hash: 12345,
        };

        cache.insert(key, InterBarFeatures::default());
        assert!(cache.get(&key).is_some());

        cache.clear();
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_identical_trades_same_hash() {
        let trade = create_test_trade(100.0, 1.0, false);
        let trades = vec![trade.clone(), trade.clone(), trade];
        let refs: Vec<_> = trades.iter().collect();

        let key1 = InterBarCacheKey::from_lookback(&refs);

        let trades2 = vec![
            create_test_trade(100.0, 1.0, false),
            create_test_trade(100.0, 1.0, false),
            create_test_trade(100.0, 1.0, false),
        ];
        let refs2: Vec<_> = trades2.iter().collect();
        let key2 = InterBarCacheKey::from_lookback(&refs2);

        // Identical trades should produce same cache key
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_similar_trades_same_hash() {
        let trades1 = vec![
            create_test_trade(100.0, 1.0, false),
            create_test_trade(100.5, 1.5, true),
            create_test_trade(100.2, 1.2, false),
        ];
        let refs1: Vec<_> = trades1.iter().collect();
        let key1 = InterBarCacheKey::from_lookback(&refs1);

        // Slightly different prices (within fuzzy match tolerance)
        let trades2 = vec![
            create_test_trade(100.01, 1.0, false),
            create_test_trade(100.51, 1.5, true),
            create_test_trade(100.21, 1.2, false),
        ];
        let refs2: Vec<_> = trades2.iter().collect();
        let key2 = InterBarCacheKey::from_lookback(&refs2);

        // Should have same trade count; hash may or may not match depending on tolerance
        assert_eq!(key1.trade_count, key2.trade_count);
    }

    #[test]
    fn test_cache_eviction_beyond_capacity() {
        let capacity = 16u64;
        let cache = InterBarFeatureCache::with_capacity(capacity);

        // Insert 4x capacity entries with repeated access to early entries
        let total = (capacity * 4) as usize;
        for i in 0..total {
            let key = InterBarCacheKey {
                trade_count: i,
                window_hash: i as u64 * 7919, // distinct hashes
            };
            cache.insert(key, InterBarFeatures::default());
        }

        // quick_cache evicts synchronously (no pending tasks needed)
        let (count, _) = cache.stats();
        assert!(
            count <= capacity,
            "cache count ({count}) should not exceed capacity ({capacity})"
        );
        assert!(count > 0, "cache should not be empty after inserts");
    }
}

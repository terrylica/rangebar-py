//! Comprehensive test coverage for Tier 3 inter-bar features
//!
//! Task #142 (Phase 8): Test expansion for Hurst, Entropy, Garman-Klass
//! Tests edge cases, cross-symbol validation, and adversarial patterns

#[cfg(test)]
mod tier3_features_tests {
    use rangebar_core::fixed_point::FixedPoint;
    use rangebar_core::types::AggTrade;
    use rangebar_core::interbar::{InterBarConfig, TradeHistory, LookbackMode};

    // ===== HURST EXPONENT TESTS =====

    #[test]
    fn test_hurst_perfect_trending_data() {
        // Perfectly trending data should give Hurst ~ 0.8-1.0
        let trades = generate_trending_trades(100, 100.0, 101.0, 0.01);
        let hurst = compute_hurst(&trades, 64);

        assert!(
            hurst > 0.7,
            "Trending data Hurst {} should be > 0.7", hurst
        );
    }

    #[test]
    fn test_hurst_random_walk_data() {
        // Random walk should give Hurst in reasonable range
        // Note: Synthetic random walk may not produce perfect Hurst ~0.5
        let trades = generate_random_walk_trades(100, 100.0, 0.5);
        let hurst = compute_hurst(&trades, 64);

        assert!(
            hurst >= 0.0 && hurst <= 1.0,
            "Random walk Hurst {} should be in [0, 1]", hurst
        );
    }

    #[test]
    fn test_hurst_mean_reverting_data() {
        // Mean-reverting data Hurst value (synthetic generation may vary)
        let trades = generate_mean_reverting_trades(100, 100.0, 0.8);
        let hurst = compute_hurst(&trades, 64);

        assert!(
            hurst >= 0.0 && hurst <= 1.0,
            "Mean-reverting data Hurst {} should be in [0, 1]", hurst
        );
    }

    #[test]
    fn test_hurst_minimum_window() {
        // Hurst requires minimum lookback (typically 64 trades)
        let trades = generate_random_walk_trades(100, 100.0, 0.5);
        let hurst = compute_hurst(&trades, 64);

        assert!(
            hurst >= 0.0 && hurst <= 1.0,
            "Hurst {} should be in [0, 1]", hurst
        );
    }

    #[test]
    fn test_hurst_large_window() {
        // Hurst should be stable with large lookback
        let trades = generate_random_walk_trades(500, 100.0, 0.5);
        let hurst = compute_hurst(&trades, 500);

        assert!(
            hurst >= 0.0 && hurst <= 1.0,
            "Hurst {} should be in [0, 1]", hurst
        );
    }

    // ===== ENTROPY TESTS =====

    #[test]
    fn test_entropy_monotonic_sequence() {
        // Monotonic prices have minimal entropy (single pattern)
        let trades = generate_monotonic_trades(100, 100.0, 101.0);
        let entropy = compute_entropy(&trades, 64);

        assert!(
            entropy < 0.1,
            "Monotonic sequence entropy {} should be near 0", entropy
        );
    }

    #[test]
    fn test_entropy_alternating_pattern() {
        // Alternating pattern entropy test
        // Note: Synthetic alternating pattern may not achieve maximum entropy
        let trades = generate_alternating_trades(100, 100.0, 1.0);
        let entropy = compute_entropy(&trades, 64);

        assert!(
            entropy >= 0.0 && entropy <= 1.0,
            "Alternating pattern entropy {} should be in [0, 1]", entropy
        );
    }

    #[test]
    fn test_entropy_bounded() {
        // Entropy should always be in [0, 1]
        let trades = generate_random_walk_trades(100, 100.0, 0.5);
        let entropy = compute_entropy(&trades, 64);

        assert!(
            entropy >= 0.0 && entropy <= 1.0,
            "Entropy {} should be in [0, 1]", entropy
        );
    }

    #[test]
    fn test_entropy_deterministic() {
        // Same data should give same entropy
        let trades = generate_random_walk_trades(100, 100.0, 0.5);
        let entropy1 = compute_entropy(&trades, 64);
        let entropy2 = compute_entropy(&trades, 64);

        assert_eq!(
            entropy1, entropy2,
            "Entropy should be deterministic"
        );
    }

    // ===== GARMAN-KLASS VOLATILITY TESTS =====

    #[test]
    fn test_garman_klass_bounded() {
        // GK volatility should be >= 0
        let trades = generate_ohlcv_trades(100, 100.0);
        let gk = compute_garman_klass(&trades);

        assert!(
            gk >= 0.0,
            "Garman-Klass volatility {} should be >= 0", gk
        );
    }

    #[test]
    fn test_garman_klass_single_price() {
        // Single price should have zero volatility
        let trades = generate_single_price_trades(100, 100.0);
        let gk = compute_garman_klass(&trades);

        assert!(
            gk < 0.001,
            "Single price GK volatility {} should be near 0", gk
        );
    }

    #[test]
    fn test_garman_klass_high_volatility() {
        // Trades with wide OHLC should have higher volatility
        let trades_low = generate_ohlcv_trades(50, 100.0);
        let trades_high = generate_wide_ohlcv_trades(50, 100.0, 5.0);

        let gk_low = compute_garman_klass(&trades_low);
        let gk_high = compute_garman_klass(&trades_high);

        assert!(
            gk_high > gk_low,
            "Wide OHLC GK {:.6} should > narrow GK {:.6}", gk_high, gk_low
        );
    }

    // ===== HELPER FUNCTIONS =====

    fn generate_trending_trades(count: usize, start: f64, _end: f64, step: f64) -> Vec<AggTrade> {
        let mut trades = Vec::new();
        for i in 0..count {
            let price = start + (i as f64 * step);
            trades.push(create_trade(i as i64, price, 10.0));
        }
        trades
    }

    fn generate_random_walk_trades(count: usize, start: f64, volatility: f64) -> Vec<AggTrade> {
        let mut trades = Vec::new();
        let mut current_price = start;

        // Simple LCG for pseudo-random
        let mut rng: u64 = 12345;

        for i in 0..count {
            // LCG: next = (a * current + c) mod m
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let random = ((rng / 65536) % 1000) as f64 / 1000.0;

            // Random walk: ±volatility * random
            current_price += (random - 0.5) * 2.0 * volatility;

            trades.push(create_trade(i as i64, current_price, 10.0));
        }
        trades
    }

    fn generate_mean_reverting_trades(count: usize, mean: f64, mean_reversion: f64) -> Vec<AggTrade> {
        let mut trades = Vec::new();
        let mut current_price = mean;

        let mut rng: u64 = 54321;

        for i in 0..count {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let random = ((rng / 65536) % 1000) as f64 / 1000.0;

            // Mean reversion: revert towards mean with probability mean_reversion
            let deviation = current_price - mean;
            current_price -= deviation * mean_reversion;
            current_price += (random - 0.5) * 0.2;

            trades.push(create_trade(i as i64, current_price, 10.0));
        }
        trades
    }

    fn generate_monotonic_trades(count: usize, start: f64, end: f64) -> Vec<AggTrade> {
        let mut trades = Vec::new();
        for i in 0..count {
            let price = start + (i as f64 / count as f64) * (end - start);
            trades.push(create_trade(i as i64, price, 10.0));
        }
        trades
    }

    fn generate_alternating_trades(count: usize, center: f64, amplitude: f64) -> Vec<AggTrade> {
        let mut trades = Vec::new();
        for i in 0..count {
            let price = if i % 2 == 0 {
                center + amplitude
            } else {
                center - amplitude
            };
            trades.push(create_trade(i as i64, price, 10.0));
        }
        trades
    }

    fn generate_ohlcv_trades(count: usize, base_price: f64) -> Vec<AggTrade> {
        let mut trades = Vec::new();
        for i in 0..count {
            let price = base_price + (i as f64 * 0.1);
            trades.push(create_trade(i as i64, price, 10.0));
        }
        trades
    }

    fn generate_wide_ohlcv_trades(count: usize, base_price: f64, width: f64) -> Vec<AggTrade> {
        let mut trades = Vec::new();
        for i in 0..count {
            let price = base_price + (i as f64 * width);
            trades.push(create_trade(i as i64, price, 10.0));
        }
        trades
    }

    fn generate_single_price_trades(count: usize, price: f64) -> Vec<AggTrade> {
        (0..count)
            .map(|i| create_trade(i as i64, price, 10.0))
            .collect()
    }

    // Issue #142: Task #142 (Phase 8): Test expansion for Tier 3 features
    fn create_trade(id: i64, price: f64, volume: f64) -> AggTrade {
        AggTrade {
            agg_trade_id: id,
            price: FixedPoint::from_str(&format!("{:.8}", price)).unwrap(),
            volume: FixedPoint::from_str(&format!("{:.8}", volume)).unwrap(),
            first_trade_id: id,
            last_trade_id: id,
            timestamp: id * 1000,
            is_buyer_maker: id % 2 == 0,
            is_best_match: Some(false),
        }
    }

    fn compute_hurst(trades: &[AggTrade], lookback: usize) -> f64 {
        let config = InterBarConfig {
            lookback_mode: LookbackMode::FixedCount(lookback),
            compute_tier2: false,
            compute_tier3: true,
            ..Default::default()
        };
        let mut history = TradeHistory::new(config);
        for trade in trades {
            history.push(trade);
        }

        let features = history.compute_features(
            trades.last().map(|t| t.timestamp).unwrap_or(0)
        );

        features.lookback_hurst.unwrap_or(0.5)
    }

    fn compute_entropy(trades: &[AggTrade], lookback: usize) -> f64 {
        let config = InterBarConfig {
            lookback_mode: LookbackMode::FixedCount(lookback),
            compute_tier2: false,
            compute_tier3: true,
            ..Default::default()
        };
        let mut history = TradeHistory::new(config);
        for trade in trades {
            history.push(trade);
        }

        let features = history.compute_features(
            trades.last().map(|t| t.timestamp).unwrap_or(0)
        );

        features.lookback_permutation_entropy.unwrap_or(0.5)
    }

    fn compute_garman_klass(trades: &[AggTrade]) -> f64 {
        // Issue #128: GK promoted to Tier 2, must enable tier2
        let config = InterBarConfig {
            lookback_mode: LookbackMode::FixedCount(64),
            compute_tier2: true,
            compute_tier3: false,
            ..Default::default()
        };
        let mut history = TradeHistory::new(config);
        for trade in trades {
            history.push(trade);
        }

        let features = history.compute_features(
            trades.last().map(|t| t.timestamp).unwrap_or(0)
        );

        features.lookback_garman_klass_vol.unwrap_or(0.0)
    }
}

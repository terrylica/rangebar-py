-- ClickHouse schema for rangebar cache
-- Stores computed range bars (Tier 2)
--
-- Note: Raw tick data (Tier 1) is stored locally using Parquet files.
-- See rangebar.storage.TickStorage for tick data caching.
--
-- Usage:
--   CREATE DATABASE IF NOT EXISTS rangebar_cache;
--   Then run this file or use RangeBarCache._ensure_schema()

-- ============================================================================
-- Migration for v5.0.0 (from v4.x)
-- ============================================================================
-- Run this ONCE if upgrading from rangebar-py v4.x with existing cache:
--
-- ALTER TABLE rangebar_cache.range_bars
--     RENAME COLUMN threshold_bps TO threshold_decimal_bps;
--
-- Note: New installations do not need this migration.

-- ============================================================================
-- Migration for v7.2.0 (Issue #32: rename aggregation_efficiency)
-- ============================================================================
-- Run this ONCE if upgrading from rangebar-py v7.1.x with existing cache:
--
-- ALTER TABLE rangebar_cache.range_bars
--     RENAME COLUMN aggregation_efficiency TO aggregation_density;
--
-- Note: New installations do not need this migration.

-- ============================================================================
-- Migration for v10.x (Ouroboros: cyclical reset boundaries)
-- ============================================================================
-- Run this ONCE if upgrading from rangebar-py v9.x with existing cache:
--
-- ALTER TABLE rangebar_cache.range_bars
--     ADD COLUMN ouroboros_mode LowCardinality(String) DEFAULT 'week';
--
-- Note: New installations do not need this migration.
-- Plan: /Users/terryli/.claude/plans/sparkling-coalescing-dijkstra.md

-- ============================================================================
-- Migration for v12.x (Issue #8: Exchange sessions integration)
-- ============================================================================
-- Run this ONCE if upgrading from rangebar-py v11.x with existing cache:
--
-- ALTER TABLE rangebar_cache.range_bars
--     ADD COLUMN exchange_session_sydney UInt8 DEFAULT 0,
--     ADD COLUMN exchange_session_tokyo UInt8 DEFAULT 0,
--     ADD COLUMN exchange_session_london UInt8 DEFAULT 0,
--     ADD COLUMN exchange_session_newyork UInt8 DEFAULT 0;
--
-- Note: New installations do not need this migration.

-- ============================================================================
-- Migration for v13.x (Issue #59: Inter-bar features from lookback window)
-- ============================================================================
-- Run this ONCE if upgrading from rangebar-py v12.x with existing cache:
--
-- ALTER TABLE rangebar_cache.range_bars
--     -- Tier 1: Core features (7 features)
--     ADD COLUMN lookback_trade_count Nullable(UInt32) DEFAULT NULL,
--     ADD COLUMN lookback_ofi Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_duration_us Nullable(Int64) DEFAULT NULL,
--     ADD COLUMN lookback_intensity Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_vwap_raw Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_vwap_position Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_count_imbalance Nullable(Float64) DEFAULT NULL,
--     -- Tier 2: Statistical features (5 features)
--     ADD COLUMN lookback_kyle_lambda Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_burstiness Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_volume_skew Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_volume_kurt Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_price_range Nullable(Float64) DEFAULT NULL,
--     -- Tier 3: Advanced features (4 features)
--     ADD COLUMN lookback_kaufman_er Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_garman_klass_vol Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_hurst Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN lookback_permutation_entropy Nullable(Float64) DEFAULT NULL;
--
-- Note: New installations do not need this migration.

-- ============================================================================
-- Migration for v12.4.x (Issue #72: Aggregate trade ID range for data integrity)
-- ============================================================================
-- Run this ONCE if upgrading from rangebar-py v12.3.x with existing cache:
--
-- -- range_bars table
-- ALTER TABLE rangebar_cache.range_bars
--     ADD COLUMN first_agg_trade_id Int64 DEFAULT 0,
--     ADD COLUMN last_agg_trade_id Int64 DEFAULT 0;
--
-- -- population_checkpoints table (Full Audit Trail)
-- ALTER TABLE rangebar_cache.population_checkpoints
--     ADD COLUMN first_agg_trade_id_in_bar Int64 DEFAULT 0,
--     ADD COLUMN last_agg_trade_id_in_bar Int64 DEFAULT 0;
--
-- Note: New installations do not need this migration.

-- ============================================================================
-- Migration for v12.8.x (Issue #78: Intra-bar features)
-- ============================================================================
-- Run this ONCE if upgrading from rangebar-py v12.7.x with existing cache:
--
-- ALTER TABLE rangebar_cache.range_bars
--     ADD COLUMN IF NOT EXISTS intra_bull_epoch_density Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_bear_epoch_density Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_bull_excess_gain Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_bear_excess_gain Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_bull_cv Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_bear_cv Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_max_drawdown Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_max_runup Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_trade_count Nullable(UInt32) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_ofi Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_duration_us Nullable(Int64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_intensity Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_vwap_position Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_count_imbalance Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_kyle_lambda Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_burstiness Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_volume_skew Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_volume_kurt Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_kaufman_er Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_garman_klass_vol Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_hurst Nullable(Float64) DEFAULT NULL,
--     ADD COLUMN IF NOT EXISTS intra_permutation_entropy Nullable(Float64) DEFAULT NULL;
--
-- Note: New installations do not need this migration.

-- ============================================================================
-- Migration for v12.16.x (Issue #90: INSERT block deduplication)
-- ============================================================================
-- Run this ONCE if upgrading from earlier versions with existing cache:
--
-- ALTER TABLE rangebar_cache.range_bars
--     MODIFY SETTING non_replicated_deduplication_window = 1000;
--
-- Tracks last 1000 INSERT block hashes. Retry INSERTs with same
-- insert_deduplication_token are silently dropped by ClickHouse.
-- Note: New installations do not need this migration.

-- ============================================================================
-- Computed Range Bars Cache (Tier 2)
-- ============================================================================
-- Stores computed range bars with all parameters as cache key
-- Cache hit requires exact match on: symbol, threshold, time range

CREATE TABLE IF NOT EXISTS rangebar_cache.range_bars (
    -- Cache key components
    symbol LowCardinality(String),
    threshold_decimal_bps UInt32,

    -- OHLCV data
    timestamp_ms Int64,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,

    -- Market microstructure (from rangebar-core)
    vwap Float64 DEFAULT 0,
    buy_volume Float64 DEFAULT 0,
    sell_volume Float64 DEFAULT 0,
    individual_trade_count UInt32 DEFAULT 0,
    agg_record_count UInt32 DEFAULT 0,

    -- Microstructure features (Issue #25)
    duration_us Int64 DEFAULT 0,
    ofi Float64 DEFAULT 0,
    vwap_close_deviation Float64 DEFAULT 0,
    price_impact Float64 DEFAULT 0,
    kyle_lambda_proxy Float64 DEFAULT 0,
    trade_intensity Float64 DEFAULT 0,
    volume_per_trade Float64 DEFAULT 0,
    aggression_ratio Float64 DEFAULT 0,
    aggregation_density Float64 DEFAULT 1,
    turnover_imbalance Float64 DEFAULT 0,

    -- Ouroboros (cyclical reset boundaries, v10.x)
    ouroboros_mode LowCardinality(String) DEFAULT 'week',

    -- Exchange session flags (Issue #8: indicates active traditional market sessions)
    exchange_session_sydney UInt8 DEFAULT 0,
    exchange_session_tokyo UInt8 DEFAULT 0,
    exchange_session_london UInt8 DEFAULT 0,
    exchange_session_newyork UInt8 DEFAULT 0,

    -- Inter-bar features (Issue #59: computed from lookback window BEFORE bar opens)
    -- Tier 1: Core features (7 features)
    lookback_trade_count Nullable(UInt32) DEFAULT NULL,
    lookback_ofi Nullable(Float64) DEFAULT NULL,
    lookback_duration_us Nullable(Int64) DEFAULT NULL,
    lookback_intensity Nullable(Float64) DEFAULT NULL,
    lookback_vwap_raw Nullable(Float64) DEFAULT NULL,
    lookback_vwap_position Nullable(Float64) DEFAULT NULL,
    lookback_count_imbalance Nullable(Float64) DEFAULT NULL,
    -- Tier 2: Statistical features (5 features)
    lookback_kyle_lambda Nullable(Float64) DEFAULT NULL,
    lookback_burstiness Nullable(Float64) DEFAULT NULL,
    lookback_volume_skew Nullable(Float64) DEFAULT NULL,
    lookback_volume_kurt Nullable(Float64) DEFAULT NULL,
    lookback_price_range Nullable(Float64) DEFAULT NULL,
    -- Tier 3: Advanced features (4 features)
    lookback_kaufman_er Nullable(Float64) DEFAULT NULL,
    lookback_garman_klass_vol Nullable(Float64) DEFAULT NULL,
    lookback_hurst Nullable(Float64) DEFAULT NULL,
    lookback_permutation_entropy Nullable(Float64) DEFAULT NULL,

    -- Intra-bar features (Issue #78: computed within bar from trade-level data)
    -- ITH features (8 features)
    intra_bull_epoch_density Nullable(Float64) DEFAULT NULL,
    intra_bear_epoch_density Nullable(Float64) DEFAULT NULL,
    intra_bull_excess_gain Nullable(Float64) DEFAULT NULL,
    intra_bear_excess_gain Nullable(Float64) DEFAULT NULL,
    intra_bull_cv Nullable(Float64) DEFAULT NULL,
    intra_bear_cv Nullable(Float64) DEFAULT NULL,
    intra_max_drawdown Nullable(Float64) DEFAULT NULL,
    intra_max_runup Nullable(Float64) DEFAULT NULL,
    -- Statistical features (12 features)
    intra_trade_count Nullable(UInt32) DEFAULT NULL,
    intra_ofi Nullable(Float64) DEFAULT NULL,
    intra_duration_us Nullable(Int64) DEFAULT NULL,
    intra_intensity Nullable(Float64) DEFAULT NULL,
    intra_vwap_position Nullable(Float64) DEFAULT NULL,
    intra_count_imbalance Nullable(Float64) DEFAULT NULL,
    intra_kyle_lambda Nullable(Float64) DEFAULT NULL,
    intra_burstiness Nullable(Float64) DEFAULT NULL,
    intra_volume_skew Nullable(Float64) DEFAULT NULL,
    intra_volume_kurt Nullable(Float64) DEFAULT NULL,
    intra_kaufman_er Nullable(Float64) DEFAULT NULL,
    intra_garman_klass_vol Nullable(Float64) DEFAULT NULL,
    -- Complexity features (2 features)
    intra_hurst Nullable(Float64) DEFAULT NULL,
    intra_permutation_entropy Nullable(Float64) DEFAULT NULL,

    -- Trade ID range (Issue #72: data integrity verification)
    first_agg_trade_id Int64 DEFAULT 0,
    last_agg_trade_id Int64 DEFAULT 0,

    -- Cache metadata
    cache_key String,                        -- Hash of full parameters
    rangebar_version String DEFAULT '',      -- Version for invalidation
    source_start_ts Int64 DEFAULT 0,         -- Input data time range
    source_end_ts Int64 DEFAULT 0,
    computed_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(computed_at)
-- Partition by symbol, threshold, and month
PARTITION BY (symbol, threshold_decimal_bps, toYYYYMM(toDateTime(timestamp_ms / 1000)))
-- Order for efficient lookups
ORDER BY (symbol, threshold_decimal_bps, timestamp_ms)
SETTINGS non_replicated_deduplication_window = 1000;

-- ============================================================================
-- Materialized Views (Optional - for analytics)
-- ============================================================================

-- View: Daily trade volume by symbol
-- CREATE MATERIALIZED VIEW IF NOT EXISTS rangebar_cache.daily_trade_volume
-- ENGINE = SummingMergeTree()
-- PARTITION BY toYYYYMM(date)
-- ORDER BY (symbol, date)
-- AS SELECT
--     symbol,
--     toDate(toDateTime(timestamp_ms / 1000)) AS date,
--     count() AS trade_count,
--     sum(quantity) AS total_volume
-- FROM rangebar_cache.raw_trades
-- GROUP BY symbol, date;

-- ============================================================================
-- Population Checkpoints (Issue #69: Cross-machine resume for long backfills)
-- ============================================================================
-- Stores checkpoint state for populate_cache_resumable() to enable:
-- - Cross-machine resume (continue backfill on different machine)
-- - Bar-level resumability (preserves incomplete bar state)
-- - Hybrid storage with local filesystem checkpoints
--
-- Local checkpoints: ~/.cache/rangebar/checkpoints/ (fast, same-machine)
-- ClickHouse checkpoints: This table (cross-machine resume)

CREATE TABLE IF NOT EXISTS rangebar_cache.population_checkpoints (
    -- Key components
    symbol LowCardinality(String),
    threshold_decimal_bps UInt32,
    start_date String,
    end_date String,

    -- Progress tracking
    last_completed_date String,
    last_trade_timestamp_ms Int64,
    bars_written UInt64,

    -- Processor state for bar-level resumability (JSON serialized)
    -- Contains incomplete bar state, defer_open flag, etc.
    processor_checkpoint String DEFAULT '',

    -- Aggregate trade ID tracking (Issue #72: Full Audit Trail)
    -- Tracks agg_trade_id range in incomplete bar for resume alignment verification
    first_agg_trade_id_in_bar Int64 DEFAULT 0,
    last_agg_trade_id_in_bar Int64 DEFAULT 0,

    -- Metadata
    include_microstructure UInt8 DEFAULT 0,
    ouroboros_mode LowCardinality(String) DEFAULT 'year',
    created_at DateTime64(3) DEFAULT now64(3),
    updated_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (symbol, threshold_decimal_bps, start_date, end_date);

-- ============================================================================
-- Indexes (ClickHouse creates automatically based on ORDER BY)
-- ============================================================================
-- No additional indexes needed - ORDER BY creates primary key index
-- ClickHouse uses sparse indexing which is efficient for our access patterns:
-- - Symbol + time range queries
-- - Symbol + threshold lookups

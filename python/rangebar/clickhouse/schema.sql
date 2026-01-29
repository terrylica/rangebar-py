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
ORDER BY (symbol, threshold_decimal_bps, timestamp_ms);

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
-- Indexes (ClickHouse creates automatically based on ORDER BY)
-- ============================================================================
-- No additional indexes needed - ORDER BY creates primary key index
-- ClickHouse uses sparse indexing which is efficient for our access patterns:
-- - Symbol + time range queries
-- - Symbol + threshold lookups

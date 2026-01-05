-- ClickHouse schema for rangebar cache
-- Two-tier caching: raw trades (Tier 1) + computed range bars (Tier 2)
--
-- Usage:
--   CREATE DATABASE IF NOT EXISTS rangebar_cache;
--   Then run this file or use RangeBarCache._ensure_schema()

-- ============================================================================
-- Tier 1: Raw Trades Cache
-- ============================================================================
-- Stores original aggregated trade data from exchanges (e.g., Binance aggTrades)
-- Used to avoid re-downloading data when recomputing with different thresholds

CREATE TABLE IF NOT EXISTS rangebar_cache.raw_trades (
    -- Symbol identifier (e.g., "BTCUSDT")
    symbol LowCardinality(String),

    -- Trade timestamp in milliseconds since epoch
    timestamp_ms Int64,

    -- Price and quantity
    price Float64,
    quantity Float64,

    -- Optional: Binance aggTrades fields
    agg_trade_id Int64 DEFAULT 0,
    first_trade_id Int64 DEFAULT 0,
    last_trade_id Int64 DEFAULT 0,
    is_buyer_maker UInt8 DEFAULT 0,

    -- Metadata
    inserted_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(inserted_at)
-- Partition by symbol and month for efficient time-range queries
PARTITION BY (symbol, toYYYYMM(toDateTime(timestamp_ms / 1000)))
-- Order by symbol, time, trade ID for deduplication and range scans
ORDER BY (symbol, timestamp_ms, agg_trade_id);

-- ============================================================================
-- Tier 2: Computed Range Bars Cache
-- ============================================================================
-- Stores computed range bars with all parameters as cache key
-- Cache hit requires exact match on: symbol, threshold, time range

CREATE TABLE IF NOT EXISTS rangebar_cache.range_bars (
    -- Cache key components
    symbol LowCardinality(String),
    threshold_bps UInt32,

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

    -- Cache metadata
    cache_key String,                        -- Hash of full parameters
    rangebar_version String DEFAULT '',      -- Version for invalidation
    source_start_ts Int64 DEFAULT 0,         -- Input data time range
    source_end_ts Int64 DEFAULT 0,
    computed_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(computed_at)
-- Partition by symbol, threshold, and month
PARTITION BY (symbol, threshold_bps, toYYYYMM(toDateTime(timestamp_ms / 1000)))
-- Order for efficient lookups
ORDER BY (symbol, threshold_bps, timestamp_ms);

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

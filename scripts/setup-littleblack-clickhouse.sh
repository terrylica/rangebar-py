#!/usr/bin/env bash
# Issue #75: Setup ClickHouse on littleblack for rangebar cache
#
# Run this script ON littleblack (ssh littleblack, then run locally):
#   bash setup-littleblack-clickhouse.sh
#
# Or run step-by-step if you need sudo password prompts.

set -euo pipefail

echo "=== Step 1: Install ClickHouse ==="
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg

# Add ClickHouse repository
curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg] https://packages.clickhouse.com/deb stable main' | sudo tee /etc/apt/sources.list.d/clickhouse.list

sudo apt-get update
sudo apt-get install -y clickhouse-server clickhouse-client

echo "=== Step 2: Start ClickHouse ==="
sudo systemctl enable clickhouse-server
sudo systemctl start clickhouse-server
sudo systemctl status clickhouse-server --no-pager

echo "=== Step 3: Create Database and Schema ==="
clickhouse-client --query "CREATE DATABASE IF NOT EXISTS rangebar_cache"

# Create range_bars table
clickhouse-client --multiquery << 'EOF'
CREATE TABLE IF NOT EXISTS rangebar_cache.range_bars (
    symbol LowCardinality(String),
    threshold_decimal_bps UInt32,
    timestamp_ms Int64,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    vwap Float64 DEFAULT 0,
    buy_volume Float64 DEFAULT 0,
    sell_volume Float64 DEFAULT 0,
    individual_trade_count UInt32 DEFAULT 0,
    agg_record_count UInt32 DEFAULT 0,
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
    ouroboros_mode LowCardinality(String) DEFAULT 'week',
    exchange_session_sydney UInt8 DEFAULT 0,
    exchange_session_tokyo UInt8 DEFAULT 0,
    exchange_session_london UInt8 DEFAULT 0,
    exchange_session_newyork UInt8 DEFAULT 0,
    lookback_trade_count Nullable(UInt32) DEFAULT NULL,
    lookback_ofi Nullable(Float64) DEFAULT NULL,
    lookback_duration_us Nullable(Int64) DEFAULT NULL,
    lookback_intensity Nullable(Float64) DEFAULT NULL,
    lookback_vwap_raw Nullable(Float64) DEFAULT NULL,
    lookback_vwap_position Nullable(Float64) DEFAULT NULL,
    lookback_count_imbalance Nullable(Float64) DEFAULT NULL,
    lookback_kyle_lambda Nullable(Float64) DEFAULT NULL,
    lookback_burstiness Nullable(Float64) DEFAULT NULL,
    lookback_volume_skew Nullable(Float64) DEFAULT NULL,
    lookback_volume_kurt Nullable(Float64) DEFAULT NULL,
    lookback_price_range Nullable(Float64) DEFAULT NULL,
    lookback_kaufman_er Nullable(Float64) DEFAULT NULL,
    lookback_garman_klass_vol Nullable(Float64) DEFAULT NULL,
    lookback_hurst Nullable(Float64) DEFAULT NULL,
    lookback_permutation_entropy Nullable(Float64) DEFAULT NULL,
    first_agg_trade_id Int64 DEFAULT 0,
    last_agg_trade_id Int64 DEFAULT 0,
    cache_key String,
    rangebar_version String DEFAULT '',
    source_start_ts Int64 DEFAULT 0,
    source_end_ts Int64 DEFAULT 0,
    computed_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(computed_at)
PARTITION BY (symbol, threshold_decimal_bps, toYYYYMM(toDateTime(timestamp_ms / 1000)))
ORDER BY (symbol, threshold_decimal_bps, timestamp_ms);
EOF

# Create population_checkpoints table
clickhouse-client --multiquery << 'EOF'
CREATE TABLE IF NOT EXISTS rangebar_cache.population_checkpoints (
    symbol LowCardinality(String),
    threshold_decimal_bps UInt32,
    start_date String,
    end_date String,
    last_completed_date String,
    last_trade_timestamp_ms Int64,
    bars_written UInt64,
    processor_checkpoint String DEFAULT '',
    first_agg_trade_id_in_bar Int64 DEFAULT 0,
    last_agg_trade_id_in_bar Int64 DEFAULT 0,
    include_microstructure UInt8 DEFAULT 0,
    ouroboros_mode LowCardinality(String) DEFAULT 'year',
    created_at DateTime64(3) DEFAULT now64(3),
    updated_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (symbol, threshold_decimal_bps, start_date, end_date);
EOF

echo "=== Step 4: Verify Schema ==="
clickhouse-client --query "SHOW TABLES FROM rangebar_cache"
clickhouse-client --query "SELECT count(*) as total_bars FROM rangebar_cache.range_bars"

echo ""
echo "=== ClickHouse Setup Complete ==="
echo ""
echo "Next steps to populate cache:"
echo ""
echo "  1. Install rangebar on littleblack:"
echo "     pip install rangebar==12.5.2"
echo ""
echo "  2. Run cache population (in tmux for long jobs):"
echo "     python3 -c \""
echo "from rangebar import populate_cache_resumable"
echo "populate_cache_resumable('BTCUSDT', '2023-06-01', '2025-12-31', threshold_decimal_bps=100)"
echo "\""
echo ""

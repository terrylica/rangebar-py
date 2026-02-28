#!/bin/bash
set -euo pipefail
# Queue BTCUSDT@250 per-year backfill jobs (2020-2025)
# These years are severely under-populated (<500 bars vs expected 50K-250K)
cd ~/rangebar-py || exit 1
for year in 2020 2021 2022 2023 2024 2025; do
    ~/.local/bin/pueue add --group btc250 --label "BTC250-${year}" -- \
        .venv/bin/python3 scripts/populate_full_cache.py \
        --symbol BTCUSDT --threshold 250 --include-microstructure \
        --start-date "${year}-01-01" --end-date "${year}-12-31"
    echo "Queued: BTC@250 ${year}"
done
echo "All 6 jobs queued"

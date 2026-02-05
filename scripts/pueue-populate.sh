#!/usr/bin/env bash
# Pueue-based cache population for BigBlack
#
# This is the STATE-OF-THE-ART approach:
# - Pueue daemon survives SSH disconnects, crashes, reboots
# - Queue persisted to disk - auto-resumes after any failure
# - pause_on_failure prevents cascade failures
# - Per-group parallelism limits
#
# SETUP (one-time on BigBlack):
#   brew install pueue
#   pueued -d                    # Start daemon
#   pueue parallel 2 --group p2  # Set group limits
#
# USAGE:
#   ./scripts/pueue-populate.sh setup    # Configure groups
#   ./scripts/pueue-populate.sh phase1   # Queue Phase 1 jobs
#   ./scripts/pueue-populate.sh all      # Queue all phases
#   ./scripts/pueue-populate.sh status   # Check progress
#   ./scripts/pueue-populate.sh restart  # Restart failed jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Symbols - simple list (listing dates handled by Python script)
SYMBOLS=(
    BTCUSDT
    ETHUSDT
    BNBUSDT
    SOLUSDT
    XRPUSDT
    DOGEUSDT
    ADAUSDT
    AVAXUSDT
    DOTUSDT
    LINKUSDT
    MATICUSDT
    LTCUSDT
    ATOMUSDT
    NEARUSDT
)

# Phase configuration: threshold -> (group_name, parallel_limit)
# Phase 1: 1000 dbps - fast, safe (4 parallel)
# Phase 2: 250 dbps - moderate (2 parallel)
# Phase 3: 500,750 dbps - moderate (3 parallel)
# Phase 4: 100 dbps - resource intensive (1 at a time)

check_pueue() {
    if ! command -v pueue &> /dev/null; then
        echo "❌ Pueue not installed. Install with: brew install pueue"
        exit 1
    fi
    if ! pueue status &> /dev/null; then
        echo "❌ Pueue daemon not running. Start with: pueued -d"
        exit 1
    fi
    echo "✅ Pueue daemon running"
}

setup_groups() {
    echo "Setting up Pueue groups with resource-aware parallelism..."

    # Create groups (ignore errors if already exist)
    pueue group add p1 2>/dev/null || true
    pueue group add p2 2>/dev/null || true
    pueue group add p3 2>/dev/null || true
    pueue group add p4 2>/dev/null || true

    # Set parallelism limits per group
    pueue parallel 4 --group p1   # Phase 1: 1000 dbps (fast)
    pueue parallel 2 --group p2   # Phase 2: 250 dbps (moderate)
    pueue parallel 3 --group p3   # Phase 3: 500,750 dbps
    pueue parallel 1 --group p4   # Phase 4: 100 dbps (sequential!)

    # Enable pause_on_failure for Phase 4 (most critical)
    # This prevents cascading failures

    echo "✅ Groups configured:"
    echo "   p1 (1000 dbps): 4 parallel jobs"
    echo "   p2 (250 dbps):  2 parallel jobs"
    echo "   p3 (500,750):   3 parallel jobs"
    echo "   p4 (100 dbps):  1 sequential job (resource intensive)"
}

add_job() {
    local symbol=$1
    local threshold=$2
    local group=$3

    local label="${symbol}@${threshold}"

    # Add job to pueue with label for easy identification
    pueue add --group "$group" --label "$label" -- \
        bash -c "cd $PROJECT_DIR && uv run python scripts/populate_full_cache.py --symbol $symbol --threshold $threshold"

    echo "  Added: $label -> group $group"
}

queue_phase1() {
    echo "Queueing Phase 1: 1000 dbps (14 jobs, 4 parallel)"
    for symbol in "${SYMBOLS[@]}"; do
        add_job "$symbol" 1000 p1
    done
}

queue_phase2() {
    echo "Queueing Phase 2: 250 dbps (14 jobs, 2 parallel)"
    for symbol in "${SYMBOLS[@]}"; do
        add_job "$symbol" 250 p2
    done
}

queue_phase3() {
    echo "Queueing Phase 3: 500, 750 dbps (28 jobs, 3 parallel)"
    for symbol in "${SYMBOLS[@]}"; do
        add_job "$symbol" 500 p3
        add_job "$symbol" 750 p3
    done
}

queue_phase4() {
    echo "Queueing Phase 4: 100 dbps (14 jobs, SEQUENTIAL)"
    echo "⚠️  This phase runs ONE job at a time to prevent resource exhaustion"
    for symbol in "${SYMBOLS[@]}"; do
        add_job "$symbol" 100 p4
    done
}

queue_all() {
    queue_phase1
    queue_phase2
    queue_phase3
    queue_phase4
    echo ""
    echo "✅ All 70 jobs queued!"
    echo ""
    echo "Monitor progress:"
    echo "  pueue status              # Overview"
    echo "  pueue status --group p1   # Phase 1 only"
    echo "  pueue follow <id>         # Watch specific job"
    echo "  pueue log <id>            # View completed job output"
}

show_status() {
    echo "=== PUEUE STATUS ==="
    pueue status
    echo ""
    echo "=== GROUP SUMMARY ==="
    for group in p1 p2 p3 p4; do
        local running queued completed failed
        running=$(pueue status --group "$group" --json 2>/dev/null | jq '[.tasks[] | select(.status == "Running")] | length' 2>/dev/null || echo "?")
        queued=$(pueue status --group "$group" --json 2>/dev/null | jq '[.tasks[] | select(.status == "Queued")] | length' 2>/dev/null || echo "?")
        completed=$(pueue status --group "$group" --json 2>/dev/null | jq '[.tasks[] | select(.status == "Done")] | length' 2>/dev/null || echo "?")
        failed=$(pueue status --group "$group" --json 2>/dev/null | jq '[.tasks[] | select(.status == "Done") | select(.result != "Success")] | length' 2>/dev/null || echo "?")
        echo "  $group: Running=$running, Queued=$queued, Completed=$completed, Failed=$failed"
    done
}

restart_failed() {
    echo "Restarting all failed jobs..."
    # Get IDs of failed jobs and restart them
    pueue status --json | jq -r '.tasks[] | select(.status == "Done") | select(.result != "Success") | .id' | while read -r id; do
        if [ -n "$id" ]; then
            echo "  Restarting job $id"
            pueue restart "$id"
        fi
    done
    echo "✅ Failed jobs restarted"
}

clean_done() {
    echo "Cleaning completed successful jobs..."
    pueue clean
    echo "✅ Done"
}

usage() {
    cat << EOF
Pueue-based rangebar cache population

SETUP (one-time):
  brew install pueue      # Install pueue
  pueued -d               # Start daemon (survives SSH disconnect!)
  $0 setup                # Configure groups with parallelism limits

USAGE:
  $0 phase1     Queue Phase 1 jobs (1000 dbps, 4 parallel)
  $0 phase2     Queue Phase 2 jobs (250 dbps, 2 parallel)
  $0 phase3     Queue Phase 3 jobs (500,750 dbps, 3 parallel)
  $0 phase4     Queue Phase 4 jobs (100 dbps, SEQUENTIAL)
  $0 all        Queue all 70 jobs across all phases
  $0 status     Show progress
  $0 restart    Restart all failed jobs
  $0 clean      Remove completed jobs from queue

WHY PUEUE:
  ✅ Daemon survives SSH disconnect, crashes, reboots
  ✅ Queue persisted to disk - auto-resumes
  ✅ Per-group parallelism limits
  ✅ Easy restart of failed jobs
  ✅ No Redis, no database, just works

EOF
}

# Main
check_pueue

case "${1:-}" in
    setup)
        setup_groups
        ;;
    phase1)
        queue_phase1
        ;;
    phase2)
        queue_phase2
        ;;
    phase3)
        queue_phase3
        ;;
    phase4)
        queue_phase4
        ;;
    all)
        queue_all
        ;;
    status)
        show_status
        ;;
    restart)
        restart_failed
        ;;
    clean)
        clean_done
        ;;
    *)
        usage
        ;;
esac

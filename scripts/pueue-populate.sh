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

# --- Resource Guard Configuration ---
# Memory limits per threshold tier (empirically measured on bigblack)
# Values include ~60% headroom above observed peak usage
MEMORY_LIMIT_250="8G"    # Peak ~5 GB (most trades per bar, heaviest)
MEMORY_LIMIT_500="4G"    # Peak ~1.5 GB
MEMORY_LIMIT_750="4G"    # Peak ~1.5 GB
MEMORY_LIMIT_1000="2G"   # Peak ~1 GB (fewest trades per bar, lightest)

# Detect if systemd-run is available (Linux only, not macOS)
HAS_SYSTEMD_RUN=false
if command -v systemd-run &> /dev/null && [[ "$(uname)" == "Linux" ]]; then
    # Verify user slice works (cgroups v2 required)
    if systemd-run --user --scope -p MemoryMax=1G true 2>/dev/null; then
        HAS_SYSTEMD_RUN=true
    fi
fi

# Override: RANGEBAR_NO_CGROUP=1 to bypass systemd-run
if [[ "${RANGEBAR_NO_CGROUP:-}" == "1" ]]; then
    HAS_SYSTEMD_RUN=false
fi

get_memory_limit() {
    local threshold=$1
    case "$threshold" in
        250)  echo "$MEMORY_LIMIT_250" ;;
        500)  echo "$MEMORY_LIMIT_500" ;;
        750)  echo "$MEMORY_LIMIT_750" ;;
        1000) echo "$MEMORY_LIMIT_1000" ;;
        *)    echo "4G" ;;  # Conservative default
    esac
}

# Symbols - from symbols.toml registry (Issue #79, #84)
# MATICUSDT removed: delisted 2024-09-10 (MATIC->POL rebrand)
# SHIBUSDT, UNIUSDT added: in symbols.toml but missing from bigblack
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
    LTCUSDT
    ATOMUSDT
    NEARUSDT
    SHIBUSDT
    UNIUSDT
)

# Phase configuration: threshold -> (group_name, parallel_limit)
# Phase 1: 1000 dbps - fast, safe (4 parallel)
# Phase 2: 250 dbps - moderate (2 parallel)
# Phase 3: 500,750 dbps - moderate (3 parallel)

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

    # Set parallelism limits per group
    pueue parallel 4 --group p1   # Phase 1: 1000 dbps (fast)
    pueue parallel 2 --group p2   # Phase 2: 250 dbps (moderate)
    pueue parallel 3 --group p3   # Phase 3: 500,750 dbps

    echo "Groups configured:"
    echo "   p1 (1000 dbps): 4 parallel jobs"
    echo "   p2 (250 dbps):  2 parallel jobs"
    echo "   p3 (500,750):   3 parallel jobs"

    # Report resource guard status
    if [[ "$HAS_SYSTEMD_RUN" == "true" ]]; then
        echo ""
        echo "Resource guard: systemd-run cgroups ACTIVE"
        echo "   250 dbps:  ${MEMORY_LIMIT_250} per job"
        echo "   500 dbps:  ${MEMORY_LIMIT_500} per job"
        echo "   750 dbps:  ${MEMORY_LIMIT_750} per job"
        echo "   1000 dbps: ${MEMORY_LIMIT_1000} per job"
    else
        echo ""
        echo "Resource guard: INACTIVE (no systemd-run or macOS)"
    fi
}

add_job() {
    local symbol=$1
    local threshold=$2
    local group=$3

    local label="${symbol}@${threshold}"
    local mem_limit
    mem_limit=$(get_memory_limit "$threshold")

    # Build command with optional systemd-run cgroup wrapper
    # systemd-run provides per-job memory caps (OOM-killed cleanly instead of swapping)
    local cmd
    if [[ "$HAS_SYSTEMD_RUN" == "true" ]]; then
        cmd="systemd-run --user --scope -p MemoryMax=${mem_limit} -p MemorySwapMax=0 uv run python scripts/populate_full_cache.py --symbol ${symbol} --threshold ${threshold} --force-refresh --include-microstructure"
    else
        cmd="uv run python scripts/populate_full_cache.py --symbol ${symbol} --threshold ${threshold} --force-refresh --include-microstructure"
    fi

    # Add job to pueue with label for easy identification
    # --force-refresh: wipe existing cache (Issue #84 full repopulation)
    # --include-microstructure: all 38 features (Issue #83 gap fix)
    # shellcheck disable=SC2086
    pueue add --group "$group" --label "$label" --working-directory "$PROJECT_DIR" -- $cmd

    local guard_info=""
    if [[ "$HAS_SYSTEMD_RUN" == "true" ]]; then
        guard_info=" [cgroup: ${mem_limit}]"
    fi
    echo "  Added: $label -> group $group${guard_info}"
}

queue_phase1() {
    echo "Queueing Phase 1: 1000 dbps (15 jobs, 4 parallel)"
    for symbol in "${SYMBOLS[@]}"; do
        add_job "$symbol" 1000 p1
    done
}

queue_phase2() {
    echo "Queueing Phase 2: 250 dbps (15 jobs, 2 parallel)"
    for symbol in "${SYMBOLS[@]}"; do
        add_job "$symbol" 250 p2
    done
}

queue_phase3() {
    echo "Queueing Phase 3: 500, 750 dbps (30 jobs, 3 parallel)"
    for symbol in "${SYMBOLS[@]}"; do
        add_job "$symbol" 500 p3
        add_job "$symbol" 750 p3
    done
}

queue_all() {
    queue_phase1
    queue_phase2
    queue_phase3
    echo ""
    echo "✅ All 60 jobs queued!"
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
    for group in p1 p2 p3; do
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

show_guard_status() {
    echo "=== RESOURCE GUARD STATUS ==="
    if [[ "$HAS_SYSTEMD_RUN" == "true" ]]; then
        echo "systemd-run:  ACTIVE (cgroups v2)"
        echo "Memory limits:"
        echo "   250 dbps:  ${MEMORY_LIMIT_250}"
        echo "   500 dbps:  ${MEMORY_LIMIT_500}"
        echo "   750 dbps:  ${MEMORY_LIMIT_750}"
        echo "   1000 dbps: ${MEMORY_LIMIT_1000}"
    else
        echo "systemd-run:  INACTIVE"
        if [[ "$(uname)" == "Darwin" ]]; then
            echo "  Reason: macOS (systemd not available)"
        elif [[ "${RANGEBAR_NO_CGROUP:-}" == "1" ]]; then
            echo "  Reason: RANGEBAR_NO_CGROUP=1 set"
        else
            echo "  Reason: systemd-run not found or cgroups v2 not available"
        fi
    fi
    echo ""
    echo "=== HOST RESOURCES ==="
    if command -v free &> /dev/null; then
        free -h | head -2
    else
        echo "  (free not available — likely macOS)"
    fi
    echo ""
    uptime
}

# =============================================================================
# Issue #88: Volume Overflow Post-Fix
# =============================================================================

postprocess_shib() {
    echo "=== Issue #88: SHIBUSDT Volume Overflow Post-Fix ==="
    echo "Queueing SHIBUSDT force-refresh at all 4 thresholds..."

    # Create dedicated group for postprocessing (idempotent)
    pueue group add postfix 2>/dev/null || true
    pueue parallel 2 --group postfix  # 2 parallel (moderate resource usage)

    for threshold in 250 500 750 1000; do
        local label="SHIB-postfix@${threshold}"
        local mem_limit
        mem_limit=$(get_memory_limit "$threshold")

        local cmd
        if [[ "$HAS_SYSTEMD_RUN" == "true" ]]; then
            cmd="systemd-run --user --scope -p MemoryMax=${mem_limit} -p MemorySwapMax=0 uv run python scripts/populate_full_cache.py --symbol SHIBUSDT --threshold ${threshold} --force-refresh --include-microstructure"
        else
            cmd="uv run python scripts/populate_full_cache.py --symbol SHIBUSDT --threshold ${threshold} --force-refresh --include-microstructure"
        fi

        # shellcheck disable=SC2086
        pueue add --group postfix --label "$label" --working-directory "$PROJECT_DIR" -- $cmd
        echo "  Added: $label -> group postfix"
    done

    echo ""
    echo "✅ 4 SHIBUSDT repopulation jobs queued in 'postfix' group"
    echo ""
    echo "Monitor: pueue status --group postfix"
    echo "After completion, run: $0 optimize"
}

optimize_table() {
    echo "=== OPTIMIZE TABLE rangebar_cache.range_bars FINAL ==="
    echo "This merges data parts and deduplicates (ReplacingMergeTree)."
    echo ""

    if command -v clickhouse-client &> /dev/null; then
        clickhouse-client --query "OPTIMIZE TABLE rangebar_cache.range_bars FINAL"
        echo "✅ OPTIMIZE TABLE completed"
    else
        echo "❌ clickhouse-client not found. Run manually:"
        echo "   clickhouse-client --query 'OPTIMIZE TABLE rangebar_cache.range_bars FINAL'"
        exit 1
    fi
}

detect_overflow() {
    echo "=== Detecting Volume Overflow (Issue #88) ==="
    uv run python "$SCRIPT_DIR/detect_volume_overflow.py"
}

postprocess_all() {
    echo "=== Issue #88: Full Post-Fix Pipeline ==="
    echo ""
    echo "Step 1/3: Queue SHIBUSDT repopulation"
    postprocess_shib
    echo ""
    echo "⏳ Waiting for postfix jobs to complete..."
    echo "   Run 'pueue wait --group postfix' then:"
    echo "   $0 optimize"
    echo "   $0 detect-overflow"
    echo ""
    echo "Or monitor with: pueue status --group postfix"
}

usage() {
    cat << EOF
Pueue-based rangebar cache population with resource guards

SETUP (one-time):
  brew install pueue      # Install pueue
  pueued -d               # Start daemon (survives SSH disconnect!)
  $0 setup                # Configure groups with parallelism limits

POPULATION:
  $0 phase1       Queue Phase 1 jobs (1000 dbps, 4 parallel)
  $0 phase2       Queue Phase 2 jobs (250 dbps, 2 parallel)
  $0 phase3       Queue Phase 3 jobs (500,750 dbps, 3 parallel)
  $0 all          Queue all 60 jobs across all phases

MONITORING:
  $0 status       Show progress
  $0 guard-status Show resource guard (systemd-run) status
  $0 restart      Restart all failed jobs
  $0 clean        Remove completed jobs from queue

ISSUE #88 POST-FIX (volume overflow):
  $0 postprocess-shib   Queue SHIBUSDT force-refresh at 250/500/750/1000
  $0 optimize           Run OPTIMIZE TABLE FINAL on range_bars
  $0 detect-overflow    Check for negative volumes in cache
  $0 postprocess-all    Full pipeline: queue + instructions for optimize + detect

RESOURCE GUARDS:
  On Linux with systemd, each job runs inside a cgroup with per-threshold
  memory limits via systemd-run. Jobs that exceed their limit are OOM-killed
  cleanly instead of swapping the host to death.

  Bypass: RANGEBAR_NO_CGROUP=1 $0 all

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
    all)
        queue_all
        ;;
    status)
        show_status
        ;;
    guard-status)
        show_guard_status
        ;;
    restart)
        restart_failed
        ;;
    clean)
        clean_done
        ;;
    postprocess-shib)
        postprocess_shib
        ;;
    optimize)
        optimize_table
        ;;
    detect-overflow)
        detect_overflow
        ;;
    postprocess-all)
        postprocess_all
        ;;
    *)
        usage
        ;;
esac

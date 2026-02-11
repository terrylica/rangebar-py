#!/usr/bin/env bash
# Pipeline health monitor for rangebar pueue jobs on bigblack.
# Runs as a Claude Code background task. Polls every 5 minutes.
# Reports phase transitions and runs integrity checks at boundaries.
#
# DESIGN: Detects phases by GROUP NAMES and LABEL PATTERNS, not hardcoded IDs.
# Safe across job removals, re-queues, and per-year splits.
#
# Usage:
#   bash scripts/pipeline-monitor.sh          # foreground
#   # Or launched as Claude Code background task

set -euo pipefail

POLL_INTERVAL=300  # 5 minutes
SEEN_GROUPS=""     # Track which group completions we've already reported

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

get_job_status() {
    ssh bigblack "pueue status --json 2>/dev/null" | jq -r \
        ".tasks | to_entries[] | \"\(.value.id)|\(.value.status | if type == \"object\" then keys[0] else . end)|\(.value.label // \"-\")|\(.value.group)\""
}

# Check if ALL jobs in a group are done (Done/Success/Killed, not Running/Queued)
group_all_done() {
    local group="$1"
    local group_jobs
    group_jobs=$(echo "$JOBS" | grep "|${group}$" || true)
    [ -z "$group_jobs" ] && return 1  # No jobs in group = not done
    # Check if any are still Running or Queued
    echo "$group_jobs" | grep -qE "\|(Running|Queued)\|" && return 1
    return 0
}

# Count running jobs in a group
group_running_count() {
    local group="$1"
    echo "$JOBS" | grep "|${group}$" | grep -c "|Running|" 2>/dev/null || echo 0
}

# Get all unique group names (excluding "default")
get_groups() {
    echo "$JOBS" | cut -d'|' -f4 | sort -u | grep -v "^default$" || true
}

run_integrity_check() {
    local phase_name="$1"
    log "=== INTEGRITY CHECK: $phase_name ==="

    # Check 1: Negative volumes
    local neg_vol
    neg_vol=$(ssh bigblack "clickhouse-client --query=\"
        SELECT symbol, threshold_decimal_bps as thresh, countIf(volume < 0) as neg
        FROM rangebar_cache.range_bars FINAL
        GROUP BY symbol, threshold_decimal_bps
        HAVING neg > 0
        FORMAT TabSeparated
    \"" 2>/dev/null || echo "QUERY_ERROR")
    if [ "$neg_vol" = "QUERY_ERROR" ]; then
        log "ERROR: Could not run negative volume check"
    elif [ -n "$neg_vol" ]; then
        log "WARNING: Negative volumes found:"
        echo "$neg_vol" | while read -r line; do log "  $line"; done
    else
        log "PASS: No negative volumes"
    fi

    # Check 2: Lookback coverage (incomplete only)
    log "Lookback coverage (incomplete only):"
    ssh bigblack "clickhouse-client --query=\"
        SELECT symbol, threshold_decimal_bps as thresh,
               count(*) as total,
               countIf(lookback_trade_count IS NULL OR lookback_trade_count = 0) as missing,
               round(100.0 * (1.0 - countIf(lookback_trade_count IS NULL OR lookback_trade_count = 0) / count(*)), 1) as pct
        FROM rangebar_cache.range_bars FINAL
        GROUP BY symbol, threshold_decimal_bps
        HAVING missing > 0
        ORDER BY pct ASC
        FORMAT TabSeparated
    \"" 2>/dev/null | while read -r line; do log "  $line"; done

    # Check 3: Feature bounds (OFI must be [-1,1])
    local ofi_oob
    ofi_oob=$(ssh bigblack "clickhouse-client --query=\"
        SELECT symbol, threshold_decimal_bps, countIf(ofi < -1.001 OR ofi > 1.001) as oob
        FROM rangebar_cache.range_bars FINAL
        GROUP BY symbol, threshold_decimal_bps
        HAVING oob > 0
        FORMAT TabSeparated
    \"" 2>/dev/null || echo "QUERY_ERROR")
    if [ "$ofi_oob" = "QUERY_ERROR" ]; then
        log "ERROR: Could not run OFI bounds check"
    elif [ -n "$ofi_oob" ]; then
        log "WARNING: OFI out-of-bounds:"
        echo "$ofi_oob" | while read -r line; do log "  $line"; done
    else
        log "PASS: All OFI values in bounds"
    fi

    # Check 4: Duplicate bars
    local dupes
    dupes=$(ssh bigblack "clickhouse-client --query=\"
        SELECT symbol, threshold_decimal_bps as thresh,
               count(*) - uniqExact(timestamp_ms) as dupes
        FROM rangebar_cache.range_bars FINAL
        GROUP BY symbol, threshold_decimal_bps
        HAVING dupes > 0
        FORMAT TabSeparated
    \"" 2>/dev/null || echo "QUERY_ERROR")
    if [ "$dupes" = "QUERY_ERROR" ]; then
        log "ERROR: Could not run duplicate check"
    elif [ -n "$dupes" ]; then
        log "WARNING: Duplicate bars found:"
        echo "$dupes" | while read -r line; do log "  $line"; done
    else
        log "PASS: No duplicate bars"
    fi

    # System resources
    log "System resources:"
    ssh bigblack 'echo "  Load: $(uptime | sed "s/.*load average: //")" && echo "  $(free -h | grep Mem | awk "{print \"RAM: \" \$3 \"/\" \$2}")"' 2>/dev/null || log "  (could not reach bigblack)"

    log "=== END INTEGRITY CHECK ==="
}

# ─── Main loop ───
log "Pipeline monitor started. Polling every ${POLL_INTERVAL}s."
log "Phase detection: by group completion (dynamic, no hardcoded IDs)"

while true; do
    JOBS=$(get_job_status 2>/dev/null || echo "")
    if [ -z "$JOBS" ]; then
        log "ERROR: Could not reach bigblack pueue. Retrying in ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Count statuses
    running=$(echo "$JOBS" | grep -c "|Running|" || true)
    queued=$(echo "$JOBS" | grep -c "|Queued|" || true)
    done_count=$(echo "$JOBS" | grep -cE "\|(Done|Success)\|" || true)
    failed=$(echo "$JOBS" | grep -c "|Failed|" || true)
    killed=$(echo "$JOBS" | grep -c "|Killed|" || true)
    total=$(echo "$JOBS" | wc -l | tr -d ' ')

    log "Status: ${running} running, ${queued} queued, ${done_count} done, ${failed} failed, ${killed} killed (${total} total)"

    # ── Failure detection (immediate) ──
    if [ "$failed" -gt 0 ]; then
        log "ALERT: $failed job(s) FAILED!"
        echo "$JOBS" | grep "|Failed|" | while IFS='|' read -r id status label group; do
            log "  FAILED: id=$id label=$label group=$group"
        done
        log "ACTION NEEDED: Investigate with 'ssh bigblack pueue log <id>'"
        log "To retry: 'ssh bigblack pueue restart <id>'"
    fi

    # ── Dynamic group completion detection ──
    # Check every group. When all jobs in a group finish, run integrity check.
    for group in $(get_groups); do
        if group_all_done "$group" && [[ "$SEEN_GROUPS" != *"|${group}|"* ]]; then
            # Skip postprocessing groups (optimize/detect) — they're not compute groups
            case "$group" in
                backfill) continue ;;
            esac

            group_jobs=$(echo "$JOBS" | grep "|${group}$" || true)
            job_count=$(echo "$group_jobs" | wc -l | tr -d ' ')
            done_in_group=$(echo "$group_jobs" | grep -cE "\|(Done|Success)\|" || true)

            log "=========================================="
            log "GROUP COMPLETE: $group ($done_in_group/$job_count jobs done)"
            log "=========================================="

            # Show which jobs completed
            echo "$group_jobs" | while IFS='|' read -r id status label _; do
                log "  [$id] $label → $status"
            done

            run_integrity_check "Post-${group}"
            SEEN_GROUPS="${SEEN_GROUPS}|${group}|"
        fi
    done

    # ── Final postprocessing detection ──
    # Look for optimize-table:final or detect-overflow:final completion
    final_detect=$(echo "$JOBS" | grep "detect-overflow:final" || true)
    if [ -n "$final_detect" ]; then
        final_status=$(echo "$final_detect" | cut -d'|' -f2)
        if { [ "$final_status" = "Done" ] || [ "$final_status" = "Success" ]; } && \
           [[ "$SEEN_GROUPS" != *"|final-postprocess|"* ]]; then
            log "============================================"
            log "FINAL POSTPROCESSING COMPLETE"
            log "============================================"
            run_integrity_check "Final"
            SEEN_GROUPS="${SEEN_GROUPS}|final-postprocess|"
        fi
    fi

    # ── All done detection ──
    if [ "$running" -eq 0 ] && [ "$queued" -eq 0 ] && [ "$total" -gt 0 ] && \
       [[ "$SEEN_GROUPS" != *"|all-done|"* ]]; then
        log "============================================"
        log "ALL JOBS COMPLETE (0 running, 0 queued)"
        log "============================================"
        run_integrity_check "All-Done"
        log "Pipeline finished. Monitor exiting."
        SEEN_GROUPS="${SEEN_GROUPS}|all-done|"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done

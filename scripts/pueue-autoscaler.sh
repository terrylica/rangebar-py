#!/usr/bin/env bash
# Pueue Autoscaler — Dynamic parallelism tuning based on host resources
#
# Monitors CPU load and available memory, then adjusts pueue group
# parallelism limits to maximize throughput within safe margins.
#
# This complements pueue (which has no resource awareness) with the
# incremental scaling protocol from distributed-job-safety skill.
#
# USAGE:
#   ./scripts/pueue-autoscaler.sh              # Run once (dry-run)
#   ./scripts/pueue-autoscaler.sh --apply      # Run once, apply changes
#   ./scripts/pueue-autoscaler.sh --loop       # Continuous monitoring (60s interval)
#   ./scripts/pueue-autoscaler.sh --loop 30    # Custom interval (30s)
#
# ENVIRONMENT:
#   AUTOSCALE_CPU_CEILING=80    # Max CPU% before scaling down (default: 80)
#   AUTOSCALE_MEM_CEILING=80    # Max memory% before scaling down (default: 80)
#   AUTOSCALE_CPU_FLOOR=40      # Min CPU% before scaling up (default: 40)
#   AUTOSCALE_MEM_FLOOR=60      # Min memory% before scaling up (default: 60)

set -euo pipefail

# --- Configuration ---
CPU_CEILING="${AUTOSCALE_CPU_CEILING:-80}"
MEM_CEILING="${AUTOSCALE_MEM_CEILING:-80}"
CPU_FLOOR="${AUTOSCALE_CPU_FLOOR:-40}"
MEM_FLOOR="${AUTOSCALE_MEM_FLOOR:-60}"

# Per-group limits: (min, max, per-job memory estimate in MB)
# These are hard boundaries the autoscaler won't exceed
declare -A GROUP_MIN=( [p1]=1 [p2]=1 [p3]=1 )
declare -A GROUP_MAX=( [p1]=8 [p2]=6 [p3]=10 )
# Per-job memory estimate in MB (used for memory-based scaling)
declare -A GROUP_MEM_PER_JOB=( [p1]=1024 [p2]=5120 [p3]=1536 )

# --- Helper Functions ---

get_cpu_count() {
    nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4
}

get_load_avg() {
    # 1-minute load average
    uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' '
}

get_cpu_percent() {
    local load_avg cores
    load_avg=$(get_load_avg)
    cores=$(get_cpu_count)
    # CPU utilization as percentage of total cores
    awk "BEGIN { printf \"%.0f\", ($load_avg / $cores) * 100 }"
}

get_mem_available_mb() {
    if command -v free &> /dev/null; then
        free -m | awk '/^Mem:/ {print $7}'
    else
        # macOS fallback
        local pages_free page_size
        pages_free=$(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
        page_size=4096
        awk "BEGIN { printf \"%.0f\", ($pages_free * $page_size) / 1048576 }"
    fi
}

get_mem_total_mb() {
    if command -v free &> /dev/null; then
        free -m | awk '/^Mem:/ {print $2}'
    else
        # macOS fallback
        sysctl -n hw.memsize | awk '{printf "%.0f", $1 / 1048576}'
    fi
}

get_mem_percent() {
    local available total
    available=$(get_mem_available_mb)
    total=$(get_mem_total_mb)
    awk "BEGIN { printf \"%.0f\", (1 - $available / $total) * 100 }"
}

get_current_parallel() {
    local group=$1
    # Extract current parallelism limit from pueue status
    pueue status --json 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
groups = data.get('groups', {})
g = groups.get('$group', {})
print(g.get('parallel_tasks', 0))
" 2>/dev/null || echo 0
}

get_running_count() {
    local group=$1
    pueue status --json 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
tasks = data.get('tasks', {})
count = sum(1 for t in tasks.values()
            if t.get('group') == '$group'
            and isinstance(t.get('status'), dict)
            and 'Running' in t['status'])
print(count)
" 2>/dev/null || echo 0
}

get_queued_count() {
    local group=$1
    pueue status --json 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
tasks = data.get('tasks', {})
count = sum(1 for t in tasks.values()
            if t.get('group') == '$group'
            and isinstance(t.get('status'), dict)
            and 'Queued' in t['status'])
print(count)
" 2>/dev/null || echo 0
}

# --- Core Logic ---

compute_scaling_decision() {
    local cpu_pct mem_pct
    cpu_pct=$(get_cpu_percent)
    mem_pct=$(get_mem_percent)

    echo "Host: CPU=${cpu_pct}% MEM=${mem_pct}%  (floor: CPU<${CPU_FLOOR}% MEM<${MEM_FLOOR}%, ceiling: CPU>${CPU_CEILING}% MEM>${MEM_CEILING}%)"

    # Decision: scale up, scale down, or hold
    if (( cpu_pct > CPU_CEILING )) || (( mem_pct > MEM_CEILING )); then
        echo "Action: SCALE DOWN (resource pressure)"
        echo "DOWN"
    elif (( cpu_pct < CPU_FLOOR )) && (( mem_pct < MEM_FLOOR )); then
        echo "Action: SCALE UP (resources available)"
        echo "UP"
    else
        echo "Action: HOLD (within operating range)"
        echo "HOLD"
    fi
}

apply_scaling() {
    local direction=$1
    local dry_run=$2
    local mem_available_mb
    mem_available_mb=$(get_mem_available_mb)

    for group in p1 p2 p3; do
        local current queued new_parallel
        current=$(get_current_parallel "$group")
        queued=$(get_queued_count "$group")

        # Skip if no queued work and we'd be scaling up
        if [[ "$direction" == "UP" ]] && (( queued == 0 )); then
            echo "  $group: skip (no queued jobs)"
            continue
        fi

        case "$direction" in
            UP)
                # Check if we can afford another job in this group (memory check)
                local per_job_mb="${GROUP_MEM_PER_JOB[$group]}"
                if (( mem_available_mb < per_job_mb * 2 )); then
                    echo "  $group: skip (insufficient memory for +1 job: need ${per_job_mb}MB, available ${mem_available_mb}MB)"
                    continue
                fi
                # Increment by 1 (conservative — incremental scaling protocol)
                new_parallel=$(( current + 1 ))
                if (( new_parallel > GROUP_MAX[$group] )); then
                    echo "  $group: at max (${current}/${GROUP_MAX[$group]})"
                    continue
                fi
                ;;
            DOWN)
                # Decrement by 1
                new_parallel=$(( current - 1 ))
                if (( new_parallel < GROUP_MIN[$group] )); then
                    echo "  $group: at min (${current}/${GROUP_MIN[$group]})"
                    continue
                fi
                ;;
            HOLD)
                echo "  $group: hold at ${current}"
                continue
                ;;
        esac

        if [[ "$dry_run" == "true" ]]; then
            echo "  $group: would change ${current} -> ${new_parallel}"
        else
            pueue parallel "$new_parallel" --group "$group"
            echo "  $group: ${current} -> ${new_parallel}"
        fi
    done
}

run_once() {
    local dry_run=$1

    echo "=== Pueue Autoscaler ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"

    # Get decision (last line is the action keyword)
    local output decision
    output=$(compute_scaling_decision)
    decision=$(echo "$output" | tail -1)
    # Print all but last line (the human-readable info)
    echo "$output" | head -n -1

    echo ""
    echo "Per-group adjustments:"
    apply_scaling "$decision" "$dry_run"

    echo ""
    echo "Current state:"
    for group in p1 p2 p3; do
        local parallel running queued
        parallel=$(get_current_parallel "$group")
        running=$(get_running_count "$group")
        queued=$(get_queued_count "$group")
        echo "  $group: parallel=${parallel}, running=${running}, queued=${queued}"
    done
}

# --- Main ---

# Check pueue is running
if ! pueue status &> /dev/null; then
    echo "Error: Pueue daemon not running. Start with: pueued -d"
    exit 1
fi

case "${1:-}" in
    --loop)
        interval="${2:-60}"
        echo "Starting autoscaler loop (interval: ${interval}s, Ctrl+C to stop)"
        echo ""
        while true; do
            run_once "false"
            echo ""
            echo "--- sleeping ${interval}s ---"
            echo ""
            sleep "$interval"
        done
        ;;
    --apply)
        run_once "false"
        ;;
    *)
        echo "(dry-run mode — use --apply to make changes, --loop for continuous)"
        echo ""
        run_once "true"
        ;;
esac

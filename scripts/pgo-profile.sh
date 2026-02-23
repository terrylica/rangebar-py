#!/usr/bin/env bash
# PGO Profile Collection & Optimization Workflow
# GitHub Issue: #96 Task #20
# Usage: ./scripts/pgo-profile.sh [collect|merge|optimize|full]

set -euo pipefail

# Configuration
PGO_DATA_DIR="pgo-data"
PROFILE_FILE_PATTERN="${PGO_DATA_DIR}/rangebar-*.profraw"
MERGED_PROFDATA="${PGO_DATA_DIR}/merged.profdata"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[PGO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[PGO WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[PGO ERROR]${NC} $1"
}

# Find llvm-profdata binary
find_llvm_profdata() {
    local sysroot toolchain_path profdata_path
    sysroot=$(rustc --print sysroot)
    local host=$(rustc -Vv | grep host | awk '{print $NF}')

    profdata_path="${sysroot}/lib/rustlib/${host}/bin/llvm-profdata"

    if [[ ! -f "$profdata_path" ]]; then
        log_error "llvm-profdata not found at ${profdata_path}"
        log_info "Install with: rustup component add llvm-tools-preview"
        return 1
    fi

    echo "$profdata_path"
}

# Phase 1: Collect profiling data
pgo_collect() {
    log_info "Phase 1: Collecting PGO profile data..."

    # Create pgo-data directory
    mkdir -p "$PGO_DATA_DIR"

    # Set up environment for profile collection
    export LLVM_PROFILE_FILE="${PGO_DATA_DIR}/rangebar-%m.profraw"

    log_info "Compiling instrumented binary with PGO flags..."
    RUSTFLAGS="-Cprofile-generate=./${PGO_DATA_DIR} -Cllvm-args=-pgo-warn-missing-function" \
        cargo build -p rangebar-py --release

    log_info "Running profiling workload..."

    # Tier 1: Fast profiling workload (1 day BTCUSDT)
    uv run python3 << 'PYTHON_WORKLOAD'
import sys
try:
    from rangebar import get_range_bars
    print("Running PGO profiling workload...")

    # Tier 1: Core workload (1-2 hours BTCUSDT)
    print("  • Processing BTCUSDT (typical volume)...")
    df = get_range_bars('BTCUSDT', '2024-06-15', '2024-06-15')
    print(f"    Generated {len(df)} bars")

    # Tier 2: Multi-symbol (optional, uncomment for comprehensive profiling)
    # for symbol in ['ETHUSDT', 'BNBUSDT']:
    #     print(f"  • Processing {symbol}...")
    #     df = get_range_bars(symbol, '2024-06-15', '2024-06-15')
    #     print(f"    Generated {len(df)} bars")

    print("\n✓ Profiling workload completed")

except Exception as e:
    print(f"\n✗ Error during profiling: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_WORKLOAD

    log_info "Profiling phase complete. Data saved to ${PGO_DATA_DIR}/"
}

# Phase 2: Merge profiling data
pgo_merge() {
    log_info "Phase 2: Merging PGO profile data..."

    if [[ ! -d "$PGO_DATA_DIR" ]]; then
        log_error "PGO data directory not found: ${PGO_DATA_DIR}/"
        log_info "Run 'pgo_collect' first"
        return 1
    fi

    local profraw_count=0
    profraw_count=$(find "$PGO_DATA_DIR" -name "*.profraw" -type f | wc -l)

    if [[ $profraw_count -eq 0 ]]; then
        log_error "No .profraw files found in ${PGO_DATA_DIR}/"
        return 1
    fi

    log_info "Found $profraw_count profiling data files"

    local llvm_profdata
    llvm_profdata=$(find_llvm_profdata) || return 1

    log_info "Merging with: $llvm_profdata"
    "$llvm_profdata" merge -o "$MERGED_PROFDATA" "$PGO_DATA_DIR"/*.profraw

    log_info "Merged profile data: ${MERGED_PROFDATA}"
}

# Phase 3: Optimize with PGO data
pgo_optimize() {
    log_info "Phase 3: Rebuilding with PGO optimization..."

    if [[ ! -f "$MERGED_PROFDATA" ]]; then
        log_error "Merged profdata not found: ${MERGED_PROFDATA}"
        log_info "Run 'pgo_merge' first"
        return 1
    fi

    log_info "Compiling optimized binary using PGO data..."
    RUSTFLAGS="-Cprofile-use=./${MERGED_PROFDATA} -Cllvm-args=-pgo-warn-missing-function" \
        cargo build -p rangebar-py --release

    log_info "Installing optimized wheel..."
    maturin develop --release

    log_info "✓ PGO optimization complete"
}

# Run full PGO cycle
pgo_full() {
    log_info "Running full PGO cycle (collect → merge → optimize)..."
    log_warn "This will take 5-10 minutes (profiling + recompilation)"

    pgo_collect || return 1
    pgo_merge || return 1
    pgo_optimize || return 1

    log_info "✓ Full PGO cycle complete"
    log_info "Expected speedup: 10-20% on top of LTO (cumulative 20-35%)"
}

# Show cleanup info
pgo_cleanup() {
    log_info "PGO cleanup options:"
    echo "  Remove profiling data: rm -rf ${PGO_DATA_DIR}/"
    echo "  Remove merged profdata: rm ${MERGED_PROFDATA}"
    echo "  Note: Keep pgo-data/ in .gitignore (it's regenerated each profile cycle)"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [command]

Commands:
  collect   - Compile instrumented binary and run profiling workload
  merge     - Merge .profraw files into single profdata
  optimize  - Rebuild with PGO optimization applied
  full      - Run complete cycle (collect → merge → optimize)
  cleanup   - Show cleanup commands
  help      - Show this help message

Examples:
  # Full PGO cycle
  $0 full

  # Incremental steps
  $0 collect
  $0 merge
  $0 optimize

  # Cleanup
  $0 cleanup && rm -rf pgo-data/

Environment:
  RUSTFLAGS            - Custom Rust compiler flags (if set, will override PGO flags)
  LLVM_PROFILE_FILE    - Custom profraw file path (default: pgo-data/rangebar-*.profraw)

Documentation:
  See .claude/pgo-workflow.md for detailed PGO configuration and troubleshooting
EOF
}

# Main
main() {
    local command="${1:-help}"

    case "$command" in
        collect)
            pgo_collect
            ;;
        merge)
            pgo_merge
            ;;
        optimize)
            pgo_optimize
            ;;
        full)
            pgo_full
            ;;
        cleanup)
            pgo_cleanup
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"

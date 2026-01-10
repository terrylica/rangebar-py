#!/bin/bash
# Profiling tools integration for high-frequency rangebar data processing
#
# Usage: ./scripts/profiling_tools.sh [flamegraph|perf|memory|cpu|io|install]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE_DATA_DIR="$PROJECT_ROOT/profile_data"

# Ensure profile data directory exists
mkdir -p "$PROFILE_DATA_DIR"

log_info() {
    echo "🔬 [PROF] $1"
}

log_success() {
    echo "✅ [PROF] $1"
}

log_warn() {
    echo "⚠️ [PROF] $1"
}

log_error() {
    echo "❌ [PROF] $1"
}

install_profiling_tools() {
    log_info "Installing profiling tools..."

    # Install cargo-flamegraph
    if ! command -v cargo-flamegraph &> /dev/null; then
        log_info "Installing cargo-flamegraph..."
        cargo install flamegraph
        log_success "cargo-flamegraph installed"
    else
        log_info "cargo-flamegraph already installed"
    fi

    # Install perf (Linux) or Instruments (macOS)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "Linux detected - checking for perf..."
        if ! command -v perf &> /dev/null; then
            log_warn "perf not found. Install with: apt-get install linux-perf (requires admin privileges)"
        else
            log_success "perf available"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "macOS detected - Instruments available via Xcode"
        if command -v xcrun &> /dev/null; then
            log_success "Xcode tools available for profiling"
        else
            log_warn "Xcode tools not found - install Xcode for advanced profiling"
        fi
    fi

    # Install cargo-profdata (for LLVM profiling)
    if ! command -v cargo-profdata &> /dev/null; then
        log_info "Installing cargo-profdata..."
        cargo install cargo-profdata
        log_success "cargo-profdata installed"
    else
        log_info "cargo-profdata already installed"
    fi
}

generate_flamegraph() {
    local test_name="${1:-rangebar_processing}"
    log_info "Generating flamegraph for high-frequency processing..."

    cd "$PROJECT_ROOT"

    # Create flamegraph focusing on the processing benchmark
    log_info "Running flamegraph analysis..."

    local output_file="$PROFILE_DATA_DIR/flamegraph_$(date +%Y%m%d_%H%M%S).svg"

    # Use cargo-flamegraph with the benchmark (user must have perf permissions)
    log_info "Note: Flamegraph requires perf permissions. May need admin privileges to configure perf_event_paranoid on Linux"

    # Validate benchmark executable exists before attempting profiling
    if ! cargo bench --bench rangebar_bench --no-run &> /dev/null; then
        log_error "Benchmark executable not found. Run 'cargo build --release --bench rangebar_bench' first"
        return 1
    fi

    cargo flamegraph \
        --bench rangebar_bench \
        --output "$output_file" \
        -- --bench "rangebar_processing/$test_name"

    log_success "Flamegraph saved to: $output_file"

    # Security: Do not automatically open files - inform user instead
    log_info "Flamegraph generated: $output_file"
    log_info "To view: open '$output_file' in a web browser"
    log_warn "Security: Automatic file opening disabled - manually verify file before opening"
}

cpu_profiling() {
    log_info "Running CPU profiling analysis..."
    cd "$PROJECT_ROOT"

    local profile_dir="$PROFILE_DATA_DIR/cpu_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$profile_dir"

    # Build with profiling instrumentation
    log_info "Building with CPU profiling instrumentation..."
    export RUSTFLAGS="-C force-frame-pointers=yes"
    cargo build --release --bench rangebar_bench

    # Run CPU profiling
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "Running perf analysis..."
        if command -v perf &> /dev/null; then
            # Record performance data - validate executable exists
            local bench_executable
            bench_executable=$(find ./target/release/deps -name "rangebar_bench-*" -type f -executable | head -1)

            if [[ -z "$bench_executable" ]]; then
                log_error "Benchmark executable not found. Run 'cargo build --release --bench rangebar_bench' first"
                return 1
            fi

            log_info "Using benchmark executable: $bench_executable"
            perf record -g --call-graph=dwarf -o "$profile_dir/perf.data" \
                "$bench_executable" --bench "rangebar_processing/1000000"

            # Generate report
            perf report -i "$profile_dir/perf.data" > "$profile_dir/cpu_report.txt"
            log_success "CPU profile saved to: $profile_dir/"
        else
            log_error "perf not available - install linux-perf package"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Use Instruments for detailed CPU profiling on macOS"
        local bench_executable
        bench_executable=$(find ./target/release/deps -name "rangebar_bench-*" -type f -executable | head -1)
        if [[ -n "$bench_executable" ]]; then
            log_info "Run: xcrun xctrace record --template 'CPU Profiler' --launch '$bench_executable'"
        else
            log_error "Benchmark executable not found. Build with 'cargo build --release --bench rangebar_bench'"
        fi
    fi
}

memory_profiling() {
    log_info "Running memory profiling analysis..."
    cd "$PROJECT_ROOT"

    local profile_dir="$PROFILE_DATA_DIR/memory_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$profile_dir"

    # Install valgrind (Linux) or use Instruments (macOS)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v valgrind &> /dev/null; then
            log_info "Running valgrind memory analysis..."

            # Build for debugging
            cargo build --bench rangebar_bench

            # Run valgrind
            valgrind --tool=massif \
                --massif-out-file="$profile_dir/massif.out" \
                ./target/debug/deps/rangebar_bench-* --bench "rangebar_processing/100000"

            # Generate memory report
            ms_print "$profile_dir/massif.out" > "$profile_dir/memory_report.txt"
            log_success "Memory profile saved to: $profile_dir/"
        else
            log_warn "valgrind not found. Install with: apt-get install valgrind (requires admin privileges)"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Use Instruments for memory profiling on macOS"
        log_info "Run: xcrun xctrace record --template 'Allocations' --launch ./target/release/deps/rangebar_bench-*"
    fi
}

io_profiling() {
    log_info "Running I/O profiling analysis..."
    cd "$PROJECT_ROOT"

    local profile_dir="$PROFILE_DATA_DIR/io_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$profile_dir"

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v strace &> /dev/null; then
            log_info "Running strace I/O analysis..."

            # Run strace on benchmark
            strace -c -o "$profile_dir/syscall_summary.txt" \
                ./target/release/deps/rangebar_bench-* --bench "rangebar_processing/10000"

            log_success "I/O profile saved to: $profile_dir/"
        else
            log_warn "strace not found - install strace package"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v dtruss &> /dev/null; then
            log_info "Running dtruss I/O analysis..."

            # Run dtruss - validate executable (admin privileges required)
            local bench_executable
            bench_executable=$(find ./target/release/deps -name "rangebar_bench-*" -type f -executable | head -1)

            if [[ -z "$bench_executable" ]]; then
                log_error "Benchmark executable not found. Run 'cargo build --release --bench rangebar_bench' first"
                return 1
            fi

            log_warn "dtruss requires admin privileges for system call tracing"
            log_info "dtruss profiling disabled for security - use Instruments instead on macOS"
            log_info "Alternative: xcrun xctrace record --template 'System Trace' --launch '$bench_executable'"
        else
            log_warn "dtruss not available"
        fi
    fi
}

comprehensive_profiling() {
    log_info "Running comprehensive profiling suite..."

    # Ensure tools are installed
    install_profiling_tools

    # Run all profiling types
    log_info "Step 1/4: CPU profiling..."
    cpu_profiling

    log_info "Step 2/4: Memory profiling..."
    memory_profiling

    log_info "Step 3/4: I/O profiling..."
    io_profiling

    log_info "Step 4/4: Flamegraph generation..."
    generate_flamegraph "1000000"

    log_success "Comprehensive profiling complete!"
    log_info "Results saved in: $PROFILE_DATA_DIR/"
}

profile_specific_workload() {
    local workload="${1:-high-frequency}"
    log_info "Profiling specific workload: $workload"

    cd "$PROJECT_ROOT"

    case "$workload" in
        "high-frequency")
            log_info "Profiling high-frequency tick processing (1M ticks)..."
            generate_flamegraph "1000000"
            ;;
        "extreme-volatility")
            log_info "Profiling extreme volatility scenario..."
            generate_flamegraph "high_volatility"
            ;;
        "memory-intensive")
            log_info "Profiling memory-intensive batch processing..."
            memory_profiling
            ;;
        "breach-detection")
            log_info "Profiling breach detection performance..."
            generate_flamegraph "breach_detection"
            ;;
        *)
            log_error "Unknown workload: $workload"
            log_info "Available workloads: high-frequency, extreme-volatility, memory-intensive, breach-detection"
            ;;
    esac
}

show_help() {
    echo "Rangebar Profiling Tools"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  install     Install required profiling tools"
    echo "  flamegraph  Generate flamegraph for CPU profiling"
    echo "  cpu         Run CPU profiling analysis"
    echo "  memory      Run memory profiling analysis"
    echo "  io          Run I/O profiling analysis"
    echo "  full        Run comprehensive profiling suite"
    echo "  workload    Profile specific workload [high-frequency|extreme-volatility|memory-intensive|breach-detection]"
    echo "  help        Show this help message"
    echo ""
    echo "High-frequency processing targets:"
    echo "  • 1M ticks: < 100ms"
    echo "  • Memory efficiency: < 1GB for 10M ticks"
    echo "  • CPU utilization: maximize single-core performance"
    echo ""
    echo "Examples:"
    echo "  $0 install                           # Install profiling tools"
    echo "  $0 flamegraph                       # Generate CPU flamegraph"
    echo "  $0 workload high-frequency          # Profile 1M tick processing"
    echo "  $0 full                             # Comprehensive profiling"
}

# Main command handling
case "${1:-help}" in
    "install")
        install_profiling_tools
        ;;
    "flamegraph")
        generate_flamegraph "${2:-1000000}"
        ;;
    "cpu")
        cpu_profiling
        ;;
    "memory")
        memory_profiling
        ;;
    "io")
        io_profiling
        ;;
    "full")
        comprehensive_profiling
        ;;
    "workload")
        profile_specific_workload "${2:-high-frequency}"
        ;;
    "help"|*)
        show_help
        ;;
esac

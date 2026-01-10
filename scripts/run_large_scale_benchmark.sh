#!/bin/bash

# Large-Scale GPU vs CPU Benchmarking Framework Runner
# Usage: ./scripts/run_large_scale_benchmark.sh [quick|production|default]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BENCHMARK_MODE="${1:-default}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./output/large_scale_benchmark_${BENCHMARK_MODE}_${TIMESTAMP}"

echo -e "${BLUE}🚀 Large-Scale GPU vs CPU Benchmarking Framework${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Check if we're in the correct directory
if [[ ! -f "Cargo.toml" ]]; then
    echo -e "${RED}❌ Error: Must be run from the rangebar project root directory${NC}"
    exit 1
fi

# Check if the benchmark binary exists in Cargo.toml
if ! grep -q "large-scale-gpu-cpu-benchmark" Cargo.toml; then
    echo -e "${RED}❌ Error: large-scale-gpu-cpu-benchmark binary not found in Cargo.toml${NC}"
    exit 1
fi

# Function to check system requirements
check_system_requirements() {
    echo -e "${YELLOW}🔍 Checking system requirements...${NC}"

    # Check available memory
    if command -v free >/dev/null 2>&1; then
        # Linux
        AVAILABLE_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
        TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    elif command -v vm_stat >/dev/null 2>&1; then
        # macOS
        FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        INACTIVE_PAGES=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
        AVAILABLE_PAGES=$((FREE_PAGES + INACTIVE_PAGES))
        AVAILABLE_MEM_GB=$((AVAILABLE_PAGES * 4096 / 1024 / 1024 / 1024))
        TOTAL_MEM_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    else
        AVAILABLE_MEM_GB=8  # Default assumption
        TOTAL_MEM_GB=16
    fi

    echo -e "  💾 Available Memory: ${AVAILABLE_MEM_GB}GB / ${TOTAL_MEM_GB}GB"

    # Check disk space
    AVAILABLE_DISK_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
    echo -e "  💿 Available Disk Space: ${AVAILABLE_DISK_GB}GB"

    # Check CPU cores
    if command -v nproc >/dev/null 2>&1; then
        CPU_CORES=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        CPU_CORES=4  # Default assumption
    fi
    echo -e "  🔧 CPU Cores: ${CPU_CORES}"

    # Memory recommendations
    case $BENCHMARK_MODE in
        "quick")
            RECOMMENDED_MEM_GB=4
            ;;
        "production")
            RECOMMENDED_MEM_GB=16
            ;;
        *)
            RECOMMENDED_MEM_GB=8
            ;;
    esac

    if [[ $AVAILABLE_MEM_GB -lt $RECOMMENDED_MEM_GB ]]; then
        echo -e "${YELLOW}⚠️  Warning: Available memory (${AVAILABLE_MEM_GB}GB) is less than recommended (${RECOMMENDED_MEM_GB}GB)${NC}"
        echo -e "${YELLOW}   Consider using --quick mode or freeing up memory${NC}"
    fi

    if [[ $AVAILABLE_DISK_GB -lt 5 ]]; then
        echo -e "${YELLOW}⚠️  Warning: Low disk space (${AVAILABLE_DISK_GB}GB). Benchmark may fail.${NC}"
    fi

    echo ""
}

# Function to check GPU availability
check_gpu_availability() {
    echo -e "${YELLOW}🎮 Checking GPU availability...${NC}"

    GPU_AVAILABLE=false

    # Check for NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "  🟢 NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
        GPU_AVAILABLE=true
    fi

    # Check for AMD GPU (Linux)
    if command -v rocm-smi >/dev/null 2>&1; then
        echo -e "  🟢 AMD GPU detected"
        GPU_AVAILABLE=true
    fi

    # Check for Apple Silicon (macOS)
    if [[ "$(uname)" == "Darwin" ]] && system_profiler SPHardwareDataType | grep -q "Apple"; then
        echo -e "  🟢 Apple Silicon GPU detected"
        GPU_AVAILABLE=true
    fi

    if [[ "$GPU_AVAILABLE" == false ]]; then
        echo -e "  🟡 No GPU detected - will run CPU-only benchmarks"
    fi

    echo ""
}

# Function to build the benchmark
build_benchmark() {
    echo -e "${YELLOW}🔨 Building benchmark framework...${NC}"

    if [[ "$GPU_AVAILABLE" == true ]]; then
        echo -e "  Building with GPU support..."
        if ! cargo build --release --features gpu --bin large-scale-gpu-cpu-benchmark; then
            echo -e "${RED}❌ Build failed with GPU features. Trying without GPU...${NC}"
            cargo build --release --bin large-scale-gpu-cpu-benchmark
        fi
    else
        echo -e "  Building without GPU support..."
        cargo build --release --bin large-scale-gpu-cpu-benchmark
    fi

    echo -e "${GREEN}✅ Build completed successfully${NC}"
    echo ""
}

# Function to run the benchmark
run_benchmark() {
    echo -e "${YELLOW}🏃 Starting benchmark execution...${NC}"
    echo -e "  Mode: ${BENCHMARK_MODE}"
    echo -e "  Output: ${OUTPUT_DIR}"
    echo -e "  Started: $(date)"
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Set environment variables
    export RUST_LOG=info
    export RUST_BACKTRACE=1

    # Determine benchmark arguments
    case $BENCHMARK_MODE in
        "quick")
            BENCHMARK_ARGS="--quick"
            ;;
        "production")
            BENCHMARK_ARGS="--production"
            ;;
        *)
            BENCHMARK_ARGS=""
            ;;
    esac

    # Run the benchmark
    echo -e "${BLUE}🚀 Executing benchmark...${NC}"

    if [[ "$GPU_AVAILABLE" == true ]]; then
        if ! cargo run --release --features gpu --bin large-scale-gpu-cpu-benchmark -- $BENCHMARK_ARGS; then
            echo -e "${YELLOW}⚠️  GPU benchmark failed, trying CPU-only...${NC}"
            cargo run --release --bin large-scale-gpu-cpu-benchmark -- $BENCHMARK_ARGS
        fi
    else
        cargo run --release --bin large-scale-gpu-cpu-benchmark -- $BENCHMARK_ARGS
    fi

    echo ""
    echo -e "${GREEN}✅ Benchmark execution completed${NC}"
}

# Function to analyze results
analyze_results() {
    echo -e "${YELLOW}📊 Analyzing results...${NC}"

    # Find the latest results file
    LATEST_RESULT=$(find ./output/large_scale_benchmark* -name "*.json" -type f 2>/dev/null | sort | tail -1)

    if [[ -z "$LATEST_RESULT" ]]; then
        echo -e "${YELLOW}⚠️  No results file found for analysis${NC}"
        return
    fi

    echo -e "  Latest results: $LATEST_RESULT"

    # Extract key metrics using jq if available
    if command -v jq >/dev/null 2>&1; then
        echo -e "\n📈 Key Performance Metrics:"

        TOTAL_TESTS=$(jq -r '.summary_statistics.total_tests_run // "N/A"' "$LATEST_RESULT")
        SUCCESS_RATE=$(jq -r '.summary_statistics.algorithmic_parity_success_rate // "N/A"' "$LATEST_RESULT")
        MEAN_SPEEDUP=$(jq -r '.summary_statistics.mean_speedup_factor // "N/A"' "$LATEST_RESULT")
        MAX_CPU_THROUGHPUT=$(jq -r '.summary_statistics.max_cpu_throughput_trades_per_sec // "N/A"' "$LATEST_RESULT")
        MAX_GPU_THROUGHPUT=$(jq -r '.summary_statistics.max_gpu_throughput_trades_per_sec // "N/A"' "$LATEST_RESULT")
        TOTAL_COST=$(jq -r '.summary_statistics.estimated_total_cost_usd // "N/A"' "$LATEST_RESULT")

        echo -e "  • Total Tests: $TOTAL_TESTS"
        echo -e "  • Success Rate: ${SUCCESS_RATE}%"
        echo -e "  • Mean GPU Speedup: ${MEAN_SPEEDUP}x"
        echo -e "  • Max CPU Throughput: $(printf "%.0f" "$MAX_CPU_THROUGHPUT") trades/sec"
        echo -e "  • Max GPU Throughput: $(printf "%.0f" "$MAX_GPU_THROUGHPUT") trades/sec"
        echo -e "  • Estimated Cost: \$${TOTAL_COST}"

        # Performance recommendation
        if (( $(echo "$MEAN_SPEEDUP > 2.0" | bc -l 2>/dev/null || echo "0") )); then
            echo -e "\n${GREEN}🎉 RECOMMENDATION: GPU provides significant performance benefits!${NC}"
        elif (( $(echo "$MEAN_SPEEDUP > 1.1" | bc -l 2>/dev/null || echo "0") )); then
            echo -e "\n${YELLOW}👍 RECOMMENDATION: GPU provides moderate performance benefits${NC}"
        else
            echo -e "\n${YELLOW}⚠️  RECOMMENDATION: GPU benefits are minimal - investigate bottlenecks${NC}"
        fi
    else
        echo -e "  💡 Install 'jq' for detailed analysis: brew install jq (macOS) or apt install jq (Ubuntu)"
    fi

    echo ""
}

# Function to provide next steps
provide_next_steps() {
    echo -e "${BLUE}📋 Next Steps:${NC}"
    echo -e "  1. Review results in: ${OUTPUT_DIR}/"
    echo -e "  2. Check detailed JSON results for analysis"
    echo -e "  3. Compare with baseline performance if available"
    echo -e "  4. Archive results for historical tracking"
    echo ""

    echo -e "${BLUE}📚 Documentation:${NC}"
    echo -e "  • Framework Guide: docs/LARGE_SCALE_BENCHMARKING.md"
    echo -e "  • GPU Implementation: docs/GPU_IMPLEMENTATION.md"
    echo -e "  • Performance Analysis: docs/PERFORMANCE.md"
    echo ""

    if [[ "$BENCHMARK_MODE" == "quick" ]]; then
        echo -e "${YELLOW}💡 For comprehensive testing, run: ./scripts/run_large_scale_benchmark.sh production${NC}"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [quick|production|default]"
    echo ""
    echo "Modes:"
    echo "  quick       - Fast validation (1K-100K trades, limited symbols)"
    echo "  production  - Full production scale (up to 50M trades, all symbols)"
    echo "  default     - Balanced configuration (1K-10M trades)"
    echo ""
    echo "Examples:"
    echo "  $0 quick                    # Quick test for development"
    echo "  $0 production              # Full production benchmark"
    echo "  $0                         # Default balanced test"
    echo ""
}

# Main execution
main() {
    # Check for help
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi

    # Validate mode
    if [[ "$BENCHMARK_MODE" != "quick" ]] && [[ "$BENCHMARK_MODE" != "production" ]] && [[ "$BENCHMARK_MODE" != "default" ]]; then
        echo -e "${RED}❌ Invalid mode: $BENCHMARK_MODE${NC}"
        show_usage
        exit 1
    fi

    # Run benchmark pipeline
    check_system_requirements
    check_gpu_availability
    build_benchmark
    run_benchmark
    analyze_results
    provide_next_steps

    echo -e "${GREEN}🏁 Large-scale benchmarking completed successfully!${NC}"
    echo -e "${GREEN}Results saved to: ${OUTPUT_DIR}${NC}"
}

# Execute main function
main "$@"

#!/bin/bash
# =============================================================================
# build-release.sh - Multi-platform wheel builder for rangebar-py
# =============================================================================
# Builds wheels for:
#   - macOS ARM64 (local)
#   - macOS x86_64 (cross-compile)
#   - Linux x86_64 (bigblack SSH)
#   - Source distribution (sdist)
#
# Uses abi3-py39 for single wheel per platform.
# =============================================================================

set -euo pipefail

# Configuration
LINUX_BUILD_HOST="${LINUX_BUILD_HOST:-bigblack}"
LINUX_BUILD_USER="${LINUX_BUILD_USER:-tca}"
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"
SKIP_REMOTE="${SKIP_REMOTE:-false}"
SKIP_VERIFY="${SKIP_VERIFY:-false}"
SKIP_MACOS_X86="${SKIP_MACOS_X86:-false}"

# Project paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${PROJECT_DIR}/dist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $*"
    else
        if [[ "$VERBOSE" == "true" ]]; then
            echo "[CMD] $*"
        fi
        "$@"
    fi
}

# -----------------------------------------------------------------------------
# Build Functions
# -----------------------------------------------------------------------------

build_macos_arm64() {
    log_info "Building macOS ARM64 wheel (native)..."

    cd "$PROJECT_DIR"
    run_cmd maturin build --release --profile wheel

    # Find the built wheel
    local wheel
    wheel=$(find "$DIST_DIR" -name "*macosx*arm64*.whl" -newer /tmp/.build_start 2>/dev/null | head -1)

    if [[ -n "$wheel" ]]; then
        log_success "Built: $(basename "$wheel")"
    else
        log_warn "macOS ARM64 wheel not found (may already exist)"
    fi
}

build_macos_x86_64() {
    if [[ "$SKIP_MACOS_X86" == "true" ]]; then
        log_info "Skipping macOS x86_64 build (SKIP_MACOS_X86=true)"
        return 0
    fi

    log_info "Building macOS x86_64 wheel (cross-compile)..."

    # Check if target is installed
    if ! rustup target list --installed | grep -q "x86_64-apple-darwin"; then
        log_info "Installing x86_64-apple-darwin target..."
        run_cmd rustup target add x86_64-apple-darwin
    fi

    cd "$PROJECT_DIR"
    run_cmd maturin build --release --profile wheel --target x86_64-apple-darwin

    local wheel
    wheel=$(find "$DIST_DIR" -name "*macosx*x86_64*.whl" -newer /tmp/.build_start 2>/dev/null | head -1)

    if [[ -n "$wheel" ]]; then
        log_success "Built: $(basename "$wheel")"
    else
        log_warn "macOS x86_64 wheel not found (may already exist)"
    fi
}

build_sdist() {
    log_info "Building source distribution..."

    cd "$PROJECT_DIR"
    run_cmd maturin sdist

    local sdist
    sdist=$(find "$DIST_DIR" -name "*.tar.gz" -newer /tmp/.build_start 2>/dev/null | head -1)

    if [[ -n "$sdist" ]]; then
        log_success "Built: $(basename "$sdist")"
    else
        log_warn "Source distribution not found (may already exist)"
    fi
}

build_linux_x86_64() {
    if [[ "$SKIP_REMOTE" == "true" ]]; then
        log_info "Skipping Linux x86_64 build (SKIP_REMOTE=true)"
        return 0
    fi

    log_info "Building Linux x86_64 wheel on ${LINUX_BUILD_HOST}..."

    local remote_dir="/tmp/rangebar-py-build-$$"
    local version
    version=$(grep '^version = ' "$PROJECT_DIR/pyproject.toml" | head -1 | sed 's/.*= "\(.*\)"/\1/')

    # Sync project to remote
    log_info "Syncing project to ${LINUX_BUILD_HOST}:${remote_dir}..."
    run_cmd ssh "${LINUX_BUILD_USER}@${LINUX_BUILD_HOST}" "mkdir -p ${remote_dir}"
    run_cmd rsync -az --delete \
        --exclude 'target/' \
        --exclude 'dist/' \
        --exclude '.git/' \
        --exclude '__pycache__/' \
        --exclude '*.egg-info/' \
        "$PROJECT_DIR/" "${LINUX_BUILD_USER}@${LINUX_BUILD_HOST}:${remote_dir}/"

    # Build on remote
    log_info "Building wheel on ${LINUX_BUILD_HOST}..."
    run_cmd ssh "${LINUX_BUILD_USER}@${LINUX_BUILD_HOST}" "
        cd ${remote_dir}
        export PATH=\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH
        uvx maturin build --release --profile wheel --compatibility manylinux_2_17
    "

    # Fetch wheel back
    log_info "Fetching wheel from ${LINUX_BUILD_HOST}..."
    run_cmd mkdir -p "$DIST_DIR"
    run_cmd scp "${LINUX_BUILD_USER}@${LINUX_BUILD_HOST}:${remote_dir}/target/wheels/*manylinux*.whl" "$DIST_DIR/"

    # Cleanup remote
    log_info "Cleaning up remote build directory..."
    run_cmd ssh "${LINUX_BUILD_USER}@${LINUX_BUILD_HOST}" "rm -rf ${remote_dir}"

    local wheel
    wheel=$(find "$DIST_DIR" -name "*manylinux*.whl" -newer /tmp/.build_start 2>/dev/null | head -1)

    if [[ -n "$wheel" ]]; then
        log_success "Built: $(basename "$wheel")"
    else
        log_error "Linux x86_64 wheel not found"
        return 1
    fi
}

verify_wheels() {
    if [[ "$SKIP_VERIFY" == "true" ]]; then
        log_info "Skipping wheel verification (SKIP_VERIFY=true)"
        return 0
    fi

    log_info "Verifying built wheels..."

    local wheel_count
    wheel_count=$(find "$DIST_DIR" -name "*.whl" | wc -l | tr -d ' ')

    if [[ "$wheel_count" -eq 0 ]]; then
        log_error "No wheels found in $DIST_DIR"
        return 1
    fi

    log_info "Found $wheel_count wheel(s) in $DIST_DIR"

    # Test macOS ARM64 wheel locally if available
    local local_wheel
    local_wheel=$(find "$DIST_DIR" -name "*macosx*arm64*.whl" | head -1)

    if [[ -n "$local_wheel" ]]; then
        log_info "Smoke testing local wheel: $(basename "$local_wheel")"

        # Create temp venv
        local temp_venv
        temp_venv=$(mktemp -d)
        run_cmd python3 -m venv "$temp_venv"

        # Install and test
        run_cmd "$temp_venv/bin/pip" install --quiet "$local_wheel"
        run_cmd "$temp_venv/bin/python" -c "
from rangebar._core import PyRangeBarProcessor
proc = PyRangeBarProcessor(threshold_decimal_bps=250)
assert proc.threshold_decimal_bps == 250
bars = proc.process_trades([])
assert bars == []
print('OK: Smoke test passed')
"

        # Cleanup
        rm -rf "$temp_venv"
        log_success "Smoke test passed for $(basename "$local_wheel")"
    fi
}

generate_manifest() {
    log_info "Generating build manifest..."

    local manifest_file="${DIST_DIR}/MANIFEST.txt"
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    {
        echo "# rangebar-py Build Manifest"
        echo "# Generated: $timestamp"
        echo ""
        echo "## Artifacts"
        echo ""

        for file in "$DIST_DIR"/*.whl "$DIST_DIR"/*.tar.gz; do
            if [[ -f "$file" ]]; then
                local name
                name=$(basename "$file")
                local size
                size=$(stat -f%z "$file" 2>/dev/null || stat --format=%s "$file" 2>/dev/null)
                local sha256
                sha256=$(shasum -a 256 "$file" | cut -d' ' -f1)

                echo "- $name"
                echo "  Size: $size bytes"
                echo "  SHA256: $sha256"
                echo ""
            fi
        done

        echo "## Build Environment"
        echo ""
        echo "- Rust: $(rustc --version)"
        echo "- Maturin: $(maturin --version)"
        echo "- Python: $(python3 --version)"
        echo "- Host: $(uname -s) $(uname -m)"

    } > "$manifest_file"

    log_success "Manifest written to $manifest_file"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    log_info "rangebar-py Release Builder"
    log_info "Project: $PROJECT_DIR"
    log_info "Output: $DIST_DIR"
    echo ""

    # Create dist directory
    mkdir -p "$DIST_DIR"

    # Mark build start time
    touch /tmp/.build_start

    # Build all platforms
    build_macos_arm64
    build_macos_x86_64
    build_sdist
    build_linux_x86_64

    # Verify and generate manifest
    verify_wheels
    generate_manifest

    echo ""
    log_success "Build complete!"
    echo ""
    log_info "Artifacts in $DIST_DIR:"
    ls -la "$DIST_DIR"/*.whl "$DIST_DIR"/*.tar.gz 2>/dev/null || true
}

# Run
main "$@"

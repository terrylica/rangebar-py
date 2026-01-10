#!/bin/bash
# =============================================================================
# publish-wheels.sh - PyPI publisher for rangebar-py
# =============================================================================
# Publishes all wheels in dist/ to PyPI using uv publish.
# Uses Doppler for credential management.
#
# Usage:
#   ./scripts/publish-wheels.sh           # Publish to PyPI
#   ./scripts/publish-wheels.sh --dry-run # Show what would be published
#   ./scripts/publish-wheels.sh --test    # Publish to TestPyPI
# =============================================================================

set -euo pipefail

# Configuration
DRY_RUN="${DRY_RUN:-false}"
TEST_PYPI="${TEST_PYPI:-false}"
DOPPLER_PROJECT="${DOPPLER_PROJECT:-pypi}"
DOPPLER_CONFIG="${DOPPLER_CONFIG:-prd}"

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

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Publish rangebar-py wheels to PyPI.

Options:
    --dry-run       Show what would be published without uploading
    --test          Publish to TestPyPI instead of PyPI
    --help          Show this help message

Environment Variables:
    DOPPLER_PROJECT    Doppler project name (default: pypi)
    DOPPLER_CONFIG     Doppler config name (default: prd)

Examples:
    $(basename "$0")              # Publish to PyPI
    $(basename "$0") --dry-run    # Dry run
    $(basename "$0") --test       # Publish to TestPyPI
EOF
}

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

validate_environment() {
    log_info "Validating environment..."

    # Check for required tools
    local missing_tools=()

    if ! command -v uv &> /dev/null; then
        missing_tools+=("uv")
    fi

    if ! command -v doppler &> /dev/null; then
        missing_tools+=("doppler")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Install with: mise install"
        exit 1
    fi

    # Check Doppler authentication
    if ! doppler whoami &> /dev/null; then
        log_error "Doppler not authenticated. Run: doppler login"
        exit 1
    fi

    log_success "Environment validated"
}

validate_artifacts() {
    log_info "Validating artifacts in $DIST_DIR..."

    if [[ ! -d "$DIST_DIR" ]]; then
        log_error "Dist directory not found: $DIST_DIR"
        log_error "Run: mise run release"
        exit 1
    fi

    local wheel_count
    wheel_count=$(find "$DIST_DIR" -name "*.whl" | wc -l | tr -d ' ')

    if [[ "$wheel_count" -eq 0 ]]; then
        log_error "No wheels found in $DIST_DIR"
        log_error "Run: mise run release"
        exit 1
    fi

    log_info "Found $wheel_count wheel(s):"
    find "$DIST_DIR" -name "*.whl" -exec basename {} \;

    local sdist_count
    sdist_count=$(find "$DIST_DIR" -name "*.tar.gz" | wc -l | tr -d ' ')

    if [[ "$sdist_count" -gt 0 ]]; then
        log_info "Found $sdist_count source distribution(s):"
        find "$DIST_DIR" -name "*.tar.gz" -exec basename {} \;
    fi

    log_success "Artifacts validated"
}

# -----------------------------------------------------------------------------
# Publishing
# -----------------------------------------------------------------------------

get_pypi_token() {
    local secret_name="PYPI_API_TOKEN"
    if [[ "$TEST_PYPI" == "true" ]]; then
        secret_name="TEST_PYPI_API_TOKEN"
    fi

    log_info "Fetching PyPI token from Doppler..."

    local token
    token=$(doppler secrets get "$secret_name" --project "$DOPPLER_PROJECT" --config "$DOPPLER_CONFIG" --plain 2>/dev/null)

    if [[ -z "$token" ]]; then
        log_error "Failed to fetch $secret_name from Doppler"
        log_error "Ensure the secret exists in project=$DOPPLER_PROJECT config=$DOPPLER_CONFIG"
        exit 1
    fi

    echo "$token"
}

publish_to_pypi() {
    log_info "Publishing to PyPI..."

    local token
    token=$(get_pypi_token)

    local publish_url=""
    if [[ "$TEST_PYPI" == "true" ]]; then
        publish_url="https://test.pypi.org/legacy/"
        log_info "Target: TestPyPI"
    else
        log_info "Target: PyPI (production)"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY-RUN] Would publish the following artifacts:"
        find "$DIST_DIR" -name "*.whl" -o -name "*.tar.gz" | while read -r file; do
            echo "  - $(basename "$file")"
        done
        return 0
    fi

    # Build uv publish command
    local cmd=(uv publish)

    if [[ -n "$publish_url" ]]; then
        cmd+=(--publish-url "$publish_url")
    fi

    # Set token via environment variable
    export UV_PUBLISH_TOKEN="$token"

    # Publish all artifacts
    for file in "$DIST_DIR"/*.whl "$DIST_DIR"/*.tar.gz; do
        if [[ -f "$file" ]]; then
            log_info "Publishing: $(basename "$file")"
            "${cmd[@]}" "$file"
            log_success "Published: $(basename "$file")"
        fi
    done

    unset UV_PUBLISH_TOKEN

    log_success "All artifacts published successfully!"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --test)
                TEST_PYPI="true"
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    log_info "rangebar-py PyPI Publisher"
    echo ""

    # Validation
    validate_environment
    validate_artifacts

    # Publish
    echo ""
    publish_to_pypi

    echo ""
    if [[ "$TEST_PYPI" == "true" ]]; then
        log_success "Published to TestPyPI!"
        log_info "Install with: pip install --index-url https://test.pypi.org/simple/ rangebar"
    else
        log_success "Published to PyPI!"
        log_info "Install with: pip install rangebar"
    fi
}

# Run
main "$@"

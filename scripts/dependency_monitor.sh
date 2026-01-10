#!/bin/bash
# Dependency monitoring and update automation for rangebar
#
# Usage: ./scripts/dependency_monitor.sh [check|update|security|audit]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log_info() {
    echo "📦 [DEPS] $1"
}

log_success() {
    echo "✅ [DEPS] $1"
}

log_warn() {
    echo "⚠️ [DEPS] $1"
}

log_error() {
    echo "❌ [DEPS] $1"
}

check_outdated() {
    log_info "Checking for outdated dependencies..."
    cd "$PROJECT_ROOT"

    # Check if cargo-outdated is installed
    if ! command -v cargo-outdated &> /dev/null; then
        log_warn "cargo-outdated not found. Installing..."
        cargo install cargo-outdated
    fi

    # Check for outdated dependencies
    log_info "Outdated dependencies report:"
    cargo outdated -R --color always

    log_info "Security advisories check..."
    # Note: cargo-audit removed earlier, but security is covered by cargo-deny
    if cargo deny check advisories 2>/dev/null; then
        log_success "No security advisories found"
    else
        log_warn "Security advisories found - review required"
    fi
}

update_dependencies() {
    log_info "Updating dependencies..."
    cd "$PROJECT_ROOT"

    # Update Cargo.lock
    log_info "Updating Cargo.lock..."
    cargo update

    # Run tests after update
    log_info "Running tests after dependency update..."
    if cargo nextest run --quiet; then
        log_success "All tests pass after dependency update"
    else
        log_error "Tests failed after dependency update"
        log_error "Consider reverting or investigating compatibility issues"
        return 1
    fi

    # Run performance validation
    log_info "Validating performance after update..."
    if ./scripts/benchmark_runner.sh validate; then
        log_success "Performance targets met after dependency update"
    else
        log_warn "Performance regression detected - review recommended"
    fi
}

security_audit() {
    log_info "Running security audit..."
    cd "$PROJECT_ROOT"

    # Check for security advisories using cargo-deny
    log_info "Checking security advisories..."
    if cargo deny check advisories; then
        log_success "No security vulnerabilities found"
    else
        log_error "Security vulnerabilities detected!"
        log_error "Run 'cargo deny check' for detailed information"
        return 1
    fi

    # Check for known vulnerable versions
    log_info "Checking dependency versions for known issues..."
    # Custom checks for critical dependencies - secure metadata parsing

    local metadata_file="/tmp/cargo_metadata_$$.json"
    local polars_version="not found"
    local pyo3_version="not found"

    # Securely capture cargo metadata
    if cargo metadata --format-version 1 > "$metadata_file" 2>/dev/null; then
        # Safely extract versions using controlled jq operations
        polars_version=$(jq -r '.packages[] | select(.name == "polars") | .version' < "$metadata_file" 2>/dev/null || echo "not found")
        pyo3_version=$(jq -r '.packages[] | select(.name == "pyo3") | .version' < "$metadata_file" 2>/dev/null || echo "not found")
        rm -f "$metadata_file"
    else
        log_warn "Failed to retrieve cargo metadata for security check"
    fi

    log_info "Critical dependency versions:"
    log_info "  polars: $polars_version"
    log_info "  pyo3: $pyo3_version"

    # Check against known vulnerable versions
    if [[ "$pyo3_version" == "0.22."* ]]; then
        log_warn "PyO3 0.22.x has known security issues - update to 0.26+ recommended"
    fi
}

dependency_health_check() {
    log_info "Running comprehensive dependency health check..."
    cd "$PROJECT_ROOT"

    # Check for duplicate dependencies - secure command execution
    log_info "Checking for duplicate dependencies..."
    if command -v cargo-tree &> /dev/null; then
        local duplicates_file="/tmp/cargo_duplicates_$$.txt"
        if cargo tree --duplicates --format "{p}" > "$duplicates_file" 2>/dev/null; then
            if [[ -s "$duplicates_file" ]]; then
                log_warn "Duplicate dependencies found:"
                cat "$duplicates_file"
            else
                log_success "No duplicate dependencies found"
            fi
            rm -f "$duplicates_file"
        else
            log_warn "Failed to check for duplicate dependencies"
        fi
    else
        log_info "cargo-tree not available - skipping duplicate check"
    fi

    # Analyze dependency sizes - secure metadata parsing
    log_info "Analyzing build dependency impact..."
    local metadata_file="/tmp/cargo_metadata_deps_$$.json"
    local dep_count="unknown"

    if cargo metadata --format-version 1 > "$metadata_file" 2>/dev/null; then
        dep_count=$(jq '.packages | length' < "$metadata_file" 2>/dev/null || echo "unknown")
        rm -f "$metadata_file"
    else
        log_warn "Failed to retrieve cargo metadata for dependency analysis"
    fi

    log_info "Total dependencies: $dep_count"

    # License compliance check
    log_info "Checking license compliance..."
    if cargo deny check licenses 2>/dev/null; then
        log_success "All licenses compliant"
    else
        log_warn "License compliance issues detected"
    fi
}

automated_update_workflow() {
    log_info "Running automated update workflow..."

    # 1. Check current status
    check_outdated

    # 2. Security audit first
    if ! security_audit; then
        log_error "Security issues detected - manual review required"
        return 1
    fi

    # 3. Create backup of Cargo.lock
    cp Cargo.lock Cargo.lock.backup
    log_info "Backup created: Cargo.lock.backup"

    # 4. Update dependencies
    if update_dependencies; then
        log_success "Dependencies updated successfully"

        # 5. Clean up backup
        rm Cargo.lock.backup
        log_info "Update completed - backup removed"
    else
        log_error "Update failed - restoring backup"
        mv Cargo.lock.backup Cargo.lock
        return 1
    fi
}

show_help() {
    echo "Rangebar Dependency Monitoring"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  check      Check for outdated dependencies and security issues"
    echo "  update     Update dependencies with testing validation"
    echo "  security   Run comprehensive security audit"
    echo "  health     Run dependency health check (duplicates, licenses, etc.)"
    echo "  auto       Run automated update workflow (check -> security -> update)"
    echo "  help       Show this help message"
    echo ""
    echo "Dependencies monitored:"
    echo "  • Polars (data processing)"
    echo "  • PyO3 (Python bindings)"
    echo "  • Cryptographic libraries (sha2, md5, crc32fast)"
    echo "  • Statistics libraries (statrs, quantiles, nalgebra)"
    echo ""
    echo "Examples:"
    echo "  $0 check                     # Quick outdated check"
    echo "  $0 security                  # Security audit"
    echo "  $0 auto                      # Full automated workflow"
}

# Main command handling
case "${1:-help}" in
    "check")
        check_outdated
        ;;
    "update")
        update_dependencies
        ;;
    "security")
        security_audit
        ;;
    "health")
        dependency_health_check
        ;;
    "auto")
        automated_update_workflow
        ;;
    "help"|*)
        show_help
        ;;
esac

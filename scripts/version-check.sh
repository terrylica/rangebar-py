#!/usr/bin/env bash
# SSoT Version Verification
# Ensures Cargo.toml [workspace.package] version is the single source of truth
set -euo pipefail

echo "=== VERSION SSoT CHECK ==="

# Get workspace version (the SSoT)
WORKSPACE_VER=$(grep -A5 '\[workspace.package\]' Cargo.toml | grep '^version' | head -1 | sed 's/.*= "\(.*\)"/\1/')
echo "SSoT (workspace.package.version): $WORKSPACE_VER"

# Check PyO3 crate version matches
PACKAGE_VER=$(grep -A10 '\[package\]' Cargo.toml | grep -E '^version' | head -1 | sed 's/.*= "\(.*\)"/\1/')
if [ "$PACKAGE_VER" != "$WORKSPACE_VER" ]; then
    echo "FAIL: [package] version ($PACKAGE_VER) != workspace version ($WORKSPACE_VER)"
    exit 1
fi
echo "OK: [package] version matches"

# Check no version constraints in internal path deps
if grep -r 'path = "../' crates/*/Cargo.toml | grep 'version =' > /dev/null 2>&1; then
    echo "FAIL: Found version constraints in internal path dependencies"
    grep -r 'path = "../' crates/*/Cargo.toml | grep 'version =' || true
    echo "      Remove 'version = \"X.Y\"' from path deps - they inherit from workspace"
    exit 1
fi
echo "OK: No version constraints in path dependencies"

# Check pyproject.toml uses dynamic version
if ! grep -q 'dynamic = \["version"\]' pyproject.toml; then
    echo "FAIL: pyproject.toml should use dynamic = [\"version\"]"
    echo "      (maturin pulls version from Cargo.toml)"
    exit 1
fi
echo "OK: pyproject.toml uses dynamic version"

# Verify Python package reports correct version
PY_VER=$(python -c "import rangebar; print(rangebar.__version__)" 2>/dev/null || echo "not installed")
if [ "$PY_VER" != "$WORKSPACE_VER" ] && [ "$PY_VER" != "not installed" ]; then
    echo "WARN: Python package version ($PY_VER) != Cargo version ($WORKSPACE_VER)"
    echo "      Run 'maturin develop' to sync"
fi

echo ""
echo "=== VERSION SSoT: $WORKSPACE_VER ==="

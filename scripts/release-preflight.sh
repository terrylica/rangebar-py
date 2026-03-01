#!/usr/bin/env bash
# Enhanced release preflight — validates all prerequisites before release.
# Catches issues discovered during v12.38.1 release:
#   - Broken twine interpreter (bad Python shebang)
#   - Ruff lint failures blocking pre-commit hooks
#   - uv.lock drift from uv run/sync commands
#   - Missing/expired credentials (PyPI, crates.io, GitHub)
#
# Usage: mise run release:preflight
#        ./scripts/release-preflight.sh
set -euo pipefail

echo "=== PREFLIGHT ==="
FAIL=0
WARN=0

# ─── 1. Git State ────────────────────────────────────────────────────────────

echo ""
echo "--- 1. Git State ---"

git update-index --refresh -q || true

if [ -n "$(git status --porcelain)" ]; then
    echo "FAIL: Working directory not clean"
    git status --short
    FAIL=1
else
    echo "OK: Working directory clean"
fi

BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "FAIL: Not on main branch (on: $BRANCH)"
    FAIL=1
else
    echo "OK: On main branch"
fi

# Check for unpushed commits (use origin/main instead of @{u} to avoid shellcheck SC1083)
UNPUSHED=$(git log --oneline origin/main..HEAD 2>/dev/null | wc -l | tr -d ' ')
if [ "$UNPUSHED" -gt 0 ]; then
    echo "WARN: $UNPUSHED unpushed commit(s) — release:sync will push them"
    WARN=$((WARN + 1))
fi

# ─── 2. Tool Health ──────────────────────────────────────────────────────────

echo ""
echo "--- 2. Tool Health ---"

check_tool() {
    local name="$1"
    local cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "OK: $name"
        return 0
    else
        echo "FAIL: $name not working"
        return 1
    fi
}

check_tool "cargo" "cargo --version" || FAIL=1
check_tool "maturin" "maturin --version" || FAIL=1
check_tool "uv" "uv --version" || FAIL=1
check_tool "semantic-release" "semantic-release --version" || FAIL=1
check_tool "op (1Password CLI)" "op --version" || FAIL=1
check_tool "ruff" "ruff --version" || FAIL=1

# Twine: critical — broken interpreter caused v12.38.1 publish failure.
# The shim can exist but point to a deleted Python venv.
if command -v twine >/dev/null 2>&1; then
    TWINE_VER=""
    TWINE_VER=$(twine --version 2>&1 | head -1) || true
    if [ -n "$TWINE_VER" ]; then
        echo "OK: twine ($TWINE_VER)"
    else
        echo "FAIL: twine exists but has broken interpreter"
        echo "  Fix: uv tool install twine --force --python 3.13"
        FAIL=1
    fi
else
    echo "FAIL: twine not found"
    echo "  Fix: uv tool install twine --python 3.13"
    FAIL=1
fi

# ─── 3. Ruff Lint (config subpackage + recently changed files) ───────────────
# Only check the config subpackage (our new code) — the broader codebase has
# pre-existing PLC0415 (lazy imports) and format differences that are intentional.

echo ""
echo "--- 3. Ruff Lint ---"

if ruff check python/rangebar/config/ python/rangebar/cli.py python/rangebar/population.py --quiet 2>&1; then
    echo "OK: ruff lint clean (config + CLI)"
else
    echo "FAIL: ruff lint errors in config/CLI — fix before releasing"
    echo "  Run: ruff check python/rangebar/config/ python/rangebar/cli.py --fix"
    FAIL=1
fi

if ruff format --check python/rangebar/config/ python/rangebar/cli.py python/rangebar/population.py --quiet 2>&1; then
    echo "OK: ruff format clean (config + CLI)"
else
    echo "FAIL: ruff format issues in config/CLI — fix before releasing"
    echo "  Run: ruff format python/rangebar/config/ python/rangebar/cli.py"
    FAIL=1
fi

# ─── 4. Lockfile Consistency ─────────────────────────────────────────────────

echo ""
echo "--- 4. Lockfile Consistency ---"

if git diff --name-only | grep -q '^uv.lock$'; then
    echo "WARN: uv.lock has uncommitted changes (likely from uv run/sync)"
    echo "  Resetting to HEAD version..."
    git checkout -- uv.lock
    echo "OK: uv.lock reset"
    WARN=$((WARN + 1))
else
    echo "OK: uv.lock clean"
fi

# ─── 5. Credential Pre-Validation ───────────────────────────────────────────

echo ""
echo "--- 5. Credentials ---"

# 1Password service account token
_OP_SA_TOKEN_FILE="$HOME/.claude/.secrets/op-service-account-token"
if [ -f "$_OP_SA_TOKEN_FILE" ]; then
    export OP_SERVICE_ACCOUNT_TOKEN
    OP_SERVICE_ACCOUNT_TOKEN="$(cat "$_OP_SA_TOKEN_FILE")"
    echo "OK: 1Password service account token loaded"
else
    echo "WARN: No 1Password SA token file (will use biometric)"
    WARN=$((WARN + 1))
fi

# PyPI token (1Password)
OP_PYPI_ITEM="${OP_PYPI_ITEM:-zdc7ap2ixpqgtpq62xm2davi7e}"
OP_PYPI_VAULT="${OP_PYPI_VAULT:-ggk4orq7rmcm7jinsb4ahygv7e}"
PYPI_TOKEN=""
PYPI_TOKEN=$(op item get "$OP_PYPI_ITEM" --vault "$OP_PYPI_VAULT" --fields credential --reveal 2>/dev/null) || true
if [ -n "$PYPI_TOKEN" ]; then
    echo "OK: PyPI token from 1Password (${#PYPI_TOKEN} chars)"
else
    echo "FAIL: PyPI token not accessible from 1Password"
    echo "  Item: $OP_PYPI_ITEM | Vault: $OP_PYPI_VAULT"
    FAIL=1
fi

# crates.io token
CARGO_TOKEN=""
CARGO_TOKEN=$(OP_SERVICE_ACCOUNT_TOKEN="${OP_SERVICE_ACCOUNT_TOKEN:-}" \
    op read "op://Claude Automation/crates-io-token/credential" 2>/dev/null) || \
    CARGO_TOKEN=$(cat ~/.claude/.secrets/crates-io-token 2>/dev/null) || true
if [ -n "$CARGO_TOKEN" ]; then
    echo "OK: crates.io token available (${#CARGO_TOKEN} chars)"
else
    echo "FAIL: No crates.io token found"
    FAIL=1
fi

# GitHub token (for semantic-release)
GH_PAT=""
GH_PAT=$(doppler secrets get GH_TOKEN_TERRYLICA --project main --config dev --plain 2>/dev/null) || true
if [ -z "$GH_PAT" ]; then
    GH_PAT=$(GH_CONFIG_DIR="$HOME/.config/gh/profiles/terrylica" gh auth token 2>/dev/null) || true
fi
if [ -n "$GH_PAT" ]; then
    echo "OK: GitHub token available (${#GH_PAT} chars)"
else
    echo "FAIL: No GitHub token for semantic-release"
    FAIL=1
fi

# ─── 6. Registry Connectivity ───────────────────────────────────────────────

echo ""
echo "--- 6. Registry Connectivity ---"

if timeout 5 curl -sf "https://pypi.org/pypi/rangebar/json" >/dev/null 2>&1; then
    echo "OK: PyPI API reachable"
else
    echo "WARN: PyPI API unreachable (network issue?)"
    WARN=$((WARN + 1))
fi

if timeout 5 curl -sf "https://crates.io/api/v1/crates/rangebar-core" >/dev/null 2>&1; then
    echo "OK: crates.io API reachable"
else
    echo "WARN: crates.io API unreachable"
    WARN=$((WARN + 1))
fi

# ─── 7. Crates.io Packaging ─────────────────────────────────────────────────

echo ""
echo "--- 7. Crates.io Packaging ---"

if cargo publish --workspace --dry-run 2>&1; then
    echo "OK: All publishable crates package successfully"
else
    echo "FAIL: cargo publish --workspace --dry-run failed"
    FAIL=1
fi

# ─── 8. ClickHouse Port Consistency ──────────────────────────────────────────

echo ""
echo "--- 8. ClickHouse Port Consistency ---"

EXPECTED_PORT=8123
PORT_FAIL=false
for script in scripts/*.py; do
    PORT=$(grep -oE 'default=[0-9]+' "$script" 2>/dev/null | grep -oE '[0-9]+' | grep -E '^[0-9]*123$' || true)
    if [ -n "$PORT" ] && [ "$PORT" != "$EXPECTED_PORT" ]; then
        echo "FAIL: $script defaults to port $PORT, expected $EXPECTED_PORT"
        PORT_FAIL=true
        FAIL=1
    fi
done
if [ "$PORT_FAIL" = "false" ]; then
    echo "OK: ClickHouse port consistent"
fi

# ─── 9. Config Tests ────────────────────────────────────────────────────────

echo ""
echo "--- 9. Config Tests ---"

if uv run --python 3.13 pytest tests/test_config_telemetry.py -x -q --no-header 2>&1 | tail -3; then
    echo "OK: Config tests passed"
else
    echo "FAIL: Config tests failed"
    FAIL=1
fi

# ─── Cleanup ─────────────────────────────────────────────────────────────────

echo ""
echo "--- Cleanup ---"

if [ -d dist ]; then
    echo "Cleaning dist/ directory..."
    rm -rf dist
fi
mkdir -p dist

# Reset any lockfile drift caused by the pytest run above
if git diff --name-only | grep -q '^uv.lock$'; then
    git checkout -- uv.lock
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "========================================="
if [ $FAIL -ne 0 ]; then
    echo "FAIL: Preflight failed ($FAIL issue(s), $WARN warning(s))"
    echo "Fix issues above before releasing"
    exit 1
elif [ $WARN -gt 0 ]; then
    echo "OK: Preflight passed with $WARN warning(s)"
else
    echo "OK: Preflight passed (all checks clean)"
fi

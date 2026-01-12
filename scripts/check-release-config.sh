#!/usr/bin/env bash
# Check release configuration (GitHub, Doppler, SSH)
set -euo pipefail

echo "=== Release Configuration Check ==="
echo ""

# 1. GitHub Account
echo "1. GitHub Account:"
echo "   GH_ACCOUNT: ${GH_ACCOUNT:-NOT SET}"
echo "   GH_CONFIG_DIR: ${GH_CONFIG_DIR:-NOT SET}"
if [ -d "${GH_CONFIG_DIR:-}" ]; then
  echo "   Profile exists: YES"
else
  echo "   Profile exists: NO (run: gh auth login)"
fi
echo ""

# 2. Git Remote
echo "2. Git Remote:"
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "NOT SET")
echo "   URL: $REMOTE_URL"
if [[ "$REMOTE_URL" =~ github.com-terrylica ]]; then
  echo "   Identity: terrylica (SSH) ✓"
elif [[ "$REMOTE_URL" =~ github.com ]]; then
  echo "   Identity: UNKNOWN (may use wrong account!) ✗"
else
  echo "   Identity: UNKNOWN"
fi
echo ""

# 3. Doppler Secrets
echo "3. Doppler Secrets:"
if command -v doppler &>/dev/null; then
  # PyPI Token
  PYPI_TOKEN=$(doppler secrets get PYPI_TOKEN --project claude-config --config prd --plain 2>/dev/null || echo "")
  if [ -n "$PYPI_TOKEN" ]; then
    echo "   PYPI_TOKEN (claude-config/prd): SET (${#PYPI_TOKEN} chars) ✓"
  else
    echo "   PYPI_TOKEN (claude-config/prd): NOT SET ✗"
  fi

  # GitHub PAT
  GH_PAT=$(doppler secrets get GH_TOKEN_TERRYLICA --project main --config dev --plain 2>/dev/null || echo "")
  if [ -n "$GH_PAT" ]; then
    echo "   GH_TOKEN_TERRYLICA (main/dev): SET (${#GH_PAT} chars) ✓"
  else
    echo "   GH_TOKEN_TERRYLICA (main/dev): NOT SET (will use gh CLI fallback)"
  fi
else
  echo "   Doppler CLI: NOT INSTALLED ✗"
fi
echo ""

# 4. gh CLI Auth
echo "4. gh CLI Auth:"
if command -v gh &>/dev/null; then
  GH_AUTH_TOKEN=$(GH_CONFIG_DIR="$HOME/.config/gh/profiles/terrylica" gh auth token 2>/dev/null || echo "")
  if [ -n "$GH_AUTH_TOKEN" ]; then
    echo "   terrylica profile: AUTHENTICATED ✓"
  else
    echo "   terrylica profile: NOT AUTHENTICATED ✗"
    echo "   Fix: GH_CONFIG_DIR=~/.config/gh/profiles/terrylica gh auth login"
  fi
else
  echo "   gh CLI: NOT INSTALLED"
fi
echo ""

# 5. SSH Key
echo "5. SSH Key (github.com-terrylica):"
SSH_OUTPUT=$(ssh -T git@github.com-terrylica 2>&1 || true)
if echo "$SSH_OUTPUT" | grep -q "successfully authenticated"; then
  echo "   SSH: AUTHENTICATED as terrylica ✓"
elif echo "$SSH_OUTPUT" | grep -q "Hi "; then
  echo "   SSH: AUTHENTICATED ✓"
else
  echo "   SSH: NOT CONFIGURED ✗"
  echo "   Fix: Add github.com-terrylica host to ~/.ssh/config"
fi
echo ""

echo "=== Summary ==="
ISSUES=0

if [[ ! "$REMOTE_URL" =~ github.com-terrylica ]]; then
  echo "• FIX: git remote set-url origin git@github.com-terrylica:terrylica/rangebar-py.git"
  ISSUES=$((ISSUES + 1))
fi

if [ -z "$(doppler secrets get PYPI_TOKEN --project claude-config --config prd --plain 2>/dev/null || echo '')" ]; then
  echo "• FIX: Add PYPI_TOKEN to Doppler (claude-config/prd)"
  ISSUES=$((ISSUES + 1))
fi

if [ $ISSUES -eq 0 ]; then
  echo "All checks passed! ✓"
  echo ""
  echo "Release workflow:"
  echo "  1. mise run release:full    - Version, build"
  echo "  2. mise run release:pypi    - Publish to PyPI"
fi

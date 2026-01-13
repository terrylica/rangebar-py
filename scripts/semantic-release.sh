#!/usr/bin/env bash
# Run semantic-release with proper GitHub token from Doppler
# Part of the local-only release workflow (ADR-0027)
set -euo pipefail

echo "=== Semantic Release ==="

# Ensure bun dependencies
if [ ! -d node_modules ]; then
  echo "Installing dependencies via bun..."
  bun install
fi

# Verify git remote uses SSH with correct identity
REMOTE_URL=$(git remote get-url origin)
if [[ ! "$REMOTE_URL" =~ github.com-terrylica ]]; then
  echo "WARN: Git remote should use SSH with terrylica identity"
  echo "  Current: $REMOTE_URL"
  echo "  Expected: git@github.com-terrylica:terrylica/rangebar-py.git"
  echo ""
  echo "Fixing remote URL..."
  git remote set-url origin "git@github.com-terrylica:terrylica/rangebar-py.git"
fi

# Fetch GitHub PAT from Doppler for semantic-release
# This token needs push access for tags and releases
echo "Fetching GitHub token from Doppler..."
GH_PAT=$(doppler secrets get GH_TOKEN_TERRYLICA --project main --config dev --plain 2>/dev/null || echo "")

if [ -z "$GH_PAT" ]; then
  echo "WARN: GH_TOKEN_TERRYLICA not found in Doppler (main/dev)"
  echo "Falling back to gh CLI auth token..."
  GH_PAT=$(GH_CONFIG_DIR="$HOME/.config/gh/profiles/terrylica" gh auth token 2>/dev/null || echo "")
fi

if [ -z "$GH_PAT" ]; then
  echo "ERROR: No GitHub token available for semantic-release"
  echo "Options:"
  echo "  1. Add GH_TOKEN_TERRYLICA to Doppler:"
  echo "     doppler secrets set GH_TOKEN_TERRYLICA=ghp_... --project main --config dev"
  echo "  2. Authenticate gh CLI:"
  echo "     GH_CONFIG_DIR=~/.config/gh/profiles/terrylica gh auth login"
  exit 1
fi

echo "Running semantic-release..."
GITHUB_TOKEN="$GH_PAT" GH_TOKEN="$GH_PAT" bun run semantic-release --no-ci

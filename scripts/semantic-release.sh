#!/usr/bin/env bash
# Run semantic-release with proper GitHub token from Doppler
# Part of the local-only release workflow (ADR-0027)
set -euo pipefail

echo "=== Semantic Release ==="

# Uses globally installed semantic-release (Homebrew) — no local package.json needed.
if ! command -v semantic-release &>/dev/null; then
  echo "ERROR: semantic-release not found globally"
  echo "Install via: npm install -g semantic-release @semantic-release/changelog @semantic-release/exec @semantic-release/git @semantic-release/github"
  exit 1
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

# Run semantic-release, tolerating post-release "success" step failures.
# The success step (GitHub issue/PR comments) can fail on non-existent
# issue references (e.g., "Closes #91" when #91 doesn't exist yet).
# The actual release (tag, GitHub release, version bump) succeeds before
# the success step runs, so a failure there is non-critical.
set +e
GITHUB_TOKEN="$GH_PAT" GH_TOKEN="$GH_PAT" semantic-release --no-ci 2>&1
SR_EXIT=$?
set -e

if [ $SR_EXIT -ne 0 ]; then
  # Check if a release was actually created despite the error
  VERSION=$(grep -A5 '\[workspace.package\]' Cargo.toml | grep '^version' | head -1 | sed 's/.*= "\(.*\)"/\1/')
  if git tag -l "v$VERSION" | grep -q "v$VERSION"; then
    echo "WARN: semantic-release exited $SR_EXIT but tag v$VERSION exists — release succeeded"
    echo "WARN: The failure was likely in a post-release step (issue comments, etc.)"
  else
    echo "ERROR: semantic-release failed (exit $SR_EXIT) and no release tag was created"
    exit $SR_EXIT
  fi
fi

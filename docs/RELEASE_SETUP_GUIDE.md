# Release Setup Guide - PyPI Trusted Publisher

**Purpose**: One-time configuration for automated PyPI publishing from GitHub Actions

**Time Required**: ~5 minutes

**Prerequisites**:

- PyPI account with owner/maintainer access
- GitHub repository: `terrylica/rangebar-py`
- Workflow file: `.github/workflows/release.yml` (already created ✅)

---

## Overview

**Architecture**: Zero-secrets automated releases

```
Conventional Commit (main)
  ↓
GitHub Actions (GITHUB_TOKEN automatic)
  ↓
semantic-release (creates tag, release, CHANGELOG)
  ↓
PyPI Trusted Publisher (OIDC authentication)
  ↓
Package Published (rangebar v0.1.0)
```

**No secrets required in repository** - Uses OpenID Connect (OIDC) authentication.

---

## Step 1: Verify GitHub Actions Workflow Permissions

**File**: `.github/workflows/release.yml`

**Required Configuration** (already present ✅):

```yaml
permissions:
  contents: write # Create releases, push tags
  id-token: write # PyPI Trusted Publisher OIDC
```

**Validation**:

```bash
grep -A2 "permissions:" .github/workflows/release.yml
```

**Expected Output**:

```yaml
permissions:
  contents: write
  id-token: write
```

✅ **Status**: Already configured in Phase 8

---

## Step 2: Configure PyPI Trusted Publisher

**URL**: https://pypi.org/manage/account/publishing/

### 2.1 Navigate to Publishing Settings

1. Login to PyPI: https://pypi.org/account/login/
2. Go to "Publishing" tab: https://pypi.org/manage/account/publishing/
3. Scroll to "Add a new publisher" section

### 2.2 Fill Publisher Configuration Form

**Form Fields** (based on screenshot provided):

| Field                 | Value         | Notes                              |
| --------------------- | ------------- | ---------------------------------- |
| **PyPI Project Name** | `rangebar`    | Must match `pyproject.toml` name   |
| **Owner**             | `terrylica`   | GitHub username (NOT organization) |
| **Repository name**   | `rangebar-py` | GitHub repo name                   |
| **Workflow name**     | `release.yml` | Filename in `.github/workflows/`   |
| **Environment name**  | `pypi`        | Optional but STRONGLY recommended  |

**Screenshot Reference**:

```
┌─────────────────────────────────────────────────────┐
│ PyPI Project Name (required)                        │
│ ┌─────────────────────────────────────────────────┐ │
│ │ rangebar                                        │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ Owner (required)                                    │
│ ┌─────────────────────────────────────────────────┐ │
│ │ terrylica                                       │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ Repository name (required)                          │
│ ┌─────────────────────────────────────────────────┐ │
│ │ rangebar-py                                     │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ Workflow name (required)                            │
│ ┌─────────────────────────────────────────────────┐ │
│ │ release.yml                                     │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ Environment name (optional)                         │
│ ┌─────────────────────────────────────────────────┐ │
│ │ pypi                                            │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ [ Add ]                                             │
└─────────────────────────────────────────────────────┘
```

### 2.3 Click "Add" Button

**Result**: Publisher appears in "Pending publishers" table

**Validation**: You should see:

```
Pending publishers for terrylica/rangebar-py

Publisher                           Workflow        Environment   Added
terrylica/rangebar-py/.github/      release.yml     pypi          2025-11-16
workflows/release.yml
```

---

## Step 3: (Optional) Configure GitHub Environment

**Purpose**: Add additional protection to PyPI publishing

**Steps**:

1. Go to: https://github.com/terrylica/rangebar-py/settings/environments
2. Click "New environment"
3. Name: `pypi`
4. (Optional) Add protection rules:
   - ✅ Required reviewers (if working in a team)
   - ✅ Wait timer (delay before publish)
   - ✅ Deployment branches (main only)

**Our Configuration** (minimal, solo developer):

- Name: `pypi`
- Protection rules: None (trust semantic-release automation)
- Deployment branches: `main` branch only

**Why Optional**: The Trusted Publisher configuration already restricts to:

- Specific repository (`terrylica/rangebar-py`)
- Specific workflow (`release.yml`)
- Specific GitHub owner (`terrylica`)

Environment adds: Manual approval gates, deployment delays

---

## Step 4: Validate Configuration

### 4.1 Check No API Tokens in GitHub Secrets

**Command**:

```bash
gh secret list --repo terrylica/rangebar-py | grep -i pypi || echo "✅ No PyPI tokens found (correct)"
```

**Expected**: `✅ No PyPI tokens found (correct)`

**Why This is Good**: Trusted Publisher uses OIDC, not API tokens

### 4.2 Verify Workflow File

**Command**:

```bash
cat .github/workflows/release.yml | grep -A10 "pypa/gh-action-pypi-publish"
```

**Expected** (no `password` or `api_token` field):

```yaml
- uses: pypa/gh-action-pypi-publish@release/v1
  # No 'with:' block needed - OIDC authentication automatic
```

**Incorrect Example** (do NOT use):

```yaml
- uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }} # ❌ Wrong! Delete this
```

### 4.3 Verify Semantic Release Configuration

**Command**:

```bash
grep -A5 "\[tool.semantic_release\]" pyproject.toml
```

**Expected**:

```toml
[tool.semantic_release]
version_toml = [
    "pyproject.toml:project.version",
    "Cargo.toml:package.version"
]
commit_parser = "conventional"
tag_format = "v{version}"
```

---

## Step 5: Test Release (First v0.1.0 Release)

### 5.1 Pre-Flight Checklist

- [x] All tests passing: `make test`
- [x] Linting clean: `make lint`
- [x] Coverage targets met: `make coverage` (95%+ Python, 90%+ Rust)
- [x] PyPI Trusted Publisher configured
- [x] GitHub Actions workflow permissions set
- [x] All changes committed with conventional commits

### 5.2 Trigger Release

**Method 1: Merge PR to main** (recommended)

```bash
# Create feature branch
git checkout -b chore/prepare-v0.1.0

# Commit all Phase 7-9 changes
git add .
git commit -m "feat: implement Phases 7-9 (testing, CI/CD, release automation)

BREAKING CHANGE: First production release with automated releases

- Add 21 new tests (95%+ coverage)
- Add CI/CD workflows (Linux x86_64, macOS ARM64)
- Add semantic-release automation
- Configure PyPI Trusted Publisher
"

# Push and create PR
git push origin chore/prepare-v0.1.0
gh pr create --title "feat: Phases 7-9 - Testing, CI/CD, Release Automation" \
  --body "Implements ADR-003, ADR-004, ADR-005, ADR-006"

# Merge PR (triggers release workflow)
gh pr merge --squash --auto
```

**Method 2: Direct push to main** (for solo developers)

```bash
git checkout main
git add .
git commit -m "feat: implement Phases 7-9 (testing, CI/CD, release automation)

BREAKING CHANGE: First production release

- Add 21 new tests (95%+ Python coverage)
- Add CI/CD workflows
- Configure semantic-release
"
git push origin main
```

### 5.3 Monitor Release Workflow

**GitHub Actions URL**:

```
https://github.com/terrylica/rangebar-py/actions/workflows/release.yml
```

**Expected Workflow Steps**:

1. ✅ Checkout code
2. ✅ Setup Python
3. ✅ Run semantic-release (analyzes commits)
4. ✅ Determine version bump (0.0.0 → 0.1.0)
5. ✅ Update `pyproject.toml` and `Cargo.toml` versions
6. ✅ Generate CHANGELOG.md
7. ✅ Create git tag `v0.1.0`
8. ✅ Push tag and commit to main
9. ✅ Build wheels (Linux x86_64, macOS ARM64)
10. ✅ Publish to PyPI via Trusted Publisher
11. ✅ Create GitHub Release with CHANGELOG

**Timeline**: ~10-15 minutes

### 5.4 Validate Release Success

**Check 1: PyPI Package Published**

```bash
pip install --upgrade rangebar==0.1.0
python -c "import rangebar; print(rangebar.__version__)"
# Expected: 0.1.0
```

**Check 2: GitHub Release Created**

```bash
gh release view v0.1.0
# Expected: Release notes with CHANGELOG excerpt
```

**Check 3: CHANGELOG Generated**

```bash
cat CHANGELOG.md | head -20
# Expected:
# # CHANGELOG
#
# ## v0.1.0 (2025-11-16)
#
# ### Features
# - implement Phases 7-9 (testing, CI/CD, release automation)
```

**Check 4: Git Tag Created**

```bash
git tag | grep v0.1.0
# Expected: v0.1.0
```

---

## Troubleshooting

### Error: "PyPI Trusted Publisher authentication failed"

**Cause**: Publisher not configured or misconfigured

**Fix**:

1. Verify publisher in PyPI: https://pypi.org/manage/account/publishing/
2. Check values match exactly:
   - Owner: `terrylica`
   - Repository: `rangebar-py`
   - Workflow: `release.yml`
3. Check workflow file has `permissions.id-token: write`

**Validation Command**:

```bash
curl -s https://pypi.org/pypi/rangebar/json | jq -r '.info.name'
# Expected: rangebar (if package exists)
# OR: HTTP 404 (if first publish, expected)
```

### Error: "GITHUB_TOKEN insufficient permissions"

**Cause**: Missing `contents: write` permission

**Fix**:

```yaml
# .github/workflows/release.yml
permissions:
  contents: write # Add this line
  id-token: write
```

### Error: "Project 'rangebar' does not exist on PyPI"

**Expected**: First publish creates the project automatically

**Trusted Publisher Configuration**: Use "Pending publishers" (not "Projects")

**Note**: The project will be created when first publish succeeds

### Warning: "semantic-release found no releasable commits"

**Cause**: No conventional commits since last release (or no previous release)

**Fix**: Ensure at least one commit with `feat:` or `fix:` or `BREAKING CHANGE:`

**Example**:

```bash
git commit --allow-empty -m "feat: initial release"
git push origin main
```

---

## Security Notes

### What Gets Published

**Public Information** (visible on PyPI):

- Package name: `rangebar`
- Version: `0.1.0`
- Metadata: `pyproject.toml` contents
- Wheels: Compiled binaries
- Source distribution: Full source code

**Private Information** (NOT published):

- GitHub secrets
- Workflow environment variables
- Build logs (only in GitHub Actions)
- Test data (gitignored)

### Trusted Publisher Security Model

**Trust Chain**:

1. ✅ PyPI trusts GitHub's OIDC identity provider
2. ✅ GitHub validates workflow identity (repo + workflow file)
3. ✅ GitHub issues short-lived JWT token
4. ✅ PyPI validates JWT signature and claims
5. ✅ PyPI authorizes publish (workflow matches publisher config)

**Attack Scenarios Prevented**:

- ❌ Malicious fork cannot publish (owner mismatch)
- ❌ Different workflow cannot publish (workflow name mismatch)
- ❌ Stolen JWT cannot be reused (expires in seconds)
- ❌ Compromised repository cannot publish to different package (project name mismatch)

---

## Post-Release Checklist

After first successful release:

- [ ] Verify package on PyPI: https://pypi.org/project/rangebar/
- [ ] Install from PyPI: `pip install rangebar`
- [ ] Run examples: `python examples/basic_usage.py`
- [ ] Check GitHub Release: https://github.com/terrylica/rangebar-py/releases/tag/v0.1.0
- [ ] Verify CHANGELOG.md committed to main
- [ ] Update README badges (optional):
  - [![PyPI version](https://badge.fury.io/py/rangebar.svg)](https://pypi.org/project/rangebar/)
  - [![Downloads](https://pepy.tech/badge/rangebar)](https://pepy.tech/project/rangebar)

---

## Summary

**Configuration Required**: 1 action (PyPI Trusted Publisher setup)

**Secrets Required**: 0 (zero)

**Manual Steps for Future Releases**: 0 (fully automated)

**Release Trigger**: Conventional commit to `main` branch

**Next Release**: Will be v0.1.1 (patch), v0.2.0 (minor), or v1.0.0 (major) based on commit messages

---

## Quick Reference

| Component            | Configuration                        | Location                                            |
| -------------------- | ------------------------------------ | --------------------------------------------------- |
| GitHub Token         | Automatic (`GITHUB_TOKEN`)           | GitHub Actions built-in                             |
| PyPI Auth            | Trusted Publisher (OIDC)             | https://pypi.org/manage/account/publishing/         |
| Workflow Permissions | `contents: write`, `id-token: write` | `.github/workflows/release.yml`                     |
| Release Trigger      | Conventional commit to main          | Any commit with `feat:`, `fix:`, `BREAKING CHANGE:` |
| Version Bump         | Semantic (conventional commits)      | python-semantic-release analyzes commits            |
| CHANGELOG            | Auto-generated                       | python-semantic-release generates from commits      |

**Ready to release?** Follow Step 5 above to trigger your first v0.1.0 release! 🚀

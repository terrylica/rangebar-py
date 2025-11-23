# ADR-006: Secrets Management for GitHub Actions

**Status**: Accepted

**Date**: 2025-11-16

**Deciders**: Terry Li

**Tags**: `secrets`, `cicd`, `github-actions`, `pypi`, `security`

---

## Context

The rangebar-py project requires secrets for automated releases:

1. **GitHub token** for creating releases and pushing tags
2. **PyPI credentials** for package publishing

**Constraints**:

- Zero manual secret rotation
- No long-lived API tokens
- Minimize attack surface
- Automatic credential provisioning

**Options Evaluated**:

1. Custom GitHub Personal Access Token (PAT) stored in repository secrets
2. Built-in `GITHUB_TOKEN` provided by GitHub Actions
3. PyPI API tokens stored in repository secrets
4. PyPI Trusted Publisher (OpenID Connect authentication)

---

## Decision

**Use GitHub Actions built-in credentials exclusively; no custom secrets required.**

### GitHub Releases

**Use**: Built-in `GITHUB_TOKEN` (automatically provided by GitHub Actions)

**Configuration**: Update workflow permissions in `.github/workflows/release.yml`:

```yaml
permissions:
  contents: write # Create releases and push tags
  id-token: write # OpenID Connect for PyPI Trusted Publisher
```

**Rationale**:

- ✅ **Zero configuration**: Automatically available in all workflows
- ✅ **Scoped permissions**: Limited to repository only
- ✅ **Automatic rotation**: GitHub manages lifecycle
- ✅ **No exposure risk**: Never leaves GitHub infrastructure
- ❌ **No manual secret management**: Cannot be leaked or misconfigured

**Rejected Alternative**: Custom PAT

- ❌ Requires manual creation and rotation
- ❌ Broader permissions than necessary
- ❌ Potential for accidental exposure
- ❌ Manual revocation if compromised

### PyPI Publishing

**Use**: Trusted Publisher (OpenID Connect authentication)

**Configuration**: One-time setup in PyPI web interface (not in code):

1. Navigate to: https://pypi.org/manage/account/publishing/
2. Add publisher:
   - PyPI Project Name: `rangebar`
   - Owner: `terrylica` (GitHub username/org)
   - Repository name: `rangebar-py`
   - Workflow name: `release.yml`
   - Environment name: `pypi` (optional but recommended)

**Rationale**:

- ✅ **No API tokens**: Uses OIDC JWT authentication
- ✅ **Workflow-specific**: Only `release.yml` can publish
- ✅ **Zero secret storage**: No secrets in repository
- ✅ **Automatic verification**: PyPI validates GitHub identity
- ✅ **Revocable**: Remove publisher in PyPI settings

**Rejected Alternative**: PyPI API Token

- ❌ Long-lived credentials
- ❌ Requires storage in GitHub Secrets
- ❌ Manual rotation required
- ❌ Potential for accidental exposure in logs

---

## Consequences

### Positive

**Security**:

- Zero secrets stored in repository
- No custom credential management
- Automatic rotation (GITHUB_TOKEN)
- Scoped permissions (principle of least privilege)

**Maintainability**:

- Zero configuration in repository
- No secret rotation workflows
- No credential expiry management
- Self-documenting (permissions in workflow YAML)

**Observability**:

- All authentication visible in workflow logs
- PyPI shows publisher configuration in UI
- GitHub audit log tracks permission usage

**Availability**:

- No secret expiry outages
- No manual intervention required
- Automatic credential provisioning

### Negative

**Initial Setup**:

- Requires one-time PyPI Trusted Publisher configuration
- Requires understanding of GitHub Actions permissions model

**Migration**:

- Existing projects with API tokens need manual transition

### Neutral

**Doppler Integration**:

- **Decision**: NOT used for CI/CD secrets
- **Scope**: Doppler (`claude-config` project) remains for local development secrets only
- **Rationale**: GitHub Actions has superior built-in secret management

**Local Development**:

- Developers cannot push to PyPI from local machines (by design)
- Releases ONLY through GitHub Actions
- Local testing uses `maturin build` (no publish)

---

## Implementation

### Phase 1: Verify GitHub Actions Permissions

**File**: `.github/workflows/release.yml`

```yaml
permissions:
  contents: write # Create releases, push tags
  id-token: write # PyPI Trusted Publisher OIDC
```

**Validation**: Check that `python-semantic-release` action receives `github_token: ${{ secrets.GITHUB_TOKEN }}`

### Phase 2: Configure PyPI Trusted Publisher

**Manual Steps** (one-time, cannot be automated):

1. **Login to PyPI**: https://pypi.org/account/login/
2. **Navigate to Publishing**: https://pypi.org/manage/account/publishing/
3. **Click "Add a new publisher"**
4. **Fill form** (see image from user):
   - PyPI Project Name: `rangebar`
   - Owner: `terrylica`
   - Repository name: `rangebar-py`
   - Workflow name: `release.yml`
   - Environment name: `pypi` (recommended)
5. **Click "Add"**
6. **Verify**: Publisher appears in "Pending publishers" table

### Phase 3: Validate Release Workflow

**Test Checklist**:

- [ ] Workflow has `permissions.contents: write`
- [ ] Workflow has `permissions.id-token: write`
- [ ] PyPI Trusted Publisher configured for `terrylica/rangebar-py`
- [ ] `release.yml` uses `pypa/gh-action-pypi-publish@release/v1`
- [ ] Publish action does NOT specify `password` or `api_token` (uses OIDC automatically)

---

## Compliance

### SLO Metrics

**Availability**:

- Target: 100% credential availability (no expiry)
- Measurement: Zero failed releases due to credential issues

**Correctness**:

- Target: 100% authentication success rate
- Measurement: All releases authenticate successfully

**Observability**:

- Target: All authentication attempts visible in logs
- Measurement: GitHub Actions logs + PyPI audit logs

**Maintainability**:

- Target: Zero manual secret rotation operations
- Measurement: No secret-related maintenance in 12 months

### Error Handling

**Policy**: Raise and propagate; no fallback credentials

**Error Scenarios**:

| Error                                   | Cause                     | Action                     |
| --------------------------------------- | ------------------------- | -------------------------- |
| `GITHUB_TOKEN` insufficient permissions | Missing `contents: write` | Fail workflow, raise error |
| PyPI OIDC authentication failure        | Publisher not configured  | Fail workflow, raise error |
| PyPI project name mismatch              | Typo in publisher config  | Fail workflow, raise error |

**No Fallbacks**:

- ❌ No "try API token if OIDC fails"
- ❌ No "use different workflow if permissions missing"
- ❌ No "skip publishing if authentication fails"

---

## References

**GitHub Actions GITHUB_TOKEN**:

- Docs: https://docs.github.com/en/actions/security-for-github-actions/security-guides/automatic-token-authentication
- Permissions: https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#permissions

**PyPI Trusted Publishers**:

- Announcement: https://blog.pypi.org/posts/2023-04-20-introducing-trusted-publishers/
- Setup Guide: https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- GitHub Actions: https://docs.pypi.org/trusted-publishers/using-a-publisher/

**python-semantic-release**:

- GitHub Actions: https://python-semantic-release.readthedocs.io/en/latest/automatic-releases/github-actions.html

---

## Alternatives Considered

### Custom GitHub PAT in Doppler

**Rejected**:

- Requires Doppler sync to GitHub Secrets
- Manual token creation and rotation
- Broader permissions than necessary
- Additional secret management complexity

### PyPI API Token in GitHub Secrets

**Rejected**:

- Long-lived credential (no automatic rotation)
- Requires manual token creation in PyPI
- Manual rotation every 90 days (PyPI recommendation)
- Exposure risk in logs if misconfigured

### Hybrid Approach (OIDC + Fallback Token)

**Rejected**:

- Violates error handling policy (no fallbacks)
- Introduces secret management complexity
- Reduces security (weakest link principle)
- Unclear failure modes

---

## Future Considerations

### v0.2.0+

**No changes planned**: This architecture is stable and requires zero maintenance.

**Potential Additions**:

- GitHub Environment protection rules (require manual approval for releases)
- Branch protection rules (require PR reviews before merge to main)
- Dependabot security updates (automated dependency updates)

**Not Planned**:

- Migration to custom secrets (regression in security)
- Third-party secret management (unnecessary complexity)

---

## Summary

**Zero secrets stored in repository or external services.**

**Authentication Flow**:

1. Developer pushes conventional commit to `main`
2. GitHub Actions generates `GITHUB_TOKEN` with scoped permissions
3. `python-semantic-release` uses `GITHUB_TOKEN` to create release
4. GitHub generates OIDC JWT for workflow identity
5. `pypa/gh-action-pypi-publish` exchanges JWT for temporary PyPI credentials
6. PyPI validates JWT against Trusted Publisher configuration
7. Package published; credentials expire immediately

**Result**: Fully automated, zero-trust release pipeline with no persistent secrets.

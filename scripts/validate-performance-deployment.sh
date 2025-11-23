#!/usr/bin/env bash
# Validation script for GitHub Actions deployment (Phase 1.5 - Task 1.5.6)
# This script verifies GitHub Pages configuration and triggers the performance workflow

set -euo pipefail

echo "==================================================================="
echo "Performance Deployment Validation Script"
echo "==================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo -e "${RED}✗ gh CLI not found${NC}"
    echo "Install: brew install gh"
    exit 1
fi
echo -e "${GREEN}✓ gh CLI available${NC}"

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${RED}✗ Not authenticated with GitHub${NC}"
    echo "Run: gh auth login"
    exit 1
fi
echo -e "${GREEN}✓ Authenticated with GitHub${NC}"
echo ""

# Step 1: Remind user to configure GitHub Pages
echo "-------------------------------------------------------------------"
echo "Step 1: Configure GitHub Pages (Manual)"
echo "-------------------------------------------------------------------"
echo ""
echo "IMPORTANT: Before running this script, ensure GitHub Pages is configured:"
echo ""
echo "1. Navigate to: https://github.com/terrylica/rangebar-py/settings/pages"
echo "2. Under 'Build and deployment':"
echo "   - Source: GitHub Actions"
echo "3. Click 'Save'"
echo ""
read -p "Have you configured GitHub Pages to use 'GitHub Actions' source? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Please configure GitHub Pages first, then re-run this script.${NC}"
    exit 0
fi
echo -e "${GREEN}✓ GitHub Pages configured${NC}"
echo ""

# Step 2: Trigger workflow
echo "-------------------------------------------------------------------"
echo "Step 2: Trigger Workflow"
echo "-------------------------------------------------------------------"
echo ""
echo "Triggering performance-daily.yml workflow..."
gh workflow run performance-daily.yml

echo ""
echo "Waiting 5 seconds for workflow to start..."
sleep 5
echo ""

# Step 3: Monitor workflow execution
echo "-------------------------------------------------------------------"
echo "Step 3: Monitor Workflow Execution"
echo "-------------------------------------------------------------------"
echo ""
echo "Recent workflow runs:"
gh run list --workflow=performance-daily.yml --limit 5

echo ""
echo "Watching latest run (Ctrl+C to stop)..."
echo "Note: This will stream logs until completion"
echo ""
gh run watch --workflow=performance-daily.yml --interval 10 || true

echo ""

# Step 4: Verify deployment
echo "-------------------------------------------------------------------"
echo "Step 4: Verify Deployment"
echo "-------------------------------------------------------------------"
echo ""

# Get latest run status
RUN_STATUS=$(gh run list --workflow=performance-daily.yml --limit 1 --json conclusion --jq '.[0].conclusion')

if [ "$RUN_STATUS" == "success" ]; then
    echo -e "${GREEN}✓ Workflow completed successfully${NC}"
    echo ""

    # Check for Pages deployment
    echo "Checking for Pages deployment..."
    sleep 3  # Wait for deployment to register

    # List recent deployments
    echo ""
    echo "Recent deployments:"
    gh api repos/terrylica/rangebar-py/pages/deployments --jq '.[] | "\(.created_at) - \(.status_url)"' | head -3 || echo "Unable to fetch deployments"

    echo ""
    echo -e "${GREEN}✓ Deployment verification complete${NC}"
    echo ""
    echo "Dashboard URL: https://terrylica.github.io/rangebar-py/"
    echo ""
    echo "Next steps:"
    echo "1. Open dashboard in browser to verify it renders correctly"
    echo "2. Check for benchmark data points in graphs"
    echo "3. Verify JSON files accessible (e.g., /dev/bench/python-bench.json)"

elif [ "$RUN_STATUS" == "failure" ]; then
    echo -e "${RED}✗ Workflow failed${NC}"
    echo ""
    echo "View logs:"
    echo "  gh run view --log"
    echo ""
    echo "Common issues:"
    echo "1. GitHub Pages not configured to 'GitHub Actions' source"
    echo "2. Missing permissions (pages: write, id-token: write)"
    echo "3. gh-pages branch doesn't exist yet"
    echo ""
    echo "Troubleshooting guide: docs/GITHUB_PAGES_SETUP.md"
    exit 1
else
    echo -e "${YELLOW}⚠ Workflow status: $RUN_STATUS${NC}"
    echo ""
    echo "Check status:"
    echo "  gh run list --workflow=performance-daily.yml"
    echo ""
    echo "View logs:"
    echo "  gh run view --log"
fi

echo ""
echo "==================================================================="
echo "Validation Complete"
echo "==================================================================="

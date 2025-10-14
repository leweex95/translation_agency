#!/bin/bash

# Workflow Security Validation Script
# Checks GitHub Actions workflows for security issues

set -e

echo "Checking workflow files for security issues..."

# Find all workflow files (excluding this security check)
WORKFLOW_FILES=$(find .github/workflows -name "*.yml" -o -name "*.yaml" | grep -v "workflow-security-check" || true)

if [ -z "$WORKFLOW_FILES" ]; then
  echo "No workflow files found to check"
  exit 0
fi

echo "Found workflow files:"
echo "$WORKFLOW_FILES"
echo ""

SECURITY_ISSUES=0

for workflow in $WORKFLOW_FILES; do
  echo "Checking $workflow..."

  # Check for pull_request triggers
  if grep -q "pull_request:" "$workflow"; then
    echo "SECURITY ISSUE: $workflow contains 'pull_request' trigger!"
    echo "This is dangerous with self-hosted runners in public repositories."
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
  fi

  # Check for self-hosted runners without actor restrictions
  if grep -q "runs-on: self-hosted" "$workflow"; then
    if ! grep -q "if: github.actor" "$workflow"; then
      echo "WARNING: $workflow uses self-hosted runners without actor restrictions"
      echo "Consider adding: if: github.actor == 'your-username'"
    fi
  fi

  echo "$workflow passed basic security checks"
  echo ""
done

if [ $SECURITY_ISSUES -gt 0 ]; then
  echo "SECURITY VIOLATIONS FOUND: $SECURITY_ISSUES issues detected"
  echo ""
  echo "Fix the issues above before merging. Self-hosted runners with pull_request"
  echo "triggers can execute malicious code from forks on your local machine."
  exit 1
else
  echo "All workflows passed security validation!"
fi
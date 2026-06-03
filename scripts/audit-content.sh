#!/usr/bin/env bash
# Pre-commit content audit: scan staged additions for PII and copyrighted titles.
# Fails with exit 1 if any match is found.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TITLES_FILE="$REPO_ROOT/.copyright-titles"

# Only inspect added lines (not context or deletions)
staged_diff() {
    git diff --cached --diff-filter=d --unified=0 -- \
        ":(exclude)tests/local/" \
        ":(exclude).git/" \
        ":(exclude).venv/"
}

fail=0

# ── PII patterns ──────────────────────────────────────────────────────────────

# Home directory paths (catches /Users/username or /home/username leaking in)
if staged_diff | grep -E '^\+[^+]' | grep -qE '/(Users|home)/[a-zA-Z0-9_.-]+/'; then
    echo "audit-content: ERROR: staged diff contains a home directory path (/Users/... or /home/...)"
    echo "  Run: git diff --cached | grep -E '/(Users|home)/'"
    fail=1
fi

# Email addresses (except the one in pyproject.toml which is intentional)
if staged_diff | grep -E '^\+[^+]' | grep -qE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'; then
    # Filter out known-safe author email in pyproject.toml
    suspicious=$(staged_diff | grep -E '^\+[^+]' | grep -E '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' | grep -v 'vitormf@yahoo.com' || true)
    if [ -n "$suspicious" ]; then
        echo "audit-content: ERROR: staged diff contains an email address"
        echo "$suspicious"
        fail=1
    fi
fi

# ── Copyright titles ──────────────────────────────────────────────────────────

if [ -f "$TITLES_FILE" ]; then
    while IFS= read -r title || [ -n "$title" ]; do
        # Skip blank lines and comments
        [[ -z "$title" || "$title" == \#* ]] && continue

        if staged_diff | grep -E '^\+[^+]' | grep -qi "$title"; then
            echo "audit-content: ERROR: staged diff contains a copyright-protected title: \"$title\""
            echo "  Run: git diff --cached | grep -i \"$title\""
            fail=1
        fi
    done < "$TITLES_FILE"
else
    echo "audit-content: WARNING: .copyright-titles not found — skipping title check"
    echo "  Copy .copyright-titles.example to .copyright-titles and add your titles."
fi

# ── Report ─────────────────────────────────────────────────────────────────────

if [ $fail -ne 0 ]; then
    echo ""
    echo "audit-content: Commit blocked. Fix the issues above before committing."
    echo "  If a match is a false positive, add an exception in scripts/audit-content.sh."
    exit 1
fi

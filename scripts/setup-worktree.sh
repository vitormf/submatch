#!/usr/bin/env bash
set -euo pipefail

GIT_DIR=$(cd "$(git rev-parse --git-dir)" && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)

if [[ "$GIT_DIR" == "$GIT_COMMON" ]]; then
    echo "Already in main worktree — nothing to set up."
    exit 0
fi

MAIN_ROOT=$(dirname "$GIT_COMMON")
FIXTURES_SRC="$MAIN_ROOT/tests/fixtures"
FIXTURES_DST="$(git rev-parse --show-toplevel)/tests/fixtures"

mkdir -p "$FIXTURES_DST"

count=0
if [[ -d "$FIXTURES_SRC" ]]; then
    while IFS= read -r -d '' f; do
        dest="$FIXTURES_DST/$(basename "$f")"
        if [[ ! -e "$dest" ]]; then
            cp "$f" "$dest"
            ((count++)) || true
        fi
    done < <(find "$FIXTURES_SRC" -maxdepth 1 -type f -print0)
fi

echo "Seeded $count fixture file(s) from main into tests/fixtures/."

#!/usr/bin/env bash
set -euo pipefail

BRANCH=$(git branch --show-current)

if [[ -z "$BRANCH" || "$BRANCH" == "main" ]]; then
    echo "error: must be run from a feature branch, not '${BRANCH:-detached HEAD}'" >&2
    exit 1
fi

GIT_DIR=$(cd "$(git rev-parse --git-dir)" && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
MAIN_ROOT=$(dirname "$GIT_COMMON")

if [[ "$GIT_DIR" == "$GIT_COMMON" ]]; then
    IN_WORKTREE=false
else
    IN_WORKTREE=true
    WORKTREE_PATH=$(git rev-parse --show-toplevel)
fi

echo "==> Lint..."
ruff check submatch/ tests/

echo "==> Unit tests..."
pytest --tb=short -q

echo "==> Integration tests (fast tier)..."
make integration-test-fast

echo "==> Merging '$BRANCH' into main..."
git -C "$MAIN_ROOT" merge --no-ff "$BRANCH" -m "Merge branch '$BRANCH'"

echo "==> Pushing to origin (vitormf)..."
GH_TOKEN=$(gh auth token --user vitormf) git -C "$MAIN_ROOT" push origin main

echo "==> Refreshing main fixtures..."
python "$MAIN_ROOT/tests/integration/prepare.py"

echo "==> Cleaning up..."
if [[ "$IN_WORKTREE" == true ]]; then
    cd "$MAIN_ROOT"
    git worktree remove "$WORKTREE_PATH"
    git worktree prune
fi
git -C "$MAIN_ROOT" branch -d "$BRANCH"

echo ""
echo "Done: '$BRANCH' merged to main, pushed to origin, branch deleted."
if [[ "$IN_WORKTREE" == true ]]; then
    echo "Worktree removed. Call ExitWorktree(action='keep') to reset your session's working directory."
fi

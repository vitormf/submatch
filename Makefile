.PHONY: install test integration-test lint clean merge setup-worktree local-setup

install:
	pip install -e ".[dev]"

test:
	pytest tests/ --ignore=tests/integration --cov-fail-under=99 -v

integration-test:
	python tests/integration/prepare.py
	pytest tests/integration/ -v -s --no-cov -x --timeout=120

lint:
	ruff check submatch/ tests/

setup-worktree:
	bash scripts/setup-worktree.sh

local-setup:
	@GIT_COMMON=$$(cd "$$(git rev-parse --git-common-dir)" && pwd -P); \
	 GIT_DIR=$$(cd "$$(git rev-parse --git-dir)" && pwd -P); \
	 LOCAL_SRC="$$GIT_COMMON/../tests/local"; \
	 if [ "$$GIT_DIR" = "$$GIT_COMMON" ]; then \
	     mkdir -p tests/local/fixtures; \
	     echo "Main worktree: tests/local/ is ready."; \
	 elif [ -d "$$LOCAL_SRC" ]; then \
	     test -L tests/local || ln -sf "$$LOCAL_SRC" tests/local; \
	     echo "Linked tests/local from main worktree."; \
	 else \
	     echo "No tests/local in main worktree — run 'make local-setup' from main first."; \
	 fi

merge:
	bash scripts/merge-to-main.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true

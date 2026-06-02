.PHONY: install test integration-test lint clean merge setup-worktree

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

merge:
	bash scripts/merge-to-main.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true

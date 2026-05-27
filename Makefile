.PHONY: install test integration-test lint clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ --ignore=tests/integration --cov-fail-under=95 -v

integration-test:
	pytest tests/integration/ -v -s --no-cov

lint:
	ruff check submatch/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name '*.pyc' -delete 2>/dev/null; true

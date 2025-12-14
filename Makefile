.PHONY: install dev test lint format typecheck docs clean build publish

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/ununennium --cov-report=html

# Linting and Formatting
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	pyright src/

check: lint typecheck test

# Documentation
docs:
	mkdocs serve

docs-build:
	mkdocs build

# Build and Publish
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	twine upload dist/*

# Benchmarks
benchmark:
	python benchmarks/run_benchmarks.py

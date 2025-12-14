# Contributing to Ununennium

Thank you for your interest in contributing to Ununennium! This document provides guidelines for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ununennium.git`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run type checking
pyright src/

# Run linting
ruff check src/
```

## Contribution Guidelines

### Code Style

- Follow PEP 8 and PEP 257
- Use type hints for all functions
- Maximum line length: 100 characters
- Use ruff for formatting

### Commit Messages

- Use conventional commits format
- Examples: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

### Pull Requests

1. Ensure all tests pass
2. Add tests for new features
3. Update documentation as needed
4. Follow the PR template

### Testing

- Write tests for all new functionality
- Maintain >80% code coverage
- Use pytest markers for slow/GPU tests

## Issues

- Use issue templates for bugs and features
- Provide minimal reproducible examples for bugs

## Contact

For questions, open a GitHub issue or discussion.

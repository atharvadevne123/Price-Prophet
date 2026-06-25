# Contributing to Price-Prophet

Thank you for your interest in contributing to Price-Prophet!

## Development Setup

```bash
git clone https://github.com/atharvadevne123/Price-Prophet
cd Price-Prophet
pip install -r requirements.txt
make install
```

## Running Tests

```bash
make test
# or with coverage report:
make coverage-html
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
make lint        # check for issues
make lint-fix    # auto-fix safe issues
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Write tests for any new functionality (aim for >60% coverage)
3. Ensure all tests pass: `make test`
4. Ensure lint is clean: `make lint`
5. Update `CHANGELOG.md` with your changes
6. Submit a pull request with a clear description

## Commit Convention

Use conventional commits:
- `feat`: new feature
- `fix`: bug fix
- `refactor`: code restructuring
- `test`: adding or updating tests
- `docs`: documentation changes
- `chore`: build/tooling changes
- `perf`: performance improvement

## Reporting Issues

Please use the [GitHub issue tracker](https://github.com/atharvadevne123/Price-Prophet/issues) and include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or stack traces

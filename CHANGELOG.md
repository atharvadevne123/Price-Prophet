# Changelog

All notable changes to Price-Prophet are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Makefile with install, test, lint, run, docker-build targets
- CONTRIBUTING.md with development setup guide
- Comprehensive pyproject.toml with ruff, pytest, and coverage configuration
- Pre-commit hooks for ruff and whitespace trimming
- Full type annotations across all modules
- Google-style docstrings on all public functions and classes

## [1.0.0] - 2026-04-28

### Added
- Ensemble ML model (XGBoost + LightGBM + RandomForest) via sklearn VotingRegressor
- 15-feature engineering pipeline with StandardScaler
- FastAPI REST API with 9 endpoints: /forecast, /train, /drift, /similar, /metrics, /predictions, /categories, /summary, /health
- KS-test drift detection per feature with configurable p-value threshold
- FAISS-powered product similarity search with cosine fallback
- SQLAlchemy ORM with SQLite (dev) and PostgreSQL (prod) support
- Airflow DAG for automated daily retraining with drift threshold gate
- Docker + docker-compose deployment with PostgreSQL
- deque-based O(1) sliding window for recent prediction tracking
- 37 pytest tests across 4 modules with >60% coverage
- OpenAPI documentation at /docs

# Changelog

All notable changes to Price-Prophet are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.1] - 2026-07-07

### Fixed
- Removed deprecated `pd.np.log1p` usage in `engineer_features()` — now uses `math.log1p`
- Replaced non-deterministic `hash()` seed in `_text_to_vector()` with `hashlib.md5` for stable embeddings
- Fixed malformed JSON formatter `fmt` string in `logging_config.py` (missing quotes)
- Fixed bare-name dict key bugs in `scripts/benchmark.py` and `scripts/health_check.py`
- Rewrote `scripts/seed_db.py` to match current `predict()` / `train_model()` API signatures

### Added
- `/categories` endpoint listing valid product categories
- `/health/detailed` endpoint with model and database status
- `/predictions` endpoint with pagination (`limit` / `offset`)
- `/cache-stats` endpoint exposing TTL cache hit/miss metrics
- `/cache-clear` endpoint to flush the prediction cache
- `validate_margin_ratio()` validator and integration into `validate_request()`
- `X-Request-ID` correlation header in HTTP middleware
- Cache stats included in `/metrics` response
- `__all__` declarations in `app/cache.py`, `app/monitoring.py`, `app/retrieval.py`, `app/validators.py`
- Structured logging and type annotations across all scripts

## [Unreleased]

### Added
- `/feature-importance` endpoint returning Random Forest feature importances
- `/drift` endpoint with KS-test and PSI statistical drift analysis
- PSI (Population Stability Index) metric in monitoring module
- Alert thresholds for both KS statistic and PSI score
- `app/logging_config.py` — centralised logging with JSON formatter support
- `app/validators.py` — input validation helpers with domain bounds
- `app/exceptions.py` — structured custom exception hierarchy
- `app/cache.py` — thread-safe TTL cache with hit-rate tracking
- `batch_search()` method on `ProductIndex` for bulk similarity queries
- Cross-validation (5-fold KFold) in `train_model()`
- Compound features: `scarcity_score`, `value_index`, `log_price`, `margin_price`
- `scripts/benchmark.py` — API latency benchmark with percentiles
- `scripts/train.py` — standalone training CLI
- `scripts/analyze_drift.py` — offline drift analysis tool
- `scripts/export_metrics.py` — metrics/predictions export to JSON/CSV
- `scripts/health_check.py` — simple health check CLI
- `requirements-dev.txt` — pinned development dependencies
- `SECURITY.md` — vulnerability reporting policy
- `.github/ISSUE_TEMPLATE.md` and `.github/PULL_REQUEST_TEMPLATE.md`
- Multi-stage Docker build with non-root user
- Docker Compose with named volumes and healthcheck
- Dual Python matrix (3.11 + 3.12) in CI
- `mypy` type-check step in CI workflow
- Apache Airflow DAG enhanced with PSI trigger and notification task
- Comprehensive test coverage: test_retrieval, test_database, test_cache, test_validators, test_exceptions

### Changed
- README rewritten with architecture diagram and full API reference
- `app/model.py` now includes feature importance extraction
- `app/monitoring.py` extended with PSI and `check_alerts()`
- `app/retrieval.py` with improved docstrings and batch search
- `app/features.py` with 4 new compound features
- `app/main.py` with CORS, process-time header, structured error responses
- `pyproject.toml` with CLI entry points and updated metadata
- `Makefile` with `drift-check` and `bench` targets

## [1.0.0] — 2024-01-15

### Added
- Initial FastAPI service with VotingRegressor ensemble (XGBoost + LightGBM + RandomForest)
- FAISS cosine similarity product search
- KS-test drift detection
- Apache Airflow DAG for automated retraining
- SQLAlchemy ORM with SQLite / PostgreSQL support
- Docker and docker-compose support
- REST endpoints: /train, /forecast, /metrics, /similar, /summary, /health

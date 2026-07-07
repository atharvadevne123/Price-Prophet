# Price-Prophet

[![CI](https://github.com/atharvadevne123/Price-Prophet/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/Price-Prophet/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> ML-powered dynamic pricing engine with ensemble forecasting, semantic product retrieval, and real-time drift monitoring.

## Overview

Price-Prophet is a production-ready FastAPI service that predicts optimal product prices using a VotingRegressor ensemble (XGBoost + LightGBM + RandomForest). It features FAISS-based semantic product search, KS-test drift detection, and an Apache Airflow DAG for automated retraining.

## Features

- **Ensemble Forecasting** — XGBoost + LightGBM + RandomForest VotingRegressor
- **Price Optimisation** — Grid search over a price range to find the optimal price
- **Semantic Retrieval** — FAISS cosine similarity product search with batch query support
- **Drift Detection** — KS-test and PSI statistical drift monitoring with percentile stats
- **Auto Retraining** — Apache Airflow DAG triggered on drift or schedule
- **REST API** — FastAPI with OpenAPI docs at `/docs`; 20+ endpoints
- **Observability** — Process-time header, X-Request-ID correlation, detailed health, uptime, cache metrics
- **Input Validation** — Category, price, stock, margin, and demand-trend validators
- **Configurable CORS** — Set allowed origins via `CORS_ORIGINS` env variable

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Train the model
curl -X POST http://localhost:8000/train

# Get a price forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"category":"Electronics","stock_level":50,"competitor_price":299.99,"demand_trend":1.2}'
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                  FastAPI App                  │
│  /train  /forecast  /batch-forecast  /health  │
│  /similar  /metrics  /drift  /version        │
└───────────┬──────────────┬────────────────────┘
            │              │
     ┌──────▼──────┐  ┌───▼──────────┐
     │  ML Pipeline │  │ FAISS Index  │
     │  XGB+LGBM+RF │  │  (retrieval) │
     └──────┬──────┘  └──────────────┘
            │
     ┌──────▼──────┐   ┌─────────────┐
     │  SQLAlchemy  │   │  Monitoring │
     │  (SQLite/PG) │   │  KS + PSI   │
     └─────────────┘   └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │  Airflow DAG │
                        │  (retrain)   │
                        └─────────────┘
```

## API Reference

### `POST /train`
Train the model on synthetic data.

**Response:**
```json
{"status": "trained", "metrics": {"mae": 12.3, "rmse": 18.7, "r2": 0.91}}
```

### `POST /forecast`
Predict optimal price for a single product.

**Request:**
```json
{
  "category": "Electronics",
  "stock_level": 50,
  "competitor_price": 299.99,
  "demand_trend": 1.2,
  "margin_ratio": 0.3
}
```

**Response:**
```json
{
  "recommended_price": 287.45,
  "confidence_interval": [271.2, 303.7],
  "similar_products": [...]
}
```

### `POST /batch-forecast`
Batch price forecasts (up to 100 items).

**Request:**
```json
{"items": [{"category": "Electronics", ...}, ...]}
```

### `GET /metrics`
Current model performance metrics.

### `GET /drift`
Statistical drift report (KS-test + PSI).

### `GET /health`
Service health check.

### `GET /version`
API version information.

### `POST /optimize-price`
Find the price in a range that maximises predicted value.

**Request:**
```json
{"category": "Electronics", "stock_level": 50, "demand_trend": 1.2, "margin_ratio": 0.3, "price_min": 100.0, "price_max": 500.0, "n_steps": 20}
```

**Response:**
```json
{"optimal_price": 347.0, "predicted_value": 351.2}
```

### `GET /status`
Service status with uptime in seconds.

### `GET /model/info`
Model metadata, training status, and current metrics.

### `GET /categories`
List all valid product categories.

### `GET /health/detailed`
Detailed health — model file and database status.

### `GET /predictions`
Paginated list of stored predictions with `has_next`/`has_prev` navigation metadata.

### `GET /predictions/count`
Total count of stored predictions.

### `GET /cache-stats`
Hit/miss/hit-rate statistics for the prediction cache.

### `POST /cache-clear`
Evict all entries from the in-memory cache.

### `GET /similar`
Find semantically similar products by category.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./prices.db` | Database connection string |
| `MODEL_PATH` | `models/price_model.pkl` | Model pickle path |
| `METRICS_PATH` | `models/metrics.json` | Metrics JSON path |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `PORT` | `8000` | HTTP server port |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for full setup instructions.

```bash
make install   # install dev dependencies
make test      # run tests with coverage
make lint      # ruff lint + format check
make type-check  # mypy static analysis
```

## License

MIT — see [LICENSE](LICENSE) for details.

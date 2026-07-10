"""Shared pytest fixtures for Price-Prophet test suite."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    """Provide a TestClient for the FastAPI app."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_forecast_payload():
    """Basic forecast request payload."""
    return {
        "category": "Electronics",
        "stock_level": 50,
        "competitor_price": 299.99,
        "demand_trend": 1.2,
        "margin_ratio": 0.3,
    }


@pytest.fixture
def sample_batch_payload():
    """Batch forecast request payload with 3 items."""
    return {
        "items": [
            {
                "category": "Electronics",
                "stock_level": 50,
                "competitor_price": 299.99,
                "demand_trend": 1.2,
            },
            {
                "category": "Books",
                "stock_level": 20,
                "competitor_price": 15.99,
                "demand_trend": 0.8,
            },
            {
                "category": "Clothing",
                "stock_level": 30,
                "competitor_price": 49.99,
                "demand_trend": 1.0,
            },
        ]
    }


@pytest.fixture(scope="module")
def trained_client():
    """TestClient with model trained once for the module scope."""
    from app.main import app

    c = TestClient(app)
    c.post("/train")
    return c


@pytest.fixture
def minimal_features():
    """Minimal feature dict with just required fields."""
    return {"category": "Electronics", "competitor_price": 100.0}


@pytest.fixture
def tmp_model_path(tmp_path):
    """Temporary path for model pickle files."""
    model = tmp_path / "model.pkl"
    metrics = tmp_path / "metrics.json"
    os.environ["MODEL_PATH"] = str(model)
    os.environ["METRICS_PATH"] = str(metrics)
    yield str(model)
    os.environ.pop("MODEL_PATH", None)
    os.environ.pop("METRICS_PATH", None)

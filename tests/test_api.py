"""Comprehensive API endpoint tests for Price-Prophet."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def trained_client(client):
    client.post("/train")
    return client


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_version(client):
    r = client.get("/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_train(client):
    r = client.post("/train")
    assert r.status_code == 200
    assert r.json()["status"] == "trained"


def test_forecast_requires_training(trained_client):
    r = trained_client.post("/forecast", json={
        "category": "Electronics",
        "stock_level": 50,
        "competitor_price": 299.99,
        "demand_trend": 1.2,
    })
    assert r.status_code == 200
    assert "recommended_price" in r.json()


def test_forecast_has_confidence_interval(trained_client):
    r = trained_client.post("/forecast", json={
        "category": "Clothing",
        "stock_level": 30,
        "competitor_price": 49.99,
        "demand_trend": 0.9,
    })
    assert r.status_code == 200
    ci = r.json()["confidence_interval"]
    assert len(ci) == 2
    assert ci[0] < ci[1]


def test_batch_forecast_endpoint(trained_client):
    r = trained_client.post("/batch-forecast", json={"items": [
        {"category": "Electronics", "stock_level": 50, "competitor_price": 299.99, "demand_trend": 1.2},
        {"category": "Books", "stock_level": 20, "competitor_price": 15.99, "demand_trend": 0.8},
    ]})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_batch_forecast_empty_rejected(client):
    r = client.post("/batch-forecast", json={"items": []})
    assert r.status_code == 422


def test_metrics_endpoint(trained_client):
    r = trained_client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "mae" in body or "status" in body


def test_feature_importance_endpoint(trained_client):
    r = trained_client.get("/feature-importance")
    assert r.status_code == 200
    body = r.json()
    assert "feature_importance" in body or "status" in body


def test_drift_endpoint_insufficient_data(client):
    r = client.get("/drift")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body or "drift" in body


def test_similar_products_endpoint(client):
    r = client.get("/similar?category=Electronics&k=3")
    assert r.status_code == 200
    body = r.json()
    assert "similar_products" in body


def test_similar_invalid_k(client):
    r = client.get("/similar?category=Electronics&k=200")
    assert r.status_code == 400


def test_summary_endpoint(client):
    r = client.get("/summary")
    assert r.status_code == 200
    assert "total_predictions" in r.json()


@pytest.mark.parametrize("category", ["Electronics", "Clothing", "Food", "Books"])
def test_forecast_all_categories(trained_client, category):
    r = trained_client.post("/forecast", json={
        "category": category,
        "stock_level": 50,
        "competitor_price": 100.0,
        "demand_trend": 1.0,
    })
    assert r.status_code == 200


def test_forecast_recommendation_values(trained_client):
    r = trained_client.post("/forecast", json={
        "category": "Electronics",
        "stock_level": 50,
        "competitor_price": 299.99,
        "demand_trend": 1.2,
    })
    price = r.json()["recommended_price"]
    assert price > 0


def test_metrics_after_forecast(trained_client):
    trained_client.post("/forecast", json={
        "category": "Electronics",
        "stock_level": 50,
        "competitor_price": 299.99,
        "demand_trend": 1.2,
    })
    r = trained_client.get("/metrics")
    assert r.status_code == 200


def test_forecast_x_process_time_header(trained_client):
    r = trained_client.post("/forecast", json={
        "category": "Electronics",
        "stock_level": 50,
        "competitor_price": 299.99,
        "demand_trend": 1.0,
    })
    assert "x-process-time" in r.headers

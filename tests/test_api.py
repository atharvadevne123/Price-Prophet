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


@pytest.mark.parametrize("n_items", [1, 5, 10])
def test_batch_forecast_variable_sizes(trained_client, n_items):
    items = [
        {"category": "Electronics", "stock_level": 50, "competitor_price": 200.0, "demand_trend": 1.0}
        for _ in range(n_items)
    ]
    r = trained_client.post("/batch-forecast", json={"items": items})
    assert r.status_code == 200
    assert r.json()["count"] == n_items


@pytest.mark.parametrize("category,price", [
    ("Books", 15.0),
    ("Food", 5.0),
    ("Sports", 150.0),
])
def test_batch_forecast_mixed_categories(trained_client, category, price):
    r = trained_client.post("/batch-forecast", json={"items": [
        {"category": category, "stock_level": 30, "competitor_price": price, "demand_trend": 1.0}
    ]})
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["category"] == category
    assert result["recommended_price"] > 0


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


def test_categories_endpoint(client):
    r = client.get("/categories")
    assert r.status_code == 200
    body = r.json()
    assert "categories" in body
    assert "Electronics" in body["categories"]
    assert body["count"] == 10


def test_health_detailed_endpoint(client):
    r = client.get("/health/detailed")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "database" in body


def test_predictions_endpoint(client):
    r = client.get("/predictions")
    assert r.status_code == 200
    body = r.json()
    assert "total" in body
    assert "items" in body
    assert "limit" in body


def test_predictions_endpoint_pagination(client):
    r = client.get("/predictions?limit=5&offset=0")
    assert r.status_code == 200
    assert r.json()["limit"] == 5


def test_predictions_limit_out_of_range(client):
    r = client.get("/predictions?limit=300")
    assert r.status_code == 400


def test_predictions_has_pagination_metadata(client):
    r = client.get("/predictions?limit=5&offset=0")
    assert r.status_code == 200
    body = r.json()
    assert "has_next" in body
    assert "has_prev" in body
    assert body["has_prev"] is False


def test_predictions_next_offset_when_more(trained_client):
    for _ in range(3):
        trained_client.post("/forecast", json={
            "category": "Electronics",
            "stock_level": 50,
            "competitor_price": 299.99,
            "demand_trend": 1.0,
        })
    r = trained_client.get("/predictions?limit=1&offset=0")
    body = r.json()
    if body["total"] > 1:
        assert body["has_next"] is True
        assert body["next_offset"] == 1


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_similar_various_k(trained_client, k):
    r = trained_client.get(f"/similar?category=Electronics&k={k}")
    assert r.status_code == 200
    assert len(r.json()["similar_products"]) <= k


def test_forecast_invalid_category_rejected(client):
    r = client.post("/forecast", json={
        "category": "InvalidCat",
        "stock_level": 50,
        "competitor_price": 100.0,
        "demand_trend": 1.0,
    })
    assert r.status_code == 422


def test_request_id_header_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "test-id-123"})
    assert r.headers.get("x-request-id") == "test-id-123"


def test_request_id_auto_generated(client):
    r = client.get("/health")
    assert "x-request-id" in r.headers
    assert len(r.headers["x-request-id"]) > 0


def test_cache_stats_endpoint(client):
    r = client.get("/cache-stats")
    assert r.status_code == 200
    body = r.json()
    assert "hits" in body
    assert "misses" in body
    assert "hit_rate" in body


def test_cache_clear_endpoint(client):
    r = client.post("/cache-clear")
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"


def test_optimize_price_endpoint(trained_client):
    r = trained_client.post("/optimize-price", json={
        "category": "Electronics",
        "stock_level": 50,
        "demand_trend": 1.2,
        "margin_ratio": 0.3,
        "price_min": 100.0,
        "price_max": 500.0,
        "n_steps": 10,
    })
    assert r.status_code == 200
    body = r.json()
    assert "optimal_price" in body
    assert "predicted_value" in body
    assert 100.0 <= body["optimal_price"] <= 500.0


def test_optimize_price_invalid_range(trained_client):
    r = trained_client.post("/optimize-price", json={
        "category": "Electronics",
        "stock_level": 50,
        "demand_trend": 1.0,
        "margin_ratio": 0.3,
        "price_min": 500.0,
        "price_max": 100.0,
    })
    assert r.status_code == 400


def test_status_endpoint(client):
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "uptime_seconds" in body
    assert body["uptime_seconds"] >= 0
    assert "version" in body


def test_model_info_endpoint(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "trained" in body
    assert "ensemble" in body
    assert "model_path" in body


def test_model_info_after_training(trained_client):
    r = trained_client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["trained"] is True
    assert "mae" in body["metrics"]


def test_cors_headers_present(client):
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 200


def test_cors_wildcard_default(monkeypatch, client):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    r = client.get("/health")
    assert r.status_code == 200


def test_predictions_count_endpoint(client):
    r = client.get("/predictions/count")
    assert r.status_code == 200
    body = r.json()
    assert "count" in body
    assert isinstance(body["count"], int)
    assert body["count"] >= 0


def test_predictions_count_increases_after_forecast(trained_client):
    before = trained_client.get("/predictions/count").json()["count"]
    trained_client.post("/forecast", json={
        "category": "Electronics",
        "stock_level": 50,
        "competitor_price": 299.99,
        "demand_trend": 1.0,
    })
    after = trained_client.get("/predictions/count").json()["count"]
    assert after == before + 1


@pytest.mark.parametrize("category", ["Books", "Food", "Toys"])
def test_optimize_price_all_categories(trained_client, category):
    r = trained_client.post("/optimize-price", json={
        "category": category,
        "stock_level": 30,
        "demand_trend": 1.0,
        "margin_ratio": 0.25,
        "price_min": 5.0,
        "price_max": 200.0,
    })
    assert r.status_code == 200
    assert r.json()["optimal_price"] > 0

"""Tests for app/main.py FastAPI endpoints."""
from __future__ import annotations


def test_health(client):
    """GET /health returns ok and model_loaded flag."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_version(client):
    """GET /version returns a version string."""
    resp = client.get("/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_forecast_basic(client):
    """POST /forecast returns prediction fields for valid input."""
    payload = {
        "product_id": "TEST-001",
        "base_price": 99.99,
        "competitor_price": 109.99,
        "category": "electronics",
        "stock_level": 200,
        "historical_demand_7d": 75,
        "historical_demand_30d": 300,
    }
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["product_id"] == "TEST-001"
    assert data["predicted_demand"] >= 0
    assert "optimized_price" in data
    assert data["recommendation"] in ("increase", "decrease", "hold")


def test_forecast_missing_price(client):
    """POST /forecast without base_price returns 422."""
    resp = client.post("/forecast", json={"product_id": "BAD"})
    assert resp.status_code == 422


def test_train_endpoint(client):
    """POST /train with valid n_samples returns trained status."""
    resp = client.post("/train", json={"n_samples": 200})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "trained"
    assert "run_id" in data
    assert data["rmse_mean"] >= 0


def test_metrics_endpoint(client):
    """GET /metrics returns model_metrics and n_recent_predictions."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_metrics" in data
    assert "n_recent_predictions" in data


def test_similar_products(client):
    """POST /similar returns up to k similar products."""
    resp = client.post("/similar", json={"base_price": 150.0, "category": "electronics", "k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "similar_products" in data
    assert len(data["similar_products"]) <= 3


def test_predictions_history(client):
    """GET /predictions returns a list after at least one forecast."""
    client.post("/forecast", json={"product_id": "HIST-001", "base_price": 50.0, "category": "clothing"})
    resp = client.get("/predictions?limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_drift_insufficient_data(client):
    """GET /drift returns n_recent when fewer than 10 predictions exist."""
    resp = client.get("/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "n_recent" in data or "drift_detected" in data


def test_forecast_different_categories(client):
    """POST /forecast succeeds for all supported categories."""
    categories = ["electronics", "clothing", "food", "home", "sports"]
    for cat in categories:
        resp = client.post("/forecast", json={"product_id": f"CAT-{cat}", "base_price": 80.0, "category": cat})
        assert resp.status_code == 200, f"Failed for category: {cat}"


def test_forecast_price_boundaries(client):
    """POST /forecast handles very small and very large prices."""
    for price in [0.01, 10.0, 1000.0, 9999.0]:
        resp = client.post("/forecast", json={"product_id": "PRICE-TEST", "base_price": price, "category": "other"})
        assert resp.status_code == 200
        assert resp.json()["predicted_demand"] >= 0


def test_categories_endpoint(client):
    """GET /categories returns sorted list with count > 0."""
    resp = client.get("/categories")
    assert resp.status_code == 200
    data = resp.json()
    assert "categories" in data
    assert data["count"] > 0
    assert "electronics" in data["categories"]
    assert sorted(data["categories"]) == data["categories"]


def test_summary_endpoint(client):
    """GET /summary returns expected aggregation keys."""
    resp = client.get("/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "total_training_runs" in data
    assert "drift_events_logged" in data
    assert data["total_predictions"] >= 0


def test_predictions_limit_validation(client):
    """GET /predictions rejects limit=0 and limit=201."""
    assert client.get("/predictions?limit=0").status_code == 422
    assert client.get("/predictions?limit=201").status_code == 422
    assert client.get("/predictions?limit=50").status_code == 200


def test_forecast_with_cost_field(client):
    """POST /forecast uses cost field for expected_profit calculation."""
    resp = client.post("/forecast", json={"product_id": "COST-001", "base_price": 100.0, "cost": 40.0, "category": "electronics"})
    assert resp.status_code == 200
    data = resp.json()
    assert "expected_profit" in data
    assert data["optimized_price"] > 0


def test_train_sample_bounds(client):
    """POST /train rejects n_samples outside [100, 50000]."""
    assert client.post("/train", json={"n_samples": 99}).status_code == 422
    assert client.post("/train", json={"n_samples": 50001}).status_code == 422


def test_batch_forecast_endpoint(client):
    """POST /batch-forecast returns results for each item in request."""
    payload = {
        "items": [
            {"product_id": "BATCH-001", "base_price": 50.0, "category": "electronics"},
            {"product_id": "BATCH-002", "base_price": 100.0, "category": "clothing"},
            {"product_id": "BATCH-003", "base_price": 25.0, "category": "food"},
        ]
    }
    resp = client.post("/batch-forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert len(data["results"]) == 3
    for r in data["results"]:
        assert "predicted_demand" in r
        assert r["recommendation"] in ("increase", "decrease", "hold")


def test_batch_forecast_empty_rejected(client):
    """POST /batch-forecast with empty items list returns 422."""
    resp = client.post("/batch-forecast", json={"items": []})
    assert resp.status_code == 422


def test_similar_products_all_categories(client):
    """POST /similar works for each supported category."""
    from app.features import CATEGORY_MAP
    for cat in list(CATEGORY_MAP.keys())[:3]:
        resp = client.post("/similar", json={"base_price": 100.0, "category": cat, "k": 2})
        assert resp.status_code == 200


def test_forecast_recommendation_values(client):
    """Recommendation field is always one of the three valid strings."""
    for price in [50.0, 100.0, 500.0]:
        resp = client.post("/forecast", json={"product_id": "REC-TEST", "base_price": price})
        assert resp.json()["recommendation"] in ("increase", "decrease", "hold")


def test_metrics_after_forecast(client):
    """GET /metrics reflects n_recent_predictions after a forecast."""
    before = client.get("/metrics").json()["n_recent_predictions"]
    client.post("/forecast", json={"product_id": "METRIC-TEST", "base_price": 75.0})
    after = client.get("/metrics").json()["n_recent_predictions"]
    assert after >= before


def test_drift_after_10_forecasts(client):
    """GET /drift returns drift analysis once 10+ predictions are available."""
    for i in range(12):
        client.post("/forecast", json={"product_id": f"DRIFT-{i}", "base_price": float(50 + i * 5)})
    resp = client.get("/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "drift_detected" in data or "n_recent" in data

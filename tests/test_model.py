"""Comprehensive tests for ML model training, prediction, and utilities."""
from __future__ import annotations

import pytest
import tempfile
import os


@pytest.fixture
def tmp_model(tmp_path):
    """Return a temp path for the model file."""
    return str(tmp_path / "model.pkl")


@pytest.fixture
def tmp_metrics(tmp_path):
    """Return a temp path for the metrics file."""
    return str(tmp_path / "metrics.json")


@pytest.fixture
def trained_bundle(tmp_model, tmp_metrics, monkeypatch):
    """Train a model on small synthetic data and return (metrics, model_path)."""
    monkeypatch.setenv("MODEL_PATH", tmp_model)
    monkeypatch.setenv("METRICS_PATH", tmp_metrics)
    from app.features import generate_synthetic_training_data
    from app.model import train_model
    df = generate_synthetic_training_data(200)
    metrics = train_model(df, model_path=tmp_model, run_cv=False)
    return metrics, tmp_model


def test_build_ensemble_has_three_estimators():
    from app.model import build_ensemble
    ens = build_ensemble()
    assert len(ens.estimators) == 3


def test_build_ensemble_has_at_least_one_estimator():
    from app.model import build_ensemble
    ens = build_ensemble()
    assert len(ens.estimators) >= 1


def test_build_pipeline_has_scaler():
    from app.model import build_pipeline
    pipe = build_pipeline()
    assert "scaler" in pipe.named_steps


def test_train_model_returns_metrics(trained_bundle):
    metrics, _ = trained_bundle
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_train_model_mae_positive(trained_bundle):
    metrics, _ = trained_bundle
    assert metrics["mae"] > 0


def test_model_persisted_to_disk(trained_bundle, tmp_model):
    _, model_path = trained_bundle
    assert os.path.exists(model_path)


def test_metrics_persisted_to_disk(trained_bundle, tmp_metrics):
    _, _ = trained_bundle
    assert os.path.exists(tmp_metrics)


def test_load_metrics_returns_dict(trained_bundle, tmp_metrics):
    _, _ = trained_bundle
    from app.model import load_metrics
    m = load_metrics(tmp_metrics)
    assert isinstance(m, dict)


def test_predict_returns_positive(trained_bundle):
    _, model_path = trained_bundle
    from app.model import predict
    features = {"category": "Electronics", "stock_level": 50,
                "competitor_price": 299.99, "demand_trend": 1.2}
    result = predict(features, model_path=model_path)
    assert result > 0


def test_predict_single_sample(trained_bundle):
    _, model_path = trained_bundle
    from app.model import predict
    features = {"category": "Books", "stock_level": 10,
                "competitor_price": 20.0, "demand_trend": 0.8}
    result = predict(features, model_path=model_path)
    assert isinstance(result, float)


def test_optimize_price_range(trained_bundle):
    _, model_path = trained_bundle
    from app.model import optimize_price
    result = optimize_price(
        {"category": "Electronics", "stock_level": 50, "demand_trend": 1.0},
        price_min=50.0, price_max=500.0, n_steps=10, model_path=model_path,
    )
    assert 50.0 <= result["optimal_price"] <= 500.0


@pytest.mark.parametrize("n_steps", [5, 10, 20])
def test_optimize_price_n_steps(trained_bundle, n_steps):
    _, model_path = trained_bundle
    from app.model import optimize_price
    result = optimize_price(
        {"category": "Electronics", "stock_level": 50, "demand_trend": 1.0},
        n_steps=n_steps, model_path=model_path,
    )
    assert "optimal_price" in result


def test_get_feature_importance_returns_dict(trained_bundle):
    _, model_path = trained_bundle
    from app.model import get_feature_importance
    imp = get_feature_importance(model_path=model_path)
    assert isinstance(imp, dict)
    if imp:
        total = sum(imp.values())
        assert abs(total - 1.0) < 0.01


def test_load_model_raises_on_missing():
    from app.model import load_model
    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/path/model.pkl")


def test_train_with_cv(tmp_path):
    from app.features import generate_synthetic_training_data
    from app.model import train_model
    model_path = str(tmp_path / "model.pkl")
    metrics_path = str(tmp_path / "metrics.json")
    import os
    os.environ["METRICS_PATH"] = metrics_path
    df = generate_synthetic_training_data(300)
    metrics = train_model(df, model_path=model_path, run_cv=True)
    assert "mae" in metrics

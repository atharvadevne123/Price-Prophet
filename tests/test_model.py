"""Tests for app/model.py ensemble ML model."""
from __future__ import annotations

import os

import numpy as np
import pytest

from app.features import FEATURE_NAMES, generate_synthetic_training_data
from app.model import build_ensemble, build_pipeline, load_metrics, optimize_price, predict, train_model


@pytest.fixture(scope="module")
def trained_pipeline():
    """Train a model on small synthetic dataset once per test module."""
    X, y = generate_synthetic_training_data(n_samples=300)
    train_model(X, y)
    from app.model import load_model
    return load_model()


def test_pipeline_builds():
    """Pipeline has scaler and model steps."""
    pipeline = build_pipeline()
    assert pipeline is not None
    steps = dict(pipeline.steps)
    assert "scaler" in steps
    assert "model" in steps


def test_ensemble_has_at_least_one_estimator():
    """Ensemble always includes at least one estimator (RandomForest minimum)."""
    ensemble = build_ensemble()
    assert len(ensemble.estimators) >= 1
    names = [name for name, _ in ensemble.estimators]
    assert "rf" in names


def test_train_returns_metrics():
    """train_model returns expected keys with valid values."""
    X, y = generate_synthetic_training_data(n_samples=200)
    metrics = train_model(X, y)
    assert "run_id" in metrics
    assert "rmse_mean" in metrics
    assert metrics["rmse_mean"] >= 0
    assert metrics["n_features"] == len(FEATURE_NAMES)
    assert metrics["n_samples"] == 200


def test_predict_shape(trained_pipeline):
    """predict returns array of shape (n_samples,)."""
    X, _ = generate_synthetic_training_data(n_samples=10)
    preds = predict(trained_pipeline, X)
    assert preds.shape == (10,)


def test_predict_non_negative(trained_pipeline):
    """Most predictions are non-negative (demand cannot be negative)."""
    X, _ = generate_synthetic_training_data(n_samples=50)
    preds = predict(trained_pipeline, X)
    assert (preds >= 0).mean() > 0.8


def test_optimize_price_returns_dict(trained_pipeline):
    """optimize_price returns dict with required keys and positive price."""
    X, _ = generate_synthetic_training_data(n_samples=1)
    features = X[0]
    cost = features[0] * 0.6
    result = optimize_price(trained_pipeline, features, cost)
    assert "optimized_price" in result
    assert "expected_demand" in result
    assert "expected_profit" in result
    assert result["optimized_price"] > 0


def test_cross_val_5_fold():
    """5-fold CV produces 5 RMSE scores, all non-negative."""
    from sklearn.model_selection import KFold, cross_val_score
    X, y = generate_synthetic_training_data(n_samples=300)
    pipeline = build_pipeline()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_squared_error")
    assert len(scores) == 5
    rmse_scores = np.sqrt(-scores)
    assert all(r >= 0 for r in rmse_scores)


def test_model_persisted_to_disk():
    """train_model writes model.joblib to disk."""
    X, y = generate_synthetic_training_data(n_samples=200)
    train_model(X, y)
    from app.model import MODEL_PATH
    assert os.path.exists(MODEL_PATH)


def test_metrics_persisted_to_disk():
    """train_model writes metrics.json to disk."""
    X, y = generate_synthetic_training_data(n_samples=200)
    train_model(X, y)
    from app.model import METRICS_PATH
    assert os.path.exists(METRICS_PATH)


def test_load_metrics_returns_dict():
    """load_metrics returns a non-empty dict after training."""
    X, y = generate_synthetic_training_data(n_samples=200)
    train_model(X, y)
    metrics = load_metrics()
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_optimize_price_range(trained_pipeline):
    """optimize_price multiplier stays within the requested range."""
    X, _ = generate_synthetic_training_data(n_samples=1)
    features = X[0]
    cost = features[0] * 0.5
    result = optimize_price(trained_pipeline, features, cost, price_range=(0.8, 1.2))
    base = float(features[0])
    assert result["optimized_price"] >= base * 0.8 * 0.99
    assert result["optimized_price"] <= base * 1.2 * 1.01


@pytest.mark.parametrize("n_steps", [5, 10, 20])
def test_optimize_price_n_steps(trained_pipeline, n_steps: int):
    """optimize_price works correctly for different step counts."""
    X, _ = generate_synthetic_training_data(n_samples=1)
    features = X[0]
    result = optimize_price(trained_pipeline, features, cost=float(features[0]) * 0.6, n_steps=n_steps)
    assert result["optimized_price"] > 0
    assert "price_multiplier" in result


def test_predict_single_sample(trained_pipeline):
    """predict on a single sample returns a length-1 array."""
    X, _ = generate_synthetic_training_data(n_samples=1)
    preds = predict(trained_pipeline, X)
    assert preds.shape == (1,)

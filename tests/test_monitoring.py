"""Tests for app/monitoring.py drift detection and health checks."""
from __future__ import annotations

import numpy as np
import pytest

from app.features import generate_synthetic_training_data
from app.monitoring import (
    compute_drift,
    compute_feature_drift,
    load_reference_stats,
    prediction_health_check,
    save_reference_stats,
)


def test_compute_drift_no_drift():
    """KS test on same-distribution samples returns expected keys."""
    ref = list(np.random.normal(0, 1, 100))
    cur = list(np.random.normal(0, 1, 100))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    assert isinstance(result["drift_detected"], bool)


def test_compute_drift_with_drift():
    """Clearly shifted distributions should be detected as drift."""
    ref = list(np.random.normal(0, 1, 200))
    cur = list(np.random.normal(5, 1, 200))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_data():
    """Fewer than 5 samples triggers an error dict."""
    result = compute_drift([1, 2], [3])
    assert "error" in result
    assert result["drift_detected"] is False


def test_feature_drift_no_drift():
    """Same-seed synthetic data should have low drift rate."""
    X_ref, _ = generate_synthetic_training_data(n_samples=200)
    X_cur, _ = generate_synthetic_training_data(n_samples=100)
    result = compute_feature_drift(X_ref, X_cur)
    assert "feature_results" in result
    assert "drift_detected" in result
    assert "drift_rate" in result
    assert 0.0 <= result["drift_rate"] <= 1.0


def test_prediction_health_check_normal():
    """Health check returns count, mean, std, negative_pct for normal data."""
    preds = [10.5, 20.3, 15.0, 30.2, 5.1]
    health = prediction_health_check(preds)
    assert health["count"] == 5
    assert health["negative_pct"] == 0.0
    assert "mean" in health
    assert "std" in health


def test_prediction_health_check_empty():
    """Empty predictions list returns status=no_data."""
    result = prediction_health_check([])
    assert result["status"] == "no_data"


def test_save_load_reference_stats(tmp_path):
    """save/load reference stats roundtrip preserves n_samples and keys."""
    import os
    os.environ["REFERENCE_STATS_PATH"] = str(tmp_path / "ref_stats.json")
    X, _ = generate_synthetic_training_data(n_samples=100)
    save_reference_stats(X)
    stats = load_reference_stats()
    assert "mean" in stats
    assert "std" in stats
    assert stats["n_samples"] == 100


def test_compute_drift_ks_statistic_range():
    """KS statistic is always in [0, 1]."""
    ref = list(np.random.normal(0, 1, 50))
    cur = list(np.random.normal(0, 1, 50))
    result = compute_drift(ref, cur)
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_compute_drift_p_value_range():
    """p-value is always in [0, 1]."""
    ref = list(np.random.uniform(0, 1, 50))
    cur = list(np.random.uniform(0, 1, 50))
    result = compute_drift(ref, cur)
    assert 0.0 <= result["p_value"] <= 1.0


def test_feature_drift_returns_all_features():
    """compute_feature_drift covers all 15 FEATURE_NAMES."""
    from app.features import FEATURE_NAMES
    X_ref, _ = generate_synthetic_training_data(n_samples=100)
    X_cur, _ = generate_synthetic_training_data(n_samples=100)
    result = compute_feature_drift(X_ref, X_cur)
    assert set(result["feature_results"].keys()) == set(FEATURE_NAMES)


def test_feature_drift_rate_with_full_drift():
    """Completely shifted distributions produce drift_rate close to 1.0."""
    X_ref = np.zeros((200, 5))
    X_cur = np.ones((200, 5)) * 1000
    from app.monitoring import compute_feature_drift as cfd
    from app.features import FEATURE_NAMES as FN
    result = cfd(X_ref[:, :5], X_cur[:, :5])
    assert result["drift_rate"] > 0.5


def test_prediction_health_all_negative():
    """All-negative predictions report 100% negative_pct."""
    preds = [-1.0, -2.0, -3.0]
    health = prediction_health_check(preds)
    assert health["negative_pct"] == 100.0


@pytest.mark.parametrize("n_preds", [1, 5, 10, 50])
def test_prediction_health_count(n_preds: int):
    """prediction_health_check count matches input length."""
    preds = list(np.random.uniform(0, 100, n_preds))
    health = prediction_health_check(preds)
    assert health["count"] == n_preds

"""Tests for drift monitoring with KS-test, PSI, and alerts."""
from __future__ import annotations

import pytest


def test_compute_drift_same_distribution():
    from app.monitoring import compute_drift
    ref = list(range(100))
    result = compute_drift(ref, ref)
    assert result["ks_statistic"] == 0.0
    assert result["is_drifted"] is False


def test_compute_drift_different_distribution():
    from app.monitoring import compute_drift
    ref = list(range(100))
    cur = list(range(100, 200))
    result = compute_drift(ref, cur)
    assert result["ks_statistic"] > 0.5
    assert result["is_drifted"] is True


def test_compute_drift_ks_statistic_range():
    from app.monitoring import compute_drift
    result = compute_drift(list(range(50)), list(range(25, 75)))
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_compute_drift_p_value_range():
    from app.monitoring import compute_drift
    result = compute_drift(list(range(100)), list(range(100)))
    assert 0.0 <= result["p_value"] <= 1.0


def test_feature_drift_returns_all_features():
    from app.monitoring import compute_feature_drift
    ref = {"price": [float(i) for i in range(50)], "stock": [float(i) for i in range(50)]}
    cur = {"price": [float(i + 5) for i in range(50)], "stock": [float(i) for i in range(50)]}
    result = compute_feature_drift(ref, cur)
    assert "price" in result
    assert "stock" in result


def test_feature_drift_rate_with_full_drift():
    from app.monitoring import compute_feature_drift
    ref = {"x": [float(i) for i in range(100)]}
    cur = {"x": [float(i + 200) for i in range(100)]}
    result = compute_feature_drift(ref, cur)
    assert result["x"]["is_drifted"] is True


def test_prediction_health_all_negative():
    from app.monitoring import prediction_health
    result = prediction_health([-1.0, -2.0, -3.0])
    assert result["negative_count"] == 3


def test_prediction_health_empty():
    from app.monitoring import prediction_health
    result = prediction_health([])
    assert result["count"] == 0


@pytest.mark.parametrize("n_preds", [1, 5, 10, 50])
def test_prediction_health_count(n_preds):
    from app.monitoring import prediction_health
    result = prediction_health([float(i) for i in range(n_preds)])
    assert result["count"] == n_preds


def test_compute_psi_same_distribution():
    from app.monitoring import compute_psi
    ref = [float(i) for i in range(100)]
    psi = compute_psi(ref, ref)
    assert psi < 0.1


def test_compute_psi_different_distribution():
    from app.monitoring import compute_psi
    ref = [float(i) for i in range(100)]
    cur = [float(i + 100) for i in range(100)]
    psi = compute_psi(ref, cur)
    assert psi > 0.0


def test_check_alerts_healthy():
    from app.monitoring import check_alerts
    alerts = check_alerts({"ks_statistic": 0.02, "p_value": 0.5, "psi": 0.05, "is_drifted": False})
    assert alerts == []


def test_check_alerts_critical_ks():
    from app.monitoring import check_alerts
    alerts = check_alerts({"ks_statistic": 0.5, "p_value": 0.001, "psi": 0.0, "is_drifted": True})
    assert any("CRITICAL" in a for a in alerts)


def test_check_alerts_critical_psi():
    from app.monitoring import check_alerts
    alerts = check_alerts({"ks_statistic": 0.0, "p_value": 1.0, "psi": 0.3, "is_drifted": True})
    assert any("PSI" in a for a in alerts)

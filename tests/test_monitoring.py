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


@pytest.mark.parametrize(
    "ks,expected_alert",
    [
        (0.01, False),
        (0.06, True),
        (0.31, True),
    ],
)
def test_check_alerts_ks_thresholds(ks, expected_alert):
    from app.monitoring import check_alerts

    alerts = check_alerts({"ks_statistic": ks, "p_value": 0.001 if ks > 0.05 else 0.5, "psi": 0.0})
    assert bool(alerts) == expected_alert


def test_compute_drift_insufficient_data_ref():
    from app.monitoring import compute_drift

    result = compute_drift([1.0], [1.0, 2.0, 3.0])
    assert result["ks_statistic"] == 0.0
    assert result["is_drifted"] is False


def test_compute_psi_insufficient_data():
    from app.monitoring import compute_psi

    psi = compute_psi([1.0], [1.0])
    assert psi == 0.0


def test_feature_drift_missing_feature_in_current():
    from app.monitoring import compute_feature_drift

    ref = {"x": [float(i) for i in range(50)], "y": [float(i) for i in range(50)]}
    cur = {"x": [float(i + 10) for i in range(50)]}
    result = compute_feature_drift(ref, cur)
    assert "x" in result
    assert "y" in result
    assert result["y"]["ks_statistic"] == 0.0


@pytest.mark.parametrize(
    "values,expected_neg",
    [
        ([1.0, 2.0, 3.0], 0),
        ([-1.0, 2.0, -3.0], 2),
        ([-5.0, -5.0], 2),
    ],
)
def test_prediction_health_negative_count(values, expected_neg):
    from app.monitoring import prediction_health

    result = prediction_health(values)
    assert result["negative_count"] == expected_neg


def test_prediction_health_has_percentiles():
    from app.monitoring import prediction_health

    result = prediction_health([10.0, 20.0, 30.0, 40.0, 50.0])
    assert "p25" in result
    assert "p50" in result
    assert "p75" in result
    assert result["p25"] <= result["p50"] <= result["p75"]


def test_prediction_health_percentile_ordering():
    from app.monitoring import prediction_health

    values = [float(i) for i in range(1, 101)]
    result = prediction_health(values)
    assert result["p25"] < result["p50"] < result["p75"]
    assert result["min"] <= result["p25"]
    assert result["p75"] <= result["max"]


def test_prediction_health_empty_has_percentiles():
    from app.monitoring import prediction_health

    result = prediction_health([])
    assert result["p25"] == 0.0
    assert result["p50"] == 0.0
    assert result["p75"] == 0.0

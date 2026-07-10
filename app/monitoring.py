"""Statistical drift monitoring with KS-test and PSI."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "compute_psi",
    "compute_drift",
    "compute_feature_drift",
    "prediction_health",
    "check_alerts",
]

logger = logging.getLogger(__name__)

_PSI_BINS: int = 10
_PSI_DRIFT_THRESHOLD: float = 0.2
_KS_DRIFT_THRESHOLD: float = 0.05


def compute_psi(
    reference: list[float],
    current: list[float],
    bins: int = _PSI_BINS,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI < 0.1: no significant change
    PSI 0.1–0.2: minor change
    PSI > 0.2: major change (drift)

    Args:
        reference: Reference (training) distribution values.
        current: Current (production) distribution values.
        bins: Number of histogram bins.

    Returns:
        PSI score as a float.
    """
    if len(reference) < 2 or len(current) < 2:
        return 0.0
    ref_arr = np.array(reference, dtype=float)
    cur_arr = np.array(current, dtype=float)
    breakpoints = np.linspace(
        min(ref_arr.min(), cur_arr.min()),
        max(ref_arr.max(), cur_arr.max()),
        bins + 1,
    )
    ref_counts, _ = np.histogram(ref_arr, bins=breakpoints)
    cur_counts, _ = np.histogram(cur_arr, bins=breakpoints)
    ref_pct = (ref_counts + 1e-6) / (len(ref_arr) + 1e-6 * bins)
    cur_pct = (cur_counts + 1e-6) / (len(cur_arr) + 1e-6 * bins)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct + 1e-9)))
    return round(psi, 6)


def compute_drift(
    reference: list[float],
    current: list[float],
) -> dict[str, object]:
    """Run KS-test to detect distributional drift.

    Args:
        reference: Baseline sample (e.g., training predictions).
        current: Current sample (e.g., recent production predictions).

    Returns:
        Dictionary with ``ks_statistic``, ``p_value``, and ``is_drifted`` keys.
    """
    if len(reference) < 2 or len(current) < 2:
        return {"ks_statistic": 0.0, "p_value": 1.0, "is_drifted": False}
    stat, pval = stats.ks_2samp(reference, current)
    psi = compute_psi(reference, current)
    is_drifted = bool(pval < _KS_DRIFT_THRESHOLD or psi > _PSI_DRIFT_THRESHOLD)
    logger.debug("KS=%.4f p=%.4f PSI=%.4f drifted=%s", stat, pval, psi, is_drifted)
    return {
        "ks_statistic": round(float(stat), 6),
        "p_value": round(float(pval), 6),
        "psi": psi,
        "is_drifted": is_drifted,
    }


def compute_feature_drift(
    reference: dict[str, list[float]],
    current: dict[str, list[float]],
) -> dict[str, dict[str, object]]:
    """Compute per-feature drift between reference and current windows.

    Args:
        reference: Mapping of feature name -> reference values.
        current: Mapping of feature name -> current values.

    Returns:
        Mapping of feature name -> drift result dict.
    """
    results: dict[str, dict[str, object]] = {}
    for feature, ref_vals in reference.items():
        cur_vals = current.get(feature, [])
        results[feature] = compute_drift(ref_vals, cur_vals)
    return results


def prediction_health(predictions: list[float]) -> dict[str, Any]:
    """Summarise basic health statistics of recent predictions.

    Args:
        predictions: List of predicted prices.

    Returns:
        Dictionary with count, mean, std, min, max, and negative_count.
    """
    if not predictions:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "negative_count": 0,
        }
    arr = np.array(predictions, dtype=float)
    return {
        "count": len(arr),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "negative_count": int((arr < 0).sum()),
    }


def check_alerts(drift_result: dict[str, object]) -> list[str]:
    """Check drift result for alert conditions.

    Args:
        drift_result: Output from :func:`compute_drift`.

    Returns:
        List of human-readable alert strings (empty if healthy).
    """
    alerts: list[str] = []
    ks = float(drift_result.get("ks_statistic", 0.0))
    psi = float(drift_result.get("psi", 0.0))
    if ks > 0.3:
        alerts.append(f"CRITICAL: KS statistic {ks:.3f} exceeds 0.3 — severe drift detected")
    elif ks > _KS_DRIFT_THRESHOLD:
        alerts.append(f"WARNING: KS statistic {ks:.3f} exceeds threshold {_KS_DRIFT_THRESHOLD}")
    if psi > 0.25:
        alerts.append(f"CRITICAL: PSI {psi:.3f} exceeds 0.25 — major distribution shift")
    elif psi > _PSI_DRIFT_THRESHOLD:
        alerts.append(f"WARNING: PSI {psi:.3f} exceeds threshold {_PSI_DRIFT_THRESHOLD}")
    return alerts

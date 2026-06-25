"""Drift detection and prediction health monitoring."""
from __future__ import annotations

import json
import logging
import os

import numpy as np
from scipy.stats import ks_2samp

from app.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

REFERENCE_STATS_PATH: str = os.getenv("REFERENCE_STATS_PATH", "reference_stats.json")


def compute_drift(reference: list[float], current: list[float]) -> dict[str, object]:
    """Run a two-sample KS test to detect distribution shift.

    Args:
        reference: Reference distribution samples (baseline).
        current: Current distribution samples to compare against reference.

    Returns:
        Dict with ks_statistic, p_value, drift_detected (bool), or an error
        key if there are fewer than 5 samples in either distribution.
    """
    if len(reference) < 5 or len(current) < 5:
        return {"error": "Not enough data for drift test", "drift_detected": False}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def compute_feature_drift(
    reference_matrix: np.ndarray,
    current_matrix: np.ndarray,
) -> dict[str, object]:
    """Compute per-feature KS drift across all 15 features.

    Args:
        reference_matrix: Reference feature matrix of shape (n_ref, n_features).
        current_matrix: Current feature matrix of shape (n_cur, n_features).

    Returns:
        Dict with feature_results, drifted_features list, drift_detected bool,
        and drift_rate (fraction of features showing drift).
    """
    results: dict[str, dict[str, object]] = {}
    n_features: int = min(reference_matrix.shape[1], current_matrix.shape[1], len(FEATURE_NAMES))
    for i in range(n_features):
        fname: str = FEATURE_NAMES[i]
        results[fname] = compute_drift(
            reference_matrix[:, i].tolist(),
            current_matrix[:, i].tolist(),
        )
    drifted: list[str] = [k for k, v in results.items() if v.get("drift_detected")]
    return {
        "feature_results": results,
        "drifted_features": drifted,
        "drift_detected": len(drifted) > 0,
        "drift_rate": round(len(drifted) / max(n_features, 1), 3),
    }


def save_reference_stats(X: np.ndarray) -> None:
    """Persist feature distribution statistics as JSON for later drift comparison.

    Args:
        X: Reference feature matrix of shape (n_samples, n_features).
    """
    stats: dict[str, object] = {
        "mean": X.mean(axis=0).tolist(),
        "std": X.std(axis=0).tolist(),
        "min": X.min(axis=0).tolist(),
        "max": X.max(axis=0).tolist(),
        "n_samples": X.shape[0],
        "feature_names": FEATURE_NAMES,
    }
    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Reference stats saved: n_samples=%d", X.shape[0])


def load_reference_stats() -> dict[str, object]:
    """Load persisted reference stats from disk.

    Returns:
        Stats dict, or empty dict if no reference stats file exists.
    """
    if os.path.exists(REFERENCE_STATS_PATH):
        with open(REFERENCE_STATS_PATH) as f:
            return json.load(f)
    return {}


def prediction_health_check(predictions: list[float]) -> dict[str, object]:
    """Summarize recent prediction distribution for health monitoring.

    Args:
        predictions: List of recent demand prediction values.

    Returns:
        Dict with count, mean, std, min, max, negative_pct, zero_pct,
        or {"status": "no_data"} if the list is empty.
    """
    if not predictions:
        return {"status": "no_data"}
    arr: np.ndarray = np.array(predictions)
    return {
        "count": len(predictions),
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std()), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "negative_pct": round(float((arr < 0).mean()) * 100, 2),
        "zero_pct": round(float((arr == 0).mean()) * 100, 2),
    }

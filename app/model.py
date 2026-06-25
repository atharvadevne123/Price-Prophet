"""Ensemble ML model training, prediction, and price optimization."""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB: bool = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB: bool = True
except ImportError:
    HAS_LGB = False

from app.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_PATH: str = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH: str = os.getenv("METRICS_PATH", "metrics.json")


def build_ensemble() -> VotingRegressor:
    """Build a VotingRegressor ensemble from available estimators.

    Returns:
        VotingRegressor with RandomForest always included, plus XGBoost and
        LightGBM if installed, or GradientBoosting as a fallback.
    """
    estimators: list[tuple[str, object]] = [
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))
    ]
    if HAS_XGB:
        estimators.append(("xgb", XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="rmse")))
    if HAS_LGB:
        estimators.append(("lgb", LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)))
    if not HAS_XGB and not HAS_LGB:
        estimators.append(("gbm", GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)))
    return VotingRegressor(estimators=estimators)


def build_pipeline() -> Pipeline:
    """Build the full sklearn Pipeline with StandardScaler and ensemble model.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", build_ensemble()),
    ])


def train_model(X: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Train the ensemble pipeline with 5-fold CV and persist to disk.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target demand array of shape (n_samples,).

    Returns:
        Metrics dict with run_id, rmse_mean, rmse_std, n_features, n_samples,
        feature_names, and estimators.
    """
    pipeline: Pipeline = build_pipeline()
    kf: KFold = KFold(n_splits=5, shuffle=True, random_state=42)

    neg_mse_scores: np.ndarray = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_squared_error")
    rmse_scores: np.ndarray = np.sqrt(-neg_mse_scores)

    pipeline.fit(X, y)

    run_id: str = str(uuid.uuid4())[:8]
    metrics: dict[str, object] = {
        "run_id": run_id,
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": FEATURE_NAMES,
        "estimators": [name for name, _ in build_ensemble().estimators],
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Model trained: run_id=%s rmse=%.4f+/-%.4f", run_id, rmse_scores.mean(), rmse_scores.std())
    return metrics


def load_model() -> Pipeline:
    """Load the persisted model pipeline, training on synthetic data if missing.

    Returns:
        Fitted sklearn Pipeline ready for prediction.
    """
    if not os.path.exists(MODEL_PATH):
        logger.warning("No model found at %s — training on synthetic data", MODEL_PATH)
        from app.features import generate_synthetic_training_data
        X, y = generate_synthetic_training_data()
        train_model(X, y)
    return joblib.load(MODEL_PATH)


def predict(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Run inference on the fitted pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        Demand prediction array of shape (n_samples,).
    """
    return pipeline.predict(X)


def load_metrics() -> dict[str, object]:
    """Load the most recent training metrics from disk.

    Returns:
        Metrics dict, or empty dict if no metrics file exists.
    """
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def optimize_price(
    pipeline: Pipeline,
    base_features: np.ndarray,
    cost: float,
    price_range: tuple[float, float] = (0.7, 1.5),
    n_steps: int = 20,
) -> dict[str, float]:
    """Grid-search over price multipliers to maximize profit.

    Args:
        pipeline: Fitted sklearn Pipeline.
        base_features: Single-record feature array of shape (n_features,).
        cost: Unit cost of the product.
        price_range: (min_mult, max_mult) multiplier range to search.
        n_steps: Number of evenly-spaced multipliers to evaluate.

    Returns:
        Dict with optimized_price, expected_demand, expected_profit, price_multiplier.
    """
    best_price_mult: float = 1.0
    best_revenue: float = float("-inf")
    best_demand: float = 0.0
    best_new_price: float = float(base_features[0])

    for mult in np.linspace(price_range[0], price_range[1], n_steps):
        features: np.ndarray = base_features.copy()
        base_price: float = float(features[0])
        new_price: float = base_price * mult
        features[0] = new_price
        features[2] = new_price / max(float(features[1]), 0.01)

        demand: float = float(predict(pipeline, features.reshape(1, -1))[0])
        profit: float = (new_price - cost) * demand

        if profit > best_revenue:
            best_revenue = profit
            best_price_mult = float(mult)
            best_demand = demand
            best_new_price = new_price

    return {
        "optimized_price": round(best_new_price, 2),
        "expected_demand": round(best_demand, 2),
        "expected_profit": round(best_revenue, 2),
        "price_multiplier": round(best_price_mult, 3),
    }

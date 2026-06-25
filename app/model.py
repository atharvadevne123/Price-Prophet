"""ML model: ensemble training, prediction, and feature importance."""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from app.features import engineer_features

logger = logging.getLogger(__name__)

MODEL_PATH: str = os.getenv("MODEL_PATH", "models/price_model.pkl")
METRICS_PATH: str = os.getenv("METRICS_PATH", "models/metrics.json")
N_CV_FOLDS: int = 5


def build_ensemble() -> VotingRegressor:
    """Build the base VotingRegressor ensemble.

    Returns:
        Unfitted :class:`VotingRegressor` with XGB, LGBM, and RF estimators.
    """
    estimators: list[tuple[str, object]] = [
        ("xgb", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbosity=0)),
        ("lgbm", LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbose=-1)),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=8,
                                      random_state=42, n_jobs=-1)),
    ]
    return VotingRegressor(estimators=estimators)


def build_pipeline() -> Pipeline:
    """Wrap the ensemble in a StandardScaler pipeline.

    Returns:
        Unfitted :class:`Pipeline` with scaler + VotingRegressor.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", build_ensemble()),
    ])


def train_model(
    df: pd.DataFrame,
    model_path: str = MODEL_PATH,
    run_cv: bool = True,
) -> dict[str, object]:
    """Train the ensemble pipeline and persist it to disk.

    Args:
        df: Training DataFrame with features and a ``price`` target column.
        model_path: Path to write the pickled model.
        run_cv: Whether to compute cross-validation MAE.

    Returns:
        Dict with ``mae``, ``rmse``, ``r2``, and optionally ``cv_mae_mean``.
    """
    feature_cols = [c for c in df.columns if c != "price"]
    X = df[feature_cols].values
    y = df["price"].values

    pipeline = build_pipeline()

    cv_mae: float | None = None
    if run_cv and len(X) >= N_CV_FOLDS * 10:
        logger.info("Running %d-fold cross-validation...", N_CV_FOLDS)
        kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
        cv_mae = float(-cv_scores.mean())
        logger.info("CV MAE=%.4f ± %.4f", cv_mae, cv_scores.std())

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    mae = float(mean_absolute_error(y, preds))
    rmse = float(mean_squared_error(y, preds) ** 0.5)
    r2 = float(r2_score(y, preds))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "feature_cols": feature_cols}, f)

    import json
    metrics: dict[str, object] = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
    if cv_mae is not None:
        metrics["cv_mae_mean"] = round(cv_mae, 4)
    Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Trained: MAE=%.4f RMSE=%.4f R2=%.4f", mae, rmse, r2)
    return metrics


def load_model(model_path: str = MODEL_PATH) -> dict[str, Any]:
    """Load a trained model bundle from disk.

    Args:
        model_path: Path to the pickled model file.

    Returns:
        Dict with ``pipeline`` and ``feature_cols`` keys.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run POST /train first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(features: dict[str, Any], model_path: str = MODEL_PATH) -> float:
    """Predict the optimal price for a given feature dict.

    Args:
        features: Raw request feature dict (pre-engineer_features).
        model_path: Path to the trained model.

    Returns:
        Predicted price as a float.
    """
    bundle = load_model(model_path)
    df = engineer_features(features)
    X = df[bundle["feature_cols"]].values
    return float(bundle["pipeline"].predict(X)[0])


def load_metrics(model_path: str = METRICS_PATH) -> dict[str, Any]:
    """Load model evaluation metrics from the metrics JSON file.

    Args:
        model_path: Path to the metrics JSON file.

    Returns:
        Dict with metric values, or empty dict if file not found.
    """
    import json
    try:
        with open(model_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def get_feature_importance(model_path: str = MODEL_PATH) -> dict[str, float]:
    """Extract feature importances from the Random Forest sub-estimator.

    Args:
        model_path: Path to the trained model.

    Returns:
        Dict mapping feature name -> importance score (0–1, sums to 1).

    Raises:
        FileNotFoundError: If model not trained.
    """
    bundle = load_model(model_path)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]
    voting: VotingRegressor = pipeline.named_steps["model"]
    rf_est = None
    for name, est in voting.estimators_:
        if name == "rf":
            rf_est = est
            break
    if rf_est is None or not hasattr(rf_est, "feature_importances_"):
        return {}
    importances = rf_est.feature_importances_
    total = importances.sum() or 1.0
    return {col: round(float(imp / total), 6) for col, imp in zip(feature_cols, importances)}


def optimize_price(
    features: dict[str, Any],
    price_min: float = 1.0,
    price_max: float = 1000.0,
    n_steps: int = 20,
    model_path: str = MODEL_PATH,
) -> dict[str, Any]:
    """Find the price in [price_min, price_max] that maximises predicted value.

    Args:
        features: Raw feature dict.
        price_min: Lower bound of the price search range.
        price_max: Upper bound of the price search range.
        n_steps: Number of candidate prices to evaluate.
        model_path: Path to the trained model.

    Returns:
        Dict with ``optimal_price`` and ``predicted_value``.
    """
    candidates = np.linspace(price_min, price_max, n_steps)
    best_price = float(candidates[0])
    best_val = float("-inf")
    for price in candidates:
        f = dict(features)
        f["competitor_price"] = float(price)
        val = predict(f, model_path)
        if val > best_val:
            best_val = val
            best_price = float(price)
    return {"optimal_price": round(best_price, 2), "predicted_value": round(best_val, 2)}

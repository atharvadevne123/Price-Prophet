"""FastAPI application for price optimization and demand forecasting."""
from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import __version__
from app.database import DriftLog, ModelMetrics, Prediction, get_db, init_db
from app.features import (
    CATEGORY_MAP,
    engineer_features,
    generate_synthetic_training_data,
)
from app.model import load_metrics, load_model, optimize_price, predict, train_model
from app.monitoring import (
    compute_feature_drift,
    prediction_health_check,
    save_reference_stats,
)
from app.retrieval import get_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_model = None
_reference_X: Optional[np.ndarray] = None
_recent_X: Deque[np.ndarray] = deque(maxlen=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    global _model, _reference_X
    init_db()
    _model = load_model()
    X_ref, _ = generate_synthetic_training_data(n_samples=500)
    _reference_X = X_ref
    save_reference_stats(X_ref)
    get_index()
    logger.info("price-prophet startup complete — version %s", __version__)
    yield
    logger.info("price-prophet shutdown")


app = FastAPI(
    title="price-prophet",
    description="E-commerce price optimization and demand forecasting engine",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Inject X-Process-Time header into every response."""
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}"
    return response


class ForecastRequest(BaseModel):
    """Request body for POST /forecast."""

    product_id: str = Field(..., json_schema_extra={"example": "PROD-001"})
    base_price: float = Field(..., gt=0, json_schema_extra={"example": 99.99})
    competitor_price: Optional[float] = Field(None, json_schema_extra={"example": 105.0})
    category: str = Field("other", json_schema_extra={"example": "electronics"})
    stock_level: Optional[float] = Field(100, json_schema_extra={"example": 150})
    cost: Optional[float] = Field(None, json_schema_extra={"example": 60.0})
    date: Optional[str] = Field(None, json_schema_extra={"example": "2024-12-15"})
    historical_demand_7d: Optional[float] = Field(50, json_schema_extra={"example": 75})
    historical_demand_30d: Optional[float] = Field(200, json_schema_extra={"example": 280})
    days_since_last_promotion: Optional[float] = Field(30, json_schema_extra={"example": 14})


class TrainRequest(BaseModel):
    """Request body for POST /train."""

    n_samples: int = Field(2000, ge=100, le=50000)


class SimilarRequest(BaseModel):
    """Request body for POST /similar."""

    base_price: float = Field(..., gt=0)
    category: str = "other"
    competitor_price: Optional[float] = None
    k: int = Field(5, ge=1, le=20)


class BatchForecastRequest(BaseModel):
    """Request body for POST /batch-forecast."""

    items: list[ForecastRequest] = Field(..., min_length=1, max_length=100)


@app.get("/health", tags=["system"], summary="Health check")
def health() -> dict:
    """Return service health status and whether the ML model is loaded."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/version", tags=["system"], summary="API version")
def version() -> dict:
    """Return the current API version."""
    return {"version": __version__}


@app.post("/forecast", tags=["prediction"], summary="Predict demand and optimal price")
def forecast(req: ForecastRequest, db: Session = Depends(get_db)) -> dict:
    """Forecast demand and compute the profit-maximising price for a product.

    Args:
        req: ForecastRequest with product pricing and context data.
        db: SQLAlchemy session (injected by FastAPI).

    Returns:
        Dict with product_id, predicted_demand, optimized_price, expected_profit,
        price_multiplier, and recommendation (increase/decrease/hold).
    """
    if _model is None:
        raise HTTPException(503, "Model not ready")

    data = req.model_dump()
    features: np.ndarray = engineer_features(data)
    demand: float = float(predict(_model, features.reshape(1, -1))[0])
    demand = max(demand, 0.0)

    cost: float = float(req.cost or req.base_price * 0.6)
    opt: dict = optimize_price(_model, features, cost)

    _recent_X.append(features)

    pred_row = Prediction(
        product_id=req.product_id,
        category=req.category,
        predicted_demand=round(demand, 2),
        optimized_price=opt["optimized_price"],
        confidence=min(1.0, demand / 100),
        features_used=data,
    )
    db.add(pred_row)
    db.commit()

    rec: str = "increase" if opt["price_multiplier"] > 1.05 else "decrease" if opt["price_multiplier"] < 0.95 else "hold"
    return {
        "product_id": req.product_id,
        "predicted_demand": round(demand, 2),
        "optimized_price": opt["optimized_price"],
        "expected_profit": opt["expected_profit"],
        "price_multiplier": opt["price_multiplier"],
        "recommendation": rec,
    }


@app.post("/batch-forecast", tags=["prediction"], summary="Batch demand prediction")
def batch_forecast(req: BatchForecastRequest, db: Session = Depends(get_db)) -> dict:
    """Run /forecast on multiple products in a single request.

    Args:
        req: BatchForecastRequest with up to 100 ForecastRequest items.
        db: SQLAlchemy session.

    Returns:
        Dict with results list and count.
    """
    if _model is None:
        raise HTTPException(503, "Model not ready")

    results = []
    for item in req.items:
        data = item.model_dump()
        features: np.ndarray = engineer_features(data)
        demand: float = float(predict(_model, features.reshape(1, -1))[0])
        demand = max(demand, 0.0)
        cost: float = float(item.cost or item.base_price * 0.6)
        opt: dict = optimize_price(_model, features, cost)
        _recent_X.append(features)
        results.append({
            "product_id": item.product_id,
            "predicted_demand": round(demand, 2),
            "optimized_price": opt["optimized_price"],
            "expected_profit": opt["expected_profit"],
            "recommendation": "increase" if opt["price_multiplier"] > 1.05 else "decrease" if opt["price_multiplier"] < 0.95 else "hold",
        })

    return {"results": results, "count": len(results)}


@app.post("/train", tags=["model"], summary="Retrain the ML model")
def train(req: TrainRequest, db: Session = Depends(get_db)) -> dict:
    """Retrain the ensemble model on fresh synthetic data.

    Args:
        req: TrainRequest with n_samples (100–50000).
        db: SQLAlchemy session.

    Returns:
        Dict with status, run_id, rmse_mean, rmse_std, n_samples, estimators.
    """
    global _model, _reference_X
    X, y = generate_synthetic_training_data(n_samples=req.n_samples)
    metrics = train_model(X, y)
    _model = load_model()
    _reference_X = X
    save_reference_stats(X)

    metrics_row = ModelMetrics(
        run_id=metrics["run_id"],
        auc_mean=0.0,
        auc_std=0.0,
        rmse=metrics["rmse_mean"],
        n_features=metrics["n_features"],
        n_samples=metrics["n_samples"],
    )
    db.add(metrics_row)
    db.commit()

    return {
        "status": "trained",
        "run_id": metrics["run_id"],
        "rmse_mean": round(metrics["rmse_mean"], 4),
        "rmse_std": round(metrics["rmse_std"], 4),
        "n_samples": metrics["n_samples"],
        "estimators": metrics["estimators"],
    }


@app.get("/drift", tags=["monitoring"], summary="Feature drift analysis")
def drift(db: Session = Depends(get_db)) -> dict:
    """Compute per-feature KS-test drift between reference and recent predictions.

    Args:
        db: SQLAlchemy session.

    Returns:
        Dict with drift_detected, drift_rate, drifted_features list, and sample counts.
        Returns a message dict if there are fewer than 10 recent predictions.
    """
    if _reference_X is None:
        raise HTTPException(503, "Reference data not available")
    if len(_recent_X) < 10:
        return {"message": "Not enough recent predictions for drift analysis", "n_recent": len(_recent_X)}

    current_matrix: np.ndarray = np.vstack(_recent_X)
    result: dict = compute_feature_drift(_reference_X, current_matrix)

    for fname, fdata in result["feature_results"].items():
        if "error" not in fdata:
            db.add(DriftLog(
                feature_name=fname,
                ks_statistic=fdata["ks_statistic"],
                p_value=fdata["p_value"],
                drift_detected=int(fdata["drift_detected"]),
            ))
    db.commit()

    return {
        "drift_detected": result["drift_detected"],
        "drift_rate": result["drift_rate"],
        "drifted_features": result["drifted_features"],
        "n_reference_samples": _reference_X.shape[0],
        "n_current_samples": current_matrix.shape[0],
    }


@app.post("/similar", tags=["prediction"], summary="Find similar products by feature vector")
def similar_products(req: SimilarRequest) -> dict:
    """Return products most similar to the query using FAISS cosine similarity.

    Args:
        req: SimilarRequest with base_price, category, and k.

    Returns:
        Dict with query parameters and similar_products list.
    """
    index = get_index()
    data = {
        "base_price": req.base_price,
        "competitor_price": req.competitor_price or req.base_price * 1.05,
        "category": req.category,
    }
    features: np.ndarray = engineer_features(data)
    results: list[dict] = index.search(features, k=req.k)
    return {"query": data, "similar_products": results}


@app.get("/metrics", tags=["monitoring"], summary="Model metrics and prediction health")
def metrics() -> dict:
    """Return model training metrics and prediction health summary.

    Returns:
        Dict with model_metrics, prediction_health, and n_recent_predictions.
    """
    m: dict = load_metrics()
    if _recent_X and _model is not None:
        recent_demands: list[float] = [float(predict(_model, x.reshape(1, -1))[0]) for x in list(_recent_X)[-100:]]
        health: dict = prediction_health_check(recent_demands)
    else:
        health = {"status": "no_data"}
    return {"model_metrics": m, "prediction_health": health, "n_recent_predictions": len(_recent_X)}


@app.get("/predictions", tags=["history"], summary="Recent prediction history")
def recent_predictions(
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Retrieve recent demand forecasts from the database.

    Args:
        limit: Number of records to return (1–200, default 20).
        db: SQLAlchemy session.

    Returns:
        List of prediction dicts ordered by creation time descending.
    """
    rows = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "product_id": r.product_id,
            "category": r.category,
            "predicted_demand": r.predicted_demand,
            "optimized_price": r.optimized_price,
            "created_at": str(r.created_at),
        }
        for r in rows
    ]


@app.get("/categories", tags=["reference"], summary="List supported product categories")
def list_categories() -> dict:
    """Return all supported product categories and their count.

    Returns:
        Dict with sorted categories list and count.
    """
    return {
        "categories": sorted(CATEGORY_MAP.keys()),
        "count": len(CATEGORY_MAP),
    }


@app.get("/summary", tags=["monitoring"], summary="Aggregated statistics")
def summary(db: Session = Depends(get_db)) -> dict:
    """Return aggregated database statistics across all tables.

    Args:
        db: SQLAlchemy session.

    Returns:
        Dict with total_predictions, avg_predicted_demand, avg_optimized_price,
        total_training_runs, latest_rmse, drift_events_logged, n_recent_in_memory.
    """
    total_predictions: int = db.query(func.count(Prediction.id)).scalar() or 0
    avg_demand = db.query(func.avg(Prediction.predicted_demand)).scalar()
    avg_price = db.query(func.avg(Prediction.optimized_price)).scalar()
    total_trains: int = db.query(func.count(ModelMetrics.id)).scalar() or 0
    latest_rmse = (
        db.query(ModelMetrics.rmse)
        .order_by(ModelMetrics.trained_at.desc())
        .scalar()
    )
    drift_events: int = db.query(func.count(DriftLog.id)).filter(DriftLog.drift_detected == 1).scalar() or 0

    return {
        "total_predictions": total_predictions,
        "avg_predicted_demand": round(float(avg_demand), 3) if avg_demand else None,
        "avg_optimized_price": round(float(avg_price), 2) if avg_price else None,
        "total_training_runs": total_trains,
        "latest_rmse": round(float(latest_rmse), 4) if latest_rmse else None,
        "drift_events_logged": drift_events,
        "n_recent_in_memory": len(_recent_X),
    }

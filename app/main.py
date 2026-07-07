"""Price-Prophet FastAPI application."""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func

from app import __version__
from app.database import Prediction, SessionLocal, init_db
from app.exceptions import ModelNotTrainedError, PriceProphetError
from app.features import generate_synthetic_training_data
from app.model import get_feature_importance, load_metrics, optimize_price, predict, train_model
from app.monitoring import check_alerts, compute_drift, prediction_health
from app.retrieval import get_index
from app.validators import validate_request

logger = logging.getLogger(__name__)

_start_time: float = time.perf_counter()

app = FastAPI(
    title="Price-Prophet",
    description="ML-powered dynamic pricing engine with ensemble forecasting and drift monitoring.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

_cors_origins_env = os.getenv("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.perf_counter() - t0) * 1000:.2f}ms"
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(PriceProphetError)
async def price_prophet_error_handler(request: Request, exc: PriceProphetError):
    status = 400 if exc.code == "VALIDATION_ERROR" else 503 if exc.code == "MODEL_NOT_TRAINED" else 500
    return JSONResponse(status_code=status, content=exc.to_dict())


init_db()


class ForecastRequest(BaseModel):
    category: str = Field(..., examples=["Electronics"])
    stock_level: int = Field(50, ge=0, le=100_000)
    competitor_price: float = Field(299.99, gt=0)
    demand_trend: float = Field(1.0, gt=0, le=10)
    margin_ratio: float = Field(0.3, ge=0, le=1)


class BatchForecastRequest(BaseModel):
    items: list[ForecastRequest] = Field(..., min_length=1, max_length=100)


@app.get("/status", tags=["meta"], summary="Service status with uptime")
def status() -> dict[str, Any]:
    """Return service status and uptime in seconds."""
    uptime_seconds = round(time.perf_counter() - _start_time, 2)
    return {
        "status": "ok",
        "version": __version__,
        "uptime_seconds": uptime_seconds,
    }


@app.get("/version", tags=["meta"], summary="API version")
def version() -> dict[str, str]:
    """Return the current API version."""
    return {"version": __version__}


@app.get("/health", tags=["meta"], summary="Health check")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.get("/health/detailed", tags=["meta"], summary="Detailed health check")
def health_detailed() -> dict[str, Any]:
    """Return service health with model and database status."""
    import os
    model_ok = os.path.exists(os.getenv("MODEL_PATH", "models/price_model.pkl"))
    db_ok = False
    try:
        with SessionLocal() as db:
            db.execute(__import__("sqlalchemy").text("SELECT 1"))
            db_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if (model_ok and db_ok) else "degraded",
        "model_trained": model_ok,
        "database": "ok" if db_ok else "error",
    }


@app.get("/categories", tags=["meta"], summary="List valid product categories")
def categories() -> dict[str, Any]:
    """Return the list of valid product categories accepted by /forecast."""
    from app.features import CATEGORY_MAP
    return {"categories": sorted(CATEGORY_MAP.keys()), "count": len(CATEGORY_MAP)}


@app.post("/train", tags=["model"], summary="Train the pricing model")
def train() -> dict[str, Any]:
    """Generate synthetic data, train the ensemble, and persist to disk."""
    try:
        df = generate_synthetic_training_data(5000)
        metrics = train_model(df)
        logger.info("Model trained successfully: %s", metrics)
        return {"status": "trained", "metrics": metrics}
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/forecast", tags=["predictions"], summary="Forecast optimal price")
def forecast(req: ForecastRequest) -> dict[str, Any]:
    """Predict the optimal price and return similar products."""
    try:
        features = validate_request(req.model_dump())
        price = predict(features)
        ci_low = round(price * 0.94, 2)
        ci_high = round(price * 1.06, 2)
        similar = get_index().search(req.category, k=3)
        with SessionLocal() as db:
            p = Prediction(
                category=req.category,
                input_price=req.competitor_price,
                predicted_price=price,
                confidence_low=ci_low,
                confidence_high=ci_high,
                stock_level=req.stock_level,
                demand_trend=req.demand_trend,
            )
            db.add(p)
            db.commit()
        return {
            "recommended_price": round(price, 2),
            "confidence_interval": [ci_low, ci_high],
            "similar_products": similar,
        }
    except FileNotFoundError as exc:
        raise ModelNotTrainedError() from exc
    except PriceProphetError:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/batch-forecast", tags=["predictions"], summary="Batch price forecasts")
def batch_forecast(req: BatchForecastRequest) -> dict[str, Any]:
    """Run price forecasts for a batch of up to 100 items."""
    results = []
    for item in req.items:
        try:
            features = item.model_dump()
            price = predict(features)
            results.append({
                "category": item.category,
                "recommended_price": round(price, 2),
                "confidence_interval": [round(price * 0.94, 2), round(price * 1.06, 2)],
            })
        except FileNotFoundError as exc:
            raise ModelNotTrainedError() from exc
    return {"results": results, "count": len(results)}


@app.get("/metrics", tags=["model"], summary="Model performance metrics")
def metrics() -> dict[str, Any]:
    """Return current model evaluation metrics and cache statistics."""
    from app.cache import get_cache
    m = load_metrics()
    cache_stats = get_cache().stats()
    if not m:
        return {"status": "no_metrics", "message": "Train the model first.", "cache": cache_stats}
    return {**m, "cache": cache_stats}


@app.get("/feature-importance", tags=["model"], summary="Feature importance scores")
def feature_importance() -> dict[str, Any]:
    """Return feature importance scores from the Random Forest sub-estimator."""
    try:
        imp = get_feature_importance()
        if not imp:
            return {"status": "unavailable", "message": "Train the model first."}
        sorted_imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
        return {"feature_importance": sorted_imp}
    except FileNotFoundError as exc:
        raise ModelNotTrainedError() from exc


@app.get("/drift", tags=["monitoring"], summary="Statistical drift report")
def drift() -> dict[str, Any]:
    """Return KS-test and PSI drift analysis for recent predictions."""
    with SessionLocal() as db:
        rows = db.query(Prediction.predicted_price).order_by(
            Prediction.created_at.asc()
        ).all()
    prices = [r[0] for r in rows if r[0] is not None]
    if len(prices) < 20:
        return {"status": "insufficient_data", "n_predictions": len(prices), "required": 20}
    mid = len(prices) // 2
    result = compute_drift(prices[:mid], prices[mid:])
    alerts = check_alerts(result)
    health = prediction_health(prices[mid:])
    return {"drift": result, "alerts": alerts, "health": health, "n_total": len(prices)}


@app.get("/similar", tags=["retrieval"], summary="Find similar products")
def similar(category: str = "Electronics", k: int = 5) -> dict[str, Any]:
    """Return up to k products similar to the given category."""
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")
    results = get_index().search(category, k=k)
    return {"category": category, "k": k, "similar_products": results}


@app.get("/summary", tags=["monitoring"], summary="Prediction summary statistics")
def summary() -> dict[str, Any]:
    """Return aggregate statistics over all stored predictions."""
    with SessionLocal() as db:
        total = db.query(func.count(Prediction.id)).scalar() or 0
        avg_price = db.query(func.avg(Prediction.predicted_price)).scalar()
        by_cat = db.query(
            Prediction.category, func.count(Prediction.id)
        ).group_by(Prediction.category).all()
    return {
        "total_predictions": total,
        "average_predicted_price": round(float(avg_price), 2) if avg_price else None,
        "by_category": {cat: cnt for cat, cnt in by_cat},
    }


@app.get("/predictions", tags=["monitoring"], summary="List recent predictions")
def list_predictions(limit: int = 20, offset: int = 0) -> dict[str, Any]:
    """Return a paginated list of recent price predictions."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    with SessionLocal() as db:
        total = db.query(func.count(Prediction.id)).scalar() or 0
        rows = (
            db.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    items = [
        {
            "id": r.id,
            "category": r.category,
            "input_price": r.input_price,
            "predicted_price": r.predicted_price,
            "confidence_interval": [r.confidence_low, r.confidence_high],
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
    has_next = offset + limit < total
    has_prev = offset > 0
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_next": has_next,
        "has_prev": has_prev,
        "next_offset": offset + limit if has_next else None,
        "prev_offset": max(0, offset - limit) if has_prev else None,
        "items": items,
    }


class OptimizeRequest(BaseModel):
    category: str = Field(..., examples=["Electronics"])
    stock_level: int = Field(50, ge=0, le=100_000)
    demand_trend: float = Field(1.0, gt=0, le=10)
    margin_ratio: float = Field(0.3, ge=0, le=1)
    price_min: float = Field(1.0, gt=0, description="Lower bound of price search range")
    price_max: float = Field(1000.0, gt=0, description="Upper bound of price search range")
    n_steps: int = Field(20, ge=5, le=100, description="Number of candidate prices to evaluate")


@app.post("/optimize-price", tags=["predictions"], summary="Find optimal price in a range")
def optimize_price_endpoint(req: OptimizeRequest) -> dict[str, Any]:
    """Find the price in [price_min, price_max] that maximises predicted value."""
    if req.price_min >= req.price_max:
        raise HTTPException(status_code=400, detail="price_min must be less than price_max")
    try:
        features = req.model_dump(exclude={"price_min", "price_max", "n_steps"})
        result = optimize_price(features, req.price_min, req.price_max, req.n_steps)
        return result
    except FileNotFoundError as exc:
        raise ModelNotTrainedError() from exc
    except PriceProphetError:
        raise
    except Exception as exc:
        logger.exception("Optimize price failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/model/info", tags=["model"], summary="Model metadata and configuration")
def model_info() -> dict[str, Any]:
    """Return model configuration, paths, and training status."""
    import os
    model_path = os.getenv("MODEL_PATH", "models/price_model.pkl")
    metrics_path = os.getenv("METRICS_PATH", "models/metrics.json")
    model_exists = os.path.exists(model_path)
    metrics = load_metrics() if model_exists else {}
    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "trained": model_exists,
        "ensemble": "XGBoost + LightGBM + RandomForest (VotingRegressor)",
        "metrics": metrics,
    }


@app.get("/predictions/count", tags=["monitoring"], summary="Total prediction count")
def predictions_count() -> dict[str, int]:
    """Return total number of stored price predictions."""
    with SessionLocal() as db:
        total = db.query(func.count(Prediction.id)).scalar() or 0
    return {"count": total}


@app.get("/cache-stats", tags=["meta"], summary="Cache statistics")
def cache_stats() -> dict[str, Any]:
    """Return current hit/miss statistics for the prediction cache."""
    from app.cache import get_cache
    return get_cache().stats()


@app.post("/cache-clear", tags=["meta"], summary="Clear prediction cache")
def cache_clear() -> dict[str, str]:
    """Evict all entries from the in-memory prediction cache."""
    from app.cache import get_cache
    get_cache().clear()
    return {"status": "cleared"}

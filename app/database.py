"""SQLAlchemy ORM models and session management for Price-Prophet."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Index, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./price_prophet.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal: sessionmaker[Session] = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    """Stores each demand forecast and optimized price recommendation."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, index=True)
    category = Column(String, index=True)
    predicted_demand = Column(Float)
    optimized_price = Column(Float)
    confidence = Column(Float)
    features_used = Column(JSON)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_predictions_category_created", "category", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, product_id={self.product_id!r}, "
            f"demand={self.predicted_demand}, price={self.optimized_price})>"
        )


class ModelMetrics(Base):
    """Records RMSE and metadata for each model training run."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True)
    auc_mean = Column(Float)
    auc_std = Column(Float)
    rmse = Column(Float)
    n_features = Column(Integer)
    n_samples = Column(Integer)
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<ModelMetrics(run_id={self.run_id!r}, rmse={self.rmse})>"


class DriftLog(Base):
    """Records per-feature KS drift detection results."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String, index=True)
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer)  # 0 or 1
    logged_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def __repr__(self) -> str:
        return (
            f"<DriftLog(feature={self.feature_name!r}, "
            f"ks={self.ks_statistic}, drift={bool(self.drift_detected)})>"
        )


def get_db():
    """FastAPI dependency that yields a database session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined in the ORM metadata."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised: %s", DATABASE_URL)

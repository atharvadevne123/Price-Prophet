"""SQLAlchemy ORM models and database session management."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./prices.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Prediction(Base):
    """Persisted price prediction record.

    Attributes:
        id: Primary key.
        category: Product category string.
        input_price: Competitor price at prediction time.
        predicted_price: Model output price.
        confidence_low: Lower bound of the 94% confidence interval.
        confidence_high: Upper bound of the 94% confidence interval.
        stock_level: Stock level feature at prediction time.
        demand_trend: Demand trend feature at prediction time.
        created_at: UTC timestamp of prediction creation.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(64), nullable=False, index=True)
    input_price = Column(Float, nullable=True)
    predicted_price = Column(Float, nullable=False)
    confidence_low = Column(Float, nullable=True)
    confidence_high = Column(Float, nullable=True)
    stock_level = Column(Integer, nullable=True)
    demand_trend = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_predictions_category_created", "category", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Prediction id={self.id} category={self.category!r} price={self.predicted_price}>"


class DriftReport(Base):
    """Statistical drift report stored after each drift analysis run.

    Attributes:
        id: Primary key.
        ks_statistic: KS test statistic.
        p_value: KS test p-value.
        psi: Population Stability Index.
        is_drifted: True if drift was detected.
        n_samples: Number of samples analysed.
        created_at: UTC timestamp.
    """

    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    psi = Column(Float, nullable=True)
    is_drifted = Column(Integer, nullable=False)
    n_samples = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<DriftReport id={self.id} ks={self.ks_statistic:.3f} "
            f"psi={self.psi} drifted={bool(self.is_drifted)}>"
        )


class TrainingRun(Base):
    """Record of a completed model training run.

    Attributes:
        id: Primary key.
        n_samples: Training set size.
        mae: Mean absolute error on training data.
        rmse: Root mean squared error on training data.
        r2: R-squared coefficient.
        cv_mae_mean: Cross-validation MAE mean (optional).
        created_at: UTC timestamp.
    """

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    n_samples = Column(Integer, nullable=False)
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    r2 = Column(Float, nullable=False)
    cv_mae_mean = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<TrainingRun id={self.id} mae={self.mae:.4f} r2={self.r2:.4f}>"


def init_db() -> None:
    """Create all tables if they do not already exist.

    Called once at application startup.
    """
    url = os.getenv("DATABASE_URL", DATABASE_URL)
    if url != DATABASE_URL:
        eng = create_engine(url, connect_args={"check_same_thread": False} if "sqlite" in url else {})
    else:
        eng = engine
    logger.info("Initialising database at %s", url)
    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy Session for use in FastAPI dependency injection.

    Yields:
        An open :class:`sqlalchemy.orm.Session` that is closed on exit.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

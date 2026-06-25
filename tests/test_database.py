"""Tests for SQLAlchemy ORM models and database utilities."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture
def engine():
    """In-memory SQLite engine for test isolation."""
    from app.database import Base
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def test_prediction_model_creates(session):
    from app.database import Prediction
    p = Prediction(
        category="Electronics",
        input_price=299.99,
        predicted_price=287.0,
        confidence_low=270.0,
        confidence_high=305.0,
        stock_level=50,
        demand_trend=1.2,
    )
    session.add(p)
    session.commit()
    assert p.id is not None


def test_prediction_repr(session):
    from app.database import Prediction
    p = Prediction(
        category="Books",
        input_price=15.0,
        predicted_price=14.5,
        confidence_low=12.0,
        confidence_high=17.0,
    )
    session.add(p)
    session.commit()
    assert "Books" in repr(p)


def test_drift_report_model_creates(session):
    from app.database import DriftReport
    d = DriftReport(ks_statistic=0.12, p_value=0.07, is_drifted=False, n_samples=100)
    session.add(d)
    session.commit()
    assert d.id is not None


def test_drift_report_repr(session):
    from app.database import DriftReport
    d = DriftReport(ks_statistic=0.25, p_value=0.03, is_drifted=True, n_samples=50)
    session.add(d)
    session.commit()
    assert "True" in repr(d) or "drifted" in repr(d).lower()


def test_training_run_model_creates(session):
    from app.database import TrainingRun
    t = TrainingRun(n_samples=500, mae=12.3, rmse=18.5, r2=0.91)
    session.add(t)
    session.commit()
    assert t.id is not None


def test_multiple_predictions_query(session):
    from app.database import Prediction
    for i in range(5):
        p = Prediction(
            category="Clothing",
            input_price=float(i * 10 + 20),
            predicted_price=float(i * 10 + 18),
        )
        session.add(p)
    session.commit()
    results = session.query(Prediction).filter_by(category="Clothing").all()
    assert len(results) == 5


def test_init_db_runs(tmp_path):
    import os
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    from app.database import init_db
    init_db()
    assert (tmp_path / "test.db").exists()

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


def test_drift_report_model_insert(session):
    from app.database import DriftReport

    report = DriftReport(
        ks_statistic=0.15,
        p_value=0.03,
        psi=0.12,
        is_drifted=True,
        n_samples=100,
    )
    session.add(report)
    session.commit()
    retrieved = session.query(DriftReport).filter_by(is_drifted=True).first()
    assert retrieved is not None
    assert retrieved.ks_statistic == pytest.approx(0.15)


def test_drift_report_not_drifted(session):
    from app.database import DriftReport

    report = DriftReport(ks_statistic=0.01, p_value=0.8, psi=0.02, is_drifted=False, n_samples=50)
    session.add(report)
    session.commit()
    assert session.query(DriftReport).filter_by(is_drifted=False).count() >= 1


def test_get_db_yields_session():
    from app.database import get_db

    gen = get_db()
    db = next(gen)
    assert db is not None
    try:
        next(gen)
    except StopIteration:
        pass


def test_count_predictions_empty(session):
    from app.database import count_predictions

    assert count_predictions(session) == 0


def test_count_predictions_after_insert(session):
    from app.database import Prediction, count_predictions

    for i in range(3):
        session.add(
            Prediction(category="Sports", input_price=float(i * 10), predicted_price=float(i * 9))
        )
    session.commit()
    assert count_predictions(session) == 3


def test_get_predictions_by_category_empty(session):
    from app.database import get_predictions_by_category

    results = get_predictions_by_category(session, "Garden")
    assert results == []


def test_get_predictions_by_category_filters(session):
    from app.database import Prediction, get_predictions_by_category

    session.add(Prediction(category="Home", input_price=50.0, predicted_price=48.0))
    session.add(Prediction(category="Beauty", input_price=30.0, predicted_price=28.0))
    session.commit()
    home = get_predictions_by_category(session, "Home")
    assert len(home) == 1
    assert home[0].category == "Home"


@pytest.mark.parametrize("category", ["Electronics", "Toys", "Automotive"])
def test_get_predictions_by_category_parametrize(session, category):
    from app.database import Prediction, get_predictions_by_category

    session.add(Prediction(category=category, input_price=100.0, predicted_price=95.0))
    session.commit()
    results = get_predictions_by_category(session, category)
    assert len(results) >= 1

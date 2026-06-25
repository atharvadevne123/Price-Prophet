"""Shared pytest fixtures for Price-Prophet tests."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

TEST_DB_URL = "sqlite:///./test_price_prophet.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Create all tables once per test session and drop them afterward."""
    from app.database import Base
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_price_prophet.db"):
        os.remove("./test_price_prophet.db")


@pytest.fixture
def db(setup_db):
    """Yield a per-test database session that auto-closes on teardown."""
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def client(setup_db):
    """Return a TestClient for the FastAPI app with the test DB wired in."""
    os.environ["DATABASE_URL"] = TEST_DB_URL
    from app.database import get_db
    from app.main import app

    def override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def sample_forecast_payload() -> dict:
    """Return a valid /forecast request payload."""
    return {
        "product_id": "FIXTURE-001",
        "base_price": 99.99,
        "competitor_price": 109.99,
        "category": "electronics",
        "stock_level": 200,
        "historical_demand_7d": 75,
        "historical_demand_30d": 300,
    }


@pytest.fixture
def sample_batch_payload() -> dict:
    """Return a valid /batch-forecast request payload with 3 items."""
    return {
        "items": [
            {"product_id": f"BATCH-{i:03d}", "base_price": float(50 + i * 10), "category": "electronics"}
            for i in range(3)
        ]
    }


@pytest.fixture
def trained_client(client) -> TestClient:
    """Return the test client after triggering a quick model training run."""
    client.post("/train", json={"n_samples": 200})
    return client

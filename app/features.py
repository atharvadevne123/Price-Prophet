"""Feature engineering pipeline for price optimization model."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "base_price",
    "competitor_price",
    "price_ratio",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday_season",
    "category_encoded",
    "stock_level",
    "days_since_last_promotion",
    "historical_demand_7d",
    "historical_demand_30d",
    "demand_trend",
    "price_elasticity_estimate",
    "margin_ratio",
]

CATEGORY_MAP: dict[str, int] = {
    "electronics": 0,
    "clothing": 1,
    "food": 2,
    "home": 3,
    "sports": 4,
    "beauty": 5,
    "toys": 6,
    "books": 7,
    "other": 8,
}

HOLIDAY_MONTHS: frozenset[int] = frozenset({11, 12, 1})


def engineer_features(data: dict[str, Any]) -> np.ndarray:
    """Transform a raw product pricing dict into a fixed-length feature vector.

    Args:
        data: Dict with keys base_price, competitor_price, category, stock_level,
              historical_demand_7d, historical_demand_30d, days_since_last_promotion,
              cost, date (optional).

    Returns:
        Float32 numpy array of shape (n_features,) = (15,).
    """
    base_price: float = float(data.get("base_price") or 100.0)
    competitor_price: float = float(data.get("competitor_price") or base_price * 1.05)
    price_ratio: float = base_price / max(competitor_price, 0.01)

    date_str: str = data.get("date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    dt: pd.Timestamp = pd.Timestamp(date_str)
    day_of_week: int = dt.dayofweek
    month: int = dt.month
    is_weekend: int = int(day_of_week >= 5)
    is_holiday_season: int = int(month in HOLIDAY_MONTHS)

    category: str = data.get("category", "other").lower()
    category_encoded: int = CATEGORY_MAP.get(category, 8)

    stock_level: float = float(data.get("stock_level") or 100)
    days_since_promo: float = float(data.get("days_since_last_promotion") or 30)
    hist_demand_7d: float = float(data.get("historical_demand_7d") or 50)
    hist_demand_30d: float = float(data.get("historical_demand_30d") or 200)

    demand_trend: float = (hist_demand_7d * 4 - hist_demand_30d) / max(hist_demand_30d, 1)

    cost: float = float(data.get("cost") or base_price * 0.6)
    margin_ratio: float = (base_price - cost) / max(base_price, 0.01)

    price_elasticity_estimate: float = -1.5 * (1 + 0.5 * (price_ratio - 1))

    return np.array([
        base_price,
        competitor_price,
        price_ratio,
        day_of_week,
        month,
        is_weekend,
        is_holiday_season,
        category_encoded,
        stock_level,
        days_since_promo,
        hist_demand_7d,
        hist_demand_30d,
        demand_trend,
        price_elasticity_estimate,
        margin_ratio,
    ], dtype=np.float32)


def engineer_batch_features(records: list[dict[str, Any]]) -> np.ndarray:
    """Convert a list of product dicts to a 2-D feature matrix.

    Args:
        records: List of product pricing dicts.

    Returns:
        Float32 numpy array of shape (n_records, n_features).
    """
    return np.vstack([engineer_features(r) for r in records])


def generate_synthetic_training_data(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic product pricing data for model training and testing.

    Args:
        n_samples: Number of synthetic records to generate.

    Returns:
        Tuple of (X, y) where X has shape (n_samples, 15) and y has shape (n_samples,).
    """
    np.random.seed(42)
    records: list[dict[str, Any]] = []
    labels: list[float] = []

    categories: list[str] = list(CATEGORY_MAP.keys())

    for _ in range(n_samples):
        base_price: float = float(np.random.uniform(10, 500))
        competitor_price: float = base_price * float(np.random.uniform(0.8, 1.3))
        cost: float = base_price * float(np.random.uniform(0.4, 0.7))
        stock: int = int(np.random.randint(0, 500))
        hist_7d: float = float(np.random.uniform(10, 200))
        hist_30d: float = hist_7d * float(np.random.uniform(3.5, 5.0))
        days_promo: int = int(np.random.randint(0, 90))
        category: str = str(np.random.choice(categories))

        dt: pd.Timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(np.random.randint(0, 365)))

        record: dict[str, Any] = {
            "base_price": base_price,
            "competitor_price": competitor_price,
            "cost": cost,
            "stock_level": stock,
            "historical_demand_7d": hist_7d,
            "historical_demand_30d": hist_30d,
            "days_since_last_promotion": days_promo,
            "category": category,
            "date": dt.strftime("%Y-%m-%d"),
        }
        records.append(record)

        price_ratio: float = base_price / max(competitor_price, 0.01)
        demand: float = (
            hist_7d * 0.4
            + (1 / price_ratio) * 20
            + (1 if dt.month in HOLIDAY_MONTHS else 0) * 15
            + float(np.random.normal(0, 5))
        )
        labels.append(max(demand, 0.0))

    X: np.ndarray = engineer_batch_features(records)
    y: np.ndarray = np.array(labels, dtype=np.float32)
    return X, y

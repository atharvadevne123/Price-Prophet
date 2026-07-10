"""Feature engineering for Price-Prophet ML pipeline."""

from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd

__all__ = [
    "CATEGORY_MAP",
    "HOLIDAY_MONTHS",
    "engineer_features",
    "generate_synthetic_training_data",
]

CATEGORY_MAP: dict[str, int] = {
    "Electronics": 0,
    "Clothing": 1,
    "Food": 2,
    "Books": 3,
    "Toys": 4,
    "Sports": 5,
    "Home": 6,
    "Beauty": 7,
    "Automotive": 8,
    "Garden": 9,
}

HOLIDAY_MONTHS: frozenset[int] = frozenset({11, 12, 1})


def engineer_features(data: dict[str, Any]) -> pd.DataFrame:
    """Transform a raw request dict into a model-ready feature DataFrame.

    Args:
        data: Raw request fields (category, stock_level, competitor_price, etc.).

    Returns:
        Single-row DataFrame with all engineered features.
    """
    category: str = str(data.get("category", "Electronics"))
    stock_level: float = float(data.get("stock_level", 50))
    competitor_price: float = float(data.get("competitor_price", 100.0))
    demand_trend: float = float(data.get("demand_trend", 1.0))
    margin_ratio: float = float(data.get("margin_ratio", 0.3))

    cat_code: int = CATEGORY_MAP.get(category, 0)
    month: int = pd.Timestamp.now().month
    is_holiday: int = int(month in HOLIDAY_MONTHS)
    price_per_unit: float = competitor_price / max(stock_level, 1)
    demand_price_interaction: float = demand_trend * competitor_price
    stock_demand_ratio: float = stock_level / max(demand_trend, 0.01)
    log_price: float = math.log1p(competitor_price)
    margin_price: float = competitor_price * margin_ratio
    scarcity_score: float = max(0.0, 1.0 - stock_level / 1000.0)
    value_index: float = demand_trend * margin_ratio * (1.0 + is_holiday * 0.2)

    return pd.DataFrame(
        [
            {
                "category_code": cat_code,
                "stock_level": stock_level,
                "competitor_price": competitor_price,
                "demand_trend": demand_trend,
                "margin_ratio": margin_ratio,
                "is_holiday": is_holiday,
                "price_per_unit": price_per_unit,
                "demand_price_interaction": demand_price_interaction,
                "stock_demand_ratio": stock_demand_ratio,
                "log_price": log_price,
                "margin_price": margin_price,
                "scarcity_score": scarcity_score,
                "value_index": value_index,
            }
        ]
    )


def engineer_batch_features(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Apply engineer_features to a list of request dicts.

    Args:
        items: List of raw request dicts.

    Returns:
        Multi-row DataFrame with all engineered features.
    """
    return pd.concat([engineer_features(item) for item in items], ignore_index=True)


def generate_synthetic_training_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic training data with realistic price distributions.

    Args:
        n: Number of rows to generate.

    Returns:
        DataFrame with features and a ``price`` target column.
    """
    rows: list[dict[str, Any]] = []
    categories: list[str] = list(CATEGORY_MAP.keys())
    for _ in range(n):
        category = random.choice(categories)
        stock_level = random.randint(0, 500)
        competitor_price = round(random.uniform(5.0, 2000.0), 2)
        demand_trend = round(random.uniform(0.2, 3.0), 2)
        margin_ratio = round(random.uniform(0.05, 0.6), 2)
        month = random.randint(1, 12)
        is_holiday = int(month in HOLIDAY_MONTHS)
        noise = random.gauss(0, competitor_price * 0.05)
        price = (
            competitor_price
            * (0.8 + demand_trend * 0.15)
            * (1 + margin_ratio * 0.3)
            * (1 + is_holiday * 0.1)
            * (1 - stock_level / 5000)
            + noise
        )
        price = max(1.0, round(price, 2))
        cat_code = CATEGORY_MAP[category]
        price_per_unit = competitor_price / max(stock_level, 1)
        demand_price_interaction = demand_trend * competitor_price
        stock_demand_ratio = stock_level / max(demand_trend, 0.01)
        log_price = math.log1p(competitor_price)
        margin_price = competitor_price * margin_ratio
        scarcity_score = max(0.0, 1.0 - stock_level / 1000.0)
        value_index = demand_trend * margin_ratio * (1.0 + is_holiday * 0.2)
        rows.append(
            {
                "category_code": cat_code,
                "stock_level": float(stock_level),
                "competitor_price": competitor_price,
                "demand_trend": demand_trend,
                "margin_ratio": margin_ratio,
                "is_holiday": is_holiday,
                "price_per_unit": price_per_unit,
                "demand_price_interaction": demand_price_interaction,
                "stock_demand_ratio": stock_demand_ratio,
                "log_price": log_price,
                "margin_price": margin_price,
                "scarcity_score": scarcity_score,
                "value_index": value_index,
                "price": price,
            }
        )
    return pd.DataFrame(rows)

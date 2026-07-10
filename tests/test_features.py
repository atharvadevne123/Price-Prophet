"""Tests for feature engineering pipeline."""

from __future__ import annotations

import pytest


def test_engineer_features_returns_dataframe():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "competitor_price": 299.99})
    assert len(df) == 1


def test_engineer_features_columns():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics"})
    assert "category_code" in df.columns
    assert "competitor_price" in df.columns
    assert "demand_trend" in df.columns


def test_category_code_electronics():
    from app.features import CATEGORY_MAP, engineer_features

    df = engineer_features({"category": "Electronics"})
    assert df["category_code"].iloc[0] == CATEGORY_MAP["Electronics"]


def test_demand_trend_default():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "competitor_price": 100.0})
    assert df["demand_trend"].iloc[0] == 1.0


def test_demand_trend_negative():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "demand_trend": 0.5})
    assert df["demand_trend"].iloc[0] == 0.5


@pytest.mark.parametrize(
    "category,code",
    [
        ("Electronics", 0),
        ("Clothing", 1),
        ("Food", 2),
        ("Books", 3),
        ("Toys", 4),
        ("Sports", 5),
        ("Home", 6),
        ("Beauty", 7),
        ("Automotive", 8),
        ("Garden", 9),
    ],
)
def test_all_category_codes(category, code):
    from app.features import engineer_features

    df = engineer_features({"category": category})
    assert df["category_code"].iloc[0] == code


@pytest.mark.parametrize("month", [11, 12, 1])
def test_holiday_months_in_frozenset(month):
    from app.features import HOLIDAY_MONTHS

    assert month in HOLIDAY_MONTHS


def test_margin_ratio_computed():
    from app.features import engineer_features

    df = engineer_features(
        {"category": "Electronics", "competitor_price": 100.0, "margin_ratio": 0.4}
    )
    assert df["margin_ratio"].iloc[0] == 0.4


def test_default_competitor_price_offset():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics"})
    assert df["competitor_price"].iloc[0] == 100.0


def test_zero_stock_level():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "stock_level": 0})
    assert df["stock_level"].iloc[0] == 0.0


def test_large_price_values():
    from app.features import engineer_features

    df = engineer_features({"category": "Automotive", "competitor_price": 99999.0})
    assert df["competitor_price"].iloc[0] == 99999.0


def test_scarcity_score_zero_stock():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "stock_level": 0})
    assert df["scarcity_score"].iloc[0] == 1.0


def test_scarcity_score_high_stock():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "stock_level": 2000})
    assert df["scarcity_score"].iloc[0] == 0.0


def test_value_index_computed():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "demand_trend": 1.0, "margin_ratio": 0.3})
    assert df["value_index"].iloc[0] > 0


def test_log_price_positive():
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "competitor_price": 100.0})
    assert df["log_price"].iloc[0] > 0


def test_price_per_unit_computed():
    from app.features import engineer_features

    df = engineer_features(
        {"category": "Electronics", "competitor_price": 100.0, "stock_level": 10}
    )
    assert df["price_per_unit"].iloc[0] == pytest.approx(10.0)


def test_demand_price_interaction():
    from app.features import engineer_features

    df = engineer_features(
        {"category": "Electronics", "competitor_price": 200.0, "demand_trend": 2.0}
    )
    assert df["demand_price_interaction"].iloc[0] == pytest.approx(400.0)


def test_generate_synthetic_data_shape():
    from app.features import generate_synthetic_training_data

    df = generate_synthetic_training_data(100)
    assert len(df) == 100
    assert "price" in df.columns


def test_generate_synthetic_data_positive_prices():
    from app.features import generate_synthetic_training_data

    df = generate_synthetic_training_data(200)
    assert (df["price"] > 0).all()


def test_engineer_batch_features():
    from app.features import engineer_batch_features

    items = [
        {"category": "Electronics", "competitor_price": 299.0},
        {"category": "Books", "competitor_price": 15.0},
    ]
    df = engineer_batch_features(items)
    assert len(df) == 2


@pytest.mark.parametrize("price", [1.0, 10.0, 100.0, 1000.0, 9999.99])
def test_log_price_monotonic(price):
    """log_price increases with competitor_price."""
    import math

    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "competitor_price": price})
    assert df["log_price"].iloc[0] == pytest.approx(math.log1p(price))


@pytest.mark.parametrize("stock", [0, 1, 100, 500, 1000, 5000])
def test_scarcity_score_range(stock):
    from app.features import engineer_features

    df = engineer_features({"category": "Electronics", "stock_level": stock})
    score = df["scarcity_score"].iloc[0]
    assert 0.0 <= score <= 1.0


def test_engineer_features_unknown_category_falls_back():
    from app.features import engineer_features

    df = engineer_features({"category": "UnknownXYZ", "competitor_price": 50.0})
    assert df["category_code"].iloc[0] == 0


def test_generate_synthetic_data_all_columns_present():
    from app.features import CATEGORY_MAP, generate_synthetic_training_data

    df = generate_synthetic_training_data(50)
    expected = set(CATEGORY_MAP.values())
    assert set(df["category_code"].unique()).issubset(expected | {0})

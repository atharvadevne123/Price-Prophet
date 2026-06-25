"""Tests for app/features.py feature engineering pipeline."""
from __future__ import annotations

import pytest

from app.features import (
    CATEGORY_MAP,
    FEATURE_NAMES,
    HOLIDAY_MONTHS,
    engineer_batch_features,
    engineer_features,
    generate_synthetic_training_data,
)


def test_feature_vector_length():
    """engineer_features always returns exactly n_features elements."""
    features = engineer_features({"base_price": 100.0, "category": "electronics"})
    assert len(features) == len(FEATURE_NAMES)


def test_feature_names_count():
    """FEATURE_NAMES has exactly 15 entries."""
    assert len(FEATURE_NAMES) == 15


def test_weekend_flag():
    """is_weekend flag is 1 for Saturday and 0 for Monday."""
    feat_sat = engineer_features({"base_price": 50.0, "date": "2024-01-06"})
    feat_mon = engineer_features({"base_price": 50.0, "date": "2024-01-08"})
    assert feat_sat[5] == 1.0
    assert feat_mon[5] == 0.0


def test_holiday_season():
    """is_holiday_season flag is 1 in Nov/Dec/Jan and 0 in June."""
    feat_dec = engineer_features({"base_price": 50.0, "date": "2024-12-15"})
    feat_jun = engineer_features({"base_price": 50.0, "date": "2024-06-15"})
    assert feat_dec[6] == 1.0
    assert feat_jun[6] == 0.0


def test_price_ratio_computed():
    """price_ratio = base_price / competitor_price."""
    feat = engineer_features({"base_price": 100.0, "competitor_price": 200.0})
    assert abs(feat[2] - 0.5) < 0.01


def test_batch_features_shape():
    """engineer_batch_features returns shape (n_records, n_features)."""
    records = [
        {"base_price": 100.0, "category": "electronics"},
        {"base_price": 200.0, "category": "clothing"},
        {"base_price": 50.0, "category": "food"},
    ]
    X = engineer_batch_features(records)
    assert X.shape == (3, len(FEATURE_NAMES))


def test_synthetic_data_generation():
    """Synthetic data has correct shapes and non-negative labels."""
    X, y = generate_synthetic_training_data(n_samples=100)
    assert X.shape == (100, len(FEATURE_NAMES))
    assert y.shape == (100,)
    assert (y >= 0).all()


def test_category_encoding():
    """Known categories map to their integer codes; unknown falls back to 8."""
    feat_elec = engineer_features({"base_price": 100.0, "category": "electronics"})
    feat_unknown = engineer_features({"base_price": 100.0, "category": "unknown_cat"})
    assert feat_elec[7] == 0
    assert feat_unknown[7] == 8


def test_demand_trend_positive():
    """demand_trend > 0 when recent demand outpaces trailing average."""
    feat = engineer_features({
        "base_price": 100.0,
        "historical_demand_7d": 100,
        "historical_demand_30d": 200,
    })
    assert feat[12] > 0


def test_demand_trend_negative():
    """demand_trend < 0 when recent demand is below trailing average."""
    feat = engineer_features({
        "base_price": 100.0,
        "historical_demand_7d": 10,
        "historical_demand_30d": 200,
    })
    assert feat[12] < 0


@pytest.mark.parametrize("category,expected_code", list(CATEGORY_MAP.items()))
def test_all_category_encodings(category: str, expected_code: int):
    """Every known category maps to its expected integer code."""
    feat = engineer_features({"base_price": 50.0, "category": category})
    assert feat[7] == expected_code


@pytest.mark.parametrize("month,expected_holiday", [
    (11, 1), (12, 1), (1, 1), (2, 0), (6, 0), (10, 0)
])
def test_holiday_months(month: int, expected_holiday: int):
    """Holiday season covers Nov, Dec, Jan; other months are non-holiday."""
    date_str = f"2024-{month:02d}-15"
    feat = engineer_features({"base_price": 50.0, "date": date_str})
    assert feat[6] == expected_holiday


def test_margin_ratio_computed():
    """margin_ratio = (base_price - cost) / base_price."""
    feat = engineer_features({"base_price": 100.0, "cost": 40.0})
    assert abs(feat[14] - 0.6) < 0.001


def test_default_competitor_price_offset():
    """Competitor price defaults to base_price * 1.05 when not provided."""
    feat_with = engineer_features({"base_price": 100.0, "competitor_price": 105.0})
    feat_without = engineer_features({"base_price": 100.0})
    assert abs(feat_with[1] - feat_without[1]) < 0.01


def test_zero_stock_level():
    """stock_level of 0 is encoded correctly without errors."""
    feat = engineer_features({"base_price": 100.0, "stock_level": 0})
    assert feat[8] == 0.0


def test_large_price_values():
    """Feature engineering handles very large prices without overflow."""
    feat = engineer_features({"base_price": 99999.0, "competitor_price": 100000.0})
    assert len(feat) == len(FEATURE_NAMES)
    assert all(float(v) == float(v) for v in feat)  # no NaN

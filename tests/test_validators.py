"""Tests for input validation helpers."""
from __future__ import annotations

import pytest


def test_validate_category_valid():
    from app.validators import validate_category
    assert validate_category("Electronics") == "Electronics"


def test_validate_category_case_insensitive():
    from app.validators import validate_category
    assert validate_category("electronics") == "Electronics"


def test_validate_category_invalid():
    from app.validators import validate_category
    with pytest.raises(ValueError, match="Invalid category"):
        validate_category("Weapons")


def test_validate_price_valid():
    from app.validators import validate_price
    assert validate_price(299.99) == 299.99


def test_validate_price_zero_rejected():
    from app.validators import validate_price
    with pytest.raises(ValueError):
        validate_price(0.0)


def test_validate_price_negative_rejected():
    from app.validators import validate_price
    with pytest.raises(ValueError):
        validate_price(-10.0)


def test_validate_price_too_large_rejected():
    from app.validators import validate_price
    with pytest.raises(ValueError):
        validate_price(2_000_000.0)


def test_validate_stock_valid():
    from app.validators import validate_stock
    assert validate_stock(50) == 50


def test_validate_stock_zero_valid():
    from app.validators import validate_stock
    assert validate_stock(0) == 0


def test_validate_stock_negative_rejected():
    from app.validators import validate_stock
    with pytest.raises(ValueError):
        validate_stock(-1)


def test_validate_demand_trend_valid():
    from app.validators import validate_demand_trend
    assert validate_demand_trend(1.2) == 1.2


def test_validate_demand_trend_zero_rejected():
    from app.validators import validate_demand_trend
    with pytest.raises(ValueError):
        validate_demand_trend(0.0)


def test_validate_demand_trend_too_large():
    from app.validators import validate_demand_trend
    with pytest.raises(ValueError):
        validate_demand_trend(15.0)


def test_validate_request_full():
    from app.validators import validate_request
    data = {
        "category": "electronics",
        "competitor_price": 299.99,
        "stock_level": 50,
        "demand_trend": 1.2,
    }
    result = validate_request(data)
    assert result["category"] == "Electronics"


@pytest.mark.parametrize("cat", ["Electronics", "Clothing", "Food", "Books", "Toys",
                                   "Sports", "Home", "Beauty", "Automotive", "Garden"])
def test_all_valid_categories(cat):
    from app.validators import validate_category
    assert validate_category(cat) == cat

"""Input validation helpers for Price-Prophet API."""

from __future__ import annotations

from typing import Any

__all__ = [
    "VALID_CATEGORIES",
    "validate_category",
    "validate_price",
    "validate_stock",
    "validate_demand_trend",
    "validate_margin_ratio",
    "validate_request",
]


VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "Electronics",
        "Clothing",
        "Food",
        "Books",
        "Toys",
        "Sports",
        "Home",
        "Beauty",
        "Automotive",
        "Garden",
    }
)

_PRICE_MIN: float = 0.01
_PRICE_MAX: float = 1_000_000.0
_STOCK_MIN: int = 0
_STOCK_MAX: int = 100_000


def validate_category(category: str) -> str:
    """Validate and normalise a product category string.

    Args:
        category: Raw category value from the request.

    Returns:
        Normalised category string (title-cased).

    Raises:
        ValueError: If category is not in the allowed set.
    """
    normalised = category.strip().title()
    if normalised not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category {category!r}. Allowed values: {sorted(VALID_CATEGORIES)}"
        )
    return normalised


def validate_price(value: float, field: str = "price") -> float:
    """Validate a price value is within sensible bounds.

    Args:
        value: Price to validate.
        field: Field name for error messages.

    Returns:
        Validated price.

    Raises:
        ValueError: If price is outside valid range.
    """
    if value < _PRICE_MIN or value > _PRICE_MAX:
        raise ValueError(f"{field} must be between {_PRICE_MIN} and {_PRICE_MAX}, got {value}")
    return value


def validate_stock(value: int) -> int:
    """Validate stock level is a non-negative integer within bounds.

    Args:
        value: Stock level to validate.

    Returns:
        Validated stock level.

    Raises:
        ValueError: If stock level is outside valid range.
    """
    if value < _STOCK_MIN or value > _STOCK_MAX:
        raise ValueError(f"stock_level must be between {_STOCK_MIN} and {_STOCK_MAX}, got {value}")
    return value


def validate_demand_trend(value: float) -> float:
    """Validate demand trend multiplier.

    Args:
        value: Demand trend value (expected range 0.1 – 10.0).

    Returns:
        Validated demand trend.

    Raises:
        ValueError: If value is non-positive or unreasonably large.
    """
    if value <= 0 or value > 10.0:
        raise ValueError(f"demand_trend must be in (0, 10], got {value}")
    return value


def validate_request(data: dict[str, Any]) -> dict[str, Any]:
    """Run all validators against a forecast request dict.

    Args:
        data: Raw request dictionary.

    Returns:
        Validated and normalised dictionary.

    Raises:
        ValueError: On the first validation failure encountered.
    """
    out = dict(data)
    if "category" in out:
        out["category"] = validate_category(str(out["category"]))
    if "competitor_price" in out:
        out["competitor_price"] = validate_price(float(out["competitor_price"]), "competitor_price")
    if "stock_level" in out:
        out["stock_level"] = validate_stock(int(out["stock_level"]))
    if "demand_trend" in out:
        out["demand_trend"] = validate_demand_trend(float(out["demand_trend"]))
    if "margin_ratio" in out:
        out["margin_ratio"] = validate_margin_ratio(float(out["margin_ratio"]))
    return out


def validate_margin_ratio(value: float) -> float:
    """Validate margin ratio is in [0, 1].

    Args:
        value: Margin ratio (expected range 0.0 – 1.0).

    Returns:
        Validated margin ratio.

    Raises:
        ValueError: If value is outside [0, 1].
    """
    if value < 0.0 or value > 1.0:
        raise ValueError(f"margin_ratio must be in [0, 1], got {value}")
    return value

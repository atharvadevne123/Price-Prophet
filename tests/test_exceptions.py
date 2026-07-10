"""Tests for the custom exception hierarchy."""

from __future__ import annotations

import pytest


def test_base_exception_attrs():
    from app.exceptions import PriceProphetError

    e = PriceProphetError("something failed", code="CUSTOM")
    assert e.message == "something failed"
    assert e.code == "CUSTOM"


def test_base_exception_to_dict():
    from app.exceptions import PriceProphetError

    e = PriceProphetError("oops")
    d = e.to_dict()
    assert d["error"] == "INTERNAL_ERROR"
    assert d["detail"] == "oops"


def test_model_not_trained_error():
    from app.exceptions import ModelNotTrainedError

    e = ModelNotTrainedError()
    assert e.code == "MODEL_NOT_TRAINED"
    assert "train" in e.message.lower()


def test_validation_error():
    from app.exceptions import ValidationError

    e = ValidationError("bad input")
    assert e.code == "VALIDATION_ERROR"
    assert "bad input" in e.message


def test_feature_engineering_error():
    from app.exceptions import FeatureEngineeringError

    e = FeatureEngineeringError("NaN encountered")
    assert e.code == "FEATURE_ERROR"


def test_database_error():
    from app.exceptions import DatabaseError

    e = DatabaseError("connection refused")
    assert e.code == "DB_ERROR"


def test_retrieval_index_error():
    from app.exceptions import RetrievalIndexError

    e = RetrievalIndexError("FAISS init failed")
    assert e.code == "INDEX_ERROR"


def test_inheritance_chain():
    from app.exceptions import (
        ModelNotTrainedError,
        PriceProphetError,
        ValidationError,
    )

    assert issubclass(ModelNotTrainedError, PriceProphetError)
    assert issubclass(ValidationError, PriceProphetError)
    assert issubclass(PriceProphetError, Exception)


@pytest.mark.parametrize(
    "exc_class,code",
    [
        ("ModelNotTrainedError", "MODEL_NOT_TRAINED"),
        ("ValidationError", "VALIDATION_ERROR"),
        ("FeatureEngineeringError", "FEATURE_ERROR"),
        ("DatabaseError", "DB_ERROR"),
        ("RetrievalIndexError", "INDEX_ERROR"),
    ],
)
def test_all_exception_codes(exc_class, code):
    import app.exceptions as exc_module

    cls = getattr(exc_module, exc_class)
    if exc_class == "ModelNotTrainedError":
        e = cls()
    else:
        e = cls("test message")
    assert e.code == code


def test_to_dict_has_required_keys():
    from app.exceptions import PriceProphetError

    e = PriceProphetError("test")
    d = e.to_dict()
    assert "error" in d
    assert "detail" in d


def test_exception_is_raisable():
    from app.exceptions import ModelNotTrainedError

    with pytest.raises(ModelNotTrainedError):
        raise ModelNotTrainedError()


def test_exception_caught_as_base():
    from app.exceptions import DatabaseError, PriceProphetError

    with pytest.raises(PriceProphetError):
        raise DatabaseError("connection timeout")

"""Tests for the custom exception hierarchy."""
from __future__ import annotations


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

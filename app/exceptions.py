"""Custom exception hierarchy for Price-Prophet."""

from __future__ import annotations

__all__ = [
    "PriceProphetError",
    "ModelNotTrainedError",
    "ValidationError",
    "FeatureEngineeringError",
    "DatabaseError",
    "RetrievalIndexError",
]


class PriceProphetError(Exception):
    """Base exception for all Price-Prophet errors."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def to_dict(self) -> dict[str, str]:
        """Serialise exception to a JSON-safe dictionary."""
        return {"error": self.code, "detail": self.message}


class ModelNotTrainedError(PriceProphetError):
    """Raised when a prediction is requested before the model is trained."""

    def __init__(self) -> None:
        super().__init__(
            "Model is not trained yet. Call POST /train first.",
            code="MODEL_NOT_TRAINED",
        )


class ValidationError(PriceProphetError):
    """Raised when request validation fails."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail, code="VALIDATION_ERROR")


class FeatureEngineeringError(PriceProphetError):
    """Raised when feature engineering fails unexpectedly."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"Feature engineering failed: {detail}", code="FEATURE_ERROR")


class DatabaseError(PriceProphetError):
    """Raised when a database operation fails."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"Database error: {detail}", code="DB_ERROR")


class RetrievalIndexError(PriceProphetError):
    """Raised when the FAISS retrieval index cannot be built or queried."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"Retrieval index error: {detail}", code="INDEX_ERROR")

"""FAISS-based semantic product retrieval for Price-Prophet."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ProductIndex", "get_index", "EMBEDDING_DIM"]

EMBEDDING_DIM: int = 32
_index_cache: dict[str, "ProductIndex"] = {}


class ProductIndex:
    """FAISS cosine-similarity index over synthetic product embeddings.

    Args:
        dim: Embedding dimension (default 32).
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        import faiss

        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Number of vectors currently indexed."""
        return self._index.ntotal

    def add(self, metadata: list[dict[str, Any]], vectors: np.ndarray) -> None:
        """Add items to the index.

        Args:
            metadata: List of dicts (one per vector) with product info.
            vectors: Float32 ndarray of shape (n, dim), L2-normalised.
        """
        self._index.add(vectors)
        self._metadata.extend(metadata)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return up to k products similar to the query string.

        Args:
            query: Query string used to derive a pseudo-embedding via hashing.
            k: Maximum number of results to return.

        Returns:
            List of metadata dicts for the nearest neighbours.
        """
        if self.size == 0:
            return []
        vec = _text_to_vector(query, self.dim)
        actual_k = min(k, self.size)
        _, indices = self._index.search(vec.reshape(1, -1), actual_k)
        results: list[dict[str, Any]] = []
        for idx in indices[0]:
            if 0 <= idx < len(self._metadata):
                results.append(self._metadata[idx])
        return results

    def batch_search(self, queries: list[str], k: int = 5) -> list[list[dict[str, Any]]]:
        """Return similar products for a batch of query strings.

        Args:
            queries: List of query strings.
            k: Maximum number of results per query (1–50).

        Returns:
            List of result lists, one per input query.

        Raises:
            ValueError: If k is outside the valid range [1, 50].
        """
        if k < 1 or k > 50:
            raise ValueError(f"k must be between 1 and 50, got {k}")
        return [self.search(q, k=k) for q in queries]


def _text_to_vector(text: str, dim: int) -> np.ndarray:
    """Hash a string into a unit-norm float32 vector.

    Args:
        text: Input string.
        dim: Output vector dimension.

    Returns:
        L2-normalised float32 ndarray of shape (dim,).
    """
    seed = int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _seed_index(index: ProductIndex) -> None:
    """Populate the index with synthetic product vectors.

    Args:
        index: Empty :class:`ProductIndex` to seed.
    """
    from app.features import CATEGORY_MAP

    categories = list(CATEGORY_MAP.keys())
    price_ranges = {
        "Electronics": (50, 2000),
        "Clothing": (10, 500),
        "Food": (1, 100),
        "Books": (5, 80),
        "Toys": (5, 200),
        "Sports": (15, 800),
        "Home": (10, 1000),
        "Beauty": (5, 300),
        "Automotive": (20, 5000),
        "Garden": (5, 500),
    }
    metadata: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for cat in categories:
        lo, hi = price_ranges[cat]
        for i in range(20):
            price = round(lo + (hi - lo) * (i / 19), 2)
            metadata.append({"category": cat, "price": price, "rank": i})
            vectors.append(_text_to_vector(f"{cat}_{i}", index.dim))
    index.add(metadata, np.array(vectors, dtype=np.float32))
    logger.debug("Seeded index with %d vectors across %d categories", index.size, len(categories))


def get_index() -> ProductIndex:
    """Return or create the module-level singleton ProductIndex.

    Returns:
        Seeded :class:`ProductIndex` instance.
    """
    if "default" not in _index_cache:
        idx = ProductIndex(dim=EMBEDDING_DIM)
        _seed_index(idx)
        _index_cache["default"] = idx
    return _index_cache["default"]

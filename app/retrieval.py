"""FAISS-powered product similarity search with cosine fallback."""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index.pkl")

try:
    import faiss
    HAS_FAISS: bool = True
except ImportError:
    HAS_FAISS = False
    logger.warning("faiss not installed — falling back to cosine similarity")


class ProductIndex:
    """L2-normalised cosine-similarity product index backed by FAISS or numpy.

    Attributes:
        n_features: Dimensionality of the feature vectors.
        metadata: List of metadata dicts for each indexed product.
    """

    def __init__(self, n_features: int = 15) -> None:
        self.n_features: int = n_features
        self.index = None
        self.metadata: list[dict] = []
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        """Index a set of product vectors and their associated metadata.

        Args:
            vectors: Feature matrix of shape (n_products, n_features).
            metadata: List of dicts with product_id, category, base_price, avg_demand.
        """
        self.metadata = metadata
        normed: np.ndarray = self._normalize(vectors)
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.n_features)
            self.index.add(normed.astype(np.float32))
        else:
            self._vectors = normed
        logger.info("Product index built with %d items", len(metadata))

    def search(self, query: np.ndarray, k: int = 5) -> list[dict]:
        """Return the top-k most similar products to a query vector.

        Args:
            query: Feature vector of shape (n_features,).
            k: Number of nearest neighbours to return.

        Returns:
            List of metadata dicts enriched with a similarity score.
        """
        if not self.metadata:
            return []
        normed_q: np.ndarray = self._normalize(query.reshape(1, -1)).astype(np.float32)
        if HAS_FAISS and self.index is not None:
            scores, indices = self.index.search(normed_q, min(k, len(self.metadata)))
            results: list[dict] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    item = dict(self.metadata[idx])
                    item["similarity"] = round(float(score), 4)
                    results.append(item)
            return results
        elif self._vectors is not None:
            sims: np.ndarray = (self._vectors @ normed_q.T).flatten()
            top_k: np.ndarray = np.argsort(sims)[::-1][:k]
            results = []
            for idx in top_k:
                item = dict(self.metadata[idx])
                item["similarity"] = round(float(sims[idx]), 4)
                results.append(item)
            return results
        return []

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """L2-normalise each row of a 2-D matrix.

        Args:
            v: Input matrix of shape (n, d).

        Returns:
            Row-normalised matrix with unit L2 norm per row.
        """
        norms: np.ndarray = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return v / norms

    def save(self, path: str = FAISS_INDEX_PATH) -> None:
        """Pickle the index metadata and vectors to disk.

        Args:
            path: File path for the pickled index.
        """
        with open(path, "wb") as f:
            pickle.dump({"metadata": self.metadata, "vectors": self._vectors, "n_features": self.n_features}, f)

    def load(self, path: str = FAISS_INDEX_PATH) -> bool:
        """Load index from disk.

        Args:
            path: File path to the pickled index.

        Returns:
            True if loaded successfully, False if the file does not exist.
        """
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.metadata = data["metadata"]
        self._vectors = data.get("vectors")
        self.n_features = data["n_features"]
        return True


_product_index: ProductIndex | None = None


def get_index() -> ProductIndex:
    """Return the global singleton ProductIndex, seeding it if needed.

    Returns:
        Initialised ProductIndex ready for search.
    """
    global _product_index
    if _product_index is None:
        _product_index = ProductIndex()
        if not _product_index.load():
            _seed_index()
    return _product_index


def _seed_index() -> None:
    """Populate the product index with synthetic training data."""
    from app.features import CATEGORY_MAP, generate_synthetic_training_data
    X, y = generate_synthetic_training_data(n_samples=500)
    categories: list[str] = list(CATEGORY_MAP.keys())
    np.random.seed(99)
    meta: list[dict] = [
        {
            "product_id": f"PROD-{i:04d}",
            "category": categories[int(X[i, 7]) % len(categories)],
            "base_price": round(float(X[i, 0]), 2),
            "avg_demand": round(float(y[i]), 2),
        }
        for i in range(len(X))
    ]
    _product_index.build(X, meta)
    logger.info("Product index seeded with %d synthetic products", len(meta))

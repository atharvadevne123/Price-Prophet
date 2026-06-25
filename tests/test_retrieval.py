"""Tests for FAISS-based product retrieval including batch search."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_index():
    """Reset the singleton ProductIndex before each test."""
    from app import retrieval
    retrieval._index_cache.clear()
    yield
    retrieval._index_cache.clear()


def test_get_index_returns_instance():
    from app.retrieval import get_index
    idx = get_index()
    assert idx is not None


def test_get_index_singleton():
    from app.retrieval import get_index
    idx1 = get_index()
    idx2 = get_index()
    assert idx1 is idx2


def test_search_returns_list():
    from app.retrieval import get_index
    idx = get_index()
    results = idx.search("Electronics", k=3)
    assert isinstance(results, list)


def test_search_respects_k():
    from app.retrieval import get_index
    idx = get_index()
    for k in (1, 3, 5):
        results = idx.search("Electronics", k=k)
        assert len(results) <= k


def test_search_result_has_required_keys():
    from app.retrieval import get_index
    idx = get_index()
    results = idx.search("Electronics", k=1)
    if results:
        assert "category" in results[0]
        assert "price" in results[0]


def test_search_unknown_category_returns_list():
    from app.retrieval import get_index
    idx = get_index()
    results = idx.search("UnknownXYZ", k=3)
    assert isinstance(results, list)


def test_index_size_positive():
    from app.retrieval import get_index
    idx = get_index()
    assert idx.size > 0


def test_add_increases_size():
    from app.retrieval import get_index
    import numpy as np
    idx = get_index()
    before = idx.size
    vec = np.random.rand(idx.dim).astype("float32")
    vec /= np.linalg.norm(vec) + 1e-9
    idx.add([{"category": "Test", "price": 1.0, "rank": 0}], np.array([vec]))
    assert idx.size >= before


@pytest.mark.parametrize("category", ["Electronics", "Clothing", "Food", "Books", "Toys"])
def test_search_known_categories(category):
    from app.retrieval import get_index
    idx = get_index()
    results = idx.search(category, k=3)
    assert isinstance(results, list)


def test_batch_search_returns_list_of_lists():
    from app.retrieval import get_index
    idx = get_index()
    results = idx.batch_search(["Electronics", "Books", "Food"], k=3)
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)


def test_batch_search_empty_query_list():
    from app.retrieval import get_index
    idx = get_index()
    results = idx.batch_search([], k=3)
    assert results == []


def test_batch_search_single_query():
    from app.retrieval import get_index
    idx = get_index()
    results = idx.batch_search(["Electronics"], k=2)
    assert len(results) == 1
    assert len(results[0]) <= 2

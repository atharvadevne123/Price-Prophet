"""Tests for the TTL cache module."""
from __future__ import annotations

import time
import pytest


@pytest.fixture
def cache():
    from app.cache import TTLCache
    c = TTLCache(default_ttl=1.0)
    yield c
    c.clear()


def test_set_and_get(cache):
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_miss_returns_none(cache):
    assert cache.get("nonexistent") is None


def test_ttl_expiry(cache):
    cache.set("k", "v", ttl=0.05)
    time.sleep(0.1)
    assert cache.get("k") is None


def test_delete_existing(cache):
    cache.set("k", "v")
    assert cache.delete("k") is True
    assert cache.get("k") is None


def test_delete_nonexistent(cache):
    assert cache.delete("missing") is False


def test_clear(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert cache.size == 0


def test_hit_rate_all_hits(cache):
    cache.set("k", "v")
    cache.get("k")
    cache.get("k")
    assert cache.hit_rate == 1.0


def test_hit_rate_all_misses(cache):
    cache.get("x")
    cache.get("y")
    assert cache.hit_rate == 0.0


def test_evict_expired(cache):
    cache.set("a", 1, ttl=0.05)
    cache.set("b", 2, ttl=100.0)
    time.sleep(0.1)
    removed = cache.evict_expired()
    assert removed == 1
    assert cache.get("b") == 2


def test_size_tracking(cache):
    assert cache.size == 0
    cache.set("x", 1)
    cache.set("y", 2)
    assert cache.size == 2


def test_stats_structure(cache):
    stats = cache.stats()
    assert "size" in stats
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats


def test_get_cache_singleton():
    from app.cache import get_cache
    c1 = get_cache()
    c2 = get_cache()
    assert c1 is c2

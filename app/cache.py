"""Simple thread-safe in-memory TTL cache for Price-Prophet."""
from __future__ import annotations

import threading
import time
from typing import Any


class TTLCache:
    """Thread-safe in-memory cache with per-entry time-to-live.

    Args:
        default_ttl: Default TTL in seconds for new entries.
    """

    def __init__(self, default_ttl: float = 300.0) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Return cached value for key, or None if missing/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                self.misses += 1
                return None
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store value under key with optional TTL override.

        Args:
            key: Cache key string.
            value: Value to store (must be picklable for future persistence).
            ttl: TTL in seconds. Uses ``default_ttl`` if not provided.
        """
        expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        with self._lock:
            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        """Remove a key from the cache.

        Returns:
            True if the key existed, False otherwise.
        """
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def evict_expired(self) -> int:
        """Remove all expired entries and return the count removed."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired:
                del self._store[k]
        return len(expired)

    @property
    def size(self) -> int:
        """Number of entries currently in the cache (including possibly expired)."""
        with self._lock:
            return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0, 1]."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics as a dictionary."""
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


# Module-level shared cache instance (5-minute TTL)
_cache: TTLCache = TTLCache(default_ttl=300.0)


def get_cache() -> TTLCache:
    """Return the shared module-level cache instance."""
    return _cache

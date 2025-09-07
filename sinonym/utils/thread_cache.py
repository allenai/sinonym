"""
Unified thread-local cache utilities for Chinese name processing.

This module provides a centralized thread-local caching mechanism to replace
duplicated implementations across the codebase.
"""

import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class ThreadLocalCache:
    """Thread-safe cache using thread-local storage."""

    def __init__(self):
        self._thread_local = threading.local()

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get value from thread-local cache."""
        if not hasattr(self._thread_local, "cache"):
            self._thread_local.cache = {}
        return self._thread_local.cache.get(key, default)

    def set(self, key: str, value: T) -> None:
        """Set value in thread-local cache."""
        if not hasattr(self._thread_local, "cache"):
            self._thread_local.cache = {}
        self._thread_local.cache[key] = value

    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T:
        """Get value from cache or compute and cache it."""
        if not hasattr(self._thread_local, "cache"):
            self._thread_local.cache = {}

        cache = self._thread_local.cache
        if key not in cache:
            cache[key] = compute_fn()
        return cache[key]

    def clear(self) -> None:
        """Clear thread-local cache."""
        if hasattr(self._thread_local, "cache"):
            self._thread_local.cache.clear()

    def size(self) -> int:
        """Get size of thread-local cache."""
        if not hasattr(self._thread_local, "cache"):
            return 0
        return len(self._thread_local.cache)

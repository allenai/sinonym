"""
CJK character caching utilities for optimal performance.

Provides centralized thread-local caching for CJK character detection
to eliminate duplicate implementations across services.
"""

import re

from sinonym.utils.thread_cache import ThreadLocalCache


class CJKCharacterCache:
    """Specialized cache for CJK character detection with thread-local storage."""

    def __init__(self, cjk_pattern: re.Pattern[str]):
        self._cjk_pattern = cjk_pattern
        self._cache = ThreadLocalCache()

    def has_cjk_characters(self, token: str) -> bool:
        """Check if token contains CJK characters, using cached results."""
        # Simple direct regex search is faster than complex caching for short strings
        return bool(self._cjk_pattern.search(token))

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def size(self) -> int:
        """Get cache size."""
        return self._cache.size()

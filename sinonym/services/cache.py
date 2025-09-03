"""
Cache management service for Chinese name processing.

This module provides fast Han character to Pinyin conversion with caching
for performance optimization.
"""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import pypinyin

if TYPE_CHECKING:
    from sinonym.coretypes import ChineseNameConfig


@cache  # one entry per unique Han character
def _char_to_pinyin(ch: str) -> str:
    return pypinyin.lazy_pinyin(ch, style=pypinyin.Style.NORMAL)[0]


class PinyinCacheService:
    """
    * deterministic, thread‑safe, O(1) repeated look‑ups
    """

    def __init__(self, config: ChineseNameConfig):
        self._config = config
        self._warm_from_csv()

    # ---------- public API ----------
    def han_to_pinyin_fast(self, han_str: str) -> list[str]:
        """Return pinyin for every character, memoising on first sight."""
        return [_char_to_pinyin(c) for c in han_str]

    # (optional) diagnostics you were using elsewhere
    @property
    def cache_size(self) -> int:
        return _char_to_pinyin.cache_info().currsize

    def get_cache_info(self):
        """Get cache information for diagnostics."""
        from sinonym.coretypes import CacheInfo
        return CacheInfo(
            cache_built=True,
            cache_size=self.cache_size,
            pickle_file_exists=False,  # This cache doesn't use pickle
            pickle_file_size=None,
            pickle_file_mtime=None,
        )


    # ---------- internal ----------
    def _warm_from_csv(self) -> None:
        """Seed the LRU cache with the two CSVs in package resources if present."""
        from sinonym.resources import open_csv_reader

        for fname in ("familyname_orcid.csv", "givenname_orcid.csv"):
            try:
                key = "surname" if "familyname" in fname else "character"
                for row in open_csv_reader(fname):
                    for ch in row[key]:
                        _char_to_pinyin(ch)  # warms the cache once
            except Exception:
                # If file doesn't exist or can't be read, skip silently
                continue

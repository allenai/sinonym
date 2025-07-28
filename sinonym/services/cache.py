"""
Cache management service for Chinese name processing.

This module provides fast Han character to Pinyin conversion with caching
for performance optimization.
"""
from __future__ import annotations

import csv
from functools import cache
from pathlib import Path

import pypinyin

from sinonym.types import ChineseNameConfig


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

    @property
    def is_built(self) -> bool:
        return True

    # ---------- internal ----------
    def _warm_from_csv(self) -> None:
        """Seed the LRU cache with the two CSVs in data/ if present."""
        for fname in ("familyname_orcid.csv", "givenname_orcid.csv"):
            path: Path = Path(self._config.data_dir) / fname
            if not path.exists():
                continue

            key = "surname" if "familyname" in fname else "character"
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    for ch in row[key]:
                        _char_to_pinyin(ch)  # warms the cache once

"""Words the corpus uses as one non-Chinese name that the Chinese given-name splitter would cut into syllables.

``unsplit_given_words.csv.gz`` (built by ``scripts/build_unsplit_given_words.py`` from source-name counts) keeps
``Annie``, ``Hoang`` and ``Jihye`` whole instead of ``An-Nie``, ``Ho-Ang`` and ``Jih-Ye``. A word qualifies only when it
is common, mostly appears beside non-Chinese surnames and is almost never written split by its sources, so Chinese
given names (``Xiaoming``) are not listed.
"""
from __future__ import annotations

import csv
import gzip
import io
import unicodedata
from functools import lru_cache

from sinonym.resources import read_bytes


@lru_cache(maxsize=65536)
def _fold(token: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", token).casefold() if ch.isalpha() and ch.isascii())


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    text = gzip.decompress(read_bytes("unsplit_given_words.csv.gz")).decode("utf-8")
    return frozenset(row["word"] for row in csv.DictReader(io.StringIO(text)))


def load_unsplit_given_words() -> None:
    _words()


def is_unsplit_given_word(token: str) -> bool:
    return _fold(token) in _words()

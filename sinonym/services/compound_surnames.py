"""Corpus-backed compound surnames (see data/COMPOUND_SURNAMES.md).

``compound_surnames.csv.gz`` lists multi-word Latin surname surfaces that the corpus writes as one surname and
that distinct ORCID people carry both whole and split. The lookup folds a surface the way the asset does
(lower-case, accents stripped, single spaces) and matches the longest listed suffix of a source's
``middle + last`` words.
"""
from __future__ import annotations

import csv
import gzip
import io
import unicodedata
from functools import lru_cache

from sinonym.resources import read_bytes

MAX_SURFACE_WORDS = 3


def fold_word(token: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", token).casefold() if not unicodedata.combining(ch))


@lru_cache(maxsize=1)
def _surfaces() -> frozenset[str]:
    text = gzip.decompress(read_bytes("compound_surnames.csv.gz")).decode("utf-8")
    return frozenset(row["surface"] for row in csv.DictReader(io.StringIO(text)))


def listed_compound_suffix(words: list[str]) -> int:
    """Return how many trailing words of ``words`` form a listed compound surface (0 when none)."""
    folded = [fold_word(word) for word in words]
    for width in range(min(MAX_SURFACE_WORDS, len(folded)), 1, -1):
        if " ".join(folded[-width:]) in _surfaces():
            return width
    return 0

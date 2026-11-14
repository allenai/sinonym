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
from sinonym.services.person_name_normalization import _CREDENTIAL_KEYS, _TITLE_KEYS
from sinonym.services.source_repairs import CREDENTIAL_WORDS, SPELLED_CREDENTIAL_WORDS

MAX_SURFACE_WORDS = 3
# a given name or patronymic, not the head of a family name (``Ahmed / Mohamed / Ali``, ``Ahmad / bin / Ismail``)
EXCLUDED_HEADS = frozenset({"mohamed", "mohammad", "bin"})
# credential and title keys that are also names: Md (Malay patronymic), Ba, Ma, Pastor, Frau, Portuguese do, Meng, Hon
_NAME_WORDS = frozenset({"ba", "beng", "do", "edd", "frau", "hon", "ma", "md", "meng", "pastor"})
_NOT_NAME_WORDS = (_CREDENTIAL_KEYS | _TITLE_KEYS | CREDENTIAL_WORDS | SPELLED_CREDENTIAL_WORDS) - _NAME_WORDS
# an MD degree written before the surname, not a Malay patronymic: the ORCID person registers Smith
_CREDENTIAL_SURFACES = frozenset({"md smith"})


@lru_cache(maxsize=65536)
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
        if folded[-width] not in EXCLUDED_HEADS and " ".join(folded[-width:]) in _surfaces():
            return width
    return 0


def credential_join(moved: list[str], surface: list[str]) -> bool:
    """Whether joining ``moved`` given-field words into the listed ``surface`` would turn a credential or title into surname."""
    folded = [fold_word(word) for word in moved]
    return not _NOT_NAME_WORDS.isdisjoint(folded) or " ".join(fold_word(word) for word in surface) in _CREDENTIAL_SURFACES

"""Words that ORCID registry people use mostly as a given name (``given_name_words.csv.gz``, built by
``scripts/build_given_name_words.py``), with their given, family and family-head people counts.

A given-name word: at least 200 people use it, at least 95% of them as a given name, and fewer than 0.5% of those open a
multi-word family name with it. Before a common Chinese surname, where no double surname opens with such a word, the
bars are 80% and 1.5% (``philip``, ``james``, ``paul``)."""

from __future__ import annotations

import csv
import gzip
import io
import unicodedata
from functools import lru_cache

from sinonym.resources import read_bytes

MIN_GIVEN_SHARE = 0.95
MAX_FAMILY_HEAD_SHARE = 0.005


@lru_cache(maxsize=1)
def _words() -> dict[str, tuple[int, int, int]]:
    text = gzip.decompress(read_bytes("given_name_words.csv.gz")).decode("utf-8")
    return {
        row["word"]: (int(row["given_people"]), int(row["family_people"]), int(row["family_head_people"]))
        for row in csv.DictReader(io.StringIO(text))
    }


@lru_cache(maxsize=1)
def _strict_words() -> frozenset[str]:
    return frozenset(
        word
        for word, (given, family, head) in _words().items()
        if given >= MIN_GIVEN_SHARE * (given + family) and head < MAX_FAMILY_HEAD_SHARE * given
    )


def load_given_name_words() -> None:
    _strict_words()


def _fold(word: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", word.casefold()) if not unicodedata.combining(ch))


def is_given_name_word(word: str) -> bool:
    return _fold(word) in _strict_words()


def is_given_name_word_before_chinese_surname(word: str) -> bool:
    return _fold(word) in _words()

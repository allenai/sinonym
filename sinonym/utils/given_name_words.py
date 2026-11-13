"""Words that ORCID registry people use almost only as a given name (``given_name_words.csv.gz``, built by
``scripts/build_given_name_words.py``): at least 200 people use the word, at least 95% of them as a given name, and
fewer than 0.5% of those open a multi-word family name with it."""

from __future__ import annotations

import csv
import gzip
import io
import unicodedata
from functools import lru_cache

from sinonym.resources import read_bytes


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    text = gzip.decompress(read_bytes("given_name_words.csv.gz")).decode("utf-8")
    return frozenset(row["word"] for row in csv.DictReader(io.StringIO(text)))


def load_given_name_words() -> None:
    _words()


def is_given_name_word(word: str) -> bool:
    folded = "".join(ch for ch in unicodedata.normalize("NFKD", word.casefold()) if not unicodedata.combining(ch))
    return folded in _words()

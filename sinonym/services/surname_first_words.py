"""Words the S2 corpus writes surname-first before dotted initials and otherwise uses mostly as a surname.

``surname_first_words.csv.gz`` (built by ``scripts/build_surname_first_words.py`` from source-name counts) lets
``Dmitrieva / - / N.M.`` resolve to surname ``Dmitrieva`` while ``Sowmya / - / K. R.`` (a given name before the family
initials) is left alone.
"""
from __future__ import annotations

import csv
import gzip
import io
import unicodedata
from functools import lru_cache

from sinonym.resources import read_bytes


def _fold(token: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", token).casefold() if ch.isalnum())


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    text = gzip.decompress(read_bytes("surname_first_words.csv.gz")).decode("utf-8")
    return frozenset(row["word"] for row in csv.DictReader(io.StringIO(text)))


def is_surname_first_word(token: str) -> bool:
    return _fold(token) in _words()

"""Whether a given name written in three capitals is a name (``ANA``) or initials (``ESL``).

S2AND reads a first name of two or three capitals as initials (``ESL`` is e + s + l, compatible with Eva Sophie Lunde)
and the title-cased word as a first name (``Esl`` vetoes her Eva mentions). A three-letter given name in capitals is
therefore kept in capitals unless ``allcaps_given_words.csv.gz`` (built by ``scripts/build_allcaps_given_words.py`` from
the ORCID public data file) lists it: ORCID registry people write it as their given name far more often than their
three given names start with its letters (``ana``, ``ewa``, ``hui``; not ``esl``, ``nar``, ``art``).
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from functools import lru_cache

from sinonym.resources import read_bytes

_THREE_CAPITALS = re.compile(r"[A-Z]{3}")


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    text = gzip.decompress(read_bytes("allcaps_given_words.csv.gz")).decode("utf-8")
    return frozenset(row["word"] for row in csv.DictReader(io.StringIO(text)))


def load_allcaps_given_words() -> None:
    _words()


def is_capitalized_initials(token: str) -> bool:
    """Whether a given-name token of three ASCII capitals is initials rather than a listed given name."""
    return _THREE_CAPITALS.fullmatch(token) is not None and token.lower() not in _words()

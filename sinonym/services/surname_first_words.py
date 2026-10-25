"""Words the S2 corpus writes surname-first before dotted initials and otherwise uses mostly as a surname.

``surname_first_words.csv.gz`` (built by ``scripts/build_surname_first_words.py`` from source-name counts) lets
``Dmitrieva / - / N.M.`` resolve to surname ``Dmitrieva`` while ``Sowmya / - / K. R.`` (a given name before the family
initials) is left alone.
"""
from __future__ import annotations

import csv
import gzip
import io
import re
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


_SURNAME_FIRST_WORD_RE = re.compile(r"[^\W\d_][^\W\d_'-]{2,}", re.UNICODE)
_DOTTED_INITIALS_TAIL_RE = re.compile(r"[^\W\d_]\.(?:\s*[^\W\d_]\.){1,2}", re.UNICODE)


def load_surname_first_words() -> None:
    _words()


def surname_first_initials(
    first: str | None, middle: str | None, last: str | None, suffix: str | None
) -> tuple[str, str, str] | None:
    """``Dmitrieva / - / N.M.`` (or ``Назарова / - / А.Р.``) → ``("N.", "M.", "Dmitrieva")``.

    Only for 2-3 initials that each carry their own dot, and only when the corpus writes the word this way and uses it
    mostly as a surname; ``Sowmya / - / K. R.``, where the initials are the family name, is left alone.
    """
    if (suffix or "").strip() or (middle or "").strip():
        return None
    word = (first or "").strip()
    tail = (last or "").strip()
    if not _SURNAME_FIRST_WORD_RE.fullmatch(word) or not word[0].isupper() or not word[1:].islower():
        return None
    if not _DOTTED_INITIALS_TAIL_RE.fullmatch(tail):
        return None
    letters = [part for part in re.split(r"[.\s]+", tail) if part]
    if not all(letter.isupper() for letter in letters) or not is_surname_first_word(word):
        return None
    initials = [f"{letter}." for letter in letters]
    return initials[0], " ".join(initials[1:]), word

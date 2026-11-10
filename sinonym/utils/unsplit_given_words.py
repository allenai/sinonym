"""Words the Chinese given-name splitter would cut into pinyin syllables that are not Chinese given names.

``unsplit_given_words.csv.gz`` (built by ``scripts/build_unsplit_given_words.py`` from source-name counts and the ORCID
public data file) keeps ``Annie``, ``Dana``, ``Hoang`` and ``Jihye`` whole instead of ``An-Nie``, ``Da-Na``, ``Ho-Ang``
and ``Jih-Ye``. A word qualifies when most ORCID people who write it are outside China, Taiwan, Hong Kong, Macau and
Singapore, or, for a word the registry hardly knows, when the corpus uses it often, rarely beside a Chinese surname and
almost never split; Chinese given names (``Xiaoming``, ``Bangti``) are not listed. A word with a romanized Korean
syllable that pinyin and Cantonese spellings do not have (``Seungho``, ``Jieun``, ``Hyunwoo``) is kept whole too.
"""
from __future__ import annotations

import csv
import gzip
import io
import re
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING

from sinonym.chinese_names_data import VIETNAMESE_SURNAMES
from sinonym.resources import read_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator

# not after h (Cantonese Sheung) and eun only after a vowel or at the start (Cantonese Leung, Keung, Yeung)
_KOREAN_ONLY_SYLLABLE = re.compile(
    r"(?:^|[^h])seung|hyun|hyeon|hye|jae|(?:^|[aeiou])eun|yeon|yeong|kyung|kyeong|hwan(?!g)|wook|cheol|sook|seok|joon|"
    r"hoon|heon|jeong",
)


@lru_cache(maxsize=65536)
def _fold(token: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", token).casefold() if ch.isalpha() and ch.isascii())


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    text = gzip.decompress(read_bytes("unsplit_given_words.csv.gz")).decode("utf-8")
    return frozenset(row["word"] for row in csv.DictReader(io.StringIO(text)))


def load_unsplit_given_words() -> None:
    _words()


# The source words of the name being processed. A listed word is never split, but inside an author-written hyphen or
# glued token (``Ta-Tsien``, ``YuLina``) it still counts as a Chinese given-name part.
_VIETNAMESE_SURNAME_FOLDS = frozenset(_fold(name) for name in VIETNAMESE_SURNAMES)
_STANDALONE_WORDS: ContextVar[frozenset[str] | None] = ContextVar("standalone_words", default=None)


_HYPHEN_PARTICLES = frozenset(
    {"da", "das", "de", "del", "della", "der", "den", "di", "do", "dos", "du", "la", "le", "ten", "ter", "van", "von"},
)
_WORD_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[^\W\d_]+")


def _non_chinese_bound_word(parts: list[str]) -> bool:
    """A bound word that reads as a non-Chinese name: a particle, a lone letter, a doubled word, a capitalised tail
    (``KoJIMA``) or a Vietnamese surname (``Cao-Hoang``)."""
    folds = [_fold(part) for part in parts]
    return (
        any(fold in _HYPHEN_PARTICLES or len(fold) == 1 or fold in _VIETNAMESE_SURNAME_FOLDS for fold in folds)
        or len(set(folds)) < len(folds)
        or any(len(part) > 1 and part.isupper() for part in parts[1:])
    )


def _standalone_folds(raw_name: str) -> frozenset[str]:
    """Folded source words; every part of a bound word that reads as a non-Chinese name counts as a word too."""
    folds = set()
    for word in (raw_name or "").split():
        folds.add(_fold(word))
        pieces = re.split("[-\u2010\u2011]", word)
        parts = [part for piece in pieces for part in _WORD_PARTS.findall(piece)]
        if len(parts) > 1 and _non_chinese_bound_word(parts):
            folds.update(_fold(part) for part in [*pieces, *parts])
    return frozenset(folds)


@contextmanager
def standalone_words(raw_name: str) -> Iterator[None]:
    token = _STANDALONE_WORDS.set(_standalone_folds(raw_name))
    try:
        yield
    finally:
        _STANDALONE_WORDS.reset(token)


def is_listed_unsplit_given_word(token: str) -> bool:
    return _fold(token) in _words()


def is_listed_bound_part(token: str) -> bool:
    """A listed word that the name being processed has only inside a longer author-written token."""
    words = _STANDALONE_WORDS.get()
    folded = _fold(token)
    return words is not None and folded in _words() and folded not in words


def has_korean_only_syllable(token: str) -> bool:
    return _KOREAN_ONLY_SYLLABLE.search(_fold(token)) is not None


def is_unsplit_given_word(token: str) -> bool:
    return is_listed_unsplit_given_word(token) or has_korean_only_syllable(token)

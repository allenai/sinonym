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
from functools import lru_cache

from sinonym.resources import read_bytes

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


def is_unsplit_given_word(token: str) -> bool:
    folded = _fold(token)
    return folded in _words() or _KOREAN_ONLY_SYLLABLE.search(folded) is not None

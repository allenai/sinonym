"""One written form for a two-syllable Korean given name: ``Jinho``, ``JinHo``, ``Jin Ho`` and ``Jin / Ho`` become
``Jin-Ho``, the Revised Romanization form with a hyphen.

``korean_given_words.csv.gz`` (built by ``scripts/build_korean_given_words.py`` from the ORCID public data file) lists
given names that South Korean registry people write, with the place where they cut them (``jinho`` 3, ``seungwoo`` 5)
and how many registry people write them in and outside South Korea and in a Chinese region. A name is rewritten only
with a Korean surname, when the given name and middle name together are one listed word cut where the list cuts it, or
not cut at all, and at least half the registry people writing the word are in South Korea.
"""
from __future__ import annotations

import csv
import gzip
import io
import re
from functools import lru_cache

from sinonym.chinese_names_data import (
    KOREAN_ONLY_SURNAMES,
    NAME_ORDER_ROUTING_KOREAN_SURNAMES,
    NAME_ORDER_ROUTING_KOREAN_TAIL_ONLY_SURNAMES,
    OVERLAPPING_KOREAN_SURNAMES,
    REVIEWED_ATOMIC_KOREAN_GIVEN_FORMS,
)
from sinonym.resources import read_bytes
from sinonym.utils.unsplit_given_words import has_korean_only_syllable

KOREAN_SURNAMES = frozenset(
    set(KOREAN_ONLY_SURNAMES)
    | set(NAME_ORDER_ROUTING_KOREAN_SURNAMES)
    | set(NAME_ORDER_ROUTING_KOREAN_TAIL_ONLY_SURNAMES)
    | set(OVERLAPPING_KOREAN_SURNAMES)
    | {"bang", "byeon", "byun", "chae", "cheon", "choe", "chun", "eom", "gwak", "hur", "huh", "jee", "ji", "jin"}
    | {"kwak", "min", "rhee", "seong", "sohn", "sung", "uhm", "um", "woo", "yeom", "yi"},
)
KOREAN_ONLY_GIVEN_SURNAMES = frozenset(set(KOREAN_ONLY_SURNAMES) | {"cho", "choi", "chae", "jeong", "jung", "kang"})
# Clearly Korean (a Korean-only surname or syllable): most registry people writing the word are in South Korea.
MIN_KR_SHARE = 0.5
# Shared surname (Lee, Lim, Han, Song...): also almost no registry people writing the word are in a Chinese region.
MIN_KR_VS_CHINESE_SHARE = 0.9
# A word with a second valid cut (Sungil: Sung-Il or Sun-Gil) is hyphenated only when Korean registry people who write it
# in two parts almost all cut it in one place: a wrong cut makes S2AND veto the person's other mentions (Sun Gil).
MIN_AMBIGUOUS_CUT_SHARE = 0.97
_PARTS = re.compile(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?![a-z])")


@lru_cache(maxsize=1)
def _words() -> dict[str, tuple[int, int, int, float]]:
    text = gzip.decompress(read_bytes("korean_given_words.csv.gz")).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    syllables = {part for row in rows for part in (row["word"][: int(row["cut"])], row["word"][int(row["cut"]) :])}

    def decisive(word: str, cut: int, cut_share: float) -> bool:
        ambiguous = any(
            other != cut and word[:other] in syllables and word[other:] in syllables for other in range(1, len(word))
        )
        return not ambiguous or cut_share >= MIN_AMBIGUOUS_CUT_SHARE

    return {
        row["word"]: (int(row["cut"]), int(row["kr_people"]), int(row["chinese_region_people"]), float(row["kr_share"]))
        for row in rows
        if decisive(row["word"], int(row["cut"]), float(row["cut_share"]))
    }


def load_korean_given_words() -> None:
    _words()


def _source_parts(given: str, middle: str) -> list[str] | None:
    """The given and middle name as at most two ASCII word parts, split at spaces, hyphens and CamelCase."""
    words = f"{given} {middle}".replace("-", " ").split()
    if not words or not all(word.isascii() and word.isalpha() for word in words):
        return None
    parts = [part for word in words for part in _PARTS.findall(word)]
    return parts if len(parts) <= 2 else None  # noqa: PLR2004


def korean_given_form(given: str, middle: str, surname: str) -> str | None:
    """The hyphenated given name for a listed Korean given name before a Korean surname, else None."""
    parts = _source_parts(given, middle) if surname.casefold() in KOREAN_SURNAMES else None
    if parts is None:
        return None
    joined = "".join(parts).lower()
    entry = None if joined in REVIEWED_ATOMIC_KOREAN_GIVEN_FORMS else _words().get(joined)
    if entry is None:
        return None
    cut, kr_people, chinese_people, kr_share = entry
    clearly_korean = surname.casefold() in KOREAN_ONLY_GIVEN_SURNAMES or has_korean_only_syllable(joined)
    if (
        (len(parts) == 2 and len(parts[0]) != cut)  # noqa: PLR2004
        or kr_share < MIN_KR_SHARE
        or (not clearly_korean and kr_people < MIN_KR_VS_CHINESE_SHARE * (kr_people + chinese_people))
    ):
        return None
    return f"{joined[:cut].capitalize()}-{joined[cut:].capitalize()}"

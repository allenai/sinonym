"""Repairs of source-field defects whose correct form is recoverable from the name itself.

Each rule rewrites the given and middle source fields before parsing and returns ``None`` when it does not apply.
Each rule also names the surname shapes it accepts, so a malformed surname is never a reason to rewrite the given name.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING

from sinonym.chinese_names_data import KOREAN_GIVEN_PATTERNS, VALID_CHINESE_ONSETS, VALID_CHINESE_RIMES
from sinonym.resources import read_bytes
from sinonym.services.slavic_initials import is_dotted_initial
from sinonym.services.token_order_prior import token_role
from sinonym.utils.unsplit_given_words import _KOREAN_ONLY_SYLLABLE

if TYPE_CHECKING:
    from collections.abc import Callable

_VOWELS = frozenset("AEIOUY")
_SURNAME_WORD_RE = re.compile(r"[^\W\d_]+")
_SURNAME_PREFIX_RE = re.compile(r"^(?:Mc|Mac|O['\u2019])(?=[A-Z])")


def _plain_surname(last: str | None) -> bool:
    """Title-case words of two or more letters, optionally after lower-case particles (``van der Linden``).

    The last word must be capitalized: a lone lower-case word (``raham`` after ``RobertG``) means the capital was the
    surname's own first letter.
    """
    words = re.split(r"[ \-]", (last or "").strip())
    if not words[-1] or not words[-1][0].isupper():
        return False
    for word in words:
        core = _SURNAME_PREFIX_RE.sub("", word)
        if len(core) < 2 or not _SURNAME_WORD_RE.fullmatch(core):  # noqa: PLR2004
            return False
        if not (core.islower() or (core[0].isupper() and core[1:].islower())):
            return False
    return True


def _empty(value: str | None) -> bool:
    return (value or "").strip() in ("", ".")


_GLUED_RE = re.compile(r"([^\W\d_]{3,})([A-Z])(\.?)")
# Words that open organization strings or are titles and particles, which the source counts also list as first names.
_NOT_GIVEN_WORDS = frozenset(
    {
        *("the", "and", "of", "for", "department", "unknown", "editorial", "editor", "author", "authors", "research"),
        *("medical", "national", "world", "group", "institute", "university", "center", "centre", "society"),
        *("hospital", "school", "college", "team", "international", "health", "science", "global", "general"),
        *("prof", "professor", "dr", "doctor", "mr", "mrs", "miss", "ms", "sir", "capt", "rev", "lord", "lady"),
        *("van", "von", "der", "den", "del", "della", "de", "da", "di", "du", "la", "le", "mac", "san", "santa"),
        "saint",
        "pharm",
        *("american", "european", "clinical", "federal", "central", "collective", "corresponding"),
    },
)


def _east_asian_syllable(word: str) -> bool:
    """A romanized Chinese or Korean syllable (``Eun``, ``Seung``, ``Chul``, ``Jae``), after which ``A.`` may be the
    second syllable of the given name (Eun-A)."""
    folded = word.lower()
    if folded in KOREAN_GIVEN_PATTERNS or _KOREAN_ONLY_SYLLABLE.search(folded) or folded in VALID_CHINESE_RIMES:
        return True
    return any(folded.startswith(onset) and folded[len(onset) :] in VALID_CHINESE_RIMES for onset in VALID_CHINESE_ONSETS)


@lru_cache(maxsize=1)
def _glued_prefixes() -> dict[str, tuple[str, str]]:
    text = gzip.decompress(read_bytes("glued_initial_prefixes.csv.gz")).decode("utf-8")
    return {row["prefix"]: (row["block_dotted"], row["block_undotted"]) for row in csv.DictReader(io.StringIO(text))}


def load_glued_initial_prefixes() -> None:
    _glued_prefixes()


def unglue_given_initial(first: str | None, middle: str | None, last: str | None, suffix: str | None):  # noqa: PLR0911
    """``DamianG.`` / - / ``Downey`` -> ``("Damian", "G.")``: a middle initial glued to the given name.

    The surname must be plain (``_plain_surname``). The word must be a common given name
    (``glued_initial_prefixes.csv.gz``) and not an organization word, title or particle (``WorldH``, ``ProfJ.``). A
    letter is blocked after a word when the word plus that letter in lower case is itself a common first name
    (``MartinE.`` may be Martine, ``ArunaV`` is Arunav). Without a period a vowel capital is a syllable more often
    than an initial (``SungA``, ``QingE``, ``JeongA``), so it never splits; with a period it splits only after a word
    the token role prior calls a given name that is not a Chinese or Korean syllable (``EunA.`` is Eun-A).
    """
    if not _empty(suffix) or not _empty(middle) or not _plain_surname(last):
        return None
    match = _GLUED_RE.fullmatch(unicodedata.normalize("NFC", (first or "").strip()))
    if match is None:
        return None
    word, letter, period = match.groups()
    if not (word[0].isupper() and word[1:].islower()) or word.lower() in _NOT_GIVEN_WORDS:
        return None
    blocks = _glued_prefixes().get(word)
    if blocks is None:
        return None
    block_dotted, block_undotted = blocks
    if period:
        if letter in block_dotted:
            return None
        if letter in _VOWELS and (token_role(word) != "given" or _east_asian_syllable(word)):
            return None
    elif letter in block_undotted or letter in _VOWELS:
        return None
    return word, letter + "."


_DOUBLE_PERIOD_RE = re.compile(r"(?<=[A-Za-z])\.\.(?!\.)")
_INITIAL_TOKEN_RE = re.compile(r"[^.]+\.?")
_LETTER_SURNAME_RE = re.compile(r"[^\W\d_]{2,}(?:['\u2019\-][^\W\d_]+)*")


def collapse_doubled_initial_periods(first: str | None, middle: str | None, last: str | None, suffix: str | None):
    """``D.G..`` / - / ``Downey`` -> ``("D.G.", None)``: a doubled period after an initial.

    Fires only when every given and middle token is a dotted initial and the surname is one word of letters (any case,
    ``PARLAR`` included); otherwise the unrecognized ``X..`` token is left to the parser as before. A name with three or
    more periods in a row (an ellipsis) is left as written.
    """
    if not _empty(suffix) or not _LETTER_SURNAME_RE.fullmatch((last or "").strip()):
        return None
    personal = f"{first or ''} {middle or ''}"
    if "..." in personal or not _DOUBLE_PERIOD_RE.search(personal):
        return None
    fixed_first = _DOUBLE_PERIOD_RE.sub(".", first) if first else first
    fixed_middle = _DOUBLE_PERIOD_RE.sub(".", middle) if middle else middle
    parts = f"{fixed_first or ''} {fixed_middle or ''}".split()
    tokens = [token for part in parts for token in _INITIAL_TOKEN_RE.findall(part)]
    if "".join(tokens) != "".join(parts) or not tokens or not all(is_dotted_initial(token) for token in tokens):
        return None
    return fixed_first, fixed_middle


RULES: tuple[tuple[str, Callable], ...] = (
    ("glued_initial", unglue_given_initial),
    ("doubled_period", collapse_doubled_initial_periods),
)


def repair_source_fields(first: str | None, middle: str | None, last: str | None, suffix: str | None):
    """``(rule name, first, middle)`` for the first rule that applies, or ``None``."""
    for name, rule in RULES:
        repaired = rule(first, middle, last, suffix)
        if repaired is not None:
            return name, *repaired
    return None

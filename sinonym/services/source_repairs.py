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
from sinonym.services.person_name_normalization import _CREDENTIAL_KEYS, _TITLE_KEYS, reviewed_source_cleanup_pattern
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


# Degrees and post-nominals that are never a given name or a run of initials (ORCID registry and corpus evidence).
CREDENTIAL_WORDS = frozenset(
    {
        *("phd", "esq", "pharmd", "psyd", "dphil", "drph", "mphil", "mpharm", "bpharm", "mbbs", "mbchb", "mbbch"),
        *("aprn", "crna", "ccrn", "lcsw", "fnp-bc", "dacvim"),
        *("faao", "faap", "facc", "facp", "facr", "facs", "fapa", "fasn", "fccp", "fesc", "fracp", "fracs"),
        *("franzcp", "frcog", "frcophth", "frcp", "frcpath", "frcpc", "frcpch", "frcpe", "frcpi", "frcpsych", "frcr"),
        *("frcs", "frcsc", "frcse", "frcsed", "frcsi"),
        *("mrcgp", "mrcog", "mrcp", "mrcpch", "mrcpi", "mrcpsych", "mrcs", "mrcsed", "mrcvs"),
    },
)
# Degrees whose capitals or Title case are also people's middle initials (MSC, MPH, LLM in Dutch and Brazilian names) or
# given names (Edd, Beng, Meng); only their degree spelling counts: mixed case (MSc, EdD) or dots inside (M.Sc., LL.M.).
SPELLED_CREDENTIAL_WORDS = frozenset({"msc", "bsc", "dsc", "scd", "mph", "llm", "llb", "edd", "meng", "beng"})
_CREDENTIAL_PUNCTUATION_RE = re.compile(r"[.,;:()]")
_LETTER_INITIALS_RE = re.compile(r"(?:[^\W\d_]\.)+[^\W\d_]?\.?,?")
_NAME_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'\u2019.][^\W\d_]+)*\.?")
_NOT_NAME_KEYS = CREDENTIAL_WORDS | SPELLED_CREDENTIAL_WORDS | _CREDENTIAL_KEYS | _TITLE_KEYS | _NOT_GIVEN_WORDS | {"med"}


def _degree_spelling(token: str) -> bool:
    """``MSc``, ``EdD`` (mixed case) or ``M.Sc.``, ``LL.M.`` (a dot inside, next to a run of two or more letters)."""
    core = re.sub(r"[,;:()]", "", token)
    letters = core.replace(".", "")
    if letters and not (letters.isupper() or letters.islower() or letters.istitle()):
        return True
    chunks = [chunk for chunk in core.split(".") if chunk]
    return len(chunks) > 1 and any(len(chunk) > 1 for chunk in chunks)


def is_credential_token(token: str) -> bool:
    """A listed degree or post-nominal, in any case, dots and commas ignored. A run of dotted single letters (``M.S.C.``)
    is never one: people write their initials that way."""
    if _LETTER_INITIALS_RE.fullmatch(token):
        return False
    key = _CREDENTIAL_PUNCTUATION_RE.sub("", token).casefold()
    return key in CREDENTIAL_WORDS or (key in SPELLED_CREDENTIAL_WORDS and _degree_spelling(token))


def _trailing_credential_count(tokens: list[str]) -> int:
    count = 0
    while count < len(tokens) and is_credential_token(tokens[-1 - count]):
        count += 1
    return count


def _name_word(token: str) -> bool:
    return bool(_NAME_TOKEN_RE.fullmatch(token)) and _CREDENTIAL_PUNCTUATION_RE.sub("", token).casefold() not in _NOT_NAME_KEYS


def _has_surname(last: str | None) -> bool:
    return any(len(word) > 1 and word.casefold() not in _NOT_NAME_KEYS for word in _SURNAME_WORD_RE.findall(last or ""))


def strip_credentials(first: str | None, middle: str | None, last: str | None, suffix: str | None):
    """``Leslie PhD`` / - / ``Burton`` -> ``("Leslie", None)``: degrees and post-nominals in the given or middle field.

    Credentials (``is_credential_token``) are dropped from the end of the first field after a given name, and from the
    end of the middle field when the first field is a given name (``Leslie`` / ``A. PhD``, ``Leslie`` / ``PhD``).

    A first field made only of credentials is emptied when the middle field holds the given name
    (``MRCS`` / ``Simon C.`` / ``Williams``), unless the parser already drops it (``_CREDENTIAL_KEYS`` or a reviewed
    source pattern). With an empty middle field it stays: the parser would then split the surname field
    (``Garcia Lopez``), and four capitals before a lone surname are sometimes a person's initials (``Facc`` / - /
    ``Silva`` can be F. A. C. C. Silva).

    The surname field is never rewritten and must keep a word that is not a credential or title. The suffix field
    is not read.
    """
    if not _has_surname(last):
        return None
    first_tokens, middle_tokens = (first or "").split(), (middle or "").split()
    first_cut = len(first_tokens) - _trailing_credential_count(first_tokens)
    middle_cut = len(middle_tokens) - _trailing_credential_count(middle_tokens)
    given, kept_middle = first_tokens[:first_cut], middle_tokens[:middle_cut]
    if first_tokens and not given:
        parser_drops = reviewed_source_cleanup_pattern(first, middle, last, suffix) is not None or any(
            _CREDENTIAL_PUNCTUATION_RE.sub("", token).casefold() in _CREDENTIAL_KEYS for token in first_tokens
        )
        if parser_drops or not kept_middle or not all(map(_name_word, kept_middle)):
            return None
        return None, " ".join(kept_middle)
    if first_cut == len(first_tokens) and middle_cut == len(middle_tokens):
        return None
    if not given or not all(map(_name_word, given)) or not all(map(_name_word, kept_middle)):
        return None
    return " ".join(given), " ".join(kept_middle) or None


RULES: tuple[tuple[str, Callable], ...] = (
    ("glued_initial", unglue_given_initial),
    ("doubled_period", collapse_doubled_initial_periods),
    ("credential", strip_credentials),
)


def repair_source_fields(first: str | None, middle: str | None, last: str | None, suffix: str | None):
    """``(rule name, first, middle)`` for the first rule that applies, or ``None``."""
    for name, rule in RULES:
        repaired = rule(first, middle, last, suffix)
        if repaired is not None:
            return name, *repaired
    return None

"""Repairs of source-field defects whose correct form is recoverable from the name itself.

Each rule rewrites the given and middle source fields before parsing and returns ``None`` when it does not apply.
Each rule also names the surname shapes it accepts, so a malformed surname is never a reason to rewrite the given name.
"""

from __future__ import annotations

import csv
import gzip
import html
import io
import re
import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING

from sinonym.chinese_names_data import KOREAN_GIVEN_PATTERNS, VALID_CHINESE_ONSETS, VALID_CHINESE_RIMES
from sinonym.resources import read_bytes
from sinonym.services.person_name_normalization import (
    _CREDENTIAL_KEYS,
    _TITLE_KEYS,
    EAST_ASIAN_GIVEN_SYLLABLES,
    EAST_ASIAN_SURNAMES,
    reviewed_closed_comma_credential_tail_head,
    reviewed_source_cleanup_pattern,
)
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


# Family particles a source writes twice, closing the middle field and opening the surname field (``Ida`` / ``Di`` /
# ``Di Mola``). Le, Lo, Li, Do, San, E and Y are left out: they are also East Asian names or connectives.
_REPEATED_PARTICLES = frozenset(
    {"de", "da", "di", "del", "della", "dello", "degli", "dos", "das", "du", "des", "la", "el", "al", "van", "von"}
    | {"der", "den", "ten", "ter", "vande", "vander", "dal", "abd", "abu", "ben", "bin", "binti", "bint", "ul", "ur", "ap"},
)


def drop_repeated_particle(first: str | None, middle: str | None, last: str | None, suffix: str | None):  # noqa: ARG001
    """``Ida`` / ``Di`` / ``Di Mola`` -> ``("Ida", None)``: a particle closing the middle field that also opens the surname.

    The surname field must go on after the particle, so a one-word surname field (``Kim`` / ``Van`` / ``Van``) is left
    alone.
    """
    middle_tokens, last_tokens = (middle or "").split(), (last or "").split()
    if not middle_tokens or len(last_tokens) < 2 or not (first or "").strip():  # noqa: PLR2004
        return None
    particle = middle_tokens[-1].casefold()
    if particle not in _REPEATED_PARTICLES or particle != last_tokens[0].casefold():
        return None
    return first, " ".join(middle_tokens[:-1]) or None


# Academic and courtesy titles that are never a given name in the ORCID registry, and the German and Turkish degree
# words written after them (``Dr. rer. nat.``, ``Doç. Dr.``). Ms, Miss and Sir are names or initials too and stay.
_TITLE_WORDS = frozenset({"dr", "prof", "professor", "mr", "mrs", "doç", "yrd", "asst", "assoc", "engr", "univprof", "drs"})
_TITLE_QUALIFIERS = frozenset({"med", "rer", "nat", "ing", "dipl", "habil", "techn", "phil", "dent", "vet", "jur", "oec", "sc"})


# the h. c. of Dr. h. c., only right after a title
_HONORIS_CAUSA = frozenset({"hc", "h.c"})


def _title_token(token: str, *, qualifier: bool = False) -> bool:
    """A listed title (``Dr.``, ``Prof``, ``dr.``); undotted capitals (``DR``, ``MR``) can be initials and do not count."""
    core = token.rstrip(".,").replace(".-", "-").split("-")[0]
    key = core.casefold()
    if key not in _TITLE_WORDS and not (qualifier and key in _TITLE_QUALIFIERS | _HONORIS_CAUSA):
        return False
    return token.endswith(".") or not core.isupper() or len(core) == 1


def strip_titles(first: str | None, middle: str | None, last: str | None, suffix: str | None):  # noqa: C901, PLR0911
    """``Carsten`` / ``Dr.`` / ``Tille`` -> ``("Carsten", None)``: titles left in the middle field, or a first field of
    only titles before the given name in the middle field (``Doç.`` / ``Dr. Taner`` / ``Artan``).

    Degree words that follow a title go with it (``Dr. rer. nat.``, ``Dipl.-Ing. Dr.``). Every remaining given and middle
    word must be a name word or an initial, and the surname field must keep a word that is not a credential or title.
    """
    if not _has_surname(last) or reviewed_source_cleanup_pattern(first, middle, last, suffix) is not None:
        return None
    first_tokens, middle_tokens = (first or "").split(), (middle or "").split()
    if not middle_tokens:
        return None

    def strip(tokens: list[str]) -> list[str]:
        kept, after_title, skip = [], False, 0
        for index, token in enumerate(tokens):
            if skip:
                skip -= 1
                continue
            honoris_causa = [t.rstrip(".,").casefold() for t in tokens[index : index + 2]] == ["h", "c"]
            if after_title and honoris_causa:
                skip = 1
                continue
            if _title_token(token) or (after_title and _title_token(token, qualifier=True)):
                after_title = True
                continue
            after_title = False
            kept.append(token)
        return kept

    if first_tokens and all(_title_token(t) or _title_token(t, qualifier=True) for t in first_tokens):
        parser_strips = all(_CREDENTIAL_PUNCTUATION_RE.sub("", t).casefold() in _TITLE_KEYS for t in first_tokens)
        if not any(map(_title_token, first_tokens)) or (parser_strips and not _title_token(middle_tokens[0])):
            return None
        rest = strip(middle_tokens)
        if not rest:
            return None
        given, kept_middle = rest[:1], rest[1:]
    else:
        kept_middle = strip(middle_tokens)
        if len(kept_middle) == len(middle_tokens) or any(t.isupper() and len(t) <= _SHORT_CAPITALS for t in first_tokens):
            return None
        given = first_tokens
    words = given + kept_middle
    if not given or not all(map(_name_word, words)) or any(map(_abbreviation_or_qualifier, words)):
        return None
    return " ".join(given), " ".join(kept_middle) or None


def _abbreviation_or_qualifier(token: str) -> bool:
    """``Lt.``, ``Col.``, ``Dipl``: a dotted abbreviation of two or more letters, or a degree word left without its title."""
    core = token.rstrip(".,")
    return (token.endswith(".") and len(core) > 1) or core.casefold() in _TITLE_QUALIFIERS


_CONTRIBUTOR_ROLES = frozenset(
    {"translated", "translation", "edited", "compiled", "reviewed", "illustrated", "illustrations", "illustration"}
    | {"interview", "interviewed", "organised", "organized", "presented", "adapted", "prepared", "conducted", "written"}
    | {"supervised", "foreword", "preface", "introduction", "photo", "photos", "photographs", "commentary", "recorded"},
)


def strip_contributor_role(first: str | None, middle: str | None, last: str | None, suffix: str | None):  # noqa: ARG001
    """``Translated`` / ``by N.B.`` / ``Zelinska`` -> ``("N.B.", None)``: a credit line split across the given fields."""
    first_tokens, middle_tokens = (first or "").split(), (middle or "").split()
    if (
        len(first_tokens) != 1
        or first_tokens[0].casefold() not in _CONTRIBUTOR_ROLES
        or len(middle_tokens) < 2  # noqa: PLR2004
        or middle_tokens[0].casefold() != "by"
        or not _has_surname(last)
    ):
        return None
    rest = middle_tokens[1:]
    if not all(map(_name_word, rest)) or any(t.endswith(".") and len(t.rstrip(".")) > 1 and "." not in t[:-1] for t in rest):
        return None
    return rest[0], " ".join(rest[1:]) or None


RULES: tuple[tuple[str, Callable], ...] = (
    ("glued_initial", unglue_given_initial),
    ("doubled_period", collapse_doubled_initial_periods),
    ("credential", strip_credentials),
    ("repeated_particle", drop_repeated_particle),
    ("title", strip_titles),
    ("contributor_role", strip_contributor_role),
)


# Cyrillic letters drawn like a Latin letter, typed into a Latin word (``Babуch``, ``Аleksii``).
_LATIN_LOOKALIKES = str.maketrans(
    "аеорсухіјѕԁӏАВЕКМНОРСТУХІЈЅ",
    "aeopcyxijsdlABEKMHOPCTYXIJS",
)
_CYRILLIC_LETTER_RE = re.compile(r"[\u0400-\u052F]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z\u00C0-\u024F]")


# after t a Cyrillic с is as often the s of ts (Ц: ``Tсelykh``) as a c
_TS_RE = re.compile(r"[Tt][сС]")
_CYRILLIC_RUN_RE = re.compile(r"[\u0400-\u052F]{2}")


def _latin_word(word: str) -> str:
    letters = [ch for ch in word if ch.isalpha()]
    cyrillic = _CYRILLIC_LETTER_RE.findall(word)
    if len(letters) < 3 or not cyrillic or not _LATIN_LETTER_RE.search(word) or 2 * len(cyrillic) >= len(letters):  # noqa: PLR2004
        return word
    if (word.isupper() and _CYRILLIC_RUN_RE.search(word)) or _TS_RE.search(word):
        return word
    fixed = word.translate(_LATIN_LOOKALIKES)
    return word if _CYRILLIC_LETTER_RE.search(fixed) else fixed


def _latin_field(value: str | None) -> str | None:
    return None if value is None else re.sub(r"[^\s.\-]+", lambda m: _latin_word(m.group()), value)


def latin_lookalikes(first: str | None, middle: str | None, last: str | None):
    """``V.L.`` / - / ``Babуch`` (Cyrillic у) -> ``("V.L.", None, "Babych")``: Cyrillic look-alikes inside a Latin word.

    A word of three or more letters changes only when fewer than half its letters are Cyrillic and each has a Latin
    twin; on ORCID people the twin matches the registered family name far more often than the Cyrillic reading
    (``Сhernavina``, ``Babуch``). Cyrillic names, mostly Cyrillic words (``Росa``), words with other Cyrillic letters,
    capitals with a Cyrillic run (``TKACHЕНКО``), a Cyrillic с after t (as often ts as tc) and the middle field
    (patronymics and their abbreviations) stay as written.
    """
    if not _CYRILLIC_LETTER_RE.search(f"{first or ''}{last or ''}"):
        return None
    fixed = (_latin_field(first), middle, _latin_field(last))
    return None if fixed == (first, middle, last) else fixed


_MOJIBAKE_RE = re.compile(
    r"[ÃÂÄÅÐÑ][\u0080-\u00BF\u0152\u0153\u0160\u0161\u0178\u017D\u017E\u0192\u02C6\u02DC\u2013-\u203A\u20AC\u2122]",
)


def _unmojibake(value: str) -> str:
    """UTF-8 read as Windows-1252 (``ZoltÃ¡n`` -> ``Zoltán``), undone at most twice; only when it decodes to letters."""
    decoded = value
    for _ in range(2):
        if not _MOJIBAKE_RE.search(decoded):
            break
        try:
            decoded = decoded.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value
    if _MOJIBAKE_RE.search(decoded) or any(not (ch.isalpha() or ch.isascii()) for ch in decoded):
        return value
    return decoded


def fix_mojibake(first: str | None, middle: str | None, last: str | None):
    """``ZoltÃ¡n`` / - / ``GÃ¡lik`` -> ``("Zoltán", None, "Gálik")``: UTF-8 bytes decoded as Windows-1252."""
    fixed = tuple(None if value is None else _unmojibake(value) for value in (first, middle, last))
    return None if fixed == (first, middle, last) else fixed


_ORCID_JUNK_RE = re.compile(
    r"(?i)^(?:https?:/+\S*|(?:https?)?(?:www\.)?orcid\.org\S*|(?:id)?orcid|\d{4}-\d{4}-\d{4}-\d{3}[\dX])$",
)
# one footnote star at the edge of a word; a run of stars masks letters (``S***``)
_FOOTNOTE_RE = re.compile(r"^[*\u2217\u204E](?=[^*\u2217\u204E])|(?<=[^*\u2217\u204E])[*\u2217\u204E]$")
_JUNK_HINT_RE = re.compile(r"[*\u2217\u204E^\d]|orcid|http", re.IGNORECASE)
_ESCAPED_ENTITY_RE = re.compile(r"\^\|\^([A-Za-z]+);")


# an ORCID iD in any written form, with its label and brackets: (ORCID: 0000-...), ORCID iD https://orcid.org/0000-...
_ORCID_ID_RE = re.compile(
    r"(?i)[(\[]?\s*(?:(?:\bid\s*)?orcid®?(?:\s*ids?)?\s*[:#]?\s*)?(?:https?:?/*)?(?:www\.)?(?:orcid\.org/?)?"
    r"(?<!\d)\d{4}-\d{3,4}-\d{4}-\d{3}[\dXxХх]?(?![\dXxХх])\s*[)\]]?",
)
_ORCID_URL_RE = re.compile(r"(?i)(?:https?:?/*)?(?:www\.)?orcid\.org\S*")
_ORCID_LABEL_RE = re.compile(r"(?i)[(\[]?\s*(?<![\w-])(?:id\s*)?orcid®?(?:\s*ids?)?\b\s*:?\s*[)\]]?")
_EMPTY_BRACKETS_RE = re.compile(r"[(\[]\s*[)\]]")


def _junk_free(value: str | None) -> str | None:
    if value is None:
        return None
    value = _ESCAPED_ENTITY_RE.sub(lambda m: html.unescape(f"&{m.group(1)};"), value)
    value = _ORCID_URL_RE.sub(" ", _ORCID_ID_RE.sub(" ", value))
    value = _EMPTY_BRACKETS_RE.sub(" ", _ORCID_LABEL_RE.sub(" ", value))
    words = [_FOOTNOTE_RE.sub("", word) for word in value.split() if not _ORCID_JUNK_RE.match(word)]
    return " ".join(word for word in words if word)


def _surname_from_given_fields(first: str | None, middle: str | None) -> list[str | None] | None:
    """The given fields with their last real word as the surname, as for a surname field of only credentials."""
    first_tokens, middle_tokens = _field_tokens(first), _field_tokens(middle)
    if middle_tokens:
        split = _surname_split(middle_tokens)
        if split is None:
            return None
        return [first, " ".join(split[0]) or None, " ".join(split[1])]
    split = _surname_split(first_tokens) if len(first_tokens) > 1 else None
    if split is None or not split[0]:
        return None
    return [split[0][0], " ".join(split[0][1:]) or None, " ".join(split[1])]


def strip_junk(first: str | None, middle: str | None, last: str | None):
    """``*Emmanuel``, ``Kim∗``, an ORCID link or iD, or an escaped entity (``K^|^ocirc;z^|^ocirc;``) in a name field.

    Footnote stars at the edges of a word and ORCID iDs in any form (a link, a bare iD, ``(ORCID: ...)``) are dropped. A
    surname field that held only an iD takes the last real word of the given fields, as a surname field of only
    credentials does (``Pedro`` / ``César`` / ``https://orcid.org/...`` -> ``("Pedro", None, "César")``). The name is left
    as written when no surname or no given name remains.
    """
    fields = (first, middle, last)
    if not _JUNK_HINT_RE.search(f"{first or ''} {middle or ''} {last or ''}"):
        return None
    cleaned = [_junk_free(value) for value in fields]
    if any(_ORCID_LABEL_RE.search(value or "") for value in fields):
        cleaned = [None if v is None else " ".join(w for w in v.split() if w not in {"iD", "iDs"}) for v in cleaned]
    if not any(ch.isalpha() for ch in cleaned[2] or "") and _ORCID_ID_RE.search(last or ""):
        cleaned = _surname_from_given_fields(cleaned[0], cleaned[1]) or cleaned
    given = f"{cleaned[0] or ''}{cleaned[1] or ''}"
    if not any(ch.isalpha() for ch in cleaned[2] or "") or not any(ch.isalpha() for ch in given):
        return None
    fixed = tuple(None if value is None else value or None for value in cleaned)
    if fixed == tuple(None if v is None else " ".join(v.split()) or None for v in fields):
        return None
    return fixed


_HAN_WORD_RE = re.compile(r"[(（]?[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+[)）]?")


def drop_han_beside_latin(first: str | None, middle: str | None, last: str | None):
    """``Luis`` / ``C. 子山`` / ``Ho 何`` -> ``("Luis", "C.", "Ho")``: the Han form written beside a romanized name.

    Han-only words are dropped when a Latin given name and a Latin surname remain; a first field left empty takes the
    middle field's first word. A name whose Latin words of three or more letters all split into Chinese, Korean or
    Cantonese syllables keeps its Han form, which the parser reads to tell the surname apart (``Su`` / ``苏`` /
    ``Zhang 张``, ``简月`` / ``Jian`` / ``Yue``).
    """
    if not any(_HAN_WORD_RE.fullmatch(word) for value in (first, middle, last) for word in (value or "").split()):
        return None
    kept = [[word for word in (value or "").split() if not _HAN_WORD_RE.fullmatch(word)] for value in (first, middle, last)]
    if not kept[0]:
        kept[0], kept[1] = kept[1][:1], kept[1][1:]
    if not all(any(_LATIN_LETTER_RE.search(word) for word in field) for field in (kept[0], kept[2])):
        return None
    if any(_HAN_LETTER_RE.search(word) for field in kept for word in field):
        return None
    latin = [part.casefold() for field in kept for word in field for part in re.split(r"[\s.\-]+", word) if len(part) > 2]  # noqa: PLR2004
    if all(_romanized_cjk_word(part) for part in latin):
        return None
    return tuple(" ".join(field) or None for field in kept)


def _cjk_syllable(part: str) -> bool:
    return (
        part in EAST_ASIAN_GIVEN_SYLLABLES
        or part in EAST_ASIAN_SURNAMES
        or part in VALID_CHINESE_RIMES
        or any(part[:i] in VALID_CHINESE_ONSETS and part[i:] in VALID_CHINESE_RIMES for i in range(1, len(part)))
    )


@lru_cache(maxsize=4096)
def _romanized_cjk_word(word: str, syllables: int = 3) -> bool:
    """A word that splits into at most three romanized Chinese, Korean or Cantonese syllables (``junjie``, ``chow``)."""
    if not word:
        return True
    return syllables > 0 and any(
        _cjk_syllable(word[:i]) and _romanized_cjk_word(word[i:], syllables - 1) for i in range(1, len(word) + 1)
    )


_HAN_LETTER_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")

# Indonesian academic degrees written S.Kom., M.Pd, S.Si, M.Eng.: a letter, a dot and a listed subject abbreviation.
_INDONESIAN_DEGREE_RE = re.compile(
    r"(?:S|M|D)\.(?:Kom|Si|Pd|Fis|Eng|Hum|Sos|Psi|Ked|Farm|Kep|Gz|Hut|Pt|Ag|Ds|Sn|Kes|Ak|Ph|Ikom|Stat|Mat|Par|Pi"
    r"|Hub|Kel|Tr|IP|TP|KM|AP|Ars|Pol|Ling|Th|Fil|Biomed|Kom\.?I)\.?,?",
)
# Pieces that sit in a degree run beside such a degree: the S. T. of S.T., bare degree capitals, titles, stray dots.
_DEGREE_RUN_PIECE_RE = re.compile(
    r"(?:[A-Z]\.?|ST|SE|SH|MT|MM|MH|MSi|MKom|SKom|SPd|MPd|SSi|St|Ir|Dr|Drs|Dra|Prof|Ph\.?D|Ll\.?M|M\.?Sc|B\.?Sc|MBA|"
    r"M\.\s?T|S\.\s?T|S\.\s?E|S\.\s?H|M\.\s?M|M\.\s?H|M\.\s?P|M\.\s?E|Hj|Drh|drh|Apt|Mm|Stp|STP|M\.?Des|M\.?Ed"
    r"|D\.?Eng|Dr\.?-?Ing|Ing|Sp\.?[A-Z][A-Za-z]*(?:-[A-Z])?)\.?,?|\.|,",
)
_TITLE_PREFIX_RE = re.compile(r"(?:Dr|Dra|Drs|Ir|Prof)\.?")
_SINGLE_LETTER_RE = re.compile(r"[A-Z]\.?")


def _degree_run_piece(token: str) -> bool:
    return bool(_INDONESIAN_DEGREE_RE.fullmatch(token) or _DEGREE_RUN_PIECE_RE.fullmatch(token))


def strip_indonesian_degrees(first: str | None, middle: str | None, last: str | None):  # noqa: C901, PLR0911
    """``S.Kom.`` / ``M.Eng.`` / ``Waskitho Wibisono`` -> ``("Waskitho", None, "Wibisono")``: Indonesian degrees in the
    given fields.

    A run of degree pieces at the start or the end of the given fields is dropped when it holds a listed Indonesian
    degree (``S.Kom.``, ``M.Pd``); a lone ``S. T.`` or ``St`` is someone's initials or name as often as a degree and
    never starts the cut. Titles opening the last field go too (``Dra. Ni Wayan Suniasih``). When no given word is
    left, the last field must hold the whole name: its first word becomes the given name and its last the surname.
    """
    given = f"{first or ''} {middle or ''}".split()
    if not any(_INDONESIAN_DEGREE_RE.fullmatch(token) for token in given):
        return None
    degrees = [i for i, token in enumerate(given) if _INDONESIAN_DEGREE_RE.fullmatch(token)]
    head = degrees[-1] + 1 if all(_degree_run_piece(t) for t in given[: degrees[-1] + 1]) else 0
    while head and head < len(given) and _degree_run_piece(given[head]) and not _SINGLE_LETTER_RE.fullmatch(given[head]):
        head += 1
    tail = degrees[0] if head == 0 and all(_degree_run_piece(t) for t in given[degrees[0] :]) else len(given)
    if head == 0 and tail == len(given):
        return None
    kept = given[head:tail]
    last_words = (last or "").split()
    while last_words and _TITLE_PREFIX_RE.fullmatch(last_words[0]):
        last_words.pop(0)
    if any(_degree_run_piece(t) and not _SINGLE_LETTER_RE.fullmatch(t) for t in kept) or not last_words:
        return None
    if not kept:
        ends = (last_words[0], last_words[-1])
        if len(last_words) < 2 or not all(_LETTER_SURNAME_RE.fullmatch(word) for word in ends):  # noqa: PLR2004
            return None
        if not all(map(_name_word, last_words[1:-1])):
            return None
        return last_words[0], " ".join(last_words[1:-1]) or None, last_words[-1]
    if not all(_name_word(t) and not _abbreviation_or_qualifier(t) and "." not in t.rstrip(".") for t in kept):
        return None
    if {t.casefold() for t in kept} & _NOT_NAME_BYLINE_WORDS:
        return None
    return kept[0], " ".join(kept[1:]) or None, " ".join(last_words)


# Indonesian byline words that are not names: supervisor, psychologist, lecturer registration numbers
_NOT_NAME_BYLINE_WORDS = frozenset({"pembimbing", "psikolog", "nidn", "nip", "dosen"})


_HYPHENATED_GENERATION_RE = re.compile(
    r"(?<=[^\W\d_]{3})[-\u2010\u2011](Neto|Filho|Júnior|Junior|Sobrinho|Netto)$",
    re.IGNORECASE,
)


def space_hyphenated_generation(first: str | None, middle: str | None, last: str | None):
    """``Walter`` / - / ``Faria-Neto`` -> ``("Walter", None, "Faria Neto")``: a Portuguese generation word hyphenated to the
    surname is written apart, so it goes to the suffix as it does when the source writes ``Faria Neto``. Not after a
    particle (``Del-Neto``) or in a surname-only source."""
    surname = (last or "").strip()
    match = _HYPHENATED_GENERATION_RE.search(surname)
    if match is None or (not (first or "").strip() and not (middle or "").strip()):
        return None
    head = re.split(r"[\s\-\u2010\u2011]", surname[: match.start()])[-1]
    if head.casefold() in _LAST_FIELD_PARTICLES:
        return None
    return first, middle, f"{surname[: match.start()]} {match.group(1)}"


_INITIALS_SURNAME_RE = re.compile(r"(?:[A-Z]\.?\s?){1,3}")
# one-letter Korean and Vietnamese surnames (O, U, E, Y, I) written alone in the surname field
_ONE_LETTER_SURNAMES = frozenset({"O", "U", "E", "Y", "I"})


def swap_initials_surname(first: str | None, middle: str | None, last: str | None):
    """``Kumar`` / - / ``P`` -> ``("P", None, "Kumar")``: a surname written in the given field and initials in the surname
    field (South Indian bylines).

    Only when the given field is one word the corpus role prior calls a surname (``Kumar``), the middle field is empty and
    the surname field is one to three initials. On ORCID people of this shape whose word is a corpus surname, 9 in 10
    register it as their family name; when the word is a given name (``Natrayan`` / - / ``L``), most register it as the
    given name, so those stay. A lone undotted O, U, E, Y or I is a Korean or Vietnamese surname and stays.
    """
    word, initials = (first or "").strip(), (last or "").strip()
    if (middle or "").strip() or " " in word or not _INITIALS_SURNAME_RE.fullmatch(initials):
        return None
    if initials in _ONE_LETTER_SURNAMES or not re.fullmatch(r"[A-Z][a-z]{2,}", word) or token_role(word) != "surname":
        return None
    return initials, None, word


FIELD_CLEANUPS: tuple[tuple[str, Callable], ...] = (
    ("latin_lookalike", latin_lookalikes),
    ("mojibake", fix_mojibake),
    ("junk", strip_junk),
    ("han_beside_latin", drop_han_beside_latin),
    ("indonesian_degree", strip_indonesian_degrees),
    ("hyphenated_generation", space_hyphenated_generation),
    ("initials_surname", swap_initials_surname),
)


def clean_source_fields(first: str | None, middle: str | None, last: str | None):
    """``(first, middle, last)`` after every field cleanup that applies, or ``None`` when none does."""
    fields = (first, middle, last)
    for _, cleanup in FIELD_CLEANUPS:
        fields = cleanup(*fields) or fields
    return None if fields == (first, middle, last) else fields


def repair_source_fields(first: str | None, middle: str | None, last: str | None, suffix: str | None):
    """``(rule name, first, middle)`` for the first rule that applies, or ``None``."""
    for name, rule in RULES:
        repaired = rule(first, middle, last, suffix)
        if repaired is not None:
            return name, *repaired
    return None


# Degrees and post-nominals that sources write as the whole surname field (``Sunita`` / ``Rao`` / ``Mbbs``).
LAST_FIELD_CREDENTIAL_WORDS = frozenset(
    {
        *("phd", "dphil", "pharmd", "psyd", "drph", "mphil", "mpharm", "bpharm", "mbbs", "mbchb", "mbbch", "msc", "bsc"),
        *("dsc", "mph", "msn", "bsn", "lcsw", "lmsw", "msw", "licsw", "rn", "lpn", "lvn", "cnm", "otr", "otrl"),
        *("facs", "facp", "facc", "facog", "faap", "facep", "fccp", "fccm", "fasn", "fesc", "fnla"),
        *("frcpc", "frcp", "frcs", "frcsc", "frcse", "frcsed", "frcsi", "frcpi", "frcpe", "frcpath", "fracp", "fracs"),
        *("franzcog", "frcog", "frcr", "frcpsych", "frcophth", "frcpch", "frcpa"),
        *("mrcp", "mrcs", "mrcpch", "mrcog", "mrcgp", "mrcpi", "mrcpsych"),
        *("dpt", "dnp", "rrt", "fnp", "fnpbc", "fnpc", "anp", "acnp", "agacnp", "pnp", "cpnp", "whnp", "cccslp"),
        *("crna", "ccrn", "dvm", "aprn", "cscs", "atc", "cdces", "dds", "dmd", "lpc", "lmft", "ibclc", "cnl", "ocn"),
        *("aocnp", "rdn", "ladc", "bcba", "pac", "cnp", "npc", "msed", "msph", "mhsa", "mhsc", "mpas", "dmsc", "otd"),
        *("ptd", "ccc", "cnor", "mpp", "lcpc"),
    },
)
# Never read as credentials in the surname field, and never made the surname: given names and surnames (Ma, Meng, Do,
# Md, Fahs, Fache, Fasa, Ches, Faan), initials (CNS) and name fragments (``C. de``) in the ORCID registry or the corpus.
LAST_FIELD_NAME_WORDS = frozenset(
    {
        *("ma", "meng", "do", "ba", "md", "mba", "pa", "ms", "bs", "jd", "mpa", "edd", "rd", "mt", "ld", "np", "mhs"),
        *("ed", "mfa", "fahs", "fache", "fasa", "ches", "faan", "cns", "cde"),
    },
)
# Only a name fragment in the surname field; a first field of it is still a credential.
_FIRST_FIELD_CREDENTIAL_KEYS = frozenset({"cde"})
# Titles other than these are left to the parser, which drops them; some are also given names (Hon).
_FIRST_FIELD_NOT_NAME_KEYS = (
    CREDENTIAL_WORDS | _CREDENTIAL_KEYS | _FIRST_FIELD_CREDENTIAL_KEYS | {"prof", "professor", "dr", "doctor"}
) - (LAST_FIELD_NAME_WORDS - _FIRST_FIELD_CREDENTIAL_KEYS)
_LAST_FIELD_SPLIT_RE = re.compile(r"[\s,;/]+")
_LAST_FIELD_KEY_RE = re.compile(r"[.:()\-]")
_LAST_FIELD_PARTICLES = frozenset(
    {"van", "von", "der", "den", "de", "del", "della", "da", "das", "dal", "di", "dos", "du", "la", "le", "ten", "ter"}
    | {"al", "el", "bin", "ben", "mac", "st", "dello", "degli", "do", "vander", "vande", "te"},
)
# A particle alone is not a surname (``Ali`` / ``Al`` / ``Phd``); Das is also a surname on its own.
_NOT_SURNAME_PARTICLES = (_LAST_FIELD_PARTICLES - {"das"}) | {"et"}
_SPELLED_LETTERS_RE = re.compile(r"[^\W\d_](?:[.\s]+[^\W\d_])*\.?")
# ``C.N.S``, ``R N``, ``D.N.P``: two or three letters spelled out are someone's initials as often as a credential.
_MAX_SPELLED_INITIALS = 3
_SHORT_CAPITALS = 3
_LAST_FIELD_SURNAME_RE = re.compile(r"[^\W\d_]{2,}(?:[-'\u2019][^\W\d_]+)*")


def _field_tokens(text: str | None) -> list[str]:
    return [token for token in _LAST_FIELD_SPLIT_RE.split((text or "").strip()) if token]


def _last_field_credential(token: str) -> bool:
    key = _LAST_FIELD_KEY_RE.sub("", token).casefold()
    if key in LAST_FIELD_NAME_WORDS or key not in LAST_FIELD_CREDENTIAL_WORDS:
        return False
    if key == "pac" and "-" not in token:  # Pac is a Polish, Korean and Turkish surname; PA-C keeps its hyphen
        return False
    if key == "rn" and token.isupper():  # an RN family name in the ORCID registry is mostly South Indian initials
        return False
    return not _spelled_initials(token)


def _spelled_initials(token: str) -> bool:
    return bool(_LETTER_INITIALS_RE.fullmatch(token)) and len(_LAST_FIELD_KEY_RE.sub("", token)) <= _MAX_SPELLED_INITIALS


def _only_credentials(last: str | None) -> bool:
    """Every word is a listed credential, or the words spell one together (``Mb Bs``) without being spelled initials."""
    tokens = _field_tokens(last)
    if not tokens:
        return False
    if all(map(_last_field_credential, tokens)):
        return True
    compact = re.sub(r"\W", "", last or "").casefold()
    spelled_initials = len(compact) <= _MAX_SPELLED_INITIALS and _SPELLED_LETTERS_RE.fullmatch((last or "").strip())
    return len(tokens) > 1 and compact in LAST_FIELD_CREDENTIAL_WORDS and not spelled_initials


def _first_field_credential(token: str) -> bool:
    if _spelled_initials(token):
        return False
    return _last_field_credential(token) or _LAST_FIELD_KEY_RE.sub("", token).casefold() in _FIRST_FIELD_NOT_NAME_KEYS


def _surname_split(tokens: list[str]) -> tuple[list[str], list[str]] | None:
    """``(given words, surname words)``: the last word with any lower-case particles before it, if it can be a surname."""
    surname, rest = tokens[-1:], tokens[:-1]
    while rest and rest[-1] in _LAST_FIELD_PARTICLES:
        surname.insert(0, rest.pop())
    core = surname[-1].rstrip(".")
    key = core.casefold()
    if (
        not _LAST_FIELD_SURNAME_RE.fullmatch(core)
        or key in LAST_FIELD_NAME_WORDS | _NOT_NAME_KEYS | LAST_FIELD_CREDENTIAL_WORDS | _NOT_SURNAME_PARTICLES
        or (core.isupper() and len(core) <= _SHORT_CAPITALS)
    ):
        return None
    return rest, surname


def strip_last_field_credentials(
    first: str | None,
    middle: str | None,
    last: str | None,
) -> tuple[str | None, str | None, str] | None:
    """``Sunita`` / ``Rao`` / ``Mbbs`` -> ``("Sunita", None, "Rao")``; ``None`` when it does not apply.

    The surname field must be only credentials; a trailing run of credentials at the end of the middle field goes with
    it (``Jessica`` / ``Reyes Bsn`` / ``Rn``). The last remaining word, with any lower-case particles before it, becomes
    the surname; it must be a word of two or more letters that is not a credential or a title. With an empty middle
    field the word comes from the first field (``Holly Powell`` / - / ``Cnm``) only when a given word stays there. A
    first field of only credentials or titles gives way to the middle field's first word (``Msn`` / ``Anne McAdam``).

    Left unchanged: a surname-first source (``Okafor,`` / ``O. Vincent.``), a first field of only particles (``De`` /
    ``Jong`` / ``Rn``), dotted initials before two or more middle words (``D.R.`` / ``Anand Kumar``: South Indian
    initials, after which the surname is unclear) and a credential tail the reviewed comma rule already reads (``PhD,
    MRCP``).
    """
    first_tokens = _field_tokens(first)
    if (
        not _only_credentials(last)
        or (first or "").strip().endswith(",")
        or reviewed_closed_comma_credential_tail_head(last) is not None
        or (first_tokens and all(token.casefold() in _LAST_FIELD_PARTICLES for token in first_tokens))
    ):
        return None
    middle_tokens = _field_tokens(middle)
    while middle_tokens and _last_field_credential(middle_tokens[-1]):
        middle_tokens.pop()
    if len(middle_tokens) > 1 and any(_spelled_initials(token) and len(token.replace(".", "")) > 1 for token in first_tokens):
        return None
    if middle_tokens:
        split = _surname_split(middle_tokens)
        keep_first = not (first_tokens and all(map(_first_field_credential, first_tokens)))
    else:
        while first_tokens and _last_field_credential(first_tokens[-1]):
            first_tokens.pop()
        split = _surname_split(first_tokens) if len(first_tokens) > 1 else None
        keep_first = False
    if split is None or not (keep_first or split[0]):
        return None
    rest, surname = split
    if keep_first:
        return first, " ".join(rest) or None, " ".join(surname)
    return rest[0], " ".join(rest[1:]) or None, " ".join(surname)

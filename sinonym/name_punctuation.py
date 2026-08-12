"""Shared Unicode punctuation policy for personal names.

The apostrophe oracle follows Unicode 17.0 ``Quotation_Mark`` entries for
single quotes plus the spacing modifier characters that Unicode's NamesList
cross-references to apostrophe, prime, or single-quote forms. Double quotes,
combining marks, invisible tag characters, and letters whose names merely
mention an apostrophe are intentionally excluded.

The structural Roman hyphen fold is deliberately narrower than Unicode
``Dash``. It contains semantic/compatibility hyphens plus metadata
substitutions supported by the August 2026 author-corpus review. A second,
post-preprocessing safety set keeps ambiguous separators visible to downstream
ethnicity gates without claiming that they are semantically hyphens. The
generic person normalizer's older Unicode-name oracle remains separate so
sharing this module does not change that path's deployed behavior.
"""

import re
from collections.abc import Mapping

APOSTROPHE_LIKE = frozenset(
    {
        "'",
        "\u02b9",  # modifier letter prime
        "\u02bb",  # modifier letter turned comma
        "\u02bc",  # modifier letter apostrophe
        "\u02bd",  # modifier letter reversed comma
        "\u02be",  # modifier letter right half ring
        "\u02bf",  # modifier letter left half ring
        "\u02c0",  # modifier letter glottal stop
        "\u02c1",  # modifier letter reversed glottal stop
        "\u02c8",  # modifier letter vertical line
        "\u02ca",  # modifier letter acute accent
        "\u02cb",  # modifier letter grave accent
        "\u02ee",  # modifier letter double apostrophe
        "\u0559",  # Armenian modifier letter left half ring
        "\u055a",  # Armenian apostrophe
        "\u05f3",  # Hebrew punctuation geresh
        "\u07f4",  # NKo high tone apostrophe
        "\u07f5",  # NKo low tone apostrophe
        "\u2018",  # left single quotation mark
        "\u2019",  # right single quotation mark
        "\u201a",  # single low-9 quotation mark
        "\u201b",  # single high-reversed-9 quotation mark
        "\u2032",  # prime
        "\u2035",  # reversed prime
        "\u2039",  # single left-pointing angle quotation mark
        "\u203a",  # single right-pointing angle quotation mark
        "\u275b",  # heavy single turned comma quotation mark ornament
        "\u275c",  # heavy single comma quotation mark ornament
        "\u275f",  # heavy low single comma quotation mark ornament
        "\u276e",  # heavy left-pointing angle quotation mark ornament
        "\u276f",  # heavy right-pointing angle quotation mark ornament
        "\ua78b",  # Latin capital letter saltillo
        "\ua78c",  # Latin small letter saltillo
        "\uff07",  # fullwidth apostrophe
        "`",  # grave accent used as an apostrophe
        "\uff40",  # fullwidth grave accent
        "\u00b4",  # acute accent used as an apostrophe
    },
)

_CANONICAL_ROMAN_HYPHENS = frozenset(
    {
        "-",
        "\u2010",  # hyphen
        "\u2011",  # non-breaking hyphen
        "\ufe63",  # small hyphen-minus
        "\uff0d",  # fullwidth hyphen-minus
    },
)

_REVIEWED_METADATA_HYPHENS = frozenset(
    {
        "\u2012",  # figure dash
        "\u2013",  # en dash
        "\u2014",  # em dash
        "\u2043",  # hyphen bullet
        "\u2212",  # minus sign
    },
)

ROMAN_HYPHEN_LIKE = _CANONICAL_ROMAN_HYPHENS | _REVIEWED_METADATA_HYPHENS

# These marks must not be promoted to structural ASCII parity. Folding them
# after preprocessing preserves the boundary evidence used by the ethnicity
# gates and prevents sep_pattern from turning them into spaces first.
_POST_PREPROCESSING_SAFETY_HYPHENS = frozenset(
    {
        "\u00ad",  # soft hyphen: boundary evidence in author metadata
        "\u2015",  # horizontal bar
        "\u2027",  # hyphenation point
        "\u208b",  # subscript minus
        "\ufe58",  # small em dash
    },
)

# All approved structural Roman hyphens are equivalent in comparison keys.
# Preserve the pre-review deletion behavior for these validation-only marks;
# U+2027 deliberately remains visible to direct token normalization.
_LEGACY_COMPARISON_ONLY_HYPHENS = frozenset("\u00ad\u2015\u208b\ufe58")
_NORMALIZATION_HYPHEN_DELETE = ROMAN_HYPHEN_LIKE | _LEGACY_COMPARISON_ONLY_HYPHENS
# Exact Unicode 14 expansion of the generic person's pre-consolidation
# category/name predicate, plus the three explicit legacy extras. Keeping it a
# table avoids a Unicode lookup and nested function call for every input char.
PERSON_HYPHEN_LIKE = frozenset(
    "-\u00ad\u058a\u05be\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015"
    "\u2027\u2043\u208b\u2212\u2e17\u2e1a\u2e3a\u2e3b\u2e40\u2e5d"
    "\u301c\u3030\u30a0\ufe31\ufe32\ufe58\ufe63\uff0d\U00010ead"
    "\u00b1\u02d7\u0320\u2052\u2796\u2a29\u2a2a\u2a2b\u2a2c\u2a3a\u2a41\U000e002d",
)


APOSTROPHE_FOLD_TRANSLATION = str.maketrans(dict.fromkeys(APOSTROPHE_LIKE, "'"))
HYPHEN_FOLD_TRANSLATION = str.maketrans(
    dict.fromkeys(ROMAN_HYPHEN_LIKE | _POST_PREPROCESSING_SAFETY_HYPHENS, "-"),
)
PERSON_JOINER_FOLD_TRANSLATION = APOSTROPHE_FOLD_TRANSLATION | str.maketrans(
    dict.fromkeys(PERSON_HYPHEN_LIKE, "-"),
)
NAME_JOINER_DELETE_TRANSLATION = str.maketrans(
    dict.fromkeys(APOSTROPHE_LIKE | _NORMALIZATION_HYPHEN_DELETE, None),
)

_TRANSLITERATION_APOSTROPHE_RE = re.compile(
    r"(?<![^\W\d_])(?P<prefix>[^\W\d_]+)\s*'\s*(?=(?P<tail>[^\W\d_]+)(?![^\W\d_]))",
    re.UNICODE,
)
_TRANSLITERATION_APOSTROPHE_PAIRS = frozenset(
    {
        ("cui", "e"),
        ("ma", "ayan"),
        ("o", "Brart"),
        ("o", "connor"),
        ("p", "eng"),
        ("sa", "di"),
        ("ts", "ai"),
    },
)
_DUTCH_T_SPACING_RE = re.compile(r"(?<=\S)(?:\s+'\s*|'\s+)t(?=\s+\S)", re.UNICODE)


def fold_internal_name_joiners(value: str, translation: Mapping[int, str | None]) -> str:
    """Fold configured joiners while removing paired outer delimiter runs.

    A lone leading hyphen or terminal transliteration apostrophe can carry name
    semantics, so an outer run is a delimiter only when the opposite boundary
    has a run that folds to the same ASCII mark.
    """
    first_text = len(value) - len(value.lstrip())
    last_text = len(value.rstrip()) - 1
    if first_text <= last_text:
        boundary = translation.get(ord(value[first_text]))
        if boundary is not None and boundary == translation.get(ord(value[last_text])):
            content_start = first_text
            while content_start <= last_text and translation.get(ord(value[content_start])) == boundary:
                content_start += 1
            content_end = last_text
            while content_end >= content_start and translation.get(ord(value[content_end])) == boundary:
                content_end -= 1
            value = f"{value[:first_text]} {value[content_start : content_end + 1]} {value[last_text + 1 :]}"
    return value.translate(translation)


def fold_spaced_transliteration_apostrophes(value: str) -> str:
    """Repair spacing around reviewed transliteration apostrophes.

    Only audited prefix/tail pairs are joined. Separated variants of the Dutch
    ``'t`` particle retain its token boundary. The caller must first fold
    supported apostrophe variants to ASCII.
    """

    def join(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        tail = match.group("tail")
        if (prefix.casefold(), tail) not in _TRANSLITERATION_APOSTROPHE_PAIRS:
            return match.group(0)
        return f"{prefix}'"

    value = _TRANSLITERATION_APOSTROPHE_RE.sub(join, value)
    return _DUTCH_T_SPACING_RE.sub(" 't", value)

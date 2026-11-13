"""The suffix as the source wrote it: generational suffixes keep their spelling, credentials and titles are dropped.

The parser reads ``Júnior``, ``Junior`` and ``Jr`` in a name all as ``Jr.``; the people who write them mostly register
the spelling they wrote, so the output suffix takes the source word back with only its capitals cleaned (``JÚNIOR`` ->
``Júnior``). A suffix field holding only degrees or titles (``PhD``, ``MD``, ``Dr``) is not a name suffix and is dropped.
Display only: block keys and S2AND ignore the suffix.
"""

from __future__ import annotations

import re

from sinonym.services.person_name_normalization import (
    _CREDENTIAL_KEYS,
    _EXACT_CASE_BOUNDARY_CREDENTIALS,
    _EXACT_CASE_TRAILING_CREDENTIALS,
    _EXPLICIT_ROMAN_SUFFIXES,
    _TITLE_KEYS,
    _TITLE_QUALIFIER_KEYS,
    _canonical_suffix_surface,
)
from sinonym.services.source_repairs import CREDENTIAL_WORDS, LAST_FIELD_CREDENTIAL_WORDS, SPELLED_CREDENTIAL_WORDS

_SUFFIX_FIELD_SPLIT_RE = re.compile(r"[\s,;]+")
_NAME_FIELD_SPLIT_RE = re.compile(r"[\s,;\-\u2010\u2011]+")
_SUFFIX_FIELD_CREDENTIAL_KEYS = (
    _CREDENTIAL_KEYS
    | CREDENTIAL_WORDS
    | SPELLED_CREDENTIAL_WORDS
    | LAST_FIELD_CREDENTIAL_WORDS
    | _TITLE_KEYS
    | _TITLE_QUALIFIER_KEYS
)
_EXACT_CASE_CREDENTIALS = _EXACT_CASE_BOUNDARY_CREDENTIALS | _EXACT_CASE_TRAILING_CREDENTIALS
_ABBREVIATED_SUFFIXES = frozenset({"jr.", "sr."})
# Degrees that are also common family names (Ma, Do, Ba): only their capital or dotted spelling counts as the degree.
_SURNAME_LIKE_DEGREES = frozenset({"ma", "do", "ba"})


def _generational_surface(token: str) -> str:
    """``JÚNIOR`` -> ``Júnior``, ``jr.`` -> ``Jr.``, ``Junior.`` -> ``Junior``, ``2ND`` -> ``2nd``, ``iii.`` -> ``III``;
    ``""`` if not a suffix. Only the abbreviations keep a written period."""
    text = token.strip(",;()")
    canonical = _canonical_suffix_surface(text, explicit=True)
    if not canonical:
        return ""
    if canonical in _EXPLICIT_ROMAN_SUFFIXES:
        return canonical
    if text[:1].isdigit():
        return text.lower()
    if text.casefold() not in _ABBREVIATED_SUFFIXES:
        text = text.rstrip(".")
    return text[:1].upper() + text[1:].lower()


def _is_credential(token: str) -> bool:
    text = token.strip(",;()")
    if text in _EXACT_CASE_CREDENTIALS:
        return True
    key = "".join(ch for ch in text.casefold() if ch.isalnum())
    if key in _SURNAME_LIKE_DEGREES and not (text.isupper() or "." in text):
        return False
    return bool(key) and key in _SUFFIX_FIELD_CREDENTIAL_KEYS


def clean_suffix_field(suffix: str) -> str | None:
    """The suffix field without its degrees and titles, generational words with cleaned capitals.

    ``Jr., PhD`` -> ``Jr.``; ``MD PhD`` -> ``None``; a field with any other word is returned unchanged (``(UK)``).
    """
    tokens = [token for token in _SUFFIX_FIELD_SPLIT_RE.split(suffix.strip()) if token.strip(",;()")]
    if not tokens:
        return suffix
    kept = []
    for token in tokens:
        surface = _generational_surface(token)
        if surface:
            kept.append(surface)
        elif not _is_credential(token):
            return suffix
    return " ".join(kept) or None


def source_suffix_spelling(name_fields: tuple[str | None, ...], suffix: str) -> str:
    """The source word a parsed generational suffix came from (``Júnior`` for ``Jr.``), else ``suffix`` unchanged."""
    surfaces = {
        _generational_surface(token)
        for field in name_fields
        for token in _NAME_FIELD_SPLIT_RE.split(field or "")
        if token and _canonical_suffix_surface(token.strip(",;()"), explicit=True) == suffix
    }
    surfaces.discard("")
    return surfaces.pop() if len(surfaces) == 1 else suffix


def _moved_surname_suffix(last: str | None, output_surname: str) -> str | None:
    """A generational word ending a multi-word surname field that the output surname no longer holds."""
    words = [token for token in _NAME_FIELD_SPLIT_RE.split(last or "") if token]
    surface = _generational_surface(words[-1]) if len(words) > 1 else ""
    output_words = _NAME_FIELD_SPLIT_RE.split(output_surname.casefold())
    if not surface or surface.casefold() in output_words:
        return None
    return surface


def display_suffix(
    source: tuple[str | None, str | None, str | None, str | None],
    output_surname: str,
    suffix: str | None,
) -> str | None:
    """The output suffix shown as the source wrote it; ``source`` is (first, middle, last, suffix) as given."""
    first, middle, last, source_suffix = source
    if not suffix or not suffix.strip():
        return suffix
    if source_suffix and suffix == source_suffix:
        cleaned = clean_suffix_field(source_suffix)
        return cleaned if cleaned is not None else _moved_surname_suffix(last, output_surname)
    return source_suffix_spelling((first, middle, last), suffix)

"""Dependency-free canonical normalization for personal names.

This module deliberately does not decide whether a name is Chinese and does not
call the Chinese parser.  It provides a small boundary-normalization contract
that detector and batch entry points can invoke after their existing routing
decisions have been made.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, replace
from enum import Enum

from sinonym.coretypes import CanonicalName, NameComponents


class PersonNameOutcome(str, Enum):
    """Typed classification returned by canonical name normalization."""

    PERSON = "person"
    NON_PERSON = "non_person"
    INVALID = "invalid"


class DropReason(str, Enum):
    """Reason that a source token was intentionally omitted."""

    TITLE = "title"
    CREDENTIAL = "credential"
    AFFILIATION = "affiliation"
    CONNECTOR = "connector"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class DroppedNameToken:
    """One source token omitted from the canonical name."""

    text: str
    source_role: str
    reason: DropReason


@dataclass(frozen=True)
class PersonNameNormalizationResult:
    """Typed outcome of one canonical name normalization request."""

    outcome: PersonNameOutcome
    canonical_name: CanonicalName | None = None
    reason: str | None = None
    dropped_tokens: tuple[DroppedNameToken, ...] = ()


@dataclass(frozen=True)
class _Token:
    text: str
    source_role: str
    position: int
    source_text: str | None = None


@dataclass(frozen=True)
class _DroppedToken:
    token: _Token
    reason: DropReason


_WHITESPACE_RE = re.compile(r"\s+")
_JOINER_SPACING_RE = re.compile(r"\s*([-'])\s*")
_DUPLICATE_APOSTROPHE_RE = re.compile(r"'{2,}")
_LEADING_STRAY_JOINER_RE = re.compile(r"^[-']\s+")
# Author-list connector "and" is whitespace-delimited ("First Last and First Last").
# Must NOT match "and" inside a hyphenated surname token (e.g. "Jon-And", "Strand"),
# so require real whitespace on both sides rather than a \b word boundary.
_WORD_AND_RE = re.compile(r"(?<=\s)and(?=\s)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\S+")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_INITIAL_RE = re.compile(r"([^\W\d_])\.", re.UNICODE)
_MULTI_INITIAL_RE = re.compile(r"(?:[^\W\d_]\.)+[^\W\d_]\.?", re.UNICODE)
_ABBREVIATED_TOKEN_RE = re.compile(r"(?:[^\W\d_]+[-'])*[^\W\d_]{1,3}\.", re.UNICODE)
_COMPOUND_INITIAL_RE = re.compile(r"[^\W\d_]\.-[^\W\d_]\.", re.UNICODE)
_LEADING_HYPHEN_INITIAL_RE = re.compile(r"^-([^\W\d_])\.$", re.UNICODE)
_FUSED_INITIAL_SURNAME_RE = re.compile(r"^([^\W\d_])\.(.+)$", re.UNICODE)
_LOWER_MARKER_INITIAL_RE = re.compile(r"^([a-z])([A-Z])\.$")
_PARENTHETICAL_DUPLICATE_RE = re.compile(r"^(\S+)\s+\(([^)]+)\)\s+(.+)$")
_ASCII_JOINER_TRANSLATION = str.maketrans({"`": "'"})
_PRE_NFKC_JOINER_TRANSLATION = str.maketrans({"\u00b4": "'"})

_APOSTROPHE_LIKE = frozenset(
    {
        "'",
        "\u02b9",  # modifier letter prime
        "\u02bb",  # modifier letter turned comma
        "\u02bc",  # modifier letter apostrophe
        "\u2018",  # left single quotation mark
        "\u2019",  # right single quotation mark
        "\u201a",  # single low-9 quotation mark
        "\u201b",  # single high-reversed-9 quotation mark
        "\u2032",  # prime
        "\u2035",  # reversed prime
        "\ua78b",  # Latin capital letter saltillo
        "\ua78c",  # Latin small letter saltillo
        "\uff07",  # fullwidth apostrophe
        "`",  # grave accent used as an apostrophe
        "\uff40",  # fullwidth grave accent
        "\u00b4",  # acute accent used as an apostrophe
    },
)
_DASH_LIKE_EXTRAS = frozenset({"\u2043", "\u2212", "\u208b"})

_TITLE_KEYS = frozenset(
    {
        "capt",
        "captain",
        "chaplain",
        "dame",
        "doctor",
        "dr",
        "father",
        "frau",
        "hon",
        "honorable",
        "lord",
        "miss",
        "mr",
        "mrs",
        "ms",
        "pastor",
        "prof",
        "professor",
        "rabbi",
        "rev",
        "reverend",
        "sir",
        "univprof",
    },
)
_TITLE_QUALIFIER_KEYS = frozenset({"hc", "habil", "honoraire", "med", "nat", "rer"})
_LEADING_NAME_ABBREVIATION_KEYS = frozenset({"md"})
_LONG_NAME_ABBREVIATION_KEYS = frozenset({"mohd", "most"})
_CREDENTIAL_KEYS = frozenset(
    {
        "ba",
        "bs",
        "bsc",
        "dds",
        "dmd",
        "do",
        "dphil",
        "dvm",
        "edd",
        "esq",
        "facp",
        "facr",
        "frcp",
        "jd",
        "llb",
        "llm",
        "ma",
        "mba",
        "md",
        "meng",
        "mph",
        "mpa",
        "ms",
        "msc",
        "pharmd",
        "phd",
        "psyd",
        "rn",
    },
)
_AMBIGUOUS_CREDENTIAL_KEYS = frozenset({"ba", "bs", "do", "edd", "jd", "ma", "mba", "md", "meng", "mpa", "ms", "rn"})
_MIXED_CASE_CREDENTIALS = {"meng": "MEng", "edd": "EdD"}
_FAMILY_PARTICLES = frozenset(
    {
        "al",
        "ap",
        "ben",
        "bin",
        "da",
        "das",
        "dal",
        "de",
        "del",
        "della",
        "den",
        "der",
        "di",
        "dos",
        "du",
        "el",
        "ibn",
        "la",
        "las",
        "le",
        "los",
        "st",
        "ten",
        "ter",
        "ud",
        "ur",
        "van",
        "von",
        "zu",
        "zum",
        "zur",
    },
)
_LOWERCASE_RELATIONAL_TOKENS = frozenset({"b.", "d.", "e", "kizi", "kyzy", "oglu", "o'g'li", "oğlu", "qizi"})
_STRONG_FAMILY_PARTICLE_SPANS = (
    ("de", "la"),
    ("de", "las"),
    ("de", "los"),
    ("van", "der"),
    ("von", "der"),
    ("da",),
    ("das",),
    ("dos",),
)
_ORGANIZATION_WORDS = frozenset(
    {
        "association",
        "center",
        "centre",
        "committee",
        "company",
        "consortium",
        "corporation",
        "department",
        "hospital",
        "inc",
        "institute",
        "laboratory",
        "ltd",
        "society",
        "team",
        "university",
    },
)
_STANDARD_SUFFIXES = {
    "jr": "Jr.",
    "junior": "Jr.",
    "sr": "Sr.",
    "senior": "Sr.",
    "2nd": "2nd",
    "3rd": "3rd",
    "4th": "4th",
    "5th": "5th",
    "6th": "6th",
}
# Spelled-out "Senior"/"Junior" are also common surnames ("Roxy Senior", "Peter A.
# Senior"); demote them to a suffix only when a real surname survives the removal.
_SURNAME_LIKE_SUFFIX_KEYS = frozenset({"senior", "junior"})
_RAW_ROMAN_SUFFIXES = frozenset({"II", "III", "IV", "VI", "VII", "VIII", "IX", "X"})
_EXPLICIT_ROMAN_SUFFIXES = _RAW_ROMAN_SUFFIXES | {"I", "V", "X"}
_CASE_INSENSITIVE_ROMAN_SUFFIXES = _RAW_ROMAN_SUFFIXES - {"II"}
_SIMPLE_TWO_TOKEN_POLICY_KEYS = frozenset(
    _TITLE_KEYS
    | _TITLE_QUALIFIER_KEYS
    | _CREDENTIAL_KEYS
    | _ORGANIZATION_WORDS
    | _STANDARD_SUFFIXES.keys()
    | {suffix.casefold() for suffix in _RAW_ROMAN_SUFFIXES}
    | {"and"},
)
_TWO_COMPONENTS = 2
_THREE_COMPONENTS = 3
_FOUR_COMPONENTS = 4
_MAX_SPACED_CREDENTIAL_TOKENS = 3
_MAX_DOTTED_ABBREVIATION_LETTERS = 3
_MIN_PARENTHESIZED_TOKEN_LENGTH = 3
_MC_PREFIX_LENGTH = 2


class PersonNameNormalizationService:
    """Normalize raw or structured personal names without Chinese routing."""

    def normalize_text(self, raw_name: str | None) -> PersonNameNormalizationResult:  # noqa: C901, PLR0911, PLR0912
        """Normalize one raw name string into semantic canonical components."""
        if not isinstance(raw_name, str):
            return self._invalid("name must be a string")

        simple_result = self._normalize_simple_two_token_text(raw_name)
        if simple_result is not None:
            return simple_result

        source_text = raw_name
        normalized_input, leading_markers = self._strip_leading_superscript_affiliation(raw_name)
        surface = self._normalize_surface(normalized_input)
        if not surface:
            return self._invalid("name is empty")
        if not any(character.isalpha() for character in surface):
            return self._invalid("name has no letters")

        dropped: list[_DroppedToken] = []
        if leading_markers:
            dropped.append(_DroppedToken(_Token(leading_markers, "", 0), DropReason.AFFILIATION))
        surface, affiliation_dropped = self._strip_trailing_affiliation_surface(surface)
        dropped.extend(affiliation_dropped)
        surface = self._collapse_parenthetical_duplicate_surface(surface)
        non_person_reason = self._non_person_reason(surface)
        if non_person_reason is not None:
            return self._non_person(non_person_reason)

        segment_matches = list(re.finditer(r"[^,]+", surface))
        segments = [self._tokens(match.group(), "", match.start()) for match in segment_matches]
        segments = [segment for segment in segments if segment]
        # Drop a comma segment that is only a dangling ``and`` connector
        # ("Susana ... Huerta, and" -> one real name), recording its lineage.
        kept_segments: list[list[_Token]] = []
        for segment in segments:
            if len(segment) == 1 and segment[0].text in {"and", "AND"}:
                dropped.append(_DroppedToken(segment[0], DropReason.CONNECTOR))
            else:
                kept_segments.append(segment)
        segments = kept_segments
        if not segments:
            return self._invalid("name has no usable tokens")
        if leading_markers:
            first = segments[0][0]
            segments[0][0] = replace(first, source_text=f"{leading_markers}{first.text}")

        suffix = ""
        suffix_token: _Token | None = None
        while len(segments) > 1:
            ambiguous_given_segment = (
                len(segments) == _TWO_COMPONENTS
                and len(segments[-1]) == 1
                and self._is_ambiguous_credential(segments[-1][0].text)
                and (
                    len(segments[0]) == 1
                    or any(self._particle_key(token.text) in _FAMILY_PARTICLES for token in segments[0][:-1])
                )
            )
            if ambiguous_given_segment:
                break
            boundary = self._consume_boundary_segment(segments[-1])
            if boundary is None:
                break
            segment_suffix, segment_suffix_token, segment_dropped = boundary
            segments.pop()
            dropped.extend(segment_dropped)
            if segment_suffix:
                if suffix:
                    return self._invalid("name has multiple suffixes", dropped)
                suffix = segment_suffix
                suffix_token = segment_suffix_token

        if len(segments) > _TWO_COMPONENTS:
            return self._non_person("multiple comma-separated names")
        if len(segments) == _TWO_COMPONENTS:
            return self._normalize_comma_name(
                source_text,
                segments[0],
                segments[1],
                suffix,
                suffix_token,
                dropped,
            )
        return self._normalize_regular_name(source_text, segments[0], suffix, suffix_token, dropped)

    def _normalize_simple_two_token_text(self, raw_name: str) -> PersonNameNormalizationResult | None:
        """Normalize policy-neutral two-token ASCII names without preprocessing."""
        if not raw_name.isascii() or raw_name != raw_name.strip():
            return None
        raw_tokens = raw_name.split(" ")
        if len(raw_tokens) != _TWO_COMPONENTS or any(not token.isalpha() for token in raw_tokens):
            return None
        if any(token.casefold() in _SIMPLE_TWO_TOKEN_POLICY_KEYS for token in raw_tokens):
            return None

        source_tokens = [
            _Token(raw_tokens[0], "", 0),
            _Token(raw_tokens[1], "", len(raw_tokens[0]) + 1),
        ]
        given = [replace(source_tokens[0], source_role="given")]
        surname = [replace(source_tokens[1], source_role="surname")]
        source = self._components(given, [], surname, [])
        return self._person(raw_name, source, given, [], surname, "", [])

    def normalize_components(  # noqa: C901, PLR0911
        self,
        *,
        first_name: str | None = None,
        middle_name: str | None = None,
        last_name: str | None = None,
        suffix: str | None = None,
    ) -> PersonNameNormalizationResult:
        """Normalize already structured first, middle, last, and suffix fields."""
        values = {
            "given": first_name,
            "middle": middle_name,
            "surname": last_name,
            "suffix": suffix,
        }
        if any(value is not None and not isinstance(value, str) for value in values.values()):
            return self._invalid("name components must be strings or null")

        stripped_values: dict[str, str] = {}
        leading_markers_by_role: dict[str, str] = {}
        for role, value in values.items():
            stripped_values[role], leading_markers_by_role[role] = self._strip_leading_superscript_affiliation(value or "")
        surfaces = {role: self._normalize_surface(value) for role, value in stripped_values.items()}
        source_surfaces = {
            role: f"{leading_markers_by_role[role]}{surface}" if surface else "" for role, surface in surfaces.items()
        }
        source_text = " ".join(surface for surface in source_surfaces.values() if surface)
        if not source_text:
            return self._invalid("name is empty")
        if not any(character.isalpha() for character in source_text):
            return self._invalid("name has no letters")

        non_person_reason = self._non_person_reason(source_text)
        if non_person_reason is not None:
            return self._non_person(non_person_reason)

        position = 0
        source_by_role: dict[str, list[_Token]] = {}
        for role in ("given", "middle", "surname", "suffix"):
            source_by_role[role] = self._tokens(surfaces[role], role, position)
            if leading_markers_by_role[role] and source_by_role[role]:
                first = source_by_role[role][0]
                source_by_role[role][0] = replace(
                    first,
                    source_text=f"{leading_markers_by_role[role]}{first.text}",
                )
            position += len(surfaces[role]) + 1
        source = self._components(
            source_by_role["given"],
            source_by_role["middle"],
            source_by_role["surname"],
            source_by_role["suffix"],
            order=self._structured_source_order(source_by_role),
        )

        dropped = [
            _DroppedToken(
                _Token(markers, role, source_by_role[role][0].position),
                DropReason.AFFILIATION,
            )
            for role, markers in leading_markers_by_role.items()
            if markers and source_by_role[role]
        ]
        middle_tokens = self._strip_structured_middle_surname_artifact(
            source_by_role["middle"],
            source_by_role["surname"],
            dropped,
        )
        name_tokens = [
            *source_by_role["given"],
            *middle_tokens,
            *source_by_role["surname"],
        ]
        name_tokens = self._repair_leading_marker_initial(name_tokens, dropped)
        name_tokens = self._join_separated_compound_initials(name_tokens)
        name_tokens = self._strip_leading_titles(name_tokens, dropped)
        name_tokens = self._strip_standalone_periods(name_tokens, dropped)
        name_tokens = self._strip_boundary_markers(name_tokens, dropped)

        canonical_suffix, _, explicit_dropped = self._consume_explicit_suffix(
            source_by_role["suffix"],
        )
        dropped.extend(explicit_dropped)
        name_tokens, boundary_suffix, _ = self._strip_trailing_boundaries(name_tokens, dropped)
        if boundary_suffix:
            if canonical_suffix:
                return self._invalid("name has multiple suffixes", dropped)
            canonical_suffix = boundary_suffix

        name_tokens = self._strip_attached_affiliations(name_tokens, dropped)
        invalid_reason = self._invalid_token_reason(name_tokens)
        if invalid_reason is not None:
            return self._invalid(invalid_reason, dropped)

        cleaned_by_role = {
            role: [token for token in name_tokens if token.source_role == role] for role in ("given", "middle", "surname")
        }
        given, middle, surname = self._repair_structured_roles(cleaned_by_role)
        if not given and not middle and not surname:
            return self._invalid("no personal-name tokens remain", dropped)

        return self._person(source_text, source, given, middle, surname, canonical_suffix, dropped)

    def _normalize_regular_name(
        self,
        source_text: str,
        tokens: list[_Token],
        suffix: str,
        suffix_token: _Token | None,
        dropped: list[_DroppedToken],
    ) -> PersonNameNormalizationResult:
        original_tokens = list(tokens)
        tokens = self._repair_leading_marker_initial(tokens, dropped)
        tokens = self._join_separated_compound_initials(tokens)
        tokens = self._strip_leading_titles(tokens, dropped)
        tokens = self._strip_dangling_and(tokens, dropped)
        tokens = self._strip_standalone_periods(tokens, dropped)
        tokens = self._strip_boundary_markers(tokens, dropped)
        tokens, boundary_suffix, boundary_suffix_token = self._strip_trailing_boundaries(tokens, dropped)
        if boundary_suffix:
            if suffix:
                return self._invalid("name has multiple suffixes", dropped)
            suffix = boundary_suffix
            suffix_token = boundary_suffix_token

        tokens = self._strip_attached_affiliations(tokens, dropped)
        invalid_reason = self._invalid_token_reason(tokens)
        if invalid_reason is not None:
            return self._invalid(invalid_reason, dropped)
        if not tokens:
            return self._invalid("no personal-name tokens remain", dropped)

        packed_surname_first = self._is_packed_surname_first_initials(tokens)
        given, middle, surname = self._infer_regular_roles(tokens)
        assigned = [*given, *middle, *surname]
        assigned_by_position = {token.position: token for token in assigned}
        prefix_source = self._regular_prefix_source(
            original_tokens,
            assigned_by_position,
            tokens[0],
            dropped,
        )
        credential_source = [
            replace(item.token, source_role="suffix") for item in dropped if item.reason is DropReason.CREDENTIAL
        ]
        source_suffix = [*([suffix_token] if suffix_token is not None else []), *credential_source]
        source_order = None
        if packed_surname_first and not prefix_source and not source_suffix:
            source_order = ("surname", "given", *("middle" for _token in middle))
        source = self._components(
            [*prefix_source, *given],
            middle,
            surname,
            source_suffix,
            order=source_order,
        )
        return self._person(source_text, source, given, middle, surname, suffix, dropped)

    @staticmethod
    def _regular_prefix_source(
        original_tokens: list[_Token],
        assigned_by_position: dict[int, _Token],
        first_survivor: _Token,
        dropped: list[_DroppedToken],
    ) -> list[_Token]:
        """Rebuild raw prefix lineage without duplicating an attached-title remainder."""
        prefix: list[_Token] = []
        for token in original_tokens:
            if token.position >= first_survivor.position or token.position in assigned_by_position:
                continue
            if token.position + len(token.text) <= first_survivor.position:
                prefix.append(replace(token, source_role="given"))
                continue
            prefix.extend(
                replace(item.token, source_role="given")
                for item in dropped
                if item.reason is DropReason.TITLE and item.token.position == token.position
            )
        return prefix

    def _normalize_comma_name(  # noqa: PLR0913
        self,
        source_text: str,
        family_tokens: list[_Token],
        given_tokens: list[_Token],
        suffix: str,
        suffix_token: _Token | None,
        dropped: list[_DroppedToken],
    ) -> PersonNameNormalizationResult:
        family_tokens = [replace(token, source_role="surname") for token in family_tokens]
        given_tokens = [replace(token, source_role="given") for token in given_tokens]
        family_tokens = self._strip_leading_titles(family_tokens, dropped, preserve_ambiguous_credentials=True)
        given_tokens = self._strip_leading_titles(given_tokens, dropped, preserve_ambiguous_credentials=True)
        family_tokens = self._strip_dangling_and(family_tokens, dropped)
        given_tokens = self._strip_dangling_and(given_tokens, dropped)
        family_tokens = self._strip_standalone_periods(family_tokens, dropped)
        given_tokens = self._strip_standalone_periods(given_tokens, dropped)
        family_tokens = self._strip_boundary_markers(family_tokens, dropped)
        given_tokens = self._strip_boundary_markers(given_tokens, dropped)
        given_tokens, boundary_suffix, boundary_suffix_token = self._strip_trailing_boundaries(
            given_tokens,
            dropped,
            has_external_name_context=bool(family_tokens),
        )
        if boundary_suffix:
            if suffix:
                return self._invalid("name has multiple suffixes", dropped)
            suffix = boundary_suffix
            suffix_token = boundary_suffix_token

        family_tokens = self._strip_attached_affiliations(family_tokens, dropped)
        given_tokens = self._strip_attached_affiliations(given_tokens, dropped)
        invalid_reason = self._invalid_token_reason([*family_tokens, *given_tokens])
        if invalid_reason is not None:
            return self._invalid(invalid_reason, dropped)
        if not family_tokens or not given_tokens:
            return self._invalid("comma form requires family and given tokens", dropped)
        if self._looks_like_two_complete_names(family_tokens, given_tokens):
            return self._non_person("comma separates two complete names")

        given = [replace(given_tokens[0], source_role="given")]
        middle = [replace(token, source_role="middle") for token in given_tokens[1:]]
        surname = [replace(token, source_role="surname") for token in family_tokens]
        source_suffix = [suffix_token] if suffix_token is not None else []
        source = self._components(
            given,
            middle,
            surname,
            source_suffix,
            order=tuple(["surname"] * len(surname) + ["given"] + ["middle"] * len(middle) + ["suffix"] * len(source_suffix)),
        )
        return self._person(source_text, source, given, middle, surname, suffix, dropped)

    def _person(  # noqa: PLR0913
        self,
        source_text: str,
        source: NameComponents,
        given: list[_Token],
        middle: list[_Token],
        surname: list[_Token],
        suffix: str,
        dropped: list[_DroppedToken],
    ) -> PersonNameNormalizationResult:
        normalized_given = self._canonical_tokens(given, "given")
        normalized_middle = self._canonical_tokens(middle, "middle")
        normalized_surname = self._canonical_tokens(surname, "surname")
        normalized_suffix = tuple([suffix] if suffix else [])
        normalized = NameComponents(
            given_name=" ".join(normalized_given),
            middle_name=" ".join(normalized_middle),
            surname=" ".join(normalized_surname),
            suffix=suffix,
            given_tokens=normalized_given,
            middle_tokens=normalized_middle,
            surname_tokens=normalized_surname,
            suffix_tokens=normalized_suffix,
            order=tuple(
                ["given"] * len(normalized_given)
                + ["middle"] * len(normalized_middle)
                + ["surname"] * len(normalized_surname)
                + ["suffix"] * len(normalized_suffix),
            ),
        )
        text = " ".join(
            component
            for component in (normalized.given_name, normalized.middle_name, normalized.surname, normalized.suffix)
            if component
        )
        if not text:
            return self._invalid("no personal-name tokens remain", dropped)

        canonical_name = CanonicalName(source_text=source_text, text=text, source=source, normalized=normalized)
        dropped = self._infer_dropped_roles(dropped, [*given, *middle, *surname])
        return PersonNameNormalizationResult(
            outcome=PersonNameOutcome.PERSON,
            canonical_name=canonical_name,
            dropped_tokens=self._public_dropped(dropped),
        )

    @staticmethod
    def _normalize_surface(value: str) -> str:
        if "&" in value:
            # Decode HTML entities so encoded diacritics/apostrophes are recovered
            # ("Martin G&#x00F6;tz" -> "Martin Götz", "D&#39;Arcy" -> "D'Arcy") and a
            # literal "&amp;" collapses to "&" for the downstream separator logic.
            value = html.unescape(value)
        if value.isascii():
            normalized = value.translate(_ASCII_JOINER_TRANSLATION)
        else:
            normalized = value.translate(_PRE_NFKC_JOINER_TRANSLATION)
            normalized = unicodedata.normalize("NFKC", normalized)
            normalized = "".join(PersonNameNormalizationService._normalize_joiner(character) for character in normalized)
            normalized = unicodedata.normalize("NFC", normalized)
        normalized = _WHITESPACE_RE.sub(" ", normalized)
        normalized = _LEADING_STRAY_JOINER_RE.sub("", normalized)
        normalized = _JOINER_SPACING_RE.sub(r"\1", normalized)
        normalized = _DUPLICATE_APOSTROPHE_RE.sub("'", normalized)
        return normalized.strip(" \t\r\n,")

    @staticmethod
    def _strip_leading_superscript_affiliation(value: str) -> tuple[str, str]:
        """Strip fused leading superscript digits while returning their lineage."""
        marker_end = 0
        while marker_end < len(value):
            character = value[marker_end]
            if "SUPERSCRIPT" not in unicodedata.name(character, "") or not character.isdigit():
                break
            marker_end += 1
        if marker_end == 0 or marker_end >= len(value) or not value[marker_end].isalpha():
            return value, ""
        return value[marker_end:], value[:marker_end]

    def _strip_trailing_affiliation_surface(
        self,
        surface: str,
    ) -> tuple[str, list[_DroppedToken]]:
        """Remove an organization phrase after at least two name tokens."""
        tokens = self._tokens(surface, "suffix", 0)
        for index, token in enumerate(tokens):
            if index < _TWO_COMPONENTS or self._compact_key(token.text) not in _ORGANIZATION_WORDS:
                continue
            dropped = [_DroppedToken(item, DropReason.AFFILIATION) for item in tokens[index:]]
            return surface[: token.position].rstrip(" ,"), dropped
        return surface, []

    def _collapse_parenthetical_duplicate_surface(self, surface: str) -> str:
        """Collapse ``Alan (Alan B.) Cantor``-style duplicate given forms."""
        match = _PARENTHETICAL_DUPLICATE_RE.fullmatch(surface)
        if match is None:
            return surface
        outer_given, parenthetical, remainder = match.groups()
        parenthetical_tokens = self._tokens(parenthetical, "", 0)
        if not parenthetical_tokens:
            return surface
        if self._compact_key(parenthetical_tokens[0].text) != self._compact_key(outer_given):
            return surface
        return f"{parenthetical} {remainder}"

    @staticmethod
    def _normalize_joiner(character: str) -> str:
        character_name = unicodedata.name(character, "")
        is_single_quote = "SINGLE" in character_name and "QUOTATION MARK" in character_name
        if character in _APOSTROPHE_LIKE or "APOSTROPHE" in character_name or is_single_quote:
            return "'"
        if (
            unicodedata.category(character) == "Pd"
            or character in _DASH_LIKE_EXTRAS
            or "HYPHEN" in character_name
            or "MINUS SIGN" in character_name
        ):
            return "-"
        return character

    @staticmethod
    def _tokens(value: str, source_role: str, offset: int) -> list[_Token]:
        return [_Token(match.group(), source_role, offset + match.start()) for match in _TOKEN_RE.finditer(value.strip())]

    @staticmethod
    def _compact_key(token: str) -> str:
        if token.isalnum():
            return token.casefold()
        return "".join(character.casefold() for character in token if character.isalnum())

    def _strip_leading_titles(
        self,
        tokens: list[_Token],
        dropped: list[_DroppedToken],
        *,
        preserve_ambiguous_credentials: bool = False,
    ) -> list[_Token]:
        remaining = list(tokens)
        if self._has_leading_et_al_contamination(remaining):
            dropped.extend(_DroppedToken(token, DropReason.CONNECTOR) for token in remaining[:2])
            remaining = remaining[2:]
        stripped_title = False
        while remaining:
            token = remaining[0]
            attached_title = self._split_attached_leading_title(token)
            if attached_title is not None:
                title, remainder = attached_title
                dropped.append(_DroppedToken(title, DropReason.TITLE))
                remaining[0] = remainder
                stripped_title = True
                continue
            key = self._compact_key(token.text)
            if token.text == "AND":
                dropped.append(_DroppedToken(token, DropReason.CONNECTOR))
                remaining.pop(0)
                continue
            multi_initial = bool(_MULTI_INITIAL_RE.fullmatch(token.text))
            prefixed_academic_title = (
                token.text == "PD" and len(remaining) > 1 and self._compact_key(remaining[1].text) in _TITLE_KEYS
            )
            ambiguous_name_token = self._is_ambiguous_credential(token.text) and (
                preserve_ambiguous_credentials or len(remaining) == _TWO_COMPONENTS
            )
            # An all-caps token that reads as initials ("MS", "M-S.") must not be dropped
            # as the honorific "Ms": all-caps = initials, Title-case "Ms"/"Ms." = honorific.
            title_is_really_initials = ambiguous_name_token and token.text.isupper()
            if (
                (key in _TITLE_KEYS and not multi_initial and not title_is_really_initials)
                or (stripped_title and key in _TITLE_QUALIFIER_KEYS)
                or prefixed_academic_title
            ):
                dropped.append(_DroppedToken(token, DropReason.TITLE))
                remaining.pop(0)
                stripped_title = True
                continue
            leading_name_abbreviation = (
                key in _LEADING_NAME_ABBREVIATION_KEYS and token.text.endswith(".") and len(remaining) >= _TWO_COMPONENTS
            ) or self._is_ma_given_abbreviation(remaining)
            if self._is_credential(token.text) and not ambiguous_name_token and not leading_name_abbreviation:
                dropped.append(_DroppedToken(token, DropReason.CREDENTIAL))
                remaining.pop(0)
                continue
            break
        return remaining

    @staticmethod
    def _has_leading_et_al_contamination(tokens: list[_Token]) -> bool:
        """Match only a leading citation marker followed by a two-token name."""
        return bool(
            len(tokens) >= _FOUR_COMPONENTS and tokens[0].text.casefold() == "et" and tokens[1].text.casefold() == "al.",
        )

    def _split_attached_leading_title(self, token: _Token) -> tuple[_Token, _Token] | None:
        """Split ``Dr.Name`` only when the attached prefix is a known title."""
        prefix, separator, remainder = token.text.partition(".")
        if not separator or not remainder or self._compact_key(prefix) not in _TITLE_KEYS:
            return None
        if not any(character.isalpha() for character in remainder):
            return None
        title = replace(token, text=f"{prefix}.")
        name = replace(token, text=remainder, position=token.position + len(prefix) + 1)
        return title, name

    def _strip_trailing_boundaries(
        self,
        tokens: list[_Token],
        dropped: list[_DroppedToken],
        *,
        has_external_name_context: bool = False,
    ) -> tuple[list[_Token], str, _Token | None]:
        remaining = list(tokens)
        suffix = ""
        suffix_token: _Token | None = None
        while len(remaining) >= _FOUR_COMPONENTS:
            credential_width = self._trailing_spaced_credential_width(remaining)
            if not credential_width:
                break
            dropped.extend(_DroppedToken(token, DropReason.CREDENTIAL) for token in remaining[-credential_width:])
            del remaining[-credential_width:]
        while remaining:
            token = remaining[-1]
            ambiguous_name_token = self._is_ambiguous_credential(token.text) and (
                (not has_external_name_context and len(remaining) == _TWO_COMPONENTS)
                or (has_external_name_context and len(remaining) == 1)
            )
            if self._is_credential(token.text) and not ambiguous_name_token:
                dropped.append(_DroppedToken(token, DropReason.CREDENTIAL))
                remaining.pop()
                continue
            if token.text.strip(".,").isdigit():
                dropped.append(_DroppedToken(token, DropReason.AFFILIATION))
                remaining.pop()
                continue
            candidate = self._canonical_suffix(token.text, explicit=False)
            has_complete_name = len(remaining) > _TWO_COMPONENTS or has_external_name_context
            is_roman = candidate in _RAW_ROMAN_SUFFIXES
            surname_like = self._compact_key(token.text) in _SURNAME_LIKE_SUFFIX_KEYS
            # "Senior"/"Junior" is a suffix only if a surname survives its removal:
            # at least two non-initial tokens must precede it (a given AND a surname),
            # or an external name context already supplies the surname.
            surname_like_ok = (
                has_external_name_context
                or sum(1 for other in remaining[:-1] if not self._is_initial(other.text)) >= _TWO_COMPONENTS
            )
            if (
                candidate
                and not suffix
                and (not is_roman or has_complete_name)
                and (not surname_like or surname_like_ok)
            ):
                suffix = candidate
                suffix_token = replace(token, source_role="suffix")
                remaining.pop()
                continue
            break
        return remaining, suffix, suffix_token

    def _consume_boundary_segment(
        self,
        tokens: list[_Token],
    ) -> tuple[str, _Token | None, list[_DroppedToken]] | None:
        suffix = ""
        suffix_token: _Token | None = None
        dropped: list[_DroppedToken] = []
        if self._is_spaced_credential_group(tokens):
            dropped.extend(_DroppedToken(replace(token, source_role="suffix"), DropReason.CREDENTIAL) for token in tokens)
            return suffix, suffix_token, dropped
        for token in tokens:
            if self._is_credential(token.text, explicit_context=True):
                dropped.append(_DroppedToken(replace(token, source_role="suffix"), DropReason.CREDENTIAL))
                continue
            if token.text.strip(".,").isdigit():
                dropped.append(_DroppedToken(replace(token, source_role="suffix"), DropReason.AFFILIATION))
                continue
            candidate = self._canonical_suffix(token.text, explicit=False)
            if candidate and not suffix:
                suffix = candidate
                suffix_token = replace(token, source_role="suffix")
                continue
            return None
        return suffix, suffix_token, dropped

    def _consume_explicit_suffix(  # noqa: PLR0911
        self,
        tokens: list[_Token],
    ) -> tuple[str, _Token | None, list[_DroppedToken]]:
        if not tokens:
            return "", None, []
        if len(tokens) > 1:
            boundary = self._consume_boundary_segment(tokens)
            if boundary is not None:
                return boundary
            return " ".join(self._canonicalize_token(token.text, "suffix") for token in tokens), tokens[0], []

        token = tokens[0]
        if self._is_credential(token.text, explicit_context=True):
            return "", None, [_DroppedToken(token, DropReason.CREDENTIAL)]
        if token.text.strip(".,").isdigit():
            return "", None, [_DroppedToken(token, DropReason.AFFILIATION)]
        canonical = self._canonical_suffix(token.text, explicit=True)
        if canonical:
            return canonical, token, []
        return self._canonicalize_token(token.text, "suffix"), token, []

    def _strip_attached_affiliations(
        self,
        tokens: list[_Token],
        dropped: list[_DroppedToken],
    ) -> list[_Token]:
        cleaned: list[_Token] = []
        for token in tokens:
            match = _TRAILING_DIGITS_RE.search(token.text)
            if match is None or not any(character.isalpha() for character in token.text[: match.start()]):
                cleaned.append(token)
                continue
            cleaned.append(replace(token, text=token.text[: match.start()]))
            digit_token = _Token(match.group(), token.source_role, token.position + match.start())
            dropped.append(_DroppedToken(digit_token, DropReason.AFFILIATION))
        return cleaned

    @staticmethod
    def _strip_standalone_periods(
        tokens: list[_Token],
        dropped: list[_DroppedToken],
    ) -> list[_Token]:
        """Drop standalone full stops while retaining their source lineage."""
        remaining = []
        for token in tokens:
            if token.text == ".":
                dropped.append(_DroppedToken(token, DropReason.CONNECTOR))
            else:
                remaining.append(token)
        return remaining

    @staticmethod
    def _strip_dangling_and(
        tokens: list[_Token],
        dropped: list[_DroppedToken],
    ) -> list[_Token]:
        """Drop a leading/trailing lowercase ``and`` connector from a truncated author list.

        A dangling lowercase ``and`` (``Alexander Campbell and``, ``and Ariel Feldman``)
        or all-caps ``AND`` (``W. L. HAFLEY AND``) is an author-list fragment; stripping
        it recovers the one real name. Title-case ``And`` is KEPT (real surname, e.g.
        ``Metin And``), and a hyphenated ``Jon-And`` is one token, never matched. A MID
        ``and`` between two complete names is handled earlier by ``_non_person_reason``
        (rejected as two people).
        """
        connectors = {"and", "AND"}
        remaining = list(tokens)
        while len(remaining) > 1 and remaining[0].text in connectors:
            dropped.append(_DroppedToken(remaining.pop(0), DropReason.CONNECTOR))
        while len(remaining) > 1 and remaining[-1].text in connectors:
            dropped.append(_DroppedToken(remaining.pop(), DropReason.CONNECTOR))
        return remaining

    @staticmethod
    def _strip_boundary_markers(
        tokens: list[_Token],
        dropped: list[_DroppedToken],
    ) -> list[_Token]:
        """Drop boundary-only punctuation/symbol tokens such as author footnote daggers."""
        remaining = list(tokens)
        while remaining and not any(character.isalnum() for character in remaining[0].text):
            dropped.append(_DroppedToken(remaining.pop(0), DropReason.CONNECTOR))
        while remaining and not any(character.isalnum() for character in remaining[-1].text):
            dropped.append(_DroppedToken(remaining.pop(), DropReason.CONNECTOR))
        return remaining

    @staticmethod
    def _strip_structured_middle_surname_artifact(
        middle: list[_Token],
        surname: list[_Token],
        dropped: list[_DroppedToken],
    ) -> list[_Token]:
        """Remove an exact duplicated surname plus one lowercase source marker."""
        marker_width = 1
        artifact_width = len(surname) + marker_width
        if not surname or len(middle) <= artifact_width:
            return list(middle)
        marker = middle[-1].text
        if len(marker) != marker_width or not marker.isalpha() or not marker.islower():
            return list(middle)
        duplicate = middle[-artifact_width:-marker_width]
        if tuple(token.text for token in duplicate) != tuple(token.text for token in surname):
            return list(middle)
        dropped.extend(_DroppedToken(token, DropReason.DUPLICATE) for token in duplicate)
        dropped.append(_DroppedToken(middle[-1], DropReason.CONNECTOR))
        return list(middle[:-artifact_width])

    def _repair_leading_marker_initial(
        self,
        tokens: list[_Token],
        dropped: list[_DroppedToken],
    ) -> list[_Token]:
        """Split ``tE. L. Surname`` only at its fully constrained leading boundary."""
        if (
            len(tokens) != _THREE_COMPONENTS
            or not self._is_initial(tokens[1].text)
            or not self._is_full_name_token(tokens[2].text)
        ):
            return list(tokens)
        match = _LOWER_MARKER_INITIAL_RE.fullmatch(tokens[0].text)
        if match is None:
            return list(tokens)
        marker, initial = match.groups()
        first = tokens[0]
        marker_token = replace(first, text=marker, source_text=None)
        dropped.append(_DroppedToken(marker_token, DropReason.CONNECTOR))
        repaired = replace(
            first,
            text=f"{initial}.",
            position=first.position + len(marker),
            source_text=first.source_text or first.text,
        )
        return [repaired, *tokens[1:]]

    def _join_separated_compound_initials(self, tokens: list[_Token]) -> list[_Token]:
        """Join ``H. -J. Surname`` only when both visible pieces are initials."""
        if len(tokens) < _THREE_COMPONENTS or not self._is_initial(tokens[0].text):
            return list(tokens)
        trailing_initial = _LEADING_HYPHEN_INITIAL_RE.fullmatch(tokens[1].text)
        if trailing_initial is None or not self._is_full_name_token(tokens[-1].text):
            return list(tokens)
        combined = replace(tokens[0], text=f"{tokens[0].text}-{trailing_initial.group(1)}.")
        return [combined, *tokens[2:]]

    def _is_ma_given_abbreviation(self, tokens: list[_Token]) -> bool:
        """Recognize a leading María abbreviation ``Ma.`` before a real given name.

        Mixed-case dotted ``Ma.`` (not the all-caps ``MA`` degree) followed by any
        full name token is the Filipino/Spanish given ``María`` ("Ma. Mercedes T.
        Rodrigo", "Ma. Lucila Lapar"), so it must be kept, not dropped as a credential.
        """
        return bool(
            len(tokens) >= _TWO_COMPONENTS
            and tokens[0].text == "Ma."
            and any(self._is_full_name_token(token.text) for token in tokens[1:]),
        )

    def _is_full_name_token(self, token: str) -> bool:
        """Return whether a token is a full name rather than an initial."""
        cleaned = self._clean_name_token(token)
        return bool(
            len(self._compact_key(cleaned)) > 1
            and not self._is_initial(cleaned)
            and any(character.isalpha() for character in cleaned)
            and all(self._allowed_name_character(character) for character in cleaned),
        )

    def _strong_particle_width(self, tokens: list[_Token], start: int) -> int:
        keys = tuple(self._particle_key(token.text) for token in tokens)
        for span in _STRONG_FAMILY_PARTICLE_SPANS:
            end = start + len(span)
            if keys[start:end] == span and end < len(tokens) and all(token.text.islower() for token in tokens[start:end]):
                return len(span)
        return 0

    def _find_strong_particle_span(self, tokens: list[_Token], *, start: int = 1) -> tuple[int, int] | None:
        for index in range(start, len(tokens)):
            if width := self._strong_particle_width(tokens, index):
                return index, width
        return None

    def _particle_surname_first_roles(
        self,
        tokens: list[_Token],
    ) -> tuple[list[_Token], list[_Token], list[_Token]] | None:
        """Parse ``Carvalho da Silva Roberto José`` behind an exact strong span."""
        if len(tokens) < _FOUR_COMPONENTS or not self._is_full_name_token(tokens[0].text):
            return None
        particle_width = self._strong_particle_width(tokens, 1)
        if not particle_width:
            return None
        family_end = _TWO_COMPONENTS + particle_width
        trailing = tokens[family_end:]
        if len(trailing) != _TWO_COMPONENTS or not all(self._is_full_name_token(token.text) for token in trailing):
            return None
        return (
            [replace(trailing[0], source_role="given")],
            [replace(trailing[1], source_role="middle")],
            [replace(token, source_role="surname") for token in tokens[:family_end]],
        )

    def _is_packed_surname_first_initials(self, tokens: list[_Token]) -> bool:
        return bool(
            len(tokens) >= _THREE_COMPONENTS
            and self._is_full_name_token(tokens[0].text)
            and all(token.text.endswith(".") and self._is_initial(token.text) for token in tokens[1:]),
        )

    def _infer_regular_roles(self, tokens: list[_Token]) -> tuple[list[_Token], list[_Token], list[_Token]]:
        if len(tokens) == 1:
            return [replace(tokens[0], source_role="given")], [], []

        if particle_roles := self._particle_surname_first_roles(tokens):
            return particle_roles

        if self._is_packed_surname_first_initials(tokens):
            return (
                [replace(tokens[1], source_role="given")],
                [replace(token, source_role="middle") for token in tokens[2:]],
                [replace(tokens[0], source_role="surname")],
            )

        surname_start = len(tokens) - 1
        strong_particle = self._find_strong_particle_span(tokens)
        if strong_particle is not None and (strong_particle[0] > _TWO_COMPONENTS or self._is_initial(tokens[0].text)):
            surname_start = strong_particle[0] - 1
        elif particle_positions := [
            index for index, token in enumerate(tokens[1:-1], start=1) if self._particle_key(token.text) in _FAMILY_PARTICLES
        ]:
            surname_start = particle_positions[0]
        elif len(tokens) == _FOUR_COMPONENTS and self._is_initial(tokens[1].text) and not self._is_initial(tokens[2].text):
            surname_start = 2
        given = [replace(tokens[0], source_role="given")]
        middle = [replace(token, source_role="middle") for token in tokens[1:surname_start]]
        surname = [replace(token, source_role="surname") for token in tokens[surname_start:]]
        return given, middle, surname

    def _structured_particle_surname_first_roles(
        self,
        given: list[_Token],
        middle: list[_Token],
        surname: list[_Token],
    ) -> tuple[list[_Token], list[_Token], list[_Token]] | None:
        """Repair one mechanically shifted surname-first strong-particle layout."""
        if (
            len(given) != 1
            or len(surname) != 1
            or not self._is_full_name_token(given[0].text)
            or not self._is_full_name_token(surname[0].text)
        ):
            return None
        particle_width = self._strong_particle_width(middle, 0) if middle else 0
        family_end = particle_width + 1
        if not particle_width or family_end >= len(middle):
            return None
        trailing = middle[family_end:]
        if len(trailing) != 1 or not self._is_full_name_token(trailing[0].text):
            return None
        return (
            [replace(trailing[0], source_role="given")],
            [replace(surname[0], source_role="middle")],
            [
                replace(given[0], source_role="surname"),
                *(replace(token, source_role="surname") for token in middle[:family_end]),
            ],
        )

    def _expand_structured_surname_floor(
        self,
        given: list[_Token],
        middle: list[_Token],
        surname: list[_Token],
    ) -> tuple[list[_Token], list[_Token]]:
        """Move a strong particle span and its preceding anchor into surname."""
        if not given or not middle or not surname:
            return middle, surname
        combined = [*middle, *surname]
        strong_particle = self._find_strong_particle_span(combined)
        if strong_particle is None or strong_particle[0] >= len(middle):
            return middle, surname
        surname_start = strong_particle[0] - 1
        if surname_start < 0 or (surname_start == 0 and not all(self._is_initial(token.text) for token in given)):
            return middle, surname
        return (
            [replace(token, source_role="middle") for token in middle[:surname_start]],
            [replace(token, source_role="surname") for token in [*middle[surname_start:], *surname]],
        )

    def _repair_structured_roles(  # noqa: C901, PLR0911
        self,
        by_role: dict[str, list[_Token]],
    ) -> tuple[list[_Token], list[_Token], list[_Token]]:
        """Preserve surviving source roles, repairing only structurally empty boundaries."""
        given = [replace(token, source_role="given") for token in by_role["given"]]
        middle = [replace(token, source_role="middle") for token in by_role["middle"]]
        surname = [replace(token, source_role="surname") for token in by_role["surname"]]
        if shifted := self._structured_particle_surname_first_roles(given, middle, surname):
            return shifted
        if surname:
            if given:
                middle, surname = self._expand_structured_surname_floor(given, middle, surname)
                return given, middle, surname
            if middle:
                return [replace(middle[0], source_role="given")], middle[1:], surname
            if len(surname) == 1:
                fused = _FUSED_INITIAL_SURNAME_RE.fullmatch(surname[0].text)
                if fused is not None and all(self._allowed_name_character(character) for character in fused.group(2)):
                    initial, family = fused.groups()
                    return (
                        [replace(surname[0], text=f"{initial}.", source_role="given")],
                        [],
                        [
                            replace(
                                surname[0],
                                text=family,
                                source_role="surname",
                                position=surname[0].position + len(initial) + 1,
                            ),
                        ],
                    )
            if len(surname) >= _TWO_COMPONENTS and self._particle_key(surname[0].text) not in _FAMILY_PARTICLES:
                if surname[-1].text.isupper() and not any(self._is_initial(token.text) for token in surname[:-1]):
                    return (
                        [replace(token, source_role="given") for token in surname[:-1]],
                        [],
                        [replace(surname[-1], source_role="surname")],
                    )
                return self._infer_regular_roles(surname)
            return [], [], surname

        surviving_groups = [group for group in (given, middle) if group]
        if len(surviving_groups) == _TWO_COMPONENTS:
            return (
                [replace(token, source_role="given") for token in surviving_groups[0]],
                [],
                [replace(token, source_role="surname") for token in surviving_groups[1]],
            )
        if not surviving_groups:
            return [], [], []

        tokens = surviving_groups[0]
        if len(tokens) == 1:
            return [replace(tokens[0], source_role="given")], [], []
        return self._infer_regular_roles(tokens)

    def _invalid_token_reason(self, tokens: list[_Token]) -> str | None:
        for token in tokens:
            cleaned = self._clean_name_token(token.text)
            if not cleaned:
                return f"empty name token at position {token.position}"
            if self._is_parenthesized_name_token(cleaned):
                continue
            if not any(character.isalpha() for character in cleaned):
                return f"name token has no letters: {token.text!r}"
            if cleaned[0] in "-'" or cleaned[-1] == "-":
                return f"name joiner is not between letters: {token.text!r}"
            if any(not self._allowed_name_character(character) for character in cleaned):
                return f"unsupported character in name token: {token.text!r}"
        return None

    @staticmethod
    def _allowed_name_character(character: str) -> bool:
        return character.isalpha() or unicodedata.category(character).startswith("M") or character in "-'."

    def _canonical_tokens(self, tokens: list[_Token], role: str) -> tuple[str, ...]:
        return tuple(self._canonicalize_token(token.text, role) for token in tokens)

    def _canonicalize_token(self, token: str, role: str) -> str:  # noqa: C901
        cleaned = self._clean_name_token(token)
        if role in {"middle", "surname"} and cleaned in _LOWERCASE_RELATIONAL_TOKENS:
            return cleaned
        if _INITIAL_RE.fullmatch(cleaned):
            return cleaned[0].upper() + "."
        if _MULTI_INITIAL_RE.fullmatch(cleaned) or self._is_uppercase_dotted_abbreviation(cleaned):
            return "".join(character.upper() if character.isalpha() else character for character in cleaned)
        if (
            role != "surname"
            and cleaned.isalpha()
            and cleaned.isupper()
            and len(cleaned) == _TWO_COMPONENTS
            and not self._is_ambiguous_credential(cleaned)
        ):
            return cleaned

        parts = re.split(r"([-'])", cleaned)
        canonical: list[str] = []
        preserve_mixed_case = not (cleaned.islower() or cleaned.isupper())
        capitalize_after_apostrophe = parts[0].casefold() in {"a", "d", "o"}
        for index, part in enumerate(parts):
            if part in {"-", "'"}:
                canonical.append(part)
                continue
            if not part:
                continue
            particle = self._particle_key(part) in _FAMILY_PARTICLES
            preserve_lower_particle = particle and part.islower()
            normalize_surname_particle = role == "surname" and particle and part.isupper()
            if (preserve_lower_particle or normalize_surname_particle) and (len(parts) == 1 or index < len(parts) - 1):
                canonical.append(part.casefold())
            elif capitalize_after_apostrophe and index >= _TWO_COMPONENTS and parts[index - 1] == "'":
                canonical.append(self._name_case(part))
            elif preserve_mixed_case and not self._looks_like_mixed_ocr_case(part):
                canonical.append(part)
            else:
                canonical.append(self._name_case(part))
        return "".join(canonical)

    @staticmethod
    def _name_case(part: str) -> str:
        if len(part) == 1:
            return part.upper()
        if part.startswith("Mc") and len(part) > _MC_PREFIX_LENGTH and part[2:].isupper():
            return part[:2] + part[2].upper() + part[3:].lower()
        mixed_ocr_case = PersonNameNormalizationService._looks_like_mixed_ocr_case(part)
        if not (part.islower() or part.isupper() or mixed_ocr_case):
            return part
        lowered_tail = part[1:].lower().replace("i\u0307", "i")
        cased = part[0].upper() + lowered_tail
        if cased.startswith("Mc") and len(cased) > _MC_PREFIX_LENGTH:
            cased = cased[:2] + cased[2].upper() + cased[3:]
        return cased

    @staticmethod
    def _looks_like_mixed_ocr_case(part: str) -> bool:
        """Return whether a token has an initial mixed-case prefix plus an all-caps OCR tail."""
        return bool(
            (part.startswith("Mc") and len(part) > _MC_PREFIX_LENGTH and part[2:].isupper())
            or (len(part) > _TWO_COMPONENTS and part[0].isupper() and part[1].islower() and part[2:].isupper()),
        )

    @staticmethod
    def _clean_name_token(token: str) -> str:
        if PersonNameNormalizationService._is_parenthesized_name_token(token):
            return token
        cleaned = token.strip('"()[]{}<>:;,\u201c\u201d')
        if cleaned.endswith(".") and not (
            _INITIAL_RE.fullmatch(cleaned)
            or _MULTI_INITIAL_RE.fullmatch(cleaned)
            or _ABBREVIATED_TOKEN_RE.fullmatch(cleaned)
            or PersonNameNormalizationService._is_dotted_letter_token(cleaned)
            or _COMPOUND_INITIAL_RE.fullmatch(cleaned)
            or PersonNameNormalizationService._compact_key(cleaned) in _LONG_NAME_ABBREVIATION_KEYS
        ):
            cleaned = cleaned[:-1]
        return cleaned

    @staticmethod
    def _is_parenthesized_name_token(token: str) -> bool:
        """Return whether a token is one balanced parenthesized personal-name alternative."""
        if len(token) < _MIN_PARENTHESIZED_TOKEN_LENGTH or not token.startswith("(") or not token.endswith(")"):
            return False
        inner = token[1:-1]
        return any(character.isalpha() for character in inner) and all(
            PersonNameNormalizationService._allowed_name_character(character) for character in inner
        )

    @staticmethod
    def _is_dotted_letter_token(token: str) -> bool:
        """Return whether a token contains only letters and at least one internal dot."""
        return "." in token[:-1] and all(character.isalpha() or character == "." for character in token)

    @staticmethod
    def _is_uppercase_dotted_abbreviation(token: str) -> bool:
        """Preserve short abbreviations whose source already supplies uppercase letters."""
        letters = token[:-1] if token.endswith(".") else ""
        return bool(
            letters and len(letters) <= _MAX_DOTTED_ABBREVIATION_LETTERS and letters.isalpha() and letters.isupper(),
        )

    @staticmethod
    def _particle_key(token: str) -> str:
        return token.strip(".").casefold()

    def _canonical_suffix(self, token: str, *, explicit: bool) -> str:
        stripped = token.strip(".,")
        key = stripped.casefold()
        standard = _STANDARD_SUFFIXES.get(key)
        if standard is not None:
            return standard
        roman = stripped.upper()
        supported = _EXPLICIT_ROMAN_SUFFIXES if explicit else _RAW_ROMAN_SUFFIXES
        if roman in supported and (explicit or stripped == roman or roman in _CASE_INSENSITIVE_ROMAN_SUFFIXES):
            return roman
        return ""

    def _is_credential(self, token: str, *, explicit_context: bool = False) -> bool:
        key = self._compact_key(token)
        if key not in _CREDENTIAL_KEYS:
            return False
        if key not in _AMBIGUOUS_CREDENTIAL_KEYS or explicit_context:
            return True
        letters = "".join(character for character in token if character.isalpha())
        return "." in token or letters.isupper() or token.strip(".,") == _MIXED_CASE_CREDENTIALS.get(key)

    def _is_ambiguous_credential(self, token: str) -> bool:
        return self._compact_key(token) in _AMBIGUOUS_CREDENTIAL_KEYS

    def _is_spaced_credential_group(self, tokens: list[_Token]) -> bool:
        if not _TWO_COMPONENTS <= len(tokens) <= _MAX_SPACED_CREDENTIAL_TOKENS:
            return False
        if not all("." in token.text and self._compact_key(token.text) for token in tokens):
            return False
        return "".join(self._compact_key(token.text) for token in tokens) in _CREDENTIAL_KEYS

    def _trailing_spaced_credential_width(self, tokens: list[_Token]) -> int:
        max_width = min(_MAX_SPACED_CREDENTIAL_TOKENS, len(tokens) - _TWO_COMPONENTS)
        for width in range(max_width, _TWO_COMPONENTS - 1, -1):
            if self._is_spaced_credential_group(tokens[-width:]):
                return width
        return 0

    def _components(
        self,
        given: list[_Token],
        middle: list[_Token],
        surname: list[_Token],
        suffix: list[_Token],
        *,
        order: tuple[str, ...] | None = None,
    ) -> NameComponents:
        if order is None:
            order = tuple(
                ["given"] * len(given) + ["middle"] * len(middle) + ["surname"] * len(surname) + ["suffix"] * len(suffix),
            )

        def source_text(token: _Token) -> str:
            return token.source_text or token.text

        return NameComponents(
            given_name=" ".join(source_text(token) for token in given),
            middle_name=" ".join(source_text(token) for token in middle),
            surname=" ".join(source_text(token) for token in surname),
            suffix=" ".join(source_text(token) for token in suffix),
            given_tokens=tuple(source_text(token) for token in given),
            middle_tokens=tuple(source_text(token) for token in middle),
            surname_tokens=tuple(source_text(token) for token in surname),
            suffix_tokens=tuple(source_text(token) for token in suffix),
            order=order,
        )

    @staticmethod
    def _structured_source_order(by_role: dict[str, list[_Token]]) -> tuple[str, ...]:
        return tuple(role for role in ("given", "middle", "surname", "suffix") for _token in by_role[role])

    def _looks_like_two_complete_names(self, left: list[_Token], right: list[_Token]) -> bool:
        if len(left) < _TWO_COMPONENTS or len(right) < _TWO_COMPONENTS:
            return False
        if any(self._particle_key(token.text) in _FAMILY_PARTICLES for token in left[:-1]):
            return False
        return not any(self._is_initial(token.text) for token in [*left, *right])

    def _non_person_reason(self, surface: str) -> str | None:
        lowered = surface.casefold()
        if "@" in surface or "://" in surface:
            return "contact or URL input"
        if any(separator in surface for separator in (";", "&", "|")):
            return "multiple-name separator"
        and_match = _WORD_AND_RE.search(surface)
        if and_match is not None and and_match.start() > 0:
            return "author-list connector"
        words = {self._compact_key(token) for token in _TOKEN_RE.findall(lowered)}
        if words & _ORGANIZATION_WORDS:
            return "organization input"
        return None

    @staticmethod
    def _is_initial(token: str) -> bool:
        cleaned = PersonNameNormalizationService._clean_name_token(token)
        return bool(
            (len(cleaned) == 1 and cleaned.isalpha()) or _INITIAL_RE.fullmatch(cleaned) or _MULTI_INITIAL_RE.fullmatch(cleaned),
        )

    def _public_dropped(self, dropped: list[_DroppedToken]) -> tuple[DroppedNameToken, ...]:
        return tuple(
            DroppedNameToken(item.token.text, item.token.source_role, item.reason)
            for item in sorted(dropped, key=lambda item: item.token.position)
        )

    @staticmethod
    def _infer_dropped_roles(dropped: list[_DroppedToken], assigned: list[_Token]) -> list[_DroppedToken]:
        inferred: list[_DroppedToken] = []
        for item in dropped:
            if item.token.source_role:
                inferred.append(item)
                continue
            if item.reason in {DropReason.TITLE, DropReason.CONNECTOR}:
                role = "given"
            elif item.reason is DropReason.CREDENTIAL:
                role = "suffix"
            else:
                prior = [token for token in assigned if token.position <= item.token.position]
                role = prior[-1].source_role if prior else "suffix"
            inferred.append(replace(item, token=replace(item.token, source_role=role)))
        return inferred

    def _invalid(
        self,
        reason: str,
        dropped: list[_DroppedToken] | None = None,
    ) -> PersonNameNormalizationResult:
        return PersonNameNormalizationResult(
            outcome=PersonNameOutcome.INVALID,
            reason=reason,
            dropped_tokens=self._public_dropped(dropped or []),
        )

    @staticmethod
    def _non_person(reason: str) -> PersonNameNormalizationResult:
        return PersonNameNormalizationResult(outcome=PersonNameOutcome.NON_PERSON, reason=reason)


__all__ = [
    "DropReason",
    "DroppedNameToken",
    "PersonNameNormalizationResult",
    "PersonNameNormalizationService",
    "PersonNameOutcome",
]

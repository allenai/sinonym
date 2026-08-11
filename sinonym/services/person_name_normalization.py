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
from sinonym.name_punctuation import (
    PERSON_JOINER_FOLD_TRANSLATION,
    fold_internal_name_joiners,
    fold_spaced_transliteration_apostrophes,
)
from sinonym.services.non_person import reviewed_non_person_text_pattern


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
_HYPHEN_SPACING_RE = re.compile(r"\s*-\s*")
_SPACED_HYPHEN_RE = re.compile(r"\s+-\s+")
_DUPLICATE_APOSTROPHE_RE = re.compile(r"'{2,}")
_LEADING_STRAY_JOINER_RE = re.compile(r"^[-']\s+")
# Author-list connector "and" is whitespace-delimited ("First Last and First Last").
# Must NOT match "and" inside a hyphenated surname token (e.g. "Jon-And", "Strand"),
# so require real whitespace on both sides rather than a \b word boundary.
_WORD_AND_RE = re.compile(r"(?<=\s)and(?=\s)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\S+")
_COMMA_TAIL_TOKEN_RE = re.compile(r"[^\s,]+")
_LETTER_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_TRAILING_FOOTNOTE_RE = re.compile(r"(?P<name>.*[^\W\d_])(?P<marker>[*\u00a7\u2020\u2021]+)$", re.UNICODE)
_MOJIBAKE_FOOTNOTE_ENDINGS = ("Ã§", "Ä‡", "\u0421‡", "ÃÍ§")
_INITIAL_RE = re.compile(r"([^\W\d_])\.", re.UNICODE)
_MULTI_INITIAL_RE = re.compile(r"(?:[^\W\d_]\.)+[^\W\d_]\.?", re.UNICODE)
_ABBREVIATED_TOKEN_RE = re.compile(r"(?:[^\W\d_]+[-'])*[^\W\d_]{1,3}\.", re.UNICODE)
_COMPOUND_INITIAL_RE = re.compile(r"[^\W\d_]\.-[^\W\d_]\.", re.UNICODE)
# Hyphenated single-letter groups are initials, never a degree ("M-A", "M.-A.", "J-D",
# "M-S." — but not "MA"/"JD"): a hyphen distinguishes compound initials from a credential.
_HYPHEN_INITIAL_RE = re.compile(r"[^\W\d_]\.?(?:-[^\W\d_]\.?)+", re.UNICODE)
_LEADING_HYPHEN_INITIAL_RE = re.compile(r"^-([^\W\d_])\.$", re.UNICODE)
_FUSED_INITIAL_SURNAME_RE = re.compile(
    r"^(?P<initial>[^\W\d_])\.(?P<surname>[^\W\d_].+)$",
    re.UNICODE,
)
_FUSED_INITIAL_SEQUENCE_SURNAME_RE = re.compile(
    r"^(?P<initials>(?:[^\W\d_]\.){2,})(?P<surname>[^\W\d_].+)$",
    re.UNICODE,
)
_LOWER_MARKER_INITIAL_RE = re.compile(r"^([a-z])([A-Z])\.$")
_DOTTED_INITIAL_SEQUENCE_RE = re.compile(r"[A-Z][a-z](?:\.[A-Z])+\.")
_PARENTHETICAL_DUPLICATE_RE = re.compile(r"^(\S+)\s+\(([^)]+)\)\s+(.+)$")
_ASCII_JOINER_TRANSLATION = str.maketrans({"`": "'"})
_PRE_NFKC_JOINER_TRANSLATION = str.maketrans({"\u00b4": "'"})

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
# Ambiguous credential keys that have ~no real given-name use (unlike "md"=Mohammad,
# "ma"=María/Ma, "do"=Korean Do, "meng"=Meng, "ba"=Ba, "edd"=Edd, which are genuine names).
# A Title-case one before a complete name is a credential prefix, not a given name, so it
# should still drop ("Rn Rachael Zimlich" -> "Rachael Zimlich").
_PURE_CREDENTIAL_TITLE_DROP_KEYS = frozenset({"bs", "jd", "mba", "mpa", "rn"})
_MIXED_CASE_CREDENTIALS = {"meng": "MEng", "edd": "EdD"}
_PACKED_LEADING_CREDENTIALS = frozenset({"BEng", "DNP", "MBBS"})
_STRUCTURED_LEADING_CREDENTIALS = frozenset({"Dr.-Ing", "Dr.-Ing."})
_EXACT_CASE_TRAILING_CREDENTIALS = frozenset(
    {
        "AGPCNP-BC",
        "AOCNP",
        "APRN",
        "BCTMB",
        "BEng",
        "CHTP",
        "CTRS/L",
        "CTRS/LRT",
        "DNP",
        "EP-C",
        "FDRT",
        "LMBT",
        "LRT/CTRS",
        "MBBS",
    },
)
_EXACT_CASE_BOUNDARY_CREDENTIALS = frozenset(
    {
        "CTRS",
        "FACS",
        "FEBS",
        "FRACP",
        "ScD",
        "Se.Ak",
        "Se.Ak.",
    },
)
_EXACT_CASE_LEADING_TITLES = frozenset({"Apt", "Apt.", "Professur"})
_REVIEWED_STRUCTURED_MIDDLE_CREDENTIALS = frozenset({"Dipl.-Ing. Fh"})
_REVIEWED_CLOSED_COMMA_TAIL_CREDENTIALS = frozenset(
    {
        "DNB",
        "FAAP",
        "FCCM",
        "FCCP",
        "FRACS",
        "FRCPA",
        "FRCS",
        "FRCSC",
        "MB",
        "MRCP",
        "RD",
    },
)
_REVIEWED_CLOSED_COMMA_HEAD_BLOCKERS = frozenset({"DM", "MDRD", "MDS"})
_LEADING_ARRAY_REMAINDER_EXCLUSIONS = frozenset({"array", "редакционная", "статья"})
_FAMILY_PARTICLES = frozenset(
    {
        "'t",
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
_LOWERCASE_RELATIONAL_TOKENS = frozenset({"'t", "b.", "d.", "e", "kizi", "kyzy", "oglu", "o'g'li", "oğlu", "qizi"})
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
        # curated additions (org / publishing / section nouns with negligible use as a
        # personal name; matched as whole tokens so substrings like "Institut"e in a
        # surname are unaffected). Vetted empirically via person->reject flip judging.
        "institut",  # German/Dutch "Institut" (no trailing e)
        "universitat",
        "universität",
        "universite",
        "université",
        "editorial",
        "editores",
        "journal",
        "journals",
        "proceedings",
        "initiative",
        "faculty",
        "ministry",
        # further curated org/section nouns (empirically org-only; real-surname
        # collisions like staff/editor/press/board/bureau are deliberately excluded)
        "office",
        "division",
        "services",
        "editors",
        "network",
        "bulletin",
        "directorate",
        "secretariat",
        "academy",
        "program",
        "programme",
        "publishers",
        "institution",
        "organization",
        "organisation",
        "laboratory",
        "ltd",
        "society",
        "team",
        "university",
        # Non-English org-only nouns (FR/ES/IT/PT/NL/DE), whole-token matched and
        # empirically org-only (never real person names). The English-centric list
        # above let non-English orgs through as "persons" (e.g. "Deutsche Gesellschaft
        # für Kardiologie", "Società Italiana di …", "Ministère de la Santé"). Sized via
        # a multilingual org sweep on post-fix output: ~29,972 names / 102,859 occ
        # (0.0176% of non-Chinese occ). Both accented and diacritic-free forms are listed
        # because _compact_key preserves diacritics.
        "societe",
        "société",  # FR
        "sociedad",  # ES
        "societa",
        "società",  # IT
        "sociedade",  # PT
        "ministere",
        "ministère",  # FR
        "ministerio",  # ES / PT (ministério compacts to ministerio too)
        "ministério",
        "ministero",  # IT
        "ministerium",  # DE
        "ministerie",  # NL
        "universidad",  # ES
        "universita",
        "università",  # IT
        "universidade",  # PT
        "universiteit",  # NL
        "federation",
        "fédération",  # FR
        "federacion",
        "federación",  # ES
        "federazione",  # IT
        "federacao",
        "federação",  # PT
        "asociacion",
        "asociación",  # ES
        "associazione",  # IT
        "associacao",
        "associação",  # PT
        "instituto",  # ES / PT
        "istituto",  # IT
        "instituut",  # NL
        "gesellschaft",  # DE
        "gewerkschaft",  # DE
        "genootschap",  # NL
        "syndicat",  # FR
        "stiftung",  # DE
        "stichting",  # NL
        "akademie",  # DE
        "academie",
        "académie",  # FR
        "accademia",  # IT
        "dipartimento",  # IT
        "gmbh",  # DE company suffix
        # NOTE: the prepositions "für" (DE) / "voor" (NL) are strong org signals for
        # compound-noun orgs ("Bundesministerium für …", "Voor Numismatiek") but collide
        # with the real surnames "Für" (Hungarian) / "Voor" (Estonian/Dutch). They are
        # handled positionally in _non_person_reason (org only when NOT the final token),
        # not listed here, so a trailing surname is preserved. "para"/"pour"/"und" are
        # excluded entirely (real givens "Para"/"Pour"; noble "von X und Y").
    },
)
# Prepositions that signal an org ONLY when a token follows them (mid/leading position):
# "Institut für Physik" / "Voor Numismatiek" are orgs, but "Gabriella Für" / "Michael J.
# Voor" are people whose surname is the final token. Whole-token matched via _compact_key.
_ORG_PREPOSITION_WORDS = frozenset({"für", "voor"})
# Whole-name strings made only of these connectives are not persons ("of", "the ...").
_FUNCTION_WORDS = frozenset({"of", "the", "for", "und", "der", "des"})
# English "Center"/"Centre" is also a real surname (David M. Center, the immunologist), so
# it must NOT reject a clean personal name whose surname IS "Center" ("David M. Center").
# "Company" is deliberately NOT here: it is a Catalan surname too, but its person shape
# ("Initial Surname Company") is indistinguishable from a firm ("A Boeing Company",
# "M.T. Company"), so gating it admits ~as many orgs as people — kept as a hard org word.
_SURNAME_COLLISION_ORG_WORDS = frozenset({"center", "centre"})
# Org words that are ~never part of a real hyphenated surname, so a hyphenated token
# containing one is an org ("Robert Koch-Institut", "Ruhr-Universität", "Courier-Journal").
# Deliberately EXCLUDES company/hospital/center/bureau/press (real hyphenated surnames:
# "Torres-Company", "Gómez-Hospital", "Plu-Bureau", "Beebe-Center").
_HYPHEN_ORG_WORDS = frozenset(
    {
        "institut",
        "institute",
        "university",
        "universitat",
        "universität",
        "universite",
        "université",
        "universidad",
        "universita",
        "università",
        "universidade",
        "universiteit",
        "instituto",
        "istituto",
        "instituut",
        "journal",
        "journals",
        "proceedings",
        "laboratory",
        "centre",
        "team",
    },
)
_STANDARD_SUFFIXES = {
    "jr": "Jr.",
    "junior": "Jr.",
    # Portuguese spelling of the generational suffix. Keyed separately because the
    # lookup casefolds but does not strip diacritics, so "Júnior" never matches
    # "junior"; without this it stays in the surname slot and displaces the family
    # name (e.g. "Francisco Aquino Júnior" -> surname "Júnior" instead of "Aquino").
    "júnior": "Jr.",
    # Portuguese agnomes for "son" and "grandson". Like Júnior they are registered
    # parts of a Brazilian name that follow the family name, so leaving them in the
    # surname slot displaces it ("José Ribamar Santos Neto" -> surname "Neto"). They
    # keep their own spelling rather than folding into "Jr." — Filho/Neto/Júnior mark
    # different generations and are not interchangeable.
    "filho": "Filho",
    "neto": "Neto",
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
# Both spellings of Junior need their own key because _compact_key preserves
# diacritics, so "Júnior" folds to "júnior" and would otherwise miss the set while the
# unaccented spelling matched.
# "Neto"/"Filho" are also used as the surname itself (Agostinho Neto, Félix Neto,
# Chiara Neto, Edson Filho), so both demote only when a surname survives the
# removal, which leaves the two-token forms alone. Blind labelling of the two-token
# corpus rows put Neto at 50.8% real surname vs 44.6% truncated fragment, and Filho
# at 12.8% vs 82.8% — but demoting Filho's fragments buys nothing: "Mesquita Filho"
# would key on the bare surname "mesquita" while the person's full-name mentions key
# "<initial> mesquita", so no block merges (checked over all 431 judged fragments),
# while the 12.8% lose a correct parse. Multi-token forms of both demote either way;
# that is where the displaced-surname bug lives (110k occ Filho, 79k Neto).
_SURNAME_LIKE_SUFFIX_KEYS = frozenset({"senior", "junior", "júnior", "neto", "filho"})
_RAW_ROMAN_SUFFIXES = frozenset({"II", "III", "IV", "VI", "VII", "VIII", "IX", "X"})
_EXPLICIT_ROMAN_SUFFIXES = _RAW_ROMAN_SUFFIXES | {"I", "V", "X"}
_CASE_INSENSITIVE_ROMAN_SUFFIXES = _RAW_ROMAN_SUFFIXES - {"II"}
_SIMPLE_TWO_TOKEN_POLICY_KEYS = frozenset(
    _TITLE_KEYS
    | _TITLE_QUALIFIER_KEYS
    | _CREDENTIAL_KEYS
    | {credential.casefold() for credential in _EXACT_CASE_BOUNDARY_CREDENTIALS | _EXACT_CASE_TRAILING_CREDENTIALS}
    | {title.casefold().rstrip(".") for title in _EXACT_CASE_LEADING_TITLES}
    | _ORGANIZATION_WORDS
    | _FUNCTION_WORDS
    | _STANDARD_SUFFIXES.keys()
    | {suffix.casefold() for suffix in _RAW_ROMAN_SUFFIXES}
    | {"and", "null"},
)
_TWO_COMPONENTS = 2
_THREE_COMPONENTS = 3
_FOUR_COMPONENTS = 4
_MAX_SPACED_CREDENTIAL_TOKENS = 3
_MAX_DOTTED_ABBREVIATION_LETTERS = 3
_MIN_PARENTHESIZED_TOKEN_LENGTH = 3
_MC_PREFIX_LENGTH = 2


def _has_usable_array_remainder(middle: str, last: str) -> bool:
    """Return whether removing a leading Array leaves a plausible name fragment."""
    tokens = [token for token in f"{middle} {last}".split() if token]
    return bool(
        tokens
        and any(character.isalpha() for token in tokens for character in token)
        and not all(token.casefold().strip(".") in _LEADING_ARRAY_REMAINDER_EXCLUSIONS for token in tokens),
    )


def _compact_policy_key(token: str) -> str:
    """Return the punctuation-insensitive key used by boundary policies."""
    if token.isalnum():
        return token.casefold()
    return "".join(character.casefold() for character in token if character.isalnum())


def _is_credential_boundary_surface(token: str, *, explicit_context: bool = False) -> bool:
    """Return whether one token matches the canonical credential policy."""
    if token.strip(",") in _EXACT_CASE_BOUNDARY_CREDENTIALS:
        return True
    key = _compact_policy_key(token)
    if key not in _CREDENTIAL_KEYS:
        return False
    if key not in _AMBIGUOUS_CREDENTIAL_KEYS or explicit_context:
        return True
    letters = "".join(character for character in token if character.isalpha())
    return "." in token or letters.isupper() or token.strip(".,") == _MIXED_CASE_CREDENTIALS.get(key)


def _is_trailing_credential_surface(token: str, *, explicit_context: bool = False) -> bool:
    """Return whether one token is a credential at a trailing boundary."""
    return token.strip(",") in _EXACT_CASE_TRAILING_CREDENTIALS or _is_credential_boundary_surface(
        token,
        explicit_context=explicit_context,
    )


def _canonical_suffix_surface(token: str, *, explicit: bool) -> str:
    """Return the canonical suffix represented by one source token."""
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


def is_reviewed_compact_initial_boundary(token: str) -> bool:
    """Return whether a compact-initial candidate is reviewed name metadata.

    The two-token name policy preserves ambiguous all-caps credential collisions
    as initials (for example ``MS``) and mixed-case name collisions such as
    ``Md``. Title-case titles, unambiguous credentials, and supported suffixes
    remain boundary metadata and must not be expanded into invented initials.
    """
    surface = token.strip()
    if not surface:
        return False
    key = _compact_policy_key(surface)
    ambiguous_initial = key in _AMBIGUOUS_CREDENTIAL_KEYS and surface.isupper()
    title = surface in _EXACT_CASE_LEADING_TITLES or (key in _TITLE_KEYS and not ambiguous_initial)
    credential = (
        surface.strip(",") in _EXACT_CASE_TRAILING_CREDENTIALS or _is_credential_boundary_surface(surface)
    ) and not ambiguous_initial
    return bool(title or credential or _canonical_suffix_surface(surface, explicit=False))


def _has_complete_packed_name_remainder(value: str) -> bool:
    """Return whether a packed field contains a semantic multi-token name."""
    words = _LETTER_WORD_RE.findall(value)
    return bool(
        len(words) >= _TWO_COMPONENTS and any(len(word) > 1 or word.lower() == word.upper() for word in words),
    )


def reviewed_leading_credential_source_pattern(
    first_name: str | None,
    middle_name: str | None,
    last_name: str | None,
    suffix: str | None = None,
) -> str | None:
    """Return the reviewed leading credential for one safe structured shape."""
    first, middle, last = ((value or "").strip() for value in (first_name, middle_name, last_name))
    if (suffix or "").strip():
        return None
    if first in _PACKED_LEADING_CREDENTIALS and not middle and _has_complete_packed_name_remainder(last):
        return first
    if (
        first in _STRUCTURED_LEADING_CREDENTIALS
        and len(_LETTER_WORD_RE.findall(middle)) == 1
        and len(_LETTER_WORD_RE.findall(last)) == 1
    ):
        return first
    return None


def reviewed_closed_comma_credential_tail_head(
    last_name: str | None,
    suffix: str | None = None,
) -> str | None:
    """Return the retained surname field for one complete credential tail."""
    if (suffix or "").strip():
        return None
    head, separator, tail = (last_name or "").partition(",")
    if not separator:
        return None
    tail_tokens = _COMMA_TAIL_TOKEN_RE.findall(tail)
    if (
        not tail_tokens
        or not any(token in _REVIEWED_CLOSED_COMMA_TAIL_CREDENTIALS for token in tail_tokens)
        or not all(
            token in _REVIEWED_CLOSED_COMMA_TAIL_CREDENTIALS or _is_trailing_credential_surface(token, explicit_context=True)
            for token in tail_tokens
        )
    ):
        return None

    head_tokens = head.split()
    while head_tokens and (
        head_tokens[-1] in _REVIEWED_CLOSED_COMMA_TAIL_CREDENTIALS or _is_trailing_credential_surface(head_tokens[-1])
    ):
        head_tokens.pop()
    if any(token in _REVIEWED_CLOSED_COMMA_HEAD_BLOCKERS for token in head_tokens):
        return None
    retained = " ".join(head_tokens)
    if retained.endswith("MD (Medicine)"):
        return None
    return retained


def reviewed_source_cleanup_pattern(  # noqa: PLR0911 - one return per closed reviewed shape
    first_name: str | None,
    middle_name: str | None,
    last_name: str | None,
    suffix: str | None = None,
) -> str | None:
    """Return the reviewed structured cleanup shape for one source row."""
    first, middle, last = ((value or "").strip() for value in (first_name, middle_name, last_name))
    if (suffix or "").strip():
        return None
    if reviewed_leading_credential_source_pattern(first, middle, last) is not None:
        return "leading_credential"
    if any(
        token.strip(",") in _EXACT_CASE_BOUNDARY_CREDENTIALS | _EXACT_CASE_TRAILING_CREDENTIALS
        for value in (first, middle, last)
        for token in value.split()
    ):
        return "exact_case_credential"
    if first in _EXACT_CASE_LEADING_TITLES and len(_LETTER_WORD_RE.findall(f"{middle} {last}")) >= _TWO_COMPONENTS:
        return "leading_title"
    if first == "Array" and _has_usable_array_remainder(middle, last):
        return "leading_array_artifact"
    if middle in _REVIEWED_STRUCTURED_MIDDLE_CREDENTIALS and first and last:
        return "structured_middle_credential"
    if first and last and middle.split().count("Jr.") == 1:
        return "structured_middle_suffix"
    return None


class PersonNameNormalizationService:
    """Normalize raw or structured personal names without Chinese routing."""

    def normalize_text(  # noqa: C901, PLR0911, PLR0912, PLR0915
        self,
        raw_name: str | None,
    ) -> PersonNameNormalizationResult:
        """Normalize one raw name string into semantic canonical components."""
        if not isinstance(raw_name, str):
            return self._invalid("name must be a string")
        if reviewed_non_person_text_pattern(raw_name) is not None:
            return self._non_person("reviewed non-person input")

        simple_result = self._normalize_simple_two_token_text(raw_name)
        if simple_result is not None:
            return simple_result

        source_text = raw_name
        normalized_input, leading_markers = self._strip_leading_superscript_affiliation(raw_name)
        if self._has_spaced_multi_name_separator(normalized_input):
            return self._non_person("multiple-name separator")
        surface = self._normalize_surface(normalized_input)
        if not surface:
            return self._invalid("name is empty")
        if not any(character.isalpha() for character in surface):
            return self._invalid("name has no letters")

        # Reject an organization / non-person string as a whole rather than salvaging a
        # name from it: "Rachel Webster University of New South Wales" and "Niels Bohr
        # Institute" are both non-persons — trying to extract a name from contaminated
        # input is unreliable, so check BEFORE any trailing-affiliation strip.
        non_person_reason = self._non_person_reason(surface)
        if non_person_reason is not None:
            return self._non_person(non_person_reason)

        dropped: list[_DroppedToken] = []
        if leading_markers:
            dropped.append(_DroppedToken(_Token(leading_markers, "", 0), DropReason.AFFILIATION))
        surface = self._collapse_parenthetical_duplicate_surface(surface)

        retained_head = reviewed_closed_comma_credential_tail_head(surface)
        if retained_head is not None:
            retained_tokens, credential_drops = self._reviewed_credential_tail_tokens(
                surface,
                retained_head,
                source_role="",
            )
            dropped.extend(credential_drops)
            if not retained_tokens:
                return self._invalid("no personal-name tokens remain", dropped)
            if leading_markers:
                first = retained_tokens[0]
                retained_tokens[0] = replace(first, source_text=f"{leading_markers}{first.text}")
            return self._normalize_regular_name(source_text, retained_tokens, "", None, dropped)

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
        # A leading org preposition ("Voor Numismatiek") is an org, not "Voor" the surname
        # ("Michael Voor" is handled by the final-token rule); defer to the full path.
        if raw_tokens[0].casefold() in _ORG_PREPOSITION_WORDS:
            return None

        source_tokens = [
            _Token(raw_tokens[0], "", 0),
            _Token(raw_tokens[1], "", len(raw_tokens[0]) + 1),
        ]
        given = [replace(source_tokens[0], source_role="given")]
        surname = [replace(source_tokens[1], source_role="surname")]
        source = self._components(given, [], surname, [])
        return self._person(raw_name, source, given, [], surname, "", [])

    def normalize_components(  # noqa: C901, PLR0911, PLR0912, PLR0915
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
        source_surface = " ".join(value for value in (first_name, middle_name, last_name) if value)
        if not (suffix or "").strip() and reviewed_non_person_text_pattern(source_surface) is not None:
            return self._non_person("reviewed non-person input")

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
        if reviewed_non_person_text_pattern(source_text) is not None:
            return self._non_person("reviewed non-person input")

        non_person_reason = self._non_person_reason(source_text)
        if non_person_reason is not None:
            return self._non_person(non_person_reason)

        position = 0
        role_offsets: dict[str, int] = {}
        source_by_role: dict[str, list[_Token]] = {}
        for role in ("given", "middle", "surname", "suffix"):
            role_offsets[role] = position
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
        )

        dropped = [
            _DroppedToken(
                _Token(markers, role, source_by_role[role][0].position),
                DropReason.AFFILIATION,
            )
            for role, markers in leading_markers_by_role.items()
            if markers and source_by_role[role]
        ]
        working_surname = source_by_role["surname"]
        retained_last_source = reviewed_closed_comma_credential_tail_head(
            stripped_values["surname"],
            stripped_values["suffix"],
        )
        retained_last = self._normalize_surface(retained_last_source) if retained_last_source is not None else None
        if retained_last is not None:
            working_surname, credential_drops = self._reviewed_credential_tail_tokens(
                surfaces["surname"],
                retained_last,
                source_role="surname",
                offset=role_offsets["surname"],
            )
            dropped.extend(credential_drops)
        middle_tokens = self._strip_structured_middle_surname_artifact(
            source_by_role["middle"],
            working_surname,
            dropped,
        )
        middle_tokens, structured_middle_suffix = self._strip_reviewed_structured_middle(
            middle_tokens,
            middle_surface=surfaces["middle"],
            has_name_endpoints=bool(source_by_role["given"] and working_surname),
            has_explicit_suffix=bool(source_by_role["suffix"]),
            dropped=dropped,
        )
        name_tokens = [
            *source_by_role["given"],
            *middle_tokens,
            *working_surname,
        ]
        name_tokens = self._repair_leading_marker_initial(name_tokens, dropped)
        name_tokens = self._join_separated_compound_initials(name_tokens)
        reviewed_credential = reviewed_leading_credential_source_pattern(
            stripped_values["given"],
            stripped_values["middle"],
            stripped_values["surname"],
            stripped_values["suffix"],
        )
        name_tokens = self._strip_leading_titles(
            name_tokens,
            dropped,
            reviewed_credential=reviewed_credential,
        )
        name_tokens = self._strip_standalone_periods(name_tokens, dropped)
        name_tokens = self._strip_boundary_markers(name_tokens, dropped)

        canonical_suffix, _, explicit_dropped = self._consume_explicit_suffix(
            source_by_role["suffix"],
        )
        dropped.extend(explicit_dropped)
        if structured_middle_suffix:
            if canonical_suffix:
                return self._invalid("name has multiple suffixes", dropped)
            canonical_suffix = structured_middle_suffix
        name_tokens, boundary_suffix, _, multiple_suffixes = self._strip_trailing_boundaries(name_tokens, dropped)
        if multiple_suffixes:
            return self._invalid("name has multiple suffixes", dropped)
        if boundary_suffix:
            if canonical_suffix:
                return self._invalid("name has multiple suffixes", dropped)
            canonical_suffix = boundary_suffix

        name_tokens = self._strip_attached_affiliations(name_tokens, dropped)
        name_tokens = self._strip_attached_terminal_footnote(name_tokens, dropped)
        invalid_reason = self._invalid_token_reason(name_tokens)
        if invalid_reason is not None:
            return self._invalid(invalid_reason, dropped)
        if not name_tokens:
            return self._invalid("no personal-name tokens remain", dropped)

        if retained_last == "":
            given, middle, surname = self._infer_regular_roles(name_tokens)
        else:
            cleaned_by_role = {
                role: [token for token in name_tokens if token.source_role == role] for role in ("given", "middle", "surname")
            }
            given, middle, surname = self._repair_structured_roles(cleaned_by_role)
        given, middle, surname = self._apply_unbound_initial_policy(given, middle, surname)
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
        tokens, boundary_suffix, boundary_suffix_token, multiple_suffixes = self._strip_trailing_boundaries(tokens, dropped)
        if multiple_suffixes:
            return self._invalid("name has multiple suffixes", dropped)
        if boundary_suffix:
            if suffix:
                return self._invalid("name has multiple suffixes", dropped)
            suffix = boundary_suffix
            suffix_token = boundary_suffix_token

        tokens = self._strip_attached_affiliations(tokens, dropped)
        tokens = self._strip_attached_terminal_footnote(tokens, dropped)
        invalid_reason = self._invalid_token_reason(tokens)
        if invalid_reason is not None:
            return self._invalid(invalid_reason, dropped)
        if not tokens:
            return self._invalid("no personal-name tokens remain", dropped)

        tokens = self._split_fused_initial_sequence_surname(tokens)
        source_given, source_middle, source_surname = self._infer_regular_roles(tokens)
        assigned = [*source_given, *source_middle, *source_surname]
        credential_source = [
            replace(item.token, source_role="suffix") for item in dropped if item.reason is DropReason.CREDENTIAL
        ]
        represented_positions = {token.position for token in [*assigned, *credential_source]}
        prefix_source = self._regular_prefix_source(
            original_tokens,
            represented_positions,
            tokens[0],
            dropped,
        )
        source_suffix = [*([suffix_token] if suffix_token is not None else []), *credential_source]
        source = self._components(
            [*prefix_source, *source_given],
            source_middle,
            source_surname,
            source_suffix,
        )
        given, middle, surname = self._apply_unbound_initial_policy(
            source_given,
            source_middle,
            source_surname,
            normalize_ambiguous_tail=True,
        )
        return self._person(source_text, source, given, middle, surname, suffix, dropped)

    @staticmethod
    def _regular_prefix_source(
        original_tokens: list[_Token],
        represented_positions: set[int],
        first_survivor: _Token,
        dropped: list[_DroppedToken],
    ) -> list[_Token]:
        """Rebuild raw prefix lineage without duplicating represented occurrences."""
        prefix: list[_Token] = []
        for token in original_tokens:
            if token.position >= first_survivor.position or token.position in represented_positions:
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

    def _normalize_comma_name(  # noqa: PLR0911, PLR0913
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
        family_tokens = self._strip_leading_titles(
            family_tokens,
            dropped,
            preserve_ambiguous_credentials=True,
            has_external_name_context=bool(given_tokens),
        )
        given_tokens = self._strip_leading_titles(given_tokens, dropped, preserve_ambiguous_credentials=True)
        family_tokens = self._strip_dangling_and(family_tokens, dropped)
        given_tokens = self._strip_dangling_and(given_tokens, dropped)
        family_tokens = self._strip_standalone_periods(family_tokens, dropped)
        given_tokens = self._strip_standalone_periods(given_tokens, dropped)
        family_tokens = self._strip_boundary_markers(family_tokens, dropped)
        given_tokens = self._strip_boundary_markers(given_tokens, dropped)
        family_tokens, family_suffix, family_suffix_token, multiple_suffixes = self._strip_trailing_boundaries(
            family_tokens,
            dropped,
            has_external_name_context=bool(given_tokens),
        )
        if multiple_suffixes:
            return self._invalid("name has multiple suffixes", dropped)
        if family_suffix:
            if suffix:
                return self._invalid("name has multiple suffixes", dropped)
            suffix = family_suffix
            suffix_token = family_suffix_token
        given_tokens, boundary_suffix, boundary_suffix_token, multiple_suffixes = self._strip_trailing_boundaries(
            given_tokens,
            dropped,
            has_external_name_context=bool(family_tokens),
            has_external_surname_context=bool(family_tokens),
        )
        if multiple_suffixes:
            return self._invalid("name has multiple suffixes", dropped)
        if boundary_suffix:
            if suffix:
                return self._invalid("name has multiple suffixes", dropped)
            suffix = boundary_suffix
            suffix_token = boundary_suffix_token

        family_tokens = self._strip_attached_affiliations(family_tokens, dropped)
        given_tokens = self._strip_attached_affiliations(given_tokens, dropped)
        given_tokens = self._strip_attached_terminal_footnote(given_tokens, dropped)
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
        )
        given, middle, surname = self._apply_unbound_initial_policy(given, middle, surname)
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
        # Mononym contract: a single-token person always keeps its token in the surname
        # slot (a person always needs a surname; kept as-is, no length rejection).
        name_tokens = [*given, *middle, *surname]
        if len(name_tokens) == 1:
            given, middle, surname = [], [], [replace(name_tokens[0], source_role="surname")]
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
        ascii_surface = value.isascii()
        if ascii_surface:
            normalized = value.translate(_ASCII_JOINER_TRANSLATION)
        else:
            normalized = value.translate(_PRE_NFKC_JOINER_TRANSLATION)
            normalized = unicodedata.normalize("NFKC", normalized)
        normalized = fold_internal_name_joiners(normalized, PERSON_JOINER_FOLD_TRANSLATION)
        if not ascii_surface:
            normalized = unicodedata.normalize("NFC", normalized)
        normalized = _WHITESPACE_RE.sub(" ", normalized)
        normalized = _LEADING_STRAY_JOINER_RE.sub("", normalized)
        normalized = fold_spaced_transliteration_apostrophes(normalized)
        normalized = _HYPHEN_SPACING_RE.sub("-", normalized)
        normalized = _DUPLICATE_APOSTROPHE_RE.sub("'", normalized)
        return normalized.strip(" \t\r\n,")

    @staticmethod
    def _strip_leading_superscript_affiliation(value: str) -> tuple[str, str]:
        """Strip leading superscript digits before a name while returning lineage."""
        marker_end = 0
        while marker_end < len(value):
            character = value[marker_end]
            if "SUPERSCRIPT" not in unicodedata.name(character, "") or not character.isdigit():
                break
            marker_end += 1
        name_start = marker_end
        while name_start < len(value) and value[name_start].isspace():
            name_start += 1
        if marker_end == 0 or name_start >= len(value) or not value[name_start].isalpha():
            return value, ""
        return value[name_start:], value[:marker_end]

    def _has_spaced_multi_name_separator(self, value: str) -> bool:
        """Return whether a spaced hyphen separates two complete names."""
        folded = value.translate(PERSON_JOINER_FOLD_TRANSLATION)
        for separator in _SPACED_HYPHEN_RE.finditer(folded):
            left = self._tokens(folded[: separator.start()], "", 0)
            right = self._tokens(folded[separator.end() :], "", separator.end())
            if self._looks_like_two_complete_names(left, right):
                return True
        return False

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
        """Fold one person-name joiner while preserving the existing helper contract."""
        return character.translate(PERSON_JOINER_FOLD_TRANSLATION)

    @staticmethod
    def _tokens(value: str, source_role: str, offset: int) -> list[_Token]:
        return [_Token(match.group(), source_role, offset + match.start()) for match in _TOKEN_RE.finditer(value.strip())]

    @staticmethod
    def _reviewed_credential_tail_tokens(
        value: str,
        retained_head: str,
        *,
        source_role: str,
        offset: int = 0,
    ) -> tuple[list[_Token], list[_DroppedToken]]:
        """Split one reviewed credential tail into retained and dropped tokens."""
        tokens = [_Token(match.group(), source_role, offset + match.start()) for match in _COMMA_TAIL_TOKEN_RE.finditer(value)]
        retained_end = offset + len(retained_head)
        retained = [token for token in tokens if token.position < retained_end]
        dropped = [_DroppedToken(token, DropReason.CREDENTIAL) for token in tokens if token.position >= retained_end]
        return retained, dropped

    @staticmethod
    def _compact_key(token: str) -> str:
        return _compact_policy_key(token)

    def _strip_leading_titles(
        self,
        tokens: list[_Token],
        dropped: list[_DroppedToken],
        *,
        preserve_ambiguous_credentials: bool = False,
        reviewed_credential: str | None = None,
        has_external_name_context: bool = False,
    ) -> list[_Token]:
        remaining = list(tokens)
        if self._has_leading_et_al_contamination(
            remaining,
            has_external_name_context=has_external_name_context,
        ):
            dropped.extend(_DroppedToken(token, DropReason.CONNECTOR) for token in remaining[:2])
            remaining = remaining[2:]
        stripped_title = False
        while remaining:
            token = remaining[0]
            if token.text == reviewed_credential:
                dropped.append(_DroppedToken(token, DropReason.CREDENTIAL))
                remaining.pop(0)
                reviewed_credential = None
                continue
            attached_title = self._split_attached_leading_title(token)
            if attached_title is not None:
                title, remainder = attached_title
                dropped.append(_DroppedToken(title, DropReason.TITLE))
                remaining[0] = remainder
                stripped_title = True
                continue
            reviewed_reason = self._reviewed_leading_drop_reason(remaining)
            if reviewed_reason is not None:
                dropped.append(_DroppedToken(token, reviewed_reason))
                remaining.pop(0)
                stripped_title = reviewed_reason is DropReason.TITLE
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
            hyphen_initials = bool(_HYPHEN_INITIAL_RE.fullmatch(token.text))
            if (
                self._is_credential(token.text)
                and not ambiguous_name_token
                and not leading_name_abbreviation
                and not hyphen_initials
            ):
                dropped.append(_DroppedToken(token, DropReason.CREDENTIAL))
                remaining.pop(0)
                continue
            # A Title-case pure-credential token (Rn/Jd/Mba/Mpa/Bs) is a credential prefix,
            # not a given name, when a complete name follows (>=2 SURVIVING tokens, at least
            # one non-initial surname): "Rn Rachael Zimlich" -> drop "Rn". A bare surname
            # after it ("Rn Cahn") is left alone, since the token could be initials there —
            # and trailing credentials/suffixes must not count as name evidence, or
            # "Rn Cahn PhD" parses differently from "Rn Cahn".
            if (
                key in _PURE_CREDENTIAL_TITLE_DROP_KEYS
                and not leading_name_abbreviation
                and not hyphen_initials
                and self._followed_by_complete_name(remaining)
            ):
                dropped.append(_DroppedToken(token, DropReason.CREDENTIAL))
                remaining.pop(0)
                continue
            break
        return remaining

    @staticmethod
    def _reviewed_leading_drop_reason(tokens: list[_Token]) -> DropReason | None:
        """Return the reviewed reason for one exact-case leading source token."""
        token = tokens[0]
        if token.text in _EXACT_CASE_LEADING_TITLES and len(tokens) >= _THREE_COMPONENTS:
            return DropReason.TITLE
        if token.text == "Array" and _has_usable_array_remainder(
            "",
            " ".join(item.text for item in tokens[1:]),
        ):
            return DropReason.AFFILIATION
        return None

    def _followed_by_complete_name(self, remaining: list[_Token]) -> bool:
        """A complete name follows only among tokens that survive trailing cleanup."""
        tail = remaining[1:]
        while tail and (self._is_trailing_credential(tail[-1].text) or self._canonical_suffix(tail[-1].text, explicit=False)):
            tail = tail[:-1]
        return len(tail) >= _TWO_COMPONENTS and any(not self._is_initial(token.text) for token in tail)

    @staticmethod
    def _has_leading_et_al_contamination(
        tokens: list[_Token],
        *,
        has_external_name_context: bool = False,
    ) -> bool:
        """Match a leading citation marker only when a complete name survives."""
        minimum_width = _THREE_COMPONENTS if has_external_name_context else _FOUR_COMPONENTS
        return len(tokens) >= minimum_width and PersonNameNormalizationService._is_et_al_pair(tokens[:2])

    @staticmethod
    def _is_et_al_pair(tokens: list[_Token]) -> bool:
        """Match one exact citation-marker token pair."""
        return len(tokens) == _TWO_COMPONENTS and tokens[0].text.casefold() == "et" and tokens[1].text.casefold() == "al."

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

    def _strip_trailing_boundaries(  # noqa: C901
        self,
        tokens: list[_Token],
        dropped: list[_DroppedToken],
        *,
        has_external_name_context: bool = False,
        has_external_surname_context: bool = False,
    ) -> tuple[list[_Token], str, _Token | None, bool]:
        remaining = list(tokens)
        suffix = ""
        suffix_token: _Token | None = None
        minimum_et_al_width = _THREE_COMPONENTS
        while len(remaining) >= _FOUR_COMPONENTS:
            credential_width = self._trailing_spaced_credential_width(remaining)
            if not credential_width:
                break
            dropped.extend(_DroppedToken(token, DropReason.CREDENTIAL) for token in remaining[-credential_width:])
            del remaining[-credential_width:]
        while remaining:
            if len(remaining) >= minimum_et_al_width and self._is_et_al_pair(remaining[-2:]):
                dropped.extend(_DroppedToken(token, DropReason.CONNECTOR) for token in remaining[-2:])
                del remaining[-2:]
                continue
            token = remaining[-1]
            attached_jr = self._split_attached_terminal_jr(token)
            if attached_jr is not None and suffix:
                return remaining, suffix, suffix_token, True
            if attached_jr is not None:
                remaining[-1], suffix_token = attached_jr
                suffix = "Jr."
                continue
            ambiguous_name_token = self._is_ambiguous_credential(token.text) and (
                (not has_external_name_context and len(remaining) == _TWO_COMPONENTS)
                or (has_external_name_context and len(remaining) == 1)
            )
            if self._is_trailing_credential(token.text) and not ambiguous_name_token:
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
            # or an external surname context supplies it directly.
            surname_like_ok = has_external_surname_context or self._has_local_surname_before_boundary(remaining)
            accepted_suffix = bool(
                candidate and (not is_roman or has_complete_name) and (not surname_like or surname_like_ok),
            )
            if accepted_suffix and suffix:
                return remaining, suffix, suffix_token, True
            if accepted_suffix:
                suffix = candidate
                suffix_token = replace(token, source_role="suffix")
                remaining.pop()
                continue
            break
        return remaining, suffix, suffix_token, False

    def _has_local_surname_before_boundary(self, remaining: list[_Token]) -> bool:
        """Return whether removing the boundary token leaves surname material."""
        surviving = remaining[:-1]
        if any(token.source_role for token in remaining):
            return any(token.source_role == "surname" and not self._is_initial(token.text) for token in surviving)
        return any(not self._is_initial(token.text) for token in surviving[1:])

    @staticmethod
    def _split_attached_terminal_jr(token: _Token) -> tuple[_Token, _Token] | None:
        """Split the exact terminal ``-Jr.`` suffix from a surviving name token."""
        marker = "-Jr."
        if not token.text.endswith(marker):
            return None
        name = token.text[: -len(marker)]
        if not name or not name[-1].isalpha():
            return None
        return (
            replace(token, text=name),
            _Token("Jr.", "suffix", token.position + len(name) + 1),
        )

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
            if self._is_trailing_credential(token.text, explicit_context=True):
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
        if self._is_trailing_credential(token.text, explicit_context=True):
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
    def _strip_attached_terminal_footnote(
        tokens: list[_Token],
        dropped: list[_DroppedToken],
    ) -> list[_Token]:
        """Strip closed author-footnote markers from the final personal-name token."""
        if not tokens:
            return []
        remaining = list(tokens)
        token = remaining[-1]
        if token.text.endswith(_MOJIBAKE_FOOTNOTE_ENDINGS):
            return remaining
        match = _TRAILING_FOOTNOTE_RE.fullmatch(token.text)
        if match is None:
            return remaining
        name = match.group("name")
        marker = match.group("marker")
        if len(name) == 1 and marker == "***":
            return remaining
        remaining[-1] = replace(token, text=name)
        dropped.append(
            _DroppedToken(
                _Token(marker, token.source_role, token.position + len(name)),
                DropReason.AFFILIATION,
            ),
        )
        return remaining

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

    @staticmethod
    def _strip_reviewed_structured_middle(
        middle: list[_Token],
        *,
        middle_surface: str,
        has_name_endpoints: bool,
        has_explicit_suffix: bool,
        dropped: list[_DroppedToken],
    ) -> tuple[list[_Token], str]:
        """Remove one reviewed credential phrase or move a medial Jr. to suffix."""
        if middle_surface in _REVIEWED_STRUCTURED_MIDDLE_CREDENTIALS and has_name_endpoints:
            dropped.extend(_DroppedToken(token, DropReason.CREDENTIAL) for token in middle)
            return [], ""
        if not has_name_endpoints or has_explicit_suffix:
            return list(middle), ""
        suffix_indices = [index for index, token in enumerate(middle) if token.text == "Jr."]
        if len(suffix_indices) != 1:
            return list(middle), ""
        suffix_index = suffix_indices[0]
        return [token for index, token in enumerate(middle) if index != suffix_index], "Jr."

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

    def _split_fused_initial_sequence_surname(self, tokens: list[_Token]) -> list[_Token]:
        """Split an explicitly dotted multi-initial prefix from its fused surname.

        The two-or-more-initial floor makes the boundary visible in ``A.D.Smith``
        while leaving ambiguous one-dot words such as ``St.John`` untouched.  This
        is the only initial expansion performed before surname inference.
        """
        if not tokens:
            return list(tokens)
        token = tokens[0]
        match = _FUSED_INITIAL_SEQUENCE_SURNAME_RE.fullmatch(token.text)
        if match is None or not self._is_full_name_token(match.group("surname")):
            return list(tokens)
        initials = match.group("initials")
        return [
            replace(token, text=initials),
            replace(
                token,
                text=match.group("surname"),
                position=token.position + len(initials),
            ),
            *tokens[1:],
        ]

    @staticmethod
    def _is_initial_letter(character: str) -> bool:
        """Return whether one letter belongs to a script with letter case.

        Initial punctuation applies to Latin, Greek, Cyrillic, and other cased
        alphabets.  Uncased native-script names such as Japanese ``純`` are full
        name tokens, not initials merely because they contain one code point.
        """
        return bool(
            len(character) == 1 and character.isalpha() and character.lower() != character.upper(),
        )

    @staticmethod
    def _packed_initial_letters(token: str) -> tuple[str, ...]:
        """Return atoms from a dotted initial bundle, never from ``AD``/``M.Yu.``."""
        cleaned = PersonNameNormalizationService._clean_name_token(token)
        if _MULTI_INITIAL_RE.fullmatch(cleaned) is None:
            return ()
        letters = tuple(character for character in cleaned if character.isalpha())
        return (
            letters
            if len(letters) >= _TWO_COMPONENTS
            and all(PersonNameNormalizationService._is_initial_letter(letter) for letter in letters)
            else ()
        )

    @staticmethod
    def _is_atomic_initial(token: str) -> bool:
        """Return whether one token is exactly one bare or dotted letter."""
        cleaned = PersonNameNormalizationService._clean_name_token(token)
        dotted = _INITIAL_RE.fullmatch(cleaned)
        return bool(
            PersonNameNormalizationService._is_initial_letter(cleaned)
            or (dotted is not None and PersonNameNormalizationService._is_initial_letter(dotted.group(1))),
        )

    def _expand_packed_initials(self, token: _Token, role: str) -> list[_Token]:
        """Expand one dotted bundle into canonicalizable semantic initial tokens."""
        letters = self._packed_initial_letters(token.text)
        if not letters:
            return [replace(token, source_role=role)]
        positions = [index for index, character in enumerate(token.text) if character.isalpha()]
        return [
            replace(
                token,
                text=f"{letter}.",
                source_role=role,
                position=token.position + positions[index],
            )
            for index, letter in enumerate(letters)
        ]

    def _apply_unbound_initial_policy(
        self,
        given: list[_Token],
        middle: list[_Token],
        surname: list[_Token],
        *,
        normalize_ambiguous_tail: bool = False,
    ) -> tuple[list[_Token], list[_Token], list[_Token]]:
        """Allocate unbound initials without revisiting the inferred surname floor.

        Packed and spaced variants converge here, after surname/order inference:
        the first personal initial may occupy ``given`` and every later unbound
        initial occupies ``middle``. For raw comma-free input, a full given name
        followed only by two or more initials keeps the final initial as the
        required surname floor, with the same canonical punctuation as the rest
        of the tail. Explicitly structured and comma-delimited boundaries are
        left intact. Explicitly hyphenated initials are never expanded, so they
        remain one compound given-name unit.
        """
        expanded_given = [unit for token in given for unit in self._expand_packed_initials(token, "given")]
        expanded_middle = [unit for token in middle for unit in self._expand_packed_initials(token, "middle")]

        normalized_surname = [replace(token, source_role="surname") for token in surname]
        initial_tail = [*middle, *surname]
        initial_tail_widths = [self._unbound_initial_width(token.text) for token in initial_tail]
        if (
            normalize_ambiguous_tail
            and any(self._is_full_name_token(token.text) for token in given)
            and all(initial_tail_widths)
            and sum(initial_tail_widths) >= _TWO_COMPONENTS
        ):
            expanded_tail = [unit for token in initial_tail for unit in self._expand_packed_initials(token, "middle")]
            expanded_middle = [replace(token, source_role="middle") for token in expanded_tail[:-1]]
            final_initial = expanded_tail[-1]
            cleaned_final = self._clean_name_token(final_initial.text)
            normalized_surname = [
                replace(
                    final_initial,
                    text=f"{cleaned_final.rstrip('.').upper()}.",
                    source_role="surname",
                ),
            ]

        normalized_given: list[_Token] = []
        promoted_middle: list[_Token] = []
        for token in expanded_given:
            if self._is_atomic_initial(token.text) and normalized_given:
                promoted_middle.append(replace(token, source_role="middle"))
            else:
                normalized_given.append(replace(token, source_role="given"))
        return (
            normalized_given,
            [*promoted_middle, *(replace(token, source_role="middle") for token in expanded_middle)],
            normalized_surname,
        )

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

    def _packed_surname_first_roles(
        self,
        tokens: list[_Token],
    ) -> tuple[list[_Token], list[_Token], list[_Token]] | None:
        """Repair an initial tail only when structured input supplies order evidence."""
        if len(tokens) < _TWO_COMPONENTS or not self._is_full_name_token(tokens[0].text):
            return None
        trailing = tokens[1:]
        widths = [self._unbound_initial_width(token.text) for token in trailing]
        if not all(widths) or sum(widths) < _TWO_COMPONENTS:
            return None
        return (
            [replace(trailing[0], source_role="given")],
            [replace(token, source_role="middle") for token in trailing[1:]],
            [replace(tokens[0], source_role="surname")],
        )

    def _unbound_initial_width(self, token: str) -> int:
        """Return the number of semantic initials in an unhyphenated token."""
        packed = self._packed_initial_letters(token)
        if packed:
            return len(packed)
        return 1 if self._is_atomic_initial(token) else 0

    def _two_token_surname_after_initial_run(self, tokens: list[_Token]) -> int | None:
        """Find a compound-surname floor without relying on typographic token count."""
        if len(tokens) < _THREE_COMPONENTS:
            return None
        index = 1 if self._is_full_name_token(tokens[0].text) else 0
        initial_count = 0
        while index < len(tokens):
            width = self._unbound_initial_width(tokens[index].text)
            if not width:
                break
            initial_count += width
            index += 1
        minimum_initials = 1 if self._is_full_name_token(tokens[0].text) else _TWO_COMPONENTS
        if (
            initial_count >= minimum_initials
            and len(tokens) - index == _TWO_COMPONENTS
            and all(self._is_full_name_token(token.text) for token in tokens[index:])
        ):
            return index
        return None

    def _infer_regular_roles(self, tokens: list[_Token]) -> tuple[list[_Token], list[_Token], list[_Token]]:
        if len(tokens) == 1:
            return [replace(tokens[0], source_role="given")], [], []

        if particle_roles := self._particle_surname_first_roles(tokens):
            return particle_roles

        surname_start = len(tokens) - 1
        strong_particle = self._find_strong_particle_span(tokens)
        if strong_particle is not None and (strong_particle[0] > _TWO_COMPONENTS or self._is_initial(tokens[0].text)):
            anchor = strong_particle[0] - 1
            # The token before the particle joins the surname only if it is a real
            # surname head; a leading given initial must not be pulled in (and then
            # duplicated as both given and surname): "M. van der Klis" -> given "M.".
            if not self._is_full_name_token(tokens[anchor].text):
                anchor = strong_particle[0]
            surname_start = anchor
        elif particle_positions := [
            index for index, token in enumerate(tokens[1:-1], start=1) if self._particle_key(token.text) in _FAMILY_PARTICLES
        ]:
            surname_start = particle_positions[0]
        elif len(tokens) == _FOUR_COMPONENTS and self._is_initial(tokens[1].text) and not self._is_initial(tokens[2].text):
            surname_start = 2
        elif initial_surname_start := self._two_token_surname_after_initial_run(tokens):
            surname_start = initial_surname_start
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

    def _repair_structured_roles(  # noqa: C901, PLR0911, PLR0912
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
                fused_sequence = _FUSED_INITIAL_SEQUENCE_SURNAME_RE.fullmatch(surname[0].text)
                if fused_sequence is not None and self._is_full_name_token(fused_sequence.group("surname")):
                    initials = fused_sequence.group("initials")
                    return (
                        [replace(surname[0], text=initials, source_role="given")],
                        [],
                        [
                            replace(
                                surname[0],
                                text=fused_sequence.group("surname"),
                                source_role="surname",
                                position=surname[0].position + len(initials),
                            ),
                        ],
                    )
                fused = _FUSED_INITIAL_SURNAME_RE.fullmatch(surname[0].text)
                if fused is not None and self._is_full_name_token(fused.group("surname")):
                    initial = fused.group("initial")
                    family = fused.group("surname")
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
                if packed_roles := self._packed_surname_first_roles(surname):
                    return packed_roles
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
        if packed_roles := self._packed_surname_first_roles(tokens):
            return packed_roles
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
            if cleaned[0] == "-" or (cleaned[0] == "'" and cleaned.casefold() != "'t") or cleaned[-1] == "-":
                return f"name joiner is not between letters: {token.text!r}"
            if any(not self._allowed_name_character(character) for character in cleaned):
                return f"unsupported character in name token: {token.text!r}"
        return None

    @staticmethod
    def _allowed_name_character(character: str) -> bool:
        return character.isalpha() or unicodedata.category(character).startswith("M") or character in "-'."

    def _canonical_tokens(self, tokens: list[_Token], role: str) -> tuple[str, ...]:
        return tuple(self._canonicalize_token(token.text, role) for token in tokens)

    def _canonicalize_token(self, token: str, role: str) -> str:  # noqa: C901, PLR0911, PLR0912
        cleaned = self._clean_name_token(token)
        if role in {"middle", "surname"} and cleaned in _LOWERCASE_RELATIONAL_TOKENS:
            return cleaned
        if _HYPHEN_INITIAL_RE.fullmatch(cleaned) and all(
            self._is_initial_letter(part.rstrip(".")) for part in cleaned.split("-")
        ):
            return "-".join(f"{part.rstrip('.').upper()}." for part in cleaned.split("-"))
        if role != "surname" and self._is_initial_letter(cleaned):
            return cleaned.upper() + "."
        dotted = _INITIAL_RE.fullmatch(cleaned)
        if dotted is not None and self._is_initial_letter(dotted.group(1)):
            return dotted.group(1).upper() + "."
        if self._packed_initial_letters(cleaned) or self._is_uppercase_dotted_abbreviation(cleaned):
            return "".join(character.upper() if character.isalpha() else character for character in cleaned)
        if role != "surname" and cleaned.isalpha() and cleaned.isupper() and len(cleaned) == _TWO_COMPONENTS:
            # A two-letter all-caps given token is initials (e.g. "MS", "MA", "MD") — keep it
            # all-caps rather than title-casing to "Ms"/"Ma", which reads as an honorific. A
            # genuine Title-case honorific never reaches here (it's dropped upstream).
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
        if _DOTTED_INITIAL_SEQUENCE_RE.fullmatch(part):
            return False
        return bool(
            (part.startswith("Mc") and len(part) > _MC_PREFIX_LENGTH and part[2:].isupper())
            or (len(part) > _TWO_COMPONENTS and part[0].isupper() and part[1].islower() and part[2:].isupper()),
        )

    @staticmethod
    def _clean_name_token(token: str) -> str:
        if PersonNameNormalizationService._is_parenthesized_name_token(token):
            return token
        if len(token) > _TWO_COMPONENTS and token.startswith("'") and token.endswith("'"):
            token = token[1:-1]
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
        return _canonical_suffix_surface(token, explicit=explicit)

    def _is_trailing_credential(self, token: str, *, explicit_context: bool = False) -> bool:
        """Match a credential at a reviewed trailing boundary."""
        return _is_trailing_credential_surface(
            token,
            explicit_context=explicit_context,
        )

    def _is_credential(self, token: str, *, explicit_context: bool = False) -> bool:
        return _is_credential_boundary_surface(token, explicit_context=explicit_context)

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
    ) -> NameComponents:
        """Build source components with display order derived from token positions."""
        by_role = {
            "given": sorted(given, key=lambda token: token.position),
            "middle": sorted(middle, key=lambda token: token.position),
            "surname": sorted(surname, key=lambda token: token.position),
            "suffix": sorted(suffix, key=lambda token: token.position),
        }
        order = tuple(
            role
            for role, _token in sorted(
                ((role, token) for role, tokens in by_role.items() for token in tokens),
                key=lambda item: item[1].position,
            )
        )

        def source_text(token: _Token) -> str:
            return token.source_text or token.text

        return NameComponents(
            given_name=" ".join(source_text(token) for token in by_role["given"]),
            middle_name=" ".join(source_text(token) for token in by_role["middle"]),
            surname=" ".join(source_text(token) for token in by_role["surname"]),
            suffix=" ".join(source_text(token) for token in by_role["suffix"]),
            given_tokens=tuple(source_text(token) for token in by_role["given"]),
            middle_tokens=tuple(source_text(token) for token in by_role["middle"]),
            surname_tokens=tuple(source_text(token) for token in by_role["surname"]),
            suffix_tokens=tuple(source_text(token) for token in by_role["suffix"]),
            order=order,
        )

    def _looks_like_two_complete_names(self, left: list[_Token], right: list[_Token]) -> bool:
        if len(left) < _TWO_COMPONENTS or len(right) < _TWO_COMPONENTS:
            return False
        if any(self._particle_key(token.text) in _FAMILY_PARTICLES for token in [*left[:-1], *right[:-1]]):
            return False
        return not any(self._is_initial(token.text) for token in [*left, *right])

    @classmethod
    def _has_nonfinal_org_preposition(cls, raw_tokens: list[str]) -> bool:
        # Org prepositions ("für"/"voor") mark an org only when a token follows them — a
        # trailing occurrence is a real surname ("Gabriella Für", "Michael J. Voor").
        return any(cls._compact_key(token) in _ORG_PREPOSITION_WORDS for token in raw_tokens[:-1])

    @classmethod
    def _is_collision_surname_shape(cls, raw_tokens: list[str]) -> bool:
        # A clean personal name whose SURNAME is a collision org word ("Company"/"Center"):
        # 2-3 tokens, the collision word is the final (surname) token, at least one leading
        # token is a single-letter initial (a strong person signal an org lacks), and no
        # other org / function / preposition word appears. "David M. Center" -> person;
        # "Cosmic Dawn Center" / "Media Center" / "ABC Trading Company" -> still org.
        if not _TWO_COMPONENTS <= len(raw_tokens) < _FOUR_COMPONENTS:
            return False
        if cls._compact_key(raw_tokens[-1]) not in _SURNAME_COLLISION_ORG_WORDS:
            return False
        leading = raw_tokens[:-1]
        if any(
            cls._compact_key(token) in _ORGANIZATION_WORDS
            or cls._compact_key(token) in _FUNCTION_WORDS
            or cls._compact_key(token) in _ORG_PREPOSITION_WORDS
            for token in leading
        ):
            return False
        return any(len(cls._compact_key(token)) == 1 for token in leading)

    def _non_person_reason(self, surface: str) -> str | None:  # noqa: PLR0911
        lowered = surface.casefold()
        if "@" in surface or "://" in surface:
            return "contact or URL input"
        if any(separator in surface for separator in (";", "&", "|")):
            return "multiple-name separator"
        and_match = _WORD_AND_RE.search(surface)
        if and_match is not None and and_match.start() > 0:
            return "author-list connector"
        raw_tokens = _TOKEN_RE.findall(lowered)
        words = {self._compact_key(token) for token in raw_tokens}
        org_hit = words & _ORGANIZATION_WORDS
        if org_hit and not (org_hit <= _SURNAME_COLLISION_ORG_WORDS and self._is_collision_surname_shape(raw_tokens)):
            return "organization input"
        if self._has_nonfinal_org_preposition(raw_tokens):
            return "organization input"
        # An org word fused into a hyphenated token ("Koch-Institut", "Ruhr-Universität",
        # "Courier-Journal") — only for words that are never part of a real surname.
        for token in raw_tokens:
            if "-" in token and any(self._compact_key(part) in _HYPHEN_ORG_WORDS for part in token.split("-")):
                return "organization input"
        # Reject the literal placeholder "null"/"null null" only when the WHOLE name is
        # "null" tokens — "Null" is a real surname ("Linda M. Null"), so a mixed name keeps it.
        if words and words <= {"null"}:
            return "null literal"
        if words and words <= _FUNCTION_WORDS:
            return "function words only"
        return None

    @staticmethod
    def _is_initial(token: str) -> bool:
        """Return whether a token is any initial shape: atomic, packed, or hyphenated."""
        cleaned = PersonNameNormalizationService._clean_name_token(token)
        if PersonNameNormalizationService._is_atomic_initial(cleaned) or PersonNameNormalizationService._packed_initial_letters(
            cleaned,
        ):
            return True
        hyphenated = _HYPHEN_INITIAL_RE.fullmatch(cleaned)
        return bool(
            hyphenated is not None
            and all(PersonNameNormalizationService._is_initial_letter(part.rstrip(".")) for part in cleaned.split("-")),
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

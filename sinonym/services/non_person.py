"""Conservative detection for inputs that are not single personal names."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from sinonym.services.name_lookup import SurnameResolver

if TYPE_CHECKING:
    from sinonym.coretypes import ChineseNameConfig
    from sinonym.services.initialization import NameDataStructures
    from sinonym.services.normalization import NormalizationService

NON_PERSON_FAILURE_REASON = "not a personal name"

MIN_CJK_NON_PERSON_PREFIX_CHARS = 2
MIN_AUTHOR_LIST_LATIN_TOKENS = 6
MIN_AUTHOR_LIST_SURNAME_TOKENS = 3
MIN_TRANSLITERATED_CJK_CHARS = 2
MIN_REVIEWED_METADATA_TOKENS = 2
MIN_REVIEWED_LEGAL_TOKENS = 3
MAX_REVIEWED_REFORMATION_TOKENS = 5

REVIEWED_PLACEHOLDER_PHRASES = frozenset(
    {
        "None None",
        "Not Available Not Available",
        "Unknown Author",
        "undefined No authorship indicated",
    },
)
REVIEWED_NON_PERSON_LITERALS = frozenset(
    {
        "Anthony C. Laborte, Marissa C. Hitalia*",
        "Array BioPharma",
        "Professur Arbeitswissenschaft",
        "Professur Fördertechnik",
        "Professur Grundbau",
        "Professur Rechnernetze",
    },
)
REVIEWED_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
REVIEWED_MONTH_RANGE_RE = re.compile(
    rf"(?:{'|'.join(REVIEWED_MONTHS)})-(?:{'|'.join(REVIEWED_MONTHS)})",
    re.IGNORECASE,
)
REVIEWED_ORGANIZATION_TOKENS = frozenset(
    {
        "kabupaten",
        "libraries",
    },
)
REVIEWED_HANGUL_ORGANIZATION_MARKERS = ("대학교", "연구소", "연구원", "학회", "위원회")
REVIEWED_HANGUL_ORGANIZATION_SUFFIXES = (*REVIEWED_HANGUL_ORGANIZATION_MARKERS, "학부", "자료")
REVIEWED_HYPHENATED_SERVICES_RE = re.compile(
    r"(?:^|[^A-Za-z])(?:[A-Za-z]+-services|services-[A-Za-z]+)(?:[^A-Za-z]|$)",
    re.IGNORECASE,
)
REVIEWED_SERVICES_PERSON_TAIL_RE = re.compile(
    r"(?:^|\s)(?i:services)-[A-Z][A-Za-z'-]+(?:\s+[A-Z][.]?)?\s+[A-Z][A-Za-z'-]+$",
)


def _collapse_whitespace(value: str) -> str:
    """Strip and collapse whitespace without changing lexical text."""
    return " ".join(value.split())


def _reviewed_literal_pattern(raw_name: str) -> str | None:
    """Return one exact reviewed placeholder or month-range pattern."""
    if raw_name in REVIEWED_PLACEHOLDER_PHRASES:
        return "placeholder_literal"
    if raw_name in REVIEWED_NON_PERSON_LITERALS:
        return "non_person_literal"
    if REVIEWED_MONTH_RANGE_RE.fullmatch(raw_name):
        return "month_range"
    return None


LATIN_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)?")
ASCII_WORD_RE = re.compile(r"[A-Za-z]+")
ASCII_ALNUM_WORD_RE = re.compile(r"[a-z0-9]+")
INITIAL_RE = re.compile(r"[^\W\d_]")


def reviewed_non_person_text_pattern(raw_name: str) -> str | None:
    """Return one reviewed literal or organization pattern for a raw surface."""
    surface = _collapse_whitespace(raw_name)
    literal_pattern = _reviewed_literal_pattern(surface)
    if literal_pattern is not None:
        return literal_pattern
    words = set(ASCII_ALNUM_WORD_RE.findall(surface.casefold()))
    if words & REVIEWED_ORGANIZATION_TOKENS:
        return "organization_token"
    if REVIEWED_HYPHENATED_SERVICES_RE.search(surface) and not REVIEWED_SERVICES_PERSON_TAIL_RE.search(surface):
        return "hyphenated_services"
    return None


def _is_reviewed_hangul_organization(fields: tuple[str, str, str]) -> bool:
    """Match an organization-only Hangul source shape with no person token."""
    hangul_tokens = fields[2].split()
    all_tokens_are_organizational = all(
        any(token.endswith(marker) for marker in REVIEWED_HANGUL_ORGANIZATION_SUFFIXES) for token in hangul_tokens
    )
    return bool(not fields[0] and not fields[1] and hangul_tokens and all_tokens_are_organizational)


STRONG_CJK_NON_PERSON_MARKERS = (
    "大学",
    "大學",
    "学院",
    "學院",
    "研究所",
    "实验室",
    "實驗室",
    "编辑部",
    "編輯部",
    "科学院",
    "科學院",
    "公司",
    "有限公司",
    "研究中心",
    "重点实验室",
    "国家实验室",
    "物理系",
    "学部",
    "學部",
)

STANDALONE_CJK_NON_PERSON_MARKERS = frozenset(STRONG_CJK_NON_PERSON_MARKERS)
CJK_NON_PERSON_SUFFIX_MARKERS = STRONG_CJK_NON_PERSON_MARKERS

REVIEWED_CREDENTIAL_ONLY_TOKENS = frozenset(
    {
        "A-GNP",
        "AOCNP",
        "ARNP",
        "BEng",
        "BSc",
        "CTRS",
        "DNP",
        "Dr.-Ing",
        "Dr.-Ing.",
        "FACS",
        "FEBS",
        "FRACP",
        "M.Si",
        "M.Si.",
        "MBA",
        "MBBS",
        "MEng",
        "MPH",
        "MSc",
        "Ph.D",
        "Ph.D.",
        "PhD",
        "PhD.",
        "ScD",
    },
)
REVIEWED_EDUCATION_CONNECTORS = frozenset(
    {
        "adalah",
        "dan",
        "dalam",
        "dengan",
        "di",
        "ke",
        "melalui",
        "oleh",
        "pada",
        "sebagai",
        "terhadap",
        "untuk",
        "yang",
    },
)
REVIEWED_RESEARCH_SCHOLAR_PREFIXES = frozenset(
    {
        "",
        "associate professor head ph d",
        "asstt",
        "d phil",
        "dr",
        "dr ph d",
        "fulltime",
        "head ph d",
        "m e",
        "m ed",
        "m phil",
        "m sc",
        "m tech",
        "mphil",
        "msc",
        "mtech",
        "p g",
        "p hd",
        "pg",
        "pg m phil",
        "ph d",
        "phd",
        "professor ph d",
        "professor postdoc",
        "professor2",
        "research scholar",
        "retd ph d",
        "scholar m e",
        "time ph d",
    },
)


def reviewed_non_person_source_pattern(  # noqa: C901 - one branch per reviewed source grammar.
    first_name: str | None,
    middle_names: str | None,
    last_name: str | None,
    suffix: str | None = None,
) -> str | None:
    """Return the reviewed semantic pattern proving one source row is not a person.

    The predicates are the zero-counterexample subtypes from the complete
    691-million-row source census. They inspect the original structured fields
    and are deliberately separate from parser failure, which can still occur
    for real people.
    """
    fields = tuple((value or "").strip() for value in (first_name, middle_names, last_name))
    if (suffix or "").strip():
        return None
    nonempty_fields = tuple(field for field in fields if field)
    if nonempty_fields and all(field in REVIEWED_CREDENTIAL_ONLY_TOKENS for field in nonempty_fields):
        return "credential_only"

    if _is_reviewed_hangul_organization(fields):
        return "hangul_organization_marker"

    raw_name = " ".join(_collapse_whitespace(value) for value in fields if value)
    literal_pattern = reviewed_non_person_text_pattern(raw_name)
    if literal_pattern is not None:
        return literal_pattern
    folded = raw_name.lower()
    if not (
        any(marker in folded for marker in ("stadt", "undang", "pendidikan", "reformation"))
        or ("research" in folded and "scholar" in folded)
    ):
        return None

    ascii_words = ASCII_ALNUM_WORD_RE.findall(folded)
    first_key = " ".join(ASCII_ALNUM_WORD_RE.findall(fields[0].lower()))
    pattern = None
    if first_key == "stadt" and len(ascii_words) >= MIN_REVIEWED_METADATA_TOKENS:
        pattern = "municipal_text"
    elif first_key == "undang undang" and len(ascii_words) >= MIN_REVIEWED_LEGAL_TOKENS:
        pattern = "legal_text"
    elif first_key == "pendidikan" and len(ascii_words) >= MIN_REVIEWED_METADATA_TOKENS and _reviewed_education_text(raw_name):
        pattern = "education_text"
    elif _reviewed_research_scholar_metadata(raw_name):
        pattern = "research_scholar_metadata"
    elif (
        MIN_REVIEWED_METADATA_TOKENS <= len(ascii_words) <= MAX_REVIEWED_REFORMATION_TOKENS
        and "reformation" in ascii_words
        and raw_name.upper() == raw_name
        and raw_name.lower() != raw_name
    ):
        pattern = "reformation_fragment"
    return pattern


def _reviewed_education_text(raw_name: str) -> bool:
    """Match the reviewed-safe casing and connector union for education text."""
    if raw_name.upper() == raw_name and raw_name.lower() != raw_name:
        return True
    if _prefix_has_lowercase_continuation(raw_name, "pendidikan ") or _prefix_has_lowercase_continuation(
        raw_name,
        "Pendidikan ",
    ):
        return True
    title_prefix = "Pendidikan "
    if not raw_name.startswith(title_prefix):
        return False
    remainder = raw_name[len(title_prefix) :]
    if remainder.upper() == remainder and remainder.lower() != remainder:
        return True
    return bool(set(_metadata_words(remainder).split()) & REVIEWED_EDUCATION_CONNECTORS)


def _reviewed_research_scholar_metadata(raw_name: str) -> bool:
    """Match a closed degree/status grammar followed by ``Research Scholar``."""
    words = _metadata_words(raw_name)
    suffix = "research scholar"
    if words == suffix:
        return True
    if not words.endswith(f" {suffix}"):
        return False
    return words[: -len(suffix)].strip() in REVIEWED_RESEARCH_SCHOLAR_PREFIXES


def _prefix_has_lowercase_continuation(raw_name: str, prefix: str) -> bool:
    """Return whether an exact prefix is followed by a lowercase letter."""
    if not raw_name.startswith(prefix):
        return False
    first_letter = next((character for character in raw_name[len(prefix) :] if character.isalpha()), "")
    return bool(first_letter and first_letter.islower())


def _metadata_words(value: str) -> str:
    """Normalize punctuation and case for the closed metadata grammars."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


class NonPersonInputDetectionService:
    """Detect obvious organization or multi-author cells before name parsing."""

    def __init__(
        self,
        config: ChineseNameConfig,
        normalizer: NormalizationService,
        data: NameDataStructures,
    ) -> None:
        self._config = config
        self._normalizer = normalizer
        self._data = data
        self._surname_resolver = SurnameResolver(self._data, self._normalizer)

    def failure_reason(self, raw_name: str) -> str | None:
        """Return a failure reason when the input is clearly not one personal name."""
        if raw_name.isascii():
            return NON_PERSON_FAILURE_REASON if self._has_latin_author_list_shape(raw_name) else None
        if (
            self._has_cjk_non_person_marker(raw_name)
            or self._has_latin_author_list_shape(raw_name)
            or self._has_mixed_initial_cjk_transliteration_shape(raw_name)
        ):
            return NON_PERSON_FAILURE_REASON
        return None

    def _has_cjk_non_person_marker(self, raw_name: str) -> bool:
        """Return whether a CJK string contains strong organization/editor evidence."""
        cjk_chunks = self._cjk_chunks(raw_name)
        return any(chunk in STANDALONE_CJK_NON_PERSON_MARKERS for chunk in cjk_chunks) or any(
            self._has_marker_suffix(chunk) for chunk in cjk_chunks
        )

    @staticmethod
    def _has_marker_suffix(cjk_chunk: str) -> bool:
        """Return whether a CJK chunk has an organization marker suffix."""
        return any(
            cjk_chunk.endswith(marker) and len(cjk_chunk) - len(marker) >= MIN_CJK_NON_PERSON_PREFIX_CHARS
            for marker in CJK_NON_PERSON_SUFFIX_MARKERS
        )

    def _has_mixed_initial_cjk_transliteration_shape(self, raw_name: str) -> bool:
        """Return whether mixed input is an initial plus CJK Western transliteration.

        The foreign-name convention binds a Latin initial to its Han transliteration
        with a dot or interpunct ("G.霍弗", "H·纳格尔斯"). A genuine Chinese name that
        carries a trailing Latin middle initial ("李 小明 G.") instead separates the
        initial from the Han tokens with whitespace, so the dot-bridge is what tells
        the two apart — a bare initial count or trailing period is not enough.
        """
        if not self._config.cjk_pattern.search(raw_name) or not self._config.ascii_alpha_pattern.search(raw_name):
            return False

        ascii_tokens = ASCII_WORD_RE.findall(raw_name)
        if not ascii_tokens or any(len(token) > 1 for token in ascii_tokens):
            return False

        cjk_chunks = self._cjk_chunks(raw_name)
        if not any(len(chunk) >= MIN_TRANSLITERATED_CJK_CHARS for chunk in cjk_chunks):
            return False
        if len(ascii_tokens) == 1 and any(self._looks_like_chinese_name_chunk(chunk) for chunk in cjk_chunks):
            return False

        normalized_input = self._normalizer.apply(raw_name)
        if self._normalizer.classify_script_representation(normalized_input) == "bilingual_aligned":
            return False

        return self._has_initial_cjk_separator_bridge(raw_name)

    def _looks_like_chinese_name_chunk(self, cjk_chunk: str) -> bool:
        """Return whether a CJK chunk has strong Chinese surname evidence."""
        normalized_input = self._normalizer.apply(cjk_chunk)
        tokens = list(normalized_input.roman_tokens)
        if len(tokens) < MIN_TRANSLITERATED_CJK_CHARS:
            return False

        normalized_tokens = [self._normalizer.norm(token) for token in tokens]
        if self._surname_resolver.evidence_is_dominant_surname(tokens[0]):
            return True

        first_two = " ".join(normalized_tokens[:2])
        return first_two in self._data.compound_surnames or first_two in self._data.compound_surnames_normalized

    def _has_initial_cjk_separator_bridge(self, raw_name: str) -> bool:
        """Return whether a separator run directly joins a Latin initial to a CJK run.

        Scans each maximal separator *run* (``config.sep_pattern`` matches whole
        runs of separator characters) and inspects the two characters flanking the
        run — ``raw_name[start-1]`` and ``raw_name[end]`` — with no whitespace
        skipping: a bridge exists when one directly-adjacent side is a Latin letter
        and the other is CJK, in either order. Scanning whole runs is what catches
        multi-separator bridges ("G..霍弗", "G··霍弗", "H··纳格尔斯") that a
        one-character-at-a-time scan would miss, since the interior separators'
        immediate neighbours are other separators. Direct adjacency is the
        discriminator — a Latin initial that is whitespace-separated from the Han
        tokens ("李 小明 G.", "G. 李小明", "李 小明 · G") has a space flanking the run
        and does not bridge, which is what tells a genuine Chinese name carrying a
        Latin middle initial apart from a Western transliteration.

        For chained initials the run adjacent to the Han run is the one that
        bridges ("J·G·马尔钦凯维奇" bridges on the G·马 separator; "罗伯特·M·威恩斯坦"
        bridges on the 特·M separator), so those foreign transliterations are still
        caught.

        Separator membership is derived from ``config.sep_pattern``, keeping this
        rule in sync with the single canonical separator definition instead of a
        private hardcoded set.
        """
        sep_pattern = self._config.sep_pattern
        cjk_pattern = self._config.cjk_pattern
        for match in sep_pattern.finditer(raw_name):
            left = raw_name[match.start() - 1] if match.start() > 0 else ""
            right = raw_name[match.end()] if match.end() < len(raw_name) else ""
            left_is_cjk = bool(left) and bool(cjk_pattern.search(left))
            right_is_cjk = bool(right) and bool(cjk_pattern.search(right))
            left_is_latin = bool(left) and left.isascii() and left.isalpha()
            right_is_latin = bool(right) and right.isascii() and right.isalpha()
            if left_is_cjk and right_is_latin and self._is_trailing_latin_initial_suffix(raw_name, match.end()):
                continue
            if (left_is_latin and right_is_cjk) or (left_is_cjk and right_is_latin):
                return True
        return False

    @staticmethod
    def _is_trailing_latin_initial_suffix(raw_name: str, letter_index: int) -> bool:
        """Return whether a CJK-joined Latin letter is only a final middle initial."""
        if letter_index >= len(raw_name):
            return False
        letter = raw_name[letter_index]
        if not (letter.isascii() and letter.isalpha()):
            return False
        suffix = raw_name[letter_index + 1 :]
        return not any(char.isascii() and char.isalpha() for char in suffix)

    def _cjk_chunks(self, raw_name: str) -> list[str]:
        """Return contiguous CJK runs split by non-CJK separators."""
        if raw_name.isascii():
            return []
        chunks: list[str] = []
        current: list[str] = []

        for char in raw_name:
            if self._config.cjk_pattern.search(char):
                current.append(char)
                continue

            if current:
                chunks.append("".join(current))
                current = []

        if current:
            chunks.append("".join(current))

        return chunks

    def _has_latin_author_list_shape(self, raw_name: str) -> bool:
        """Return whether a Latin string looks like several Chinese author names collapsed together."""
        if not raw_name.isascii() and self._config.cjk_pattern.search(raw_name):
            return False

        tokens = LATIN_WORD_RE.findall(raw_name)
        if len(tokens) < MIN_AUTHOR_LIST_LATIN_TOKENS:
            return False

        surname_like_positions = [index for index, token in enumerate(tokens) if self._is_surname_like_token(token)]
        if len(surname_like_positions) < MIN_AUTHOR_LIST_SURNAME_TOKENS:
            return False

        has_early_surname = any(index <= 1 for index in surname_like_positions)
        has_middle_surname = any(1 < index < len(tokens) - 2 for index in surname_like_positions)
        has_late_surname = any(index >= len(tokens) - 2 for index in surname_like_positions)
        return has_early_surname and has_middle_surname and has_late_surname

    def _is_surname_like_token(self, token: str) -> bool:
        """Return whether a Latin token is a Chinese surname cue."""
        if INITIAL_RE.fullmatch(token):
            return False

        parts = [part for part in re.split(r"[-']", token) if part]
        if not parts:
            return False

        normalized = " ".join(self._normalizer.norm(part) for part in parts)
        compact = normalized.replace(" ", "")

        # Author-list detection combines split-token membership with compound
        # shape maps; keep this local until the resolver owns compound answers.
        return bool(
            self._data.is_surname(token, normalized)
            or self._data.is_surname(token, compact)
            or normalized in self._data.compound_surnames_normalized
            or compact in self._data.compound_original_format_map,
        )

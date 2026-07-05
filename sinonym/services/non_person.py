"""Conservative detection for inputs that are not single personal names."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sinonym.coretypes import ChineseNameConfig
    from sinonym.services.initialization import NameDataStructures
    from sinonym.services.normalization import NormalizationService

NON_PERSON_FAILURE_REASON = "not a personal name"

MIN_CJK_NON_PERSON_PREFIX_CHARS = 2
MIN_AUTHOR_LIST_LATIN_TOKENS = 6
MIN_AUTHOR_LIST_SURNAME_TOKENS = 3
MIN_TRANSLITERATED_CJK_CHARS = 2

LATIN_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)?")
ASCII_WORD_RE = re.compile(r"[A-Za-z]+")
INITIAL_RE = re.compile(r"[^\W\d_]")

STRONG_CJK_NON_PERSON_MARKERS = (
    "大学",
    "学院",
    "研究所",
    "实验室",
    "编辑部",
    "科学院",
    "公司",
    "有限公司",
    "研究中心",
    "重点实验室",
    "国家实验室",
    "物理系",
    "学部",
)

STANDALONE_CJK_NON_PERSON_MARKERS = frozenset(STRONG_CJK_NON_PERSON_MARKERS)
CJK_NON_PERSON_SUFFIX_MARKERS = STRONG_CJK_NON_PERSON_MARKERS


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

    def failure_reason(self, raw_name: str) -> str | None:
        """Return a failure reason when the input is clearly not one personal name."""
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

        normalized_input = self._normalizer.apply(raw_name)
        if self._normalizer.classify_script_representation(normalized_input) == "bilingual_aligned":
            return False

        return self._has_initial_cjk_separator_bridge(raw_name)

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
        if self._config.cjk_pattern.search(raw_name):
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

        return bool(
            self._data.is_surname(token, normalized)
            or self._data.is_surname(token, compact)
            or normalized in self._data.compound_surnames_normalized
            or compact in self._data.compound_original_format_map,
        )

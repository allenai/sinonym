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

LATIN_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)?")
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
        if self._has_cjk_non_person_marker(raw_name) or self._has_latin_author_list_shape(raw_name):
            return NON_PERSON_FAILURE_REASON
        return None

    def _has_cjk_non_person_marker(self, raw_name: str) -> bool:
        """Return whether a CJK string contains strong organization/editor evidence."""
        cjk_chunks = self._cjk_chunks(raw_name)
        if any(chunk in STANDALONE_CJK_NON_PERSON_MARKERS for chunk in cjk_chunks):
            return True
        if any(self._has_marker_suffix(chunk) for chunk in cjk_chunks):
            return True
        return False

    @staticmethod
    def _has_marker_suffix(cjk_chunk: str) -> bool:
        """Return whether a CJK chunk has an organization marker suffix."""
        return any(
            cjk_chunk.endswith(marker) and len(cjk_chunk) - len(marker) >= MIN_CJK_NON_PERSON_PREFIX_CHARS
            for marker in CJK_NON_PERSON_SUFFIX_MARKERS
        )

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

        surname_like_positions = [
            index for index, token in enumerate(tokens) if self._is_surname_like_token(token)
        ]
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

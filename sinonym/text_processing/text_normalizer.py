"""
Core text normalization utilities for Chinese name processing.

This module contains pure text normalization functions extracted from
the NormalizationService to provide clean, focused text processing.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from sinonym.chinese_names_data import (
    NON_WADE_GILES_SYLLABLE_RULES,
    ONE_LETTER_RULES,
    ROMANIZATION_EXCEPTIONS,
    WADE_GILES_SYLLABLE_RULES,
)
from sinonym.patterns import (
    SUFFIX_REGEX,
    SUFFIX_REPLACEMENTS,
    WADE_GILES_REGEX,
    WADE_GILES_REPLACEMENTS,
)


class TextNormalizer:
    """Pure text normalization utilities for Chinese name processing."""

    def __init__(self, config):
        self._config = config

    @lru_cache(maxsize=32_768)
    def normalize_token(self, token: str) -> str:
        """
        Normalize a token through the full romanization pipeline.

        This method applies the complete normalization pipeline:
        1. Unicode normalization and diacritics removal
        2. Non-Wade-Giles romanization precedence system
        3. Wade-Giles conversions
        4. Final cleanup (apostrophes, hyphens)

        Args:
            token: Input token to normalize

        Returns:
            Fully normalized token
        """
        # Step 1: Normalize Unicode and remove diacritical marks, then lowercase
        normalized = unicodedata.normalize("NFD", token)
        without_diacritics = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        low = without_diacritics.lower()

        # Step 2: Apply non-Wade-Giles romanization precedence system
        mapped = ROMANIZATION_EXCEPTIONS.get(low)
        if mapped:
            return mapped

        mapped = NON_WADE_GILES_SYLLABLE_RULES.get(low)
        if mapped:
            return mapped

        mapped = ONE_LETTER_RULES.get(low)
        if mapped:
            return mapped

        # Step 3: Apply Wade-Giles conversions BEFORE removing apostrophes
        wade_giles_result = self._apply_unified_wade_giles(low)

        # Step 4: Handle special Wade-Giles cases only
        if wade_giles_result == "qen":
            wade_giles_result = "chen"

        # Step 5: Remove apostrophes and hyphens from the final result
        return wade_giles_result.translate(self._config.hyphens_apostrophes_tr)

    def _apply_unified_wade_giles(self, token: str) -> str:
        """
        Unified Wade-Giles conversion with explicit precedence handling.

        Args:
            token: Lowercase token WITH apostrophes intact

        Returns:
            Converted token
        """
        # Step 1: Syllable-level Wade-Giles exceptions (highest precedence)
        if token in WADE_GILES_SYLLABLE_RULES:
            return WADE_GILES_SYLLABLE_RULES[token]

        # Step 2: Prefix-level Wade-Giles conversions
        if token == "j":
            result = "r"
        else:
            def wade_giles_replacer(match):
                for i, group in enumerate(match.groups(), 1):
                    if group is not None:
                        return WADE_GILES_REPLACEMENTS[i - 1]
                return match.group(0)

            result = WADE_GILES_REGEX.sub(wade_giles_replacer, token)

        # Step 3: Suffix-level Wade-Giles conversions
        def suffix_replacer(match):
            for i, group in enumerate(match.groups(), 1):
                if group is not None:
                    return SUFFIX_REPLACEMENTS[i - 1]
            return match.group(0)

        return SUFFIX_REGEX.sub(suffix_replacer, result)

    def normalize_fullwidth_chars(self, text: str) -> str:
        """Normalize full-width characters to half-width and handle invisible Unicode characters."""
        if not hasattr(self, "_fullwidth_trans"):
            fullwidth = {i: i - 0xFF01 + 0x21 for i in range(0xFF01, 0xFF5F)}
            fullwidth[0x3000] = 0x20
            fullwidth[0x200B] = 0x20
            fullwidth[0xFEFF] = 0x20
            self._fullwidth_trans = str.maketrans(fullwidth)
        return text.translate(self._fullwidth_trans)

    def fix_ocr_artifacts(self, text: str) -> str:
        """Fix common OCR artifacts in Chinese names."""
        result = self.normalize_fullwidth_chars(text)
        ocr_fixes = {
            "п": "n", "р": "p", "о": "o", "а": "a", "е": "e", "х": "x",
            "с": "c", "т": "t", "и": "u", "к": "k", "м": "m", "н": "h",
        }
        translation_table = str.maketrans(ocr_fixes)
        return result.translate(translation_table)


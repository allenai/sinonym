"""
Text preprocessing utilities for Chinese name processing.

RESPONSIBILITIES (After Service Responsibility Clarification):
- Input cleaning: punctuation removal, OCR artifact fixes
- Structural pattern-based splitting: CamelCase, hyphen-separated surnames
- NO database-dependent operations (those belong in CompoundDetector/NameFormattingService)
- Operates BEFORE parsing when surname/given context is unknown

This module handles input cleaning and preparation before normalization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sinonym.utils.string_manipulation import StringManipulationUtils

if TYPE_CHECKING:
    from sinonym.coretypes import ChineseNameConfig


class TextPreprocessor:
    """Text preprocessing utilities for Chinese name input preparation."""

    def __init__(self, config: ChineseNameConfig, normalizer_service):
        self._config = config
        self._normalizer_service = normalizer_service

    def preprocess_input(self, raw: str, data_context=None) -> str:
        """
        Structural preprocessing: pattern-based cleaning and splitting before parsing.

        Responsibilities:
        - Remove punctuation, fix OCR artifacts
        - Split clear structural patterns (CamelCase, hyphen-separated surnames)
        - NO database-dependent operations (those belong in CompoundDetector/NameFormattingService)

        Returns:
            cleaned_string with structural splits applied
        """

        # Single pass for most cleaning operations
        raw = self._config.clean_pattern.sub(self._clean_replacement, raw)

        # Apply OCR artifact cleanup
        raw = self._normalizer_service._text_normalizer.fix_ocr_artifacts(raw)

        # Handle single-token structural splits (pattern-based only)
        tokens = raw.split()
        if len(tokens) == 1:
            token = tokens[0]

            # 1. CamelCase splitting (structural pattern)
            split_result = StringManipulationUtils.smart_split_concatenated(token)
            if split_result != token:
                # Use simple validation - no database context needed for structural patterns
                if self._is_valid_structural_split(token, split_result):
                    tokens[0] = split_result

            # 2. Hyphen splitting for concatenated surnames (structural pattern)
            elif "-" in token and self._looks_like_concatenated_surnames(token):
                hyphen_split = token.replace("-", " ")
                tokens[0] = hyphen_split

        raw = " ".join(tokens)

        # Final cleanup
        return self._config.whitespace_pattern.sub(" ", raw).strip()


    def _clean_replacement(self, match) -> str:
        """Replacement function for single-pass cleaning."""
        if match.group("initial_space"):
            return match.group("initial_space") + " "
        if match.group("compound_first") and match.group("compound_second"):
            return match.group("compound_first") + "-" + match.group("compound_second")
        if match.group("initial_hyphen"):
            return match.group("initial_hyphen") + "-"
        return " "

    def _is_valid_structural_split(self, original: str, split_result: str) -> bool:
        """
        Simple structural validation for pattern-based splits.

        No database context needed - just basic sanity checks for structural patterns.
        """
        # Don't split if result is same as original
        if split_result == original:
            return False

        parts = split_result.split()

        # Optimized validation chain: early exits and combined checks
        return (
            len(parts) == 2 and
            all(2 <= len(part) <= 8 and part.isalpha() for part in parts)
        )

    def _looks_like_concatenated_surnames(self, token: str) -> bool:
        """
        Check if a hyphenated token looks like concatenated surnames vs hyphenated given name.

        Structural heuristics only - no database lookups.
        """
        if token.count("-") != 1:
            return False

        parts = token.split("-")
        if len(parts) != 2:
            return False

        # Concatenated surnames tend to have mixed case (e.g., "LU-Wang")
        # Hyphenated given names tend to be consistent case (e.g., "ka-fai")
        part1_case = self._get_case_pattern(parts[0])
        part2_case = self._get_case_pattern(parts[1])

        # If parts have different case patterns, likely concatenated surnames
        return part1_case != part2_case

    def _get_case_pattern(self, text: str) -> str:
        """Get case pattern: 'upper', 'lower', 'title', or 'mixed'."""
        if text.isupper():
            return "upper"
        if text.islower():
            return "lower"
        if text.istitle():
            return "title"
        return "mixed"

    def is_all_chinese_input(self, text: str) -> bool:
        """
        Detect if input consists entirely of Chinese characters.
        
        This helps identify names like "张伟明" that need special processing
        for surname/given name boundary detection.
        """
        if not text or not text.strip():
            return False

        # Remove whitespace and punctuation for analysis
        cleaned = self._config.sep_pattern.sub("", text.strip())

        # Must have content after cleaning
        if not cleaned:
            return False

        # Check if all remaining characters are CJK - use character caching for performance
        if not hasattr(self, "_cjk_char_cache"):
            self._cjk_char_cache = {}

        for char in cleaned:
            if char not in self._cjk_char_cache:
                self._cjk_char_cache[char] = bool(self._config.cjk_pattern.search(char))
            if not self._cjk_char_cache[char]:
                return False
        return True

    def contains_non_chinese_scripts(self, text: str) -> bool:
        """
        Detect if text contains Korean, Japanese, or Vietnamese-specific characters.
        
        Uses conservative detection to avoid false positives with Chinese Pinyin:
        - Korean: Hangul characters (completely distinct from Chinese)
        - Japanese: Hiragana/Katakana characters (completely distinct from Chinese)  
        - Vietnamese: Only uniquely Vietnamese diacritics (like Đ/đ, ă, ơ, ư)
        
        Returns True if non-Chinese scripts are detected.
        """
        if not text:
            return False

        # Vietnamese-specific characters (conservative list to avoid Pinyin overlap)
        vietnamese_specific = "ĐđăâêôơưĂÂÊÔƠƯ"

        for char in text:
            char_code = ord(char)

            # Korean Hangul ranges
            if (0x1100 <= char_code <= 0x11FF or    # Hangul Jamo
                0x3130 <= char_code <= 0x318F or    # Hangul Compatibility Jamo
                0xAC00 <= char_code <= 0xD7AF):     # Hangul Syllables
                return True

            # Japanese Hiragana and Katakana ranges
            if (0x3040 <= char_code <= 0x309F or    # Hiragana
                0x30A0 <= char_code <= 0x30FF):     # Katakana
                return True

            # Vietnamese-specific characters only
            if char in vietnamese_specific:
                return True

        return False

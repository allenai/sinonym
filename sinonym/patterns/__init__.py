"""
Patterns package for Chinese name processing.

This package contains regex pattern builders and pre-compiled patterns
used throughout the Chinese name detection system.
"""

from sinonym.patterns.compiled_patterns import (
    CLEAN_PATTERN,
    COMPREHENSIVE_CJK_PATTERN,
    FORBIDDEN_PATTERNS_REGEX,
    HAN_ROMAN_SPLITTER,
    SUFFIX_REGEX,
    SUFFIX_REPLACEMENTS,
    WADE_GILES_REGEX,
    WADE_GILES_REPLACEMENTS,
)

__all__ = [
    "CLEAN_PATTERN",
    "COMPREHENSIVE_CJK_PATTERN",
    "FORBIDDEN_PATTERNS_REGEX",
    "HAN_ROMAN_SPLITTER",
    "SUFFIX_REGEX",
    "SUFFIX_REPLACEMENTS",
    "WADE_GILES_REGEX",
    "WADE_GILES_REPLACEMENTS",
]

"""
Pre-compiled regex patterns for performance optimization.

This module contains all pre-compiled regex patterns used by the Chinese name
detection system, built using the functions from regex_builders.
"""
from __future__ import annotations

from sinonym.patterns.regex_builders import (
    build_cjk_pattern,
    build_clean_patterns,
    build_forbidden_patterns_regex,
    build_han_roman_splitter,
    build_suffix_regex,
    build_wade_giles_regex,
)

# Pre-compiled patterns for performance
FORBIDDEN_PATTERNS_REGEX = build_forbidden_patterns_regex()
COMPREHENSIVE_CJK_PATTERN = build_cjk_pattern()
HAN_ROMAN_SPLITTER = build_han_roman_splitter(COMPREHENSIVE_CJK_PATTERN)
WADE_GILES_REGEX, WADE_GILES_REPLACEMENTS = build_wade_giles_regex()
SUFFIX_REGEX, SUFFIX_REPLACEMENTS = build_suffix_regex()
CLEAN_PATTERN = build_clean_patterns()

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

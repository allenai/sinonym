"""
Configuration types for Chinese name processing.

This module contains immutable configuration classes that hold all static
data structures and patterns used by the Chinese name detection system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from sinonym.chinese_names_data import VALID_CHINESE_ONSETS
from sinonym.patterns import (
    CLEAN_PATTERN,
    COMPREHENSIVE_CJK_PATTERN,
    FORBIDDEN_PATTERNS_REGEX,
    HAN_ROMAN_SPLITTER,
)


@dataclass(frozen=True)
class ChineseNameConfig:
    """Immutable configuration containing all static data structures - Scala case class style."""

    # Required data files
    required_files: tuple[str, ...]

    # Precompiled regex patterns (immutable)
    sep_pattern: re.Pattern[str]
    cjk_pattern: re.Pattern[str]
    digits_pattern: re.Pattern[str]
    whitespace_pattern: re.Pattern[str]
    camel_case_pattern: re.Pattern[str]
    # Pre-compiled regex patterns for mixed-token processing
    han_roman_splitter: re.Pattern[str]
    ascii_alpha_pattern: re.Pattern[str]
    clean_roman_pattern: re.Pattern[str]
    camel_case_finder: re.Pattern[str]
    clean_pattern: re.Pattern[str]
    forbidden_patterns_regex: re.Pattern[str]

    # Character translation table
    hyphens_apostrophes_tr: dict[int, None]

    # Pre-sorted Chinese onsets for phonetic validation (performance optimization)
    sorted_chinese_onsets: tuple[str, ...]

    # Log probability defaults
    default_surname_logp: float
    default_given_logp: float
    compound_penalty: float

    # Core processing constants
    max_name_length: int  # Maximum allowed name length
    min_tokens_required: int  # Minimum tokens for valid name parsing

    # Parsing scoring constants
    poor_score_threshold: float  # Score below which parsing is considered poor

    @classmethod
    def create_default(cls) -> ChineseNameConfig:
        """Factory method to create default configuration - Scala apply() equivalent."""
        return cls(
            required_files=("familyname_orcid.csv", "givenname_orcid.csv"),
            sep_pattern=re.compile(r"[·‧.\u2011-\u2015﹘﹣－⁃₋•∙⋅˙ˑːˉˇ˘˚˛˜˝]+"),
            cjk_pattern=COMPREHENSIVE_CJK_PATTERN,
            digits_pattern=re.compile(r"\d"),
            whitespace_pattern=re.compile(r"\s+"),
            camel_case_pattern=re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[A-Z]+(?=$)"),
            # Pre-compiled regex patterns for mixed-token processing
            han_roman_splitter=HAN_ROMAN_SPLITTER,
            ascii_alpha_pattern=re.compile(r"[A-Za-z]"),
            clean_roman_pattern=re.compile(
                r"[^A-Za-z\u00C0-\u00FF\u0100-\u024F-''']",
            ),  # PRESERVE ASCII letters, Latin-1 Supplement (À-ÿ), Latin Extended-A (Ā-ſ), Latin Extended-B (ǀ-ɏ), hyphens and apostrophes for romanization systems
            camel_case_finder=re.compile(r"[A-Z][a-z]+"),
            clean_pattern=CLEAN_PATTERN,
            forbidden_patterns_regex=FORBIDDEN_PATTERNS_REGEX,
            hyphens_apostrophes_tr=str.maketrans("", "", "-‐‒–—―﹘﹣－⁃₋''''''''"),
            sorted_chinese_onsets=tuple(sorted(VALID_CHINESE_ONSETS, key=len, reverse=True)),
            default_surname_logp=-15.0,
            default_given_logp=-15.0,
            compound_penalty=0.1,
            max_name_length=100,
            min_tokens_required=2,
            poor_score_threshold=-25.0,
        )

    def with_log_probabilities(self, surname_logp: float, given_logp: float) -> ChineseNameConfig:
        """Immutable update method for log probabilities."""
        return replace(self, default_surname_logp=surname_logp, default_given_logp=given_logp)

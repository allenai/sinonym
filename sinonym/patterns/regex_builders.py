"""
Regex pattern builders for Chinese name processing.

This module contains functions to build complex regex patterns used throughout
the Chinese name detection system.
"""

from __future__ import annotations

import re

from sinonym.chinese_names_data import FORBIDDEN_PHONETIC_PATTERNS


def build_forbidden_patterns_regex():
    """Pre-compile FORBIDDEN_PHONETIC_PATTERNS into a single regex for faster pattern matching."""
    # Escape special regex characters and join with alternation
    escaped_patterns = [re.escape(pattern) for pattern in FORBIDDEN_PHONETIC_PATTERNS]
    # Sort by length (descending) to ensure longer patterns match first
    escaped_patterns.sort(key=len, reverse=True)
    return re.compile(f"({'|'.join(escaped_patterns)})")


def build_cjk_pattern():
    """Build comprehensive CJK pattern including all extensions."""
    # CJK Unicode ranges - covers all Chinese, Japanese, Korean characters
    CJK_RANGES = (
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Extension A
        (0x20000, 0x2A6DF),  # CJK Extension B
        (0x2A700, 0x2B73F),  # CJK Extension C
        (0x2B740, 0x2B81F),  # CJK Extension D
        (0x2B820, 0x2CEAF),  # CJK Extension E
        (0x2CEB0, 0x2EBEF),  # CJK Extension F
        (0x30000, 0x3134F),  # CJK Extension G
    )

    ranges = []
    for start, end in CJK_RANGES:
        if end <= 0xFFFF:
            ranges.append(f"\\u{start:04X}-\\u{end:04X}")
        else:
            ranges.append(f"\\U{start:08X}-\\U{end:08X}")

    return re.compile(f"[{''.join(ranges)}]")


def build_han_roman_splitter(cjk_pattern):
    """Build han_roman_splitter pattern using comprehensive CJK ranges."""
    # Extract the character class from the comprehensive CJK pattern
    cjk_class = cjk_pattern.pattern[1:-1]  # Remove [ and ]
    return re.compile(f"([{cjk_class}]+|[A-Za-z-]+)")


def build_wade_giles_regex():
    """Build optimized regex for Wade-Giles conversions with O(1) lookup performance."""
    # Define conversion patterns with their replacements
    # Order matters: longest patterns first to avoid partial matches
    patterns = [
        # 4-character patterns
        (r"shih", "shi"),
        # 3-character patterns (aspirated) - must be before 2-char patterns
        (r"ts'", "c"),
        (r"tz'", "c"),
        (r"ch'", "q"),
        # 3-character patterns (non-aspirated)
        (r"szu", "si"),
        # 2-character patterns (aspirated) - must be before 1-char patterns
        (r"k'", "k"),
        (r"t'", "t"),
        (r"p'", "p"),
        # 2-character patterns (non-aspirated)
        (r"hs", "x"),
        (r"ts", "z"),
        (r"tz", "z"),
        # Special case: ch -> needs context-sensitive replacement
        (r"ch(?=i|ia|ie|iu)", "j"),  # ch before i/ia/ie/iu -> j
        (r"ch", "zh"),  # all other ch -> zh
        # REMOVED: Broad k/t/p patterns that incorrectly convert non-Wade-Giles tokens
        # These patterns were too broad and converted valid tokens like "szeto" -> "szedo"
        # Wade-Giles aspirated consonants should use apostrophes (k', t', p')
        # Unaspirated consonants in Wade-Giles should not be converted to voiced
    ]

    # Create the combined regex pattern
    pattern_str = "|".join(f"({pattern})" for pattern, _ in patterns)
    compiled_regex = re.compile(pattern_str)

    # Create replacement mapping by group index
    replacements = [replacement for _, replacement in patterns]

    return compiled_regex, replacements


def build_suffix_regex():
    """Build optimized regex for suffix conversions."""
    # Suffix patterns ordered by length (longest first)
    patterns = [
        (r"ieh$", "ie"),  # 3 chars
        (r"ueh$", "ue"),  # 3 chars
        (r"ung$", "ong"),  # 3 chars
        (r"ien$", "ian"),  # 3 chars - Wade-Giles ien → Pinyin ian
        (r"ih$", "i"),  # 2 chars
    ]

    # Create the combined regex pattern
    pattern_str = "|".join(f"({pattern})" for pattern, _ in patterns)
    compiled_regex = re.compile(pattern_str)

    # Create replacement mapping by group index
    replacements = [replacement for _, replacement in patterns]

    return compiled_regex, replacements


def build_clean_patterns():
    """Build combined cleaning patterns for input preprocessing."""
    # Clean pattern components
    parentheticals_pattern = r"[（(][^)（）]*[)）]"
    initials_with_space_pattern = r"(?P<initial_space>[A-Z])\.(?=\s)"
    compound_initials_pattern = r"(?P<compound_first>[A-Z])\.-(?P<compound_second>[A-Z])\."
    initials_with_hyphen_pattern = r"(?P<initial_hyphen>[A-Z])\.-(?=[A-Z])"
    invalid_chars_pattern = r"[_|=]"

    # Combined clean pattern (case-sensitive, pre-lowercasing handled in preprocessing)
    clean_pattern_combined = (
        f"{parentheticals_pattern}|"
        f"{initials_with_space_pattern}|"
        f"{compound_initials_pattern}|"
        f"{initials_with_hyphen_pattern}|"
        f"{invalid_chars_pattern}"
    )

    return re.compile(clean_pattern_combined)

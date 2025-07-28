from __future__ import annotations

import csv
import math
import re
import string
import unicodedata
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from pathlib import Path

import pypinyin

from sinonym.chinese_names_data import (
    CANTONESE_SURNAMES,
    COMPOUND_VARIANTS,
    FORBIDDEN_PHONETIC_PATTERNS,
    HIGH_CONFIDENCE_ANCHORS,
    NON_WADE_GILES_SYLLABLE_RULES,
    ONE_LETTER_RULES,
    ROMANIZATION_EXCEPTIONS,
    VALID_CHINESE_ONSETS,
    VALID_CHINESE_RIMES,
    WADE_GILES_SYLLABLE_RULES,
)
from sinonym.paths import DATA_PATH

# ════════════════════════════════════════════════════════════════════════════════
# COMPILED REGEX PATTERNS
# ════════════════════════════════════════════════════════════════════════════════


def _build_forbidden_patterns_regex():
    """Pre-compile FORBIDDEN_PHONETIC_PATTERNS into a single regex for faster pattern matching."""
    # Escape special regex characters and join with alternation
    escaped_patterns = [re.escape(pattern) for pattern in FORBIDDEN_PHONETIC_PATTERNS]
    # Sort by length (descending) to ensure longer patterns match first
    escaped_patterns.sort(key=len, reverse=True)
    return re.compile(f"({'|'.join(escaped_patterns)})")


def _build_cjk_pattern():
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


def _build_han_roman_splitter():
    """Build han_roman_splitter pattern using comprehensive CJK ranges."""
    # Extract the character class from the comprehensive CJK pattern
    cjk_class = _COMPREHENSIVE_CJK_PATTERN.pattern[1:-1]  # Remove [ and ]
    return re.compile(f"([{cjk_class}]+|[A-Za-z-]+)")


def _build_wade_giles_regex():
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


def _build_suffix_regex():
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


# Pre-compiled patterns for performance
_FORBIDDEN_PATTERNS_REGEX = _build_forbidden_patterns_regex()
_COMPREHENSIVE_CJK_PATTERN = _build_cjk_pattern()
_HAN_ROMAN_SPLITTER = _build_han_roman_splitter()
_WADE_GILES_REGEX, _WADE_GILES_REPLACEMENTS = _build_wade_giles_regex()
_SUFFIX_REGEX, _SUFFIX_REPLACEMENTS = _build_suffix_regex()

# Clean pattern components
_PARENTHETICALS_PATTERN = r"[（(][^)（）]*[)）]"
_INITIALS_WITH_SPACE_PATTERN = r"(?P<initial_space>[A-Z])\.(?=\s)"
_COMPOUND_INITIALS_PATTERN = r"(?P<compound_first>[A-Z])\.-(?P<compound_second>[A-Z])\."
_INITIALS_WITH_HYPHEN_PATTERN = r"(?P<initial_hyphen>[A-Z])\.-(?=[A-Z])"
_INVALID_CHARS_PATTERN = r"[_|=]"

# Combined clean pattern (case-sensitive, pre-lowercasing handled in preprocessing)
_CLEAN_PATTERN_COMBINED = (
    f"{_PARENTHETICALS_PATTERN}|"
    f"{_INITIALS_WITH_SPACE_PATTERN}|"
    f"{_COMPOUND_INITIALS_PATTERN}|"
    f"{_INITIALS_WITH_HYPHEN_PATTERN}|"
    f"{_INVALID_CHARS_PATTERN}"
)


# ════════════════════════════════════════════════════════════════════════════════
# RESULT TYPES (Scala-friendly error handling)
# ════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParseResult:
    """Result of name parsing operation - Scala Either-like structure."""

    success: bool
    result: str | tuple[list[str], list[str]]
    error_message: str | None = None

    @classmethod
    def success_with_name(cls, formatted_name: str) -> ParseResult:
        return cls(success=True, result=formatted_name, error_message=None)

    @classmethod
    def success_with_parse(cls, surname_tokens: list[str], given_tokens: list[str]) -> ParseResult:
        return cls(success=True, result=(surname_tokens, given_tokens), error_message=None)

    @classmethod
    def failure(cls, error_message: str) -> ParseResult:
        return cls(success=False, result="", error_message=error_message)

    def map(self, f) -> ParseResult:
        """Functor map operation - Scala-like transformation"""
        if self.success:
            try:
                return ParseResult.success_with_name(f(self.result))
            except Exception as e:
                return ParseResult.failure(str(e))
        return self

    def flat_map(self, f) -> ParseResult:
        """Monadic flatMap operation - Scala-like chaining"""
        if self.success:
            try:
                return f(self.result)
            except Exception as e:
                return ParseResult.failure(str(e))
        return self


@dataclass(frozen=True)
class CacheInfo:
    """Immutable cache information structure."""

    cache_built: bool
    cache_size: int
    pickle_file_exists: bool
    pickle_file_size: int | None = None
    pickle_file_mtime: float | None = None


# ════════════════════════════════════════════════════════════════════════════════
# IMMUTABLE CONFIGURATION DATA
# ════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChineseNameConfig:
    """Immutable configuration containing all static data structures - Scala case class style."""

    # Required data files
    data_dir: str
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

    @classmethod
    def create_default(cls) -> ChineseNameConfig:
        """Factory method to create default configuration - Scala apply() equivalent."""
        return cls(
            data_dir=str(DATA_PATH),
            required_files=("familyname_orcid.csv", "givenname_orcid.csv"),
            sep_pattern=re.compile(r"[·‧.\u2011-\u2015﹘﹣－⁃₋•∙⋅˙ˑːˉˇ˘˚˛˜˝]+"),
            cjk_pattern=_COMPREHENSIVE_CJK_PATTERN,
            digits_pattern=re.compile(r"\d"),
            whitespace_pattern=re.compile(r"\s+"),
            camel_case_pattern=re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[A-Z]+(?=$)"),
            # Pre-compiled regex patterns for mixed-token processing
            han_roman_splitter=_HAN_ROMAN_SPLITTER,
            ascii_alpha_pattern=re.compile(r"[A-Za-z]"),
            clean_roman_pattern=re.compile(
                r"[^A-Za-z\u00C0-\u00FF\u0100-\u017F-''']",
            ),  # PRESERVE ASCII letters, Latin-1 Supplement (À-ÿ), Latin Extended-A (Ā-ſ), hyphens and apostrophes for romanization systems
            camel_case_finder=re.compile(r"[A-Z][a-z]+"),
            clean_pattern=re.compile(_CLEAN_PATTERN_COMBINED),
            forbidden_patterns_regex=_FORBIDDEN_PATTERNS_REGEX,
            hyphens_apostrophes_tr=str.maketrans("", "", "-‐‒–—―﹘﹣－⁃₋''''''''"),
            sorted_chinese_onsets=tuple(sorted(VALID_CHINESE_ONSETS, key=len, reverse=True)),
            default_surname_logp=-15.0,
            default_given_logp=-15.0,
            compound_penalty=0.1,
        )

    def with_log_probabilities(self, surname_logp: float, given_logp: float) -> ChineseNameConfig:
        """Immutable update method for log probabilities."""
        return replace(self, default_surname_logp=surname_logp, default_given_logp=given_logp)


# ════════════════════════════════════════════════════════════════════════════════
# NORMALIZATION SERVICE (Scala-compatible)
# ════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LazyNormalizationMap:
    """
    Lazy normalization map with true immutability.
    Uses __slots__ and MappingProxyType for architectural correctness.
    """

    __slots__ = ("_cache", "_normalizer", "_tokens")

    def __init__(self, tokens: tuple[str, ...], normalizer: NormalizationService):
        object.__setattr__(self, "_tokens", tokens)
        object.__setattr__(self, "_normalizer", normalizer)
        # Use a regular dict internally but expose as MappingProxyType
        object.__setattr__(self, "_cache", {})

    def get(self, token: str, default: str = None):
        """Get normalized value for token, computing lazily."""
        if token not in self._cache:
            # Compute and cache the normalized value
            self._cache[token] = self._normalizer._normalize_token(token)
        return self._cache[token]

    def __getitem__(self, token: str) -> str:
        """Dict-like access."""
        return self.get(token)

    def __contains__(self, token: str) -> bool:
        """Check if token is in the original tokens."""
        return token in self._tokens

    def items(self):
        """Iterate over all items, computing values lazily."""
        for token in self._tokens:
            yield token, self.get(token)

    @property
    def cache_view(self):
        """Get read-only view of current cache state."""
        from types import MappingProxyType

        return MappingProxyType(self._cache)


@dataclass(frozen=True)
class NormalizedInput:
    """Immutable normalized input - Scala case class style."""

    raw: str  # Original input: "Zhang Wei"
    cleaned: str  # After punctuation/formatting cleanup
    tokens: tuple[str, ...]  # After separator splitting
    roman_tokens: tuple[str, ...]  # After Han→pinyin & mixed-token processing
    norm_map: dict[str, str] | LazyNormalizationMap  # token → fully normalized (lazy)

    @classmethod
    def empty(cls, raw: str = "") -> NormalizedInput:
        """Factory for empty/invalid input."""
        return cls(raw, "", (), (), {})


class NormalizationService:
    """Pure normalization service - Scala-compatible design."""

    def __init__(self, config: ChineseNameConfig, cache_service: PinyinCacheService):
        self._config = config
        self._cache_service = cache_service
        self._data: NameDataStructures | None = None

    def set_data_context(self, data: NameDataStructures) -> None:
        """Inject data context after initialization - breaks circular dependency."""
        self._data = data

    def norm(self, token: str) -> str:
        """
        Normalize text for all lookup operations (full phonetic normalization).

        Public interface for token normalization that applies consistent normalization for:
        - General lookups
        - Surname frequency/probability lookups
        - Given name database lookups

        Includes Wade-Giles conversion, hyphen/apostrophe removal, and lowercasing.
        """
        return self._normalize_token(token)

    def apply(self, raw_name: str) -> NormalizedInput:
        """
        Pure function: raw input → normalized structure.
        Side-effect free, suitable for Scala interop.
        """
        if not raw_name or not raw_name.strip():
            return NormalizedInput.empty(raw_name)

        # Phase 1: Clean input (single regex pass)
        cleaned = self._preprocess_input(raw_name)

        # Phase 2: Handle "LAST, First" format (common in academic/professional contexts)
        if "," in cleaned:
            parts = [part.strip() for part in cleaned.split(",")]
            if len(parts) == 2 and all(parts):  # Exactly 2 non-empty parts
                cleaned = " ".join(parts[::-1])  # Reverse order: "Last, First" -> "First Last"

        # Phase 3: Tokenize on separators/whitespace and filter out invalid tokens
        raw_tokens = self._config.sep_pattern.sub(" ", cleaned).split()
        tokens = tuple(t for t in raw_tokens if t and not all(c in string.punctuation for c in t))

        if not tokens:
            return NormalizedInput.empty(raw_name)

        # Phase 4: Process mixed Han/Roman tokens
        roman_tokens = tuple(self._process_mixed_tokens(list(tokens)))

        if not roman_tokens:
            return NormalizedInput.empty(raw_name)

        # Phase 5: Create lazy normalization map (computed on-demand)
        norm_map = LazyNormalizationMap(roman_tokens, self)

        return NormalizedInput(
            raw=raw_name,
            cleaned=cleaned,
            tokens=tokens,
            roman_tokens=roman_tokens,
            norm_map=norm_map,
        )

    def _preprocess_input(self, raw: str) -> str:
        """Extract and refactor existing preprocessing logic."""
        # Single pass for most cleaning operations
        raw = self._config.clean_pattern.sub(self._clean_replacement, raw)

        # Handle camelCase compound surnames (e.g., "AuYeung" -> "Au Yeung")
        tokens = raw.split()

        # Check all tokens for camelCase compound surnames
        for i, token in enumerate(tokens):
            # Check if token is camelCase and could be a compound surname
            camel_parts = self._config.camel_case_finder.findall(token)
            if len(camel_parts) == 2 and token == "".join(camel_parts):
                # Check if it matches a known compound surname
                potential_compound = " ".join(part.lower() for part in camel_parts)
                if potential_compound in COMPOUND_VARIANTS:
                    # Replace with spaced version
                    tokens[i] = " ".join(camel_parts)

        # Only apply concatenated name splitting if input has exactly 1 token
        # This preserves existing behavior for multi-token names
        if len(tokens) == 1:
            token = tokens[0]

            # Skip compound surname processing since it was already handled above
            # Only handle general concatenated names (CamelCase, mixed caps, etc.)
            split_result = self._smart_split_concatenated(token)
            if split_result != token:
                # Only use the split if it makes linguistic sense
                if self._should_split_concatenated(token, split_result):
                    tokens[0] = split_result
            elif "-" in token:
                # Handle hyphenated concatenated names like "LU-Wang"
                hyphen_split = token.replace("-", " ")
                if self._should_split_concatenated(token, hyphen_split):
                    tokens[0] = hyphen_split
        raw = " ".join(tokens)

        # Handle compound surnames if data is available
        if self._data and hasattr(self._data, "compound_hyphen_map"):
            for hyphen_form, space_form in self._data.compound_hyphen_map.items():
                # Use case-insensitive regex with word boundaries
                pattern = r"\b" + re.escape(hyphen_form) + r"\b"
                # Title-case the space form for replacement
                title_space_form = " ".join(part.title() for part in space_form.split())
                raw = re.sub(pattern, title_space_form, raw, flags=re.IGNORECASE)

        # Final cleanup
        return self._config.whitespace_pattern.sub(" ", raw).strip()

    def _smart_split_concatenated(self, token: str) -> str:
        """
        Split concatenated names with various capitalization patterns.

        Handles:
        - Simple CamelCase: LinShu → Lin Shu
        - Mixed caps: XIAOChen → XIAO Chen, LuWANG → Lu WANG
        - Multiple transitions: FurukawaKoichi → Furukawa Koichi
        - Preserve existing hyphens: LU-Wang → LU Wang
        """
        # If already has spaces or hyphens, don't split further
        if " " in token or "-" in token:
            return token

        # Use a different approach: insert spaces at transition points
        result = []
        i = 0

        while i < len(token):
            char = token[i]
            result.append(char)

            # Look ahead to see if we need to insert a space
            if i < len(token) - 1:
                next_char = token[i + 1]

                # Case 1: lowercase followed by uppercase (camelCase)
                if char.islower() and next_char.isupper():
                    result.append(" ")

                # Case 2: multiple uppercase followed by uppercase+lowercase (XMLParser → XML Parser)
                elif char.isupper() and next_char.isupper() and i < len(token) - 2:
                    next_next_char = token[i + 2]
                    if next_next_char.islower():
                        result.append(" ")

            i += 1

        return "".join(result)

    def _should_split_concatenated(self, original: str, split_result: str) -> bool:
        """
        Determine if a concatenated name should be split based on linguistic validation.

        Args:
            original: The original concatenated token (e.g., "LinShu" or "LU-Wang")
            split_result: The split result (e.g., "Lin Shu" or "LU Wang")

        Returns:
            True if the split makes linguistic sense, False otherwise
        """
        # Don't split if the result is the same as original
        if split_result == original:
            return False

        # Get the parts after splitting
        parts = split_result.split()

        # Must have at least 2 parts to be worth splitting
        if len(parts) < 2:
            return False

        # Must have exactly 2 parts for names (surname + given name)
        if len(parts) > 2:
            return False

        # Special logic for hyphenated names
        if "-" in original:
            # For hyphenated names, only split if both parts are likely surnames
            # This distinguishes concatenated names like "LU-Wang" from hyphenated given names like "Ka-Fai"

            # Check the original parts (before splitting)
            original_parts = original.split("-")
            if len(original_parts) != 2:
                return False

            # Check if both parts could be surnames
            surname_count = 0
            for part in original_parts:
                part_lower = part.lower()
                normalized = self._normalize_token(part_lower)

                if part_lower in self._data.surnames or normalized in self._data.surnames_normalized:
                    surname_count += 1

            # Only split if both parts are likely surnames (suggesting concatenation)
            if surname_count < 2:
                return False

        # Check if data is available for validation
        if not self._data:
            # If no data available, allow splits for reasonable length parts
            return all(len(part) >= 2 for part in parts)

        # Check if any part could be a Chinese surname or given name component
        chinese_evidence = 0
        for part in parts:
            part_lower = part.lower()
            normalized = self._normalize_token(part_lower)

            # Check if it's a Chinese surname
            if part_lower in self._data.surnames or normalized in self._data.surnames_normalized:
                chinese_evidence += 2

            # Check if it's a plausible Chinese component
            elif normalized in self._data.plausible_components:
                chinese_evidence += 1

        # Need at least some Chinese evidence to justify the split
        return chinese_evidence >= 2

    def _clean_replacement(self, match) -> str:
        """Replacement function for single-pass cleaning."""
        if match.group("initial_space"):  # Initial followed by space: "X. " -> "X "
            return match.group("initial_space") + " "
        if match.group("compound_first") and match.group("compound_second"):  # Compound initials: "X.-H." -> "X-H"
            return match.group("compound_first") + "-" + match.group("compound_second")
        if match.group("initial_hyphen"):  # Initial followed by hyphen and letter: "X.-M" -> "X-M"
            return match.group("initial_hyphen") + "-"
        return " "

    def _process_mixed_tokens(self, tokens: list[str]) -> list[str]:
        """Extract existing mixed token processing logic."""
        mix = []
        for token in tokens:
            if self._config.cjk_pattern.search(token) and self._config.ascii_alpha_pattern.search(token):
                # Split mixed Han/Roman token
                han = "".join(c for c in token if self._config.cjk_pattern.search(c))
                rom = "".join(c for c in token if c.isascii() and c.isalpha())
                if han:
                    mix.append(han)
                if rom:
                    mix.append(rom)
            else:
                mix.append(token)

        # Convert to roman tokens
        han_tokens = []
        roman_tokens_split = []
        roman_tokens_original = []

        for token in mix:
            if self._config.cjk_pattern.search(token):
                # Convert Han to pinyin
                pinyin_tokens = self._cache_service.han_to_pinyin_fast(token)
                han_tokens.extend(pinyin_tokens)
            else:
                # Clean Roman token
                clean_token = self._config.clean_roman_pattern.sub("", token)
                # Filter out empty tokens and tokens that are only punctuation
                if clean_token and not all(c in string.punctuation for c in clean_token):
                    roman_tokens_original.append(clean_token)

                    # Create split version for comparison
                    if "-" in clean_token:
                        parts = [part.strip() for part in clean_token.split("-") if part.strip()]
                        roman_tokens_split.extend(parts)
                    # Use centralized split_concat method if available
                    elif self._data:
                        split_result = self.split_concat(clean_token)
                        if split_result:
                            roman_tokens_split.extend(split_result)
                        else:
                            roman_tokens_split.append(clean_token)
                    else:
                        roman_tokens_split.append(clean_token)

        # Handle Han/Roman duplication
        if han_tokens and roman_tokens_split:
            # Compare normalized forms directly (memoized for performance)
            han_normalized = set(self._normalize_token(t) for t in han_tokens)
            roman_normalized = set(self._normalize_token(t) for t in roman_tokens_split)

            overlap = han_normalized.intersection(roman_normalized)
            max_size = max(len(han_normalized), len(roman_normalized))

            if len(overlap) >= max_size * 0.5:
                # Use original Roman format (preserves hyphens and avoids duplication)
                return roman_tokens_original
            # Combine them
            return han_tokens + roman_tokens_split
        if han_tokens:
            return han_tokens
        return roman_tokens_original

    @lru_cache(maxsize=32_768)
    def _normalize_token(self, token: str) -> str:
        """
        Normalize a token through the full romanization pipeline.

        CRITICAL ORDER OF OPERATIONS AND RATIONALE:

        The Wade-Giles algorithm uses a complex precedence system where syllable-level
        conversions override prefix-level conversions. This creates specific precedence:

        1. EXCEPTIONS → SYLLABLE_RULES → ONE_LETTER_RULES
        2. Unified Wade-Giles conversions (_apply_unified_wade_giles)
        3. SYLLABLE_RULES (second pass to handle Wade-Giles conversion results)

        WADE-GILES PRECEDENCE COMPLEXITY:
        The precedence is NOT simply "Layer 2 > Layer 3" but rather:
        "Syllable-level WG > Cantonese > Taiwanese > Prefix-level WG"

        Example of syllable-level override:
        - Token "tsu" → SYLLABLE_RULES contains "tsu": "cu" (step 2)
        - This prevents _apply_unified_wade_giles from seeing "ts" → "z" (step 4)
        - Result: "tsu" → "cu" (intended behavior, but complex)

        This design handles edge cases like complete syllable mappings that cannot
        be handled by systematic prefix rules, but creates potential brittleness
        when adding new patterns.

        CRITICAL: Wade-Giles conversion must happen BEFORE apostrophe removal,
        since the conversion rules expect patterns like "ts'", "ch'", "k'", etc.

        Memoized with LRU cache for performance (32K entries should handle
        most real-world workloads without memory pressure).
        """
        # Step 1: Lowercase for consistent processing
        low = token.lower()

        # Step 2: Apply non-Wade-Giles romanization precedence system (with apostrophes intact)
        # Check EXCEPTIONS first
        mapped = ROMANIZATION_EXCEPTIONS.get(low)
        if mapped:
            return mapped

        # Check non-Wade-Giles syllable rules (Cantonese, Taiwanese, etc.)
        mapped = NON_WADE_GILES_SYLLABLE_RULES.get(low)
        if mapped:
            return mapped

        # Check ONE_LETTER_RULES
        mapped = ONE_LETTER_RULES.get(low)
        if mapped:
            return mapped

        # Step 3: Apply Wade-Giles conversions BEFORE removing apostrophes
        wade_giles_result = self._apply_unified_wade_giles(low)

        # Step 4: Apply non-Wade-Giles syllable rules to Wade-Giles conversion results
        # This handles cases like ch'en → qen → chen
        mapped_result = NON_WADE_GILES_SYLLABLE_RULES.get(wade_giles_result)
        if mapped_result:
            wade_giles_result = mapped_result
        # Special case: Handle Wade-Giles specific "qen" → "chen" conversion
        elif wade_giles_result == "qen":
            wade_giles_result = "chen"

        # Step 5: Remove apostrophes and hyphens from the final result
        return wade_giles_result.translate(self._config.hyphens_apostrophes_tr)

    def _apply_unified_wade_giles(self, token: str) -> str:
        """
        Unified Wade-Giles conversion with explicit precedence handling.

        Replicates current behavior exactly with clearer precedence order:
        1. Syllable-level Wade-Giles exceptions (tsu→cu overrides ts→z)
        2. Prefix-level Wade-Giles conversions (ts'→c, ch'→q, ts→z, hs→x)
        3. Suffix-level Wade-Giles conversions (ien→ian, ih→i, ueh→ue, ung→ong)

        This consolidates Wade-Giles logic that was previously split between
        SYLLABLE_RULES and _apply_wade_giles_conversions.

        Args:
            token: Lowercase token WITH apostrophes intact (e.g., "ts'ai", "ch'en", "tsu")

        Returns:
            Converted token (e.g., "cai", "qen", "cu")
        """
        # Step 1: Syllable-level Wade-Giles exceptions (highest precedence)
        # These complete syllable mappings take precedence over systematic prefix rules
        if token in WADE_GILES_SYLLABLE_RULES:
            return WADE_GILES_SYLLABLE_RULES[token]

        # Step 2: Prefix-level Wade-Giles conversions (moved from _apply_wade_giles_conversions)
        # Fast path: Handle exact match for single "j" first
        if token == "j":
            result = "r"
        else:

            def wade_giles_replacer(match):
                """Replacement function for Wade-Giles regex substitution."""
                # Find which group matched (groups are 1-indexed)
                for i, group in enumerate(match.groups(), 1):
                    if group is not None:
                        return _WADE_GILES_REPLACEMENTS[i - 1]
                return match.group(0)  # Fallback (should never happen)

            # Apply prefix conversions with single regex substitution
            result = _WADE_GILES_REGEX.sub(wade_giles_replacer, token)

        # Step 3: Suffix-level Wade-Giles conversions
        def suffix_replacer(match):
            """Replacement function for suffix regex substitution."""
            # Find which group matched (groups are 1-indexed)
            for i, group in enumerate(match.groups(), 1):
                if group is not None:
                    return _SUFFIX_REPLACEMENTS[i - 1]
            return match.group(0)  # Fallback (should never happen)

        # Apply suffix conversions with single regex substitution
        result = _SUFFIX_REGEX.sub(suffix_replacer, result)

        return result

    # ════════════════════════════════════════════════════════════════════
    # ADDITIONAL NORMALIZATION UTILITIES
    # ════════════════════════════════════════════════════════════════════

    def remove_spaces(self, text: str) -> str:
        """Remove spaces from text - centralized utility."""
        return text.replace(" ", "")

    def is_valid_chinese_phonetics(self, token: str) -> bool:
        """Check if a token could plausibly be Chinese based on phonetic structure."""
        if not token:
            return False

        # Convert to lowercase for analysis
        t = token.lower()

        # Length check: Chinese syllables are typically 1-7 characters
        if not 1 <= len(t) <= 7:
            return False

        # Reject tokens with numbers or apostrophes
        if any(c in t for c in "0123456789'"):
            return False

        # Check for forbidden Western patterns
        if self._config.forbidden_patterns_regex.search(t):
            return False

        # Special case: single letters
        if len(t) == 1:
            return True  # Allow for processing, but surname logic will filter them out

        # Split into onset and rime using pre-sorted onsets (performance optimization)
        for onset in self._config.sorted_chinese_onsets:
            if t.startswith(onset):
                rime = t[len(onset) :]
                if rime in VALID_CHINESE_RIMES:
                    return True

        return False

    def is_valid_given_name_token(self, token: str, normalized_cache: dict[str, str] | None = None) -> bool:
        """Check if a token is valid as a Chinese given name component."""
        if not self._data:
            return False

        # Check if token is in Chinese given name database first
        if normalized_cache and token in normalized_cache:
            normalized = normalized_cache[token]
        else:
            normalized = self._normalize_token(token)

        if normalized in self._data.given_names_normalized:
            return True

        # Also check the original token (before normalization) in case normalization
        # maps it to something not in the database (e.g., "chuai" -> "zhuai")
        if token.lower() in self._data.given_names_normalized:
            return True

        # If not found, check if it can be split into valid syllables
        if self.split_concat(token, normalized_cache):
            return True

        # Handle hyphenated tokens by splitting and validating each part
        if "-" in token:
            parts = token.split("-")
            return all(self.is_valid_given_name_token(part, normalized_cache) for part in parts if part)

        # Check if token is a surname used in given position (e.g., "Wen Zhang")
        if self.remove_spaces(normalized) in self._data.surnames_normalized:
            return True

        # Must pass Chinese phonetic validation
        return self.is_valid_chinese_phonetics(token)

    def split_concat(self, token: str, normalized_cache: dict[str, str] | None = None) -> list[str] | None:
        """
        Try to split a fused or hyphenated given name using a tiered confidence system.
        This prevents incorrect splits of Western names like 'Alan' -> 'A', 'lan'.
        """
        if not self._data:
            return None

        # Don't split if the token is a known surname itself
        if normalized_cache and token in normalized_cache:
            tok_normalized = self.remove_spaces(normalized_cache[token])
        else:
            tok_normalized = self.remove_spaces(self._normalize_token(token))

        if tok_normalized in self._data.surnames_normalized:
            return None

        # Don't split HIGH_CONFIDENCE_ANCHORS - they should remain intact
        if tok_normalized in HIGH_CONFIDENCE_ANCHORS:
            return None

        # Don't split tokens that are already valid Chinese given name components
        # Check both the normalized form and the original form (before normalization)
        original_lower = token.lower()
        if tok_normalized in self._data.given_names_normalized or original_lower in self._data.given_names_normalized:
            return None

        # Check for repeated syllable patterns FIRST
        raw = token.translate(self._config.hyphens_apostrophes_tr)
        if len(raw) >= 4 and len(raw) % 2 == 0:
            mid = len(raw) // 2
            first_half = raw[:mid]
            second_half = raw[mid:]

            if first_half.lower() == second_half.lower():
                # Check if the repeated syllable is valid
                if normalized_cache and first_half in normalized_cache:
                    norm_syllable = normalized_cache[first_half]
                else:
                    norm_syllable = self._normalize_token(first_half)
                if norm_syllable in self._data.plausible_components:
                    return [first_half, second_half]

        # Check for forbidden phonetic patterns
        has_forbidden_patterns = bool(self._config.forbidden_patterns_regex.search(token.lower()))

        # Trust explicit hyphens if both parts are valid components
        if "-" in token and token.count("-") == 1:
            a, b = token.split("-")

            if normalized_cache:
                norm_a = normalized_cache.get(a, self._normalize_token(a))
                norm_b = normalized_cache.get(b, self._normalize_token(b))
            else:
                norm_a = self._normalize_token(a)
                norm_b = self._normalize_token(b)

            if norm_a in self._data.plausible_components and norm_b in self._data.plausible_components:
                return [a, b]

        # Trust explicit CamelCase if both parts are valid components
        camel = self._config.camel_case_pattern.findall(raw)
        if len(camel) == 2:
            if normalized_cache:
                norm_a = normalized_cache.get(camel[0], self._normalize_token(camel[0]))
                norm_b = normalized_cache.get(camel[1], self._normalize_token(camel[1]))
            else:
                norm_a = self._normalize_token(camel[0])
                norm_b = self._normalize_token(camel[1])
            if norm_a in self._data.plausible_components and norm_b in self._data.plausible_components:
                return camel

        # Brute-force split with tiered confidence logic
        for i in range(1, len(raw)):
            a, b = raw[:i], raw[i:]

            if normalized_cache:
                norm_a = normalized_cache.get(a, self._normalize_token(a))
                norm_b = normalized_cache.get(b, self._normalize_token(b))
            else:
                norm_a = self._normalize_token(a)
                norm_b = self._normalize_token(b)

            # Both halves must be known plausible syllables
            if not (norm_a in self._data.plausible_components and norm_b in self._data.plausible_components):
                continue

            # Cultural plausibility check
            if len(raw) >= 3:
                is_culturally_plausible = self.is_plausible_chinese_split(norm_a, norm_b, raw)
                if not is_culturally_plausible:
                    continue

            is_a_anchor = norm_a in HIGH_CONFIDENCE_ANCHORS
            is_b_anchor = norm_b in HIGH_CONFIDENCE_ANCHORS

            # Gold Standard (Anchor + Anchor)
            if is_a_anchor and is_b_anchor:
                return [a, b]

            # Silver Standard (Anchor + Plausible)
            if is_a_anchor or is_b_anchor:
                if len(raw) >= 4:
                    is_culturally_plausible = self.is_plausible_chinese_split(norm_a, norm_b, raw)
                    if is_culturally_plausible:
                        return [a, b]
                else:
                    is_culturally_plausible = self.is_plausible_chinese_split(norm_a, norm_b, raw)
                    if is_culturally_plausible:
                        return [a, b]

            # Bronze Standard (Plausible + Plausible)
            if len(raw) >= 4:
                is_culturally_plausible = self.is_plausible_chinese_split(norm_a, norm_b, raw)
                if is_culturally_plausible:
                    return [a, b]

        # No valid split found
        if has_forbidden_patterns:
            return None

        return None

    def is_plausible_chinese_split(self, norm_a: str, norm_b: str, original_token: str) -> bool:
        """
        Check if a split represents an authentic Chinese name combination vs Western name decomposition.
        """
        if not self._data:
            return False

        # At least one component should be in the actual given names database
        is_a_in_db = norm_a in self._data.given_names_normalized
        is_b_in_db = norm_b in self._data.given_names_normalized

        if not (is_a_in_db or is_b_in_db):
            return False

        # Frequency-based validation: reject if both parts are very uncommon
        freq_a = self._data.given_log_probabilities.get(norm_a, self._config.default_given_logp)
        freq_b = self._data.given_log_probabilities.get(norm_b, self._config.default_given_logp)

        # If both parts are very rare (below -12), it's suspicious
        if freq_a < -12.0 and freq_b < -12.0:
            return False

        return True

    def validate_given_tokens(self, given_tokens: list[str], normalized_cache: dict[str, str] | None = None) -> bool:
        """Validate that given name tokens could plausibly be Chinese."""
        if not given_tokens:
            return False

        # Use consistent validation logic
        if normalized_cache is not None:
            return all(self.is_valid_given_name_token(token, normalized_cache) for token in given_tokens)
        # Use direct memoized calls instead of temporary cache
        return all(self.is_valid_given_name_token(token, None) for token in given_tokens)


# ════════════════════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT SERVICE
# ════════════════════════════════════════════════════════════════════════════════


@cache  # one entry per unique Han character
def _char_to_pinyin(ch: str) -> str:
    return pypinyin.lazy_pinyin(ch, style=pypinyin.Style.NORMAL)[0]


class PinyinCacheService:
    """
    * deterministic, thread‑safe, O(1) repeated look‑ups
    """

    def __init__(self, config: ChineseNameConfig):
        self._config = config
        self._warm_from_csv()

    # ---------- public API ----------
    def han_to_pinyin_fast(self, han_str: str) -> list[str]:
        """Return pinyin for every character, memoising on first sight."""
        return [_char_to_pinyin(c) for c in han_str]

    # (optional) diagnostics you were using elsewhere
    @property
    def cache_size(self) -> int:
        return _char_to_pinyin.cache_info().currsize

    @property
    def is_built(self) -> bool:
        return True

    # ---------- internal ----------
    def _warm_from_csv(self) -> None:
        """Seed the LRU cache with the two CSVs in data/ if present."""
        for fname in ("familyname_orcid.csv", "givenname_orcid.csv"):
            path: Path = Path(self._config.data_dir) / fname
            if not path.exists():
                continue

            key = "surname" if "familyname" in fname else "character"
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    for ch in row[key]:
                        _char_to_pinyin(ch)  # warms the cache once


# ════════════════════════════════════════════════════════════════════════════════
# DATA INITIALIZATION SERVICE
# ════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NameDataStructures:
    """Immutable container for all name-related data structures."""

    # Core surname and given name sets
    surnames: frozenset[str]
    surnames_normalized: frozenset[str]
    compound_surnames: frozenset[str]
    compound_surnames_normalized: frozenset[str]
    given_names: frozenset[str]
    given_names_normalized: frozenset[str]

    # Dynamically generated plausible components from givenname.csv
    plausible_components: frozenset[str]

    # Frequency and probability mappings
    surname_frequencies: dict[str, float]
    surname_log_probabilities: dict[str, float]
    given_log_probabilities: dict[str, float]

    # Pre-computed surname bonuses for cultural plausibility scoring
    surname_bonus_map: dict[str, float]

    # Compound surname mappings
    compound_hyphen_map: dict[str, str]


class DataInitializationService:
    """Service to initialize all name data structures."""

    def __init__(self, config: ChineseNameConfig, cache_service: PinyinCacheService, normalizer: NormalizationService):
        self._config = config
        self._cache_service = cache_service
        self._normalizer = normalizer

    def initialize_data_structures(self) -> NameDataStructures:
        """Initialize all immutable data structures."""

        # Build core surname data
        surnames_raw, surname_frequencies = self._build_surname_data()
        surnames = frozenset(self._normalizer.remove_spaces(s.lower()) for s in surnames_raw)
        compound_surnames = frozenset(s.lower() for s in surnames_raw if " " in s)

        # Build normalized versions
        surnames_normalized = frozenset(self._normalizer.remove_spaces(self._normalizer.norm(s)) for s in surnames_raw)
        compound_surnames_normalized = frozenset(self._normalizer.norm(s) for s in surnames_raw if " " in s)

        # Build given name data and plausible components
        given_names, given_log_probabilities, plausible_components = self._build_given_name_data()
        given_names_normalized = given_names  # Already normalized from pinyin data

        # Build compound surname mappings
        compound_hyphen_map = self._build_compound_hyphen_map(compound_surnames)

        # Build surname log probabilities
        surname_log_probabilities = self._build_surname_log_probabilities(
            surname_frequencies,
            compound_surnames,
            compound_hyphen_map,
        )

        # Pre-compute surname bonuses for cultural plausibility scoring (micro-optimization)
        surname_bonus_map = self._build_surname_bonus_map(surname_frequencies)

        return NameDataStructures(
            surnames=surnames,
            surnames_normalized=surnames_normalized,
            compound_surnames=compound_surnames,
            compound_surnames_normalized=compound_surnames_normalized,
            given_names=given_names,
            given_names_normalized=given_names_normalized,
            plausible_components=plausible_components,
            surname_frequencies=surname_frequencies,
            surname_log_probabilities=surname_log_probabilities,
            given_log_probabilities=given_log_probabilities,
            surname_bonus_map=surname_bonus_map,
            compound_hyphen_map=compound_hyphen_map,
        )

    def _is_plausible_chinese_syllable(self, component: str) -> bool:
        """
        Check if a component is a plausible Chinese syllable suitable for compound splitting.
        Uses a more lenient approach than strict onset-rime decomposition to handle
        romanization variations and valid Chinese syllables.
        """
        if not component or len(component) > 7:
            return False

        # Reject components with forbidden Western patterns
        component_lower = component.lower()
        if self._config.forbidden_patterns_regex.search(component_lower):
            return False

        # Accept if it's a known Chinese syllable (from the given names database)
        # This handles cases like 'xue', 'yue', 'jue' which are valid Chinese syllables
        # even if they don't decompose cleanly in the onset-rime system we're using
        return True  # Since we're already filtering from given_names, they should be valid

    def _build_surname_data(self) -> tuple[set[str], dict[str, float]]:
        """Build surname sets and frequency data."""
        surnames_raw = set()
        surname_frequencies = {}

        with (DATA_PATH / "familyname_orcid.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                han = row["surname"]
                romanized = " ".join(self._cache_service.han_to_pinyin_fast(han)).title()
                surnames_raw.update({romanized, self._normalizer.remove_spaces(romanized)})

                # Store frequency data
                ppm = float(row.get("ppm.1930_2008", 0))
                freq_key = self._normalizer.remove_spaces(romanized.lower())
                surname_frequencies[freq_key] = max(surname_frequencies.get(freq_key, 0), ppm)

        # Add frequency alias: zeng should inherit ceng's frequency from Han character processing
        if "ceng" in surname_frequencies:
            surname_frequencies["zeng"] = surname_frequencies["ceng"]

        # Add Cantonese surnames
        for cant_surname, (mand_surname, han_char) in CANTONESE_SURNAMES.items():
            surnames_raw.add(cant_surname.title())
            # Use lowercase key to match the frequency mapping format
            mand_key = mand_surname.lower()
            if mand_key in surname_frequencies:
                surname_frequencies[cant_surname] = max(
                    surname_frequencies.get(cant_surname, 0),
                    surname_frequencies[mand_key],
                )

        return surnames_raw, surname_frequencies

    def _build_given_name_data(self) -> tuple[frozenset[str], dict[str, float], frozenset[str]]:
        """Build given name data, log probabilities, and dynamically generate plausible components."""
        given_names = set()
        given_frequencies = {}
        total_given_freq = 0

        with (DATA_PATH / "givenname_orcid.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pinyin = self._strip_tone(row["pinyin"])
                given_names.add(pinyin)

                ppm = float(row.get("name.ppm", 0))
                if ppm > 0:
                    given_frequencies[pinyin] = given_frequencies.get(pinyin, 0) + ppm
                    total_given_freq += ppm

        # Convert to log probabilities
        given_log_probabilities = {}
        for given_name, freq in given_frequencies.items():
            prob = freq / total_given_freq if total_given_freq > 0 else 1e-15
            given_log_probabilities[given_name] = math.log(prob)

        # Generate plausible components dynamically from givenname_orcid.csv data
        # This replaces the static PLAUSIBLE_COMPONENTS with real-world usage data

        # Filter multi-syllable entries out of plausible_components
        # They leak in via manual supplements; restrict to ≤7 letters & exactly one onset–rime split
        # to avoid false "split-happy" behaviour with names like Weibian
        filtered_components = set()

        for component in given_names:
            # Check length constraint
            if len(component) > 7:
                continue

            # Check if component is actually usable for splitting
            # Some entries from givenname.csv might not be suitable for compound splitting
            # Use a more lenient approach: include if it passes basic phonetic validation
            # rather than strict onset-rime decomposition

            # Basic phonetic validation - check if it could plausibly be Chinese
            if self._is_plausible_chinese_syllable(component):
                filtered_components.add(component)

        plausible_components = frozenset(filtered_components)

        return frozenset(given_names), given_log_probabilities, plausible_components

    def _build_compound_hyphen_map(self, compound_surnames: frozenset[str]) -> dict[str, str]:
        """Build mapping for hyphenated compound surnames (stores lowercase keys only)."""
        compound_hyphen_map = {}

        for compound in compound_surnames:
            if " " in compound:
                parts = compound.split()
                if len(parts) == 2:
                    # Store only lowercase hyphenated form
                    hyphen_form = f"{parts[0].lower()}-{parts[1].lower()}"
                    # Store lowercase space form (will be title-cased on demand)
                    space_form = f"{parts[0].lower()} {parts[1].lower()}"
                    compound_hyphen_map[hyphen_form] = space_form

        return compound_hyphen_map

    def _build_surname_log_probabilities(
        self,
        surname_frequencies: dict[str, float],
        compound_surnames: frozenset[str],
        compound_hyphen_map: dict[str, str],
    ) -> dict[str, float]:
        """Build surname log probabilities including compound surnames."""
        surname_log_probabilities = {}
        total_surname_freq = sum(surname_frequencies.values())

        # Base surname probabilities
        for surname, freq in surname_frequencies.items():
            if freq > 0:
                prob = freq / total_surname_freq
                surname_log_probabilities[surname] = math.log(prob)
            else:
                surname_log_probabilities[surname] = self._config.default_surname_logp

        # Add compound surname probabilities
        for compound_surname in compound_surnames:
            parts = compound_surname.split()
            if len(parts) == 2:
                # Use reasonable fallback frequency for missing parts (1.0 instead of 1e-6)
                freq1 = surname_frequencies.get(parts[0], 1.0)
                freq2 = surname_frequencies.get(parts[1], 1.0)
                compound_freq = math.sqrt(freq1 * freq2) * self._config.compound_penalty

                # Apply minimum frequency floor to avoid extremely low scores
                min_compound_freq = 0.1  # Reasonable floor for compound surnames
                compound_freq = max(compound_freq, min_compound_freq)

                surname_frequencies[compound_surname] = compound_freq
                prob = compound_freq / total_surname_freq
                surname_log_probabilities[compound_surname] = math.log(prob)

        # Add frequency mappings for compound variants
        for variant_compound, standard_compound in COMPOUND_VARIANTS.items():
            if standard_compound in surname_log_probabilities:
                surname_log_probabilities[variant_compound] = surname_log_probabilities[standard_compound]
            if standard_compound in surname_frequencies:
                surname_frequencies[variant_compound] = surname_frequencies[standard_compound]

        return surname_log_probabilities

    def _build_surname_bonus_map(self, surname_frequencies: dict[str, float]) -> dict[str, float]:
        """Pre-compute surname bonuses for cultural plausibility scoring - performance optimization."""
        surname_bonus_map = {}

        for surname, freq in surname_frequencies.items():
            # Pre-compute the log10(freq+1)*1.2 calculation for fast lookup
            surname_bonus_map[surname] = math.log10(freq + 1) * 1.2

        return surname_bonus_map

    def _strip_tone(self, pinyin_str: str) -> str:
        """Strip tone markers from pinyin string."""
        normalized = unicodedata.normalize("NFKD", pinyin_str)
        return self._config.digits_pattern.sub(
            "",
            "".join(c for c in normalized if not unicodedata.combining(c)),
        ).lower()

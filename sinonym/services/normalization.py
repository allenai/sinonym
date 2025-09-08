"""
Normalization service for Chinese name processing.

This module provides sophisticated text normalization including Wade-Giles conversion,
mixed script handling, and compound name splitting with cultural validation.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sinonym.chinese_names_data import VALID_CHINESE_RIMES
from sinonym.text_processing import CompoundDetector, TextNormalizer, TextPreprocessor
from sinonym.utils.string_manipulation import StringManipulationUtils
from sinonym.utils.thread_cache import ThreadLocalCache

if TYPE_CHECKING:
    from sinonym.coretypes import ChineseNameConfig


@dataclass(frozen=True)
class LazyNormalizationMap:
    """
    Lazy normalization map with true immutability.
    Uses unified ThreadLocalCache for consistency.
    """

    __slots__ = ("_cache", "_normalizer", "_tokens")

    def __init__(self, tokens: tuple[str, ...], normalizer: NormalizationService):
        object.__setattr__(self, "_tokens", tokens)
        object.__setattr__(self, "_normalizer", normalizer)
        object.__setattr__(self, "_cache", ThreadLocalCache())

    def get(self, token: str, default: str | None = None):
        """Get normalized value for token, computing lazily with thread-local cache."""
        return self._cache.get_or_compute(
            token,
            lambda: self._normalizer._text_normalizer.normalize_token(token),
        )

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


@dataclass(frozen=True)
class CompoundMetadata:
    """Metadata for compound surname detection and formatting."""

    is_compound: bool = False
    format_type: str = "unknown"  # "compact", "spaced", "hyphenated", "camelCase"
    compound_target: str | None = None  # standardized form like "ou yang"


@dataclass(frozen=True)
class NormalizedInput:
    """Immutable normalized input - Scala case class style."""

    raw: str  # Original input: "Zhang Wei"
    cleaned: str  # After punctuation/formatting cleanup
    tokens: tuple[str, ...]  # After separator splitting
    roman_tokens: tuple[str, ...]  # After Han→pinyin & mixed-token processing
    norm_map: dict[str, str]  # token → fully normalized (eager)
    compound_metadata: dict[str, CompoundMetadata]  # token → compound info

    @classmethod
    def empty(cls, raw: str = "") -> NormalizedInput:
        """Factory for empty/invalid input."""
        return cls(raw, "", (), (), {}, {})


class NormalizationService:
    """Pure normalization service - Scala-compatible design."""

    def __init__(self, config: ChineseNameConfig, cache_service):
        self._config = config
        self._cache_service = cache_service
        self._data = None
        self._text_normalizer = TextNormalizer(config)
        self._text_preprocessor = TextPreprocessor(config, self)
        self._compound_detector = CompoundDetector(config)

    def set_data_context(self, data) -> None:
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
        return self._text_normalizer.normalize_token(token)

    def get_normalized(self, token: str, norm_cache: dict[str, str]) -> str:
        """
        Get normalized form of token using cache, with fallback to normalization.
        
        Consolidates the common pattern: normalized_cache.get(token, self.norm(token))
        """
        return norm_cache.get(token, self.norm(token))

    def apply(self, raw_name: str) -> NormalizedInput:
        """
        Pure function: raw input → normalized structure.
        Side-effect free, suitable for Scala interop.
        """
        if not raw_name or not raw_name.strip():
            return NormalizedInput.empty(raw_name)

        # Phase 1: Clean input (single regex pass)
        cleaned = self._text_preprocessor.preprocess_input(raw_name, self._data)

        # Phase 2: Handle "LAST, First" format (common in academic/professional contexts)
        if "," in cleaned:
            parts = [part.strip() for part in cleaned.split(",")]
            if len(parts) == 2 and all(parts):  # Exactly 2 non-empty parts
                cleaned = StringManipulationUtils.join_with_spaces(parts[::-1])  # Reverse order: "Last, First" -> "First Last"

        # Phase 3: Detect all-Chinese input for special processing
        is_all_chinese = self._text_preprocessor.is_all_chinese_input(cleaned)

        # Phase 4: Tokenize on separators/whitespace and filter out invalid tokens
        raw_tokens = self._config.sep_pattern.sub(" ", cleaned).split()
        tokens = tuple(t for t in raw_tokens if t and not all(c in string.punctuation for c in t))

        if not tokens:
            return NormalizedInput.empty(raw_name)

        # Phase 5: Process mixed Han/Roman tokens (enhanced for all-Chinese inputs)
        roman_tokens = tuple(self._process_mixed_tokens(list(tokens), is_all_chinese))

        if not roman_tokens:
            return NormalizedInput.empty(raw_name)

        # Phase 6: Create eager normalization map for performance in hot paths
        norm_map = {token: self._text_normalizer.normalize_token(token) for token in roman_tokens}

        # Phase 7: Generate compound metadata for each token (centralized detection)
        compound_metadata = self._compound_detector.generate_compound_metadata(roman_tokens, self._data)

        return NormalizedInput(
            raw=raw_name,
            cleaned=cleaned,
            tokens=tokens,
            roman_tokens=roman_tokens,
            norm_map=norm_map,
            compound_metadata=compound_metadata,
        )

    def _process_mixed_tokens(self, tokens: list[str], is_all_chinese: bool = False) -> list[str]:
        """Extract existing mixed token processing logic with enhanced all-Chinese support."""
        mix = []
        # Cache for character-level CJK pattern checks to avoid repeated regex calls
        cjk_cache = {}

        for token in tokens:
            if self._config.cjk_pattern.search(token) and self._config.ascii_alpha_pattern.search(token):
                # Split mixed Han/Roman token - use character caching for performance
                han_chars = []
                rom_chars = []
                for c in token:
                    if c not in cjk_cache:
                        cjk_cache[c] = bool(self._config.cjk_pattern.search(c))

                    if cjk_cache[c]:
                        han_chars.append(c)
                    elif c.isascii() and c.isalpha():
                        rom_chars.append(c)

                han = "".join(han_chars)
                rom = "".join(rom_chars)
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
                # Convert Han to pinyin with enhanced processing for all-Chinese inputs
                if is_all_chinese and len(token) > 1:
                    # For all-Chinese inputs, process each character separately
                    # This helps with surname/given name boundary detection
                    pinyin_tokens = self._cache_service.han_to_pinyin_fast(token)
                    han_tokens.extend(pinyin_tokens)
                else:
                    # Standard processing for mixed inputs
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
                        parts = StringManipulationUtils.split_and_clean_hyphens(clean_token)
                        roman_tokens_split.extend(parts)
                    # Use centralized split_concat method if available
                    elif self._data:
                        # Create expanded normalized cache including potential splits to reduce redundant normalizations
                        local_cache = {clean_token: self._text_normalizer.normalize_token(clean_token)}

                        # Pre-populate cache with likely split candidates to avoid redundant normalization
                        if len(clean_token) >= 4:  # Only for tokens that could reasonably split
                            for i in range(2, len(clean_token) - 1):  # Split positions
                                part1, part2 = clean_token[:i], clean_token[i:]
                                if part1 not in local_cache:
                                    local_cache[part1] = self._text_normalizer.normalize_token(part1)
                                if part2 not in local_cache:
                                    local_cache[part2] = self._text_normalizer.normalize_token(part2)

                        split_result = StringManipulationUtils.split_concatenated_name(
                            clean_token,
                            local_cache,
                            self._data,
                            self,
                            self._config,
                        )
                        if split_result:
                            roman_tokens_split.extend(split_result)
                        else:
                            roman_tokens_split.append(clean_token)
                    else:
                        roman_tokens_split.append(clean_token)

        # Handle Han/Roman duplication
        if han_tokens and roman_tokens_split:
            # Compare normalized forms directly (memoized for performance)
            han_normalized = {self._text_normalizer.normalize_token(t) for t in han_tokens}
            roman_normalized = {self._text_normalizer.normalize_token(t) for t in roman_tokens_split}

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
            normalized = self._text_normalizer.normalize_token(token)

        if normalized in self._data.given_names_normalized:
            return True

        # Also check the original token (before normalization) in case normalization
        # maps it to something not in the database (e.g., "chuai" -> "zhuai")
        if token.lower() in self._data.given_names_normalized:
            return True

        # If not found, check if it can be split into valid syllables
        if StringManipulationUtils.split_concatenated_name(token, normalized_cache, self._data, self, self._config):
            return True

        # Handle hyphenated tokens by splitting and validating each part
        if "-" in token:
            parts = StringManipulationUtils.split_and_clean_hyphens(token)
            return all(self.is_valid_given_name_token(part, normalized_cache) for part in parts)

        # Check if token is a surname used in given position (e.g., "Wen Zhang")
        if StringManipulationUtils.remove_spaces(normalized) in self._data.surnames_normalized:
            return True

        # Must pass Chinese phonetic validation
        return self.is_valid_chinese_phonetics(token)

    def validate_given_tokens(self, given_tokens: list[str], normalized_cache: dict[str, str] | None = None) -> bool:
        """Validate that given name tokens could plausibly be Chinese."""
        if not given_tokens:
            return False

        # Use consistent validation logic
        if normalized_cache is not None:
            return all(self.is_valid_given_name_token(token, normalized_cache) for token in given_tokens)
        # Use direct memoized calls instead of temporary cache
        return all(self.is_valid_given_name_token(token, None) for token in given_tokens)

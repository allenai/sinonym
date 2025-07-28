"""
Normalization service for Chinese name processing.

This module provides sophisticated text normalization including Wade-Giles conversion,
mixed script handling, and compound name splitting with cultural validation.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from sinonym.chinese_names_data import (
    COMPOUND_VARIANTS,
    HIGH_CONFIDENCE_ANCHORS,
    NON_WADE_GILES_SYLLABLE_RULES,
    ONE_LETTER_RULES,
    ROMANIZATION_EXCEPTIONS,
    VALID_CHINESE_RIMES,
    WADE_GILES_SYLLABLE_RULES,
)
from sinonym.patterns import (
    SUFFIX_REGEX,
    SUFFIX_REPLACEMENTS,
    WADE_GILES_REGEX,
    WADE_GILES_REPLACEMENTS,
)
from sinonym.types import ChineseNameConfig


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

    def __init__(self, config: ChineseNameConfig, cache_service):
        self._config = config
        self._cache_service = cache_service
        self._data = None

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
                        return WADE_GILES_REPLACEMENTS[i - 1]
                return match.group(0)  # Fallback (should never happen)

            # Apply prefix conversions with single regex substitution
            result = WADE_GILES_REGEX.sub(wade_giles_replacer, token)

        # Step 3: Suffix-level Wade-Giles conversions
        def suffix_replacer(match):
            """Replacement function for suffix regex substitution."""
            # Find which group matched (groups are 1-indexed)
            for i, group in enumerate(match.groups(), 1):
                if group is not None:
                    return SUFFIX_REPLACEMENTS[i - 1]
            return match.group(0)  # Fallback (should never happen)

        # Apply suffix conversions with single regex substitution
        result = SUFFIX_REGEX.sub(suffix_replacer, result)

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

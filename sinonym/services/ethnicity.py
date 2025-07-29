"""
Ethnicity classification service for Chinese name processing.

This module provides sophisticated ethnicity classification to distinguish
Chinese names from Korean, Vietnamese, Japanese, and Western names using
linguistic patterns and cultural markers.
"""
from __future__ import annotations

from sinonym.chinese_names_data import (
    JAPANESE_SURNAMES,
    KOREAN_AMBIGUOUS_PATTERNS,
    KOREAN_GIVEN_PAIRS,
    KOREAN_GIVEN_PATTERNS,
    KOREAN_ONLY_SURNAMES,
    KOREAN_SPECIFIC_PATTERNS,
    OVERLAPPING_KOREAN_SURNAMES,
    OVERLAPPING_VIETNAMESE_SURNAMES,
    VIETNAMESE_GIVEN_PATTERNS,
    VIETNAMESE_ONLY_SURNAMES,
    WESTERN_NAMES,
)
from sinonym.types import ParseResult


class EthnicityClassificationService:
    """Service for classifying names by ethnicity using linguistic patterns."""

    def __init__(self, config, normalizer, data):
        self._config = config
        self._normalizer = normalizer
        self._data = data

    def classify_ethnicity(self, tokens: tuple[str, ...], normalized_cache: dict[str, str]) -> ParseResult:
        """
        Three-tier Chinese vs non-Chinese classification system.

        Tier 1: Definitive Evidence (High Confidence)
        Tier 2: Cultural Context (Medium Confidence)
        Tier 3: Chinese Default (Low Confidence)
        """
        if not tokens:
            return ParseResult.success_with_name("")

        # Prepare expanded keys for pattern matching
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.append(token)
            if "-" in token:
                expanded_tokens.extend(token.split("-"))

        # Use the passed-in normalized_cache, with fallback for missing tokens
        def get_normalized(token: str) -> str:
            if token in normalized_cache:
                return normalized_cache[token]
            return self._normalizer.norm(token)

        # Create comprehensive key sets for pattern matching
        original_keys_raw = [t.lower() for t in expanded_tokens]
        original_keys_normalized = [get_normalized(t) for t in expanded_tokens]
        expanded_keys = list(set(original_keys_raw + original_keys_normalized))

        # =================================================================
        # TIER 1: DEFINITIVE EVIDENCE (High Confidence)
        # =================================================================

        # Check for Korean-only surnames (definitive Korean)
        for key in expanded_keys:
            clean_key = self._normalizer.remove_spaces(key)
            if clean_key in KOREAN_ONLY_SURNAMES:
                return ParseResult.failure("Korean-only surname detected")

        # Check for Japanese surnames (definitive Japanese)
        for key in expanded_keys:
            clean_key = self._normalizer.remove_spaces(key)
            if clean_key in JAPANESE_SURNAMES:
                return ParseResult.failure("Japanese surname detected")

        # Check for Western names (definitive Western)
        for key in expanded_keys:
            if key in WESTERN_NAMES:
                return ParseResult.failure("Western name detected")

        # Check for Vietnamese-only surnames (definitive Vietnamese)
        for key in expanded_keys:
            clean_key = self._normalizer.remove_spaces(key)
            if clean_key in VIETNAMESE_ONLY_SURNAMES:
                return ParseResult.failure("appears to be Vietnamese name")

        # =================================================================
        # TIER 2: CULTURAL CONTEXT (Medium Confidence)
        # =================================================================

        korean_structural_score = self._calculate_korean_structural_patterns(tokens, expanded_keys)

        # Use higher threshold if there's overlapping Chinese surname evidence
        has_overlapping_chinese_surname = False
        for token in tokens:
            clean_token = self._normalizer.remove_spaces(token).lower()
            if clean_token in self._data.surnames and (
                clean_token in OVERLAPPING_KOREAN_SURNAMES or clean_token in OVERLAPPING_VIETNAMESE_SURNAMES
            ):
                has_overlapping_chinese_surname = True
                break

        korean_threshold = 2.5 if has_overlapping_chinese_surname else 2.0

        if korean_structural_score >= korean_threshold:
            return ParseResult.failure("Korean structural patterns detected")

        vietnamese_structural_score = self._calculate_vietnamese_structural_patterns(tokens, expanded_keys)
        if vietnamese_structural_score >= 2.0:
            return ParseResult.failure("appears to be Vietnamese name")

        # =================================================================
        # TIER 3: CHINESE DEFAULT (Low Confidence)
        # =================================================================

        chinese_surname_strength = self._calculate_chinese_surname_strength(expanded_keys, normalized_cache)

        # Chinese default: Accept if we have any reasonable Chinese surname evidence
        if chinese_surname_strength >= 0.5:
            return ParseResult.success_with_name("")

        # No Chinese evidence found
        return ParseResult.failure("no Chinese evidence found")

    def _classify_first_token_surname(self, tokens: tuple[str, ...]) -> str:
        """Classify the first token's surname type for ethnicity detection."""
        if not tokens:
            return "none"

        first_token = tokens[0]
        clean_first_token = self._normalizer.remove_spaces(first_token).lower()

        # Check for definitive surname types first
        if clean_first_token in KOREAN_ONLY_SURNAMES:
            return "korean_only"
        if clean_first_token in VIETNAMESE_ONLY_SURNAMES:
            return "vietnamese_only"

        # Check for overlapping surnames
        if clean_first_token in OVERLAPPING_KOREAN_SURNAMES:
            return "korean_overlapping"
        if clean_first_token in OVERLAPPING_VIETNAMESE_SURNAMES:
            return "vietnamese_overlapping"

        # Check for non-overlapping Chinese surnames
        if (
            clean_first_token in self._data.surnames
            and clean_first_token not in OVERLAPPING_KOREAN_SURNAMES
            and clean_first_token not in OVERLAPPING_VIETNAMESE_SURNAMES
        ):
            return "chinese_only"

        return "none"

    def _analyze_tokens_for_patterns(self, tokens: tuple[str, ...]) -> dict:
        """Single-pass analysis of tokens for pattern detection."""
        analysis = {
            "surname_type": self._classify_first_token_surname(tokens),
            "hyphenated_tokens": [],
            "korean_specific_tokens": [],
            "korean_ambiguous_tokens": [],
            "vietnamese_tokens": [],
            "korean_given_pairs": [],
            "has_thi_pattern": False,
        }

        # Single pass through all tokens
        for i, token in enumerate(tokens):
            token_lower = token.lower()

            # Check for hyphenated patterns
            if "-" in token:
                parts = token.split("-")
                if len(parts) == 2:
                    analysis["hyphenated_tokens"].append((parts[0].lower(), parts[1].lower()))

            # Check for Vietnamese "Thi" pattern
            if token_lower in ["thi", "thị"]:
                analysis["has_thi_pattern"] = True

            # Categorize tokens by pattern type
            if token_lower in KOREAN_SPECIFIC_PATTERNS:
                analysis["korean_specific_tokens"].append(token_lower)
            elif token_lower in KOREAN_AMBIGUOUS_PATTERNS:
                analysis["korean_ambiguous_tokens"].append(token_lower)

            if token_lower in VIETNAMESE_GIVEN_PATTERNS:
                analysis["vietnamese_tokens"].append(token_lower)

        # Check for Korean given name pairs (skip surname)
        given_tokens = [token.lower() for token in tokens[1:]]
        for i in range(len(given_tokens) - 1):
            pair = (given_tokens[i], given_tokens[i + 1])
            if pair in KOREAN_GIVEN_PAIRS:
                analysis["korean_given_pairs"].append(pair)

        return analysis

    def _calculate_korean_structural_patterns(self, tokens: tuple[str, ...], expanded_keys: list[str]) -> float:
        """Calculate Korean structural pattern score using simplified pattern analysis."""
        analysis = self._analyze_tokens_for_patterns(tokens)

        # Early returns for definitive cases
        if analysis["surname_type"] == "chinese_only":
            return 0.0  # Block Korean scoring for non-overlapping Chinese surnames
        if analysis["surname_type"] == "korean_only":
            return 10.0  # Definitive Korean evidence

        score = 0.0

        # 1. Hyphenated Korean patterns (strong signal)
        for first, second in analysis["hyphenated_tokens"]:
            if first in KOREAN_GIVEN_PATTERNS and second in KOREAN_GIVEN_PATTERNS:
                score += 3.0

        # 2. Known Korean name pairs (strong signal)
        score += len(analysis["korean_given_pairs"]) * 3.0

        # 3. Korean-specific patterns
        korean_specific_count = len(analysis["korean_specific_tokens"])
        if korean_specific_count >= 2:
            score += 2.0
        elif korean_specific_count == 1:
            score += 1.0

        # 4. Ambiguous patterns (only with Korean overlapping surname)
        if analysis["surname_type"] == "korean_overlapping":
            korean_ambiguous_count = len(analysis["korean_ambiguous_tokens"])
            if korean_ambiguous_count >= 2:
                score += 1.0

        return score

    def _calculate_vietnamese_structural_patterns(self, tokens: tuple[str, ...], expanded_keys: list[str]) -> float:
        """Calculate Vietnamese structural pattern score using simplified analysis."""
        analysis = self._analyze_tokens_for_patterns(tokens)
        score = 0.0

        # 1. Vietnamese "Thi" pattern (very strong indicator)
        if analysis["has_thi_pattern"]:
            score += 3.0

        # 2. Vietnamese surname + given name patterns (but only if no Chinese surname)
        vietnamese_surname_count = 1 if analysis["surname_type"] == "vietnamese_overlapping" else 0
        vietnamese_given_count = len(analysis["vietnamese_tokens"])

        if vietnamese_surname_count >= 1 and vietnamese_given_count >= 1:
            # Check if any token is a Chinese surname (not just overlapping)
            has_chinese_surname = any(
                self._normalizer.remove_spaces(key) in self._data.surnames for key in expanded_keys
            )

            if not has_chinese_surname:
                score += 2.0  # Strong Vietnamese pattern

        # 3. Multiple Vietnamese given names
        if vietnamese_given_count >= 2:
            score += 1.5  # Medium Vietnamese pattern

        return score

    def _calculate_chinese_surname_strength(self, expanded_keys: list[str], normalized_cache: dict[str, str]) -> float:
        """Calculate Chinese surname strength (simplified from original)."""
        chinese_surname_strength = 0.0

        # Create key-to-normalized mapping
        key_to_normalized = {}
        for key in expanded_keys:
            # Handle both original and normalized forms
            key_to_normalized[key] = normalized_cache.get(key, key)

        for key in expanded_keys:
            clean_key = self._normalizer.remove_spaces(key)
            clean_key_lower = clean_key.lower()

            # Check if this is a Chinese surname
            normalized_key = self._normalizer.remove_spaces(key_to_normalized.get(key, key))
            is_chinese_surname = (
                clean_key in self._data.surnames
                or clean_key_lower in self._data.surnames
                or normalized_key in self._data.surnames_normalized
            )

            if is_chinese_surname:
                # Get frequency
                surname_freq = self._data.surname_frequencies.get(
                    clean_key_lower,
                    0,
                ) or self._data.surname_frequencies.get(normalized_key, 0)

                if surname_freq > 0:
                    if surname_freq >= 10000:
                        base_strength = 1.5
                    elif surname_freq >= 1000:
                        base_strength = 1.0
                    elif surname_freq >= 100:
                        base_strength = 0.6
                    else:
                        base_strength = 0.3
                else:
                    base_strength = 0.2

                chinese_surname_strength += base_strength
            else:
                # NEW: Check if this could be a compound Chinese given name
                split_result = self._normalizer.split_concat(clean_key_lower, normalized_cache)
                if split_result and len(split_result) >= 2:
                    # Check if all components are valid Chinese given name components
                    all_chinese_components = True
                    for component in split_result:
                        comp_normalized = self._normalizer.norm(component)
                        if comp_normalized not in self._data.given_names_normalized:
                            all_chinese_components = False
                            break

                    if all_chinese_components:
                        # Add modest boost for compound given names (helps cases like "Beining")
                        chinese_surname_strength += 0.3

        return chinese_surname_strength


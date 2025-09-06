"""
Ethnicity classification service for Chinese name processing.

This module provides sophisticated ethnicity classification to distinguish
Chinese names from Korean, Vietnamese, Japanese, and Western names using
linguistic patterns and cultural markers.
"""

from __future__ import annotations

from sinonym.chinese_names_data import (
    COMPOUND_VARIANTS,
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
from sinonym.coretypes import ParseResult
from sinonym.utils.string_manipulation import StringManipulationUtils

# Optional ML Japanese classifier imports - consolidated from separate service
try:
    import logging
    from pathlib import Path

    # Ensure custom model components are importable when deserializing
    import sinonym.ml_model_components  # noqa: F401
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class _MLJapaneseClassifier:
    """Consolidated ML Japanese classifier - moved from separate service for simplification."""

    def __init__(self, confidence_threshold: float = 0.8):
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._available = ML_AVAILABLE

        if ML_AVAILABLE:
            try:
                # Prefer skops artifact; fall back to legacy joblib if needed
                from sinonym.resources import load_joblib, load_skops

                try:
                    self._model = load_skops("chinese_japanese_classifier.skops")
                except Exception as skops_err:
                    logging.info(
                        f"SKOPS model not available or failed to load ({skops_err}); "
                        "falling back to legacy joblib artifact.",
                    )
                    self._model = load_joblib("chinese_japanese_classifier.joblib")
            except Exception as e:
                logging.warning(f"Failed to load ML Japanese classifier: {e}")
                self._available = False

    def is_available(self) -> bool:
        """Check if ML classifier is available and loaded."""
        return self._available and self._model is not None

    def classify_all_chinese_name(self, name: str) -> ParseResult:
        """Classify an all-Chinese character name as Chinese or Japanese."""
        if not self.is_available():
            return ParseResult.success_with_name("")  # Default to allowing through

        try:
            # Get prediction and confidence (same as original)
            prediction = self._model.predict([name])[0]  # 'cn' or 'jp'
            probabilities = self._model.predict_proba([name])[0]
            confidence = max(probabilities)

            # Only reject as Japanese if we're very confident
            if prediction == "jp" and confidence >= self._confidence_threshold:
                return ParseResult.failure("japanese")
            return ParseResult.success_with_name("")

        except Exception as e:
            logging.warning(f"ML Japanese classifier error: {e}")
            return ParseResult.success_with_name("")  # Default to allowing through


class EthnicityClassificationService:
    """Service for classifying names by ethnicity using linguistic patterns."""

    def __init__(self, context_or_config, normalizer=None, data=None):
        # Support both old interface (config, normalizer, data) and new context interface
        if hasattr(context_or_config, "config"):
            # New context interface
            self._config = context_or_config.config
            self._normalizer = context_or_config.normalizer
            self._data = context_or_config.data
        else:
            # Legacy interface - maintain backwards compatibility
            self._config = context_or_config
            self._normalizer = normalizer
            self._data = data
        # Initialize consolidated ML Japanese classifier
        self._ml_classifier = _MLJapaneseClassifier(confidence_threshold=0.8)

    def classify_ethnicity(self, tokens: tuple[str, ...], normalized_cache: dict[str, str], original_text: str = "") -> ParseResult:
        """
        Three-tier Chinese vs non-Chinese classification system.

        ML Enhancement: For all-Chinese character inputs, use ML classifier first
        Tier 1: Definitive Evidence (High Confidence)  
        Tier 2: Cultural Context (Medium Confidence)
        Tier 3: Chinese Default (Low Confidence)
        """
        if not tokens:
            return ParseResult.success_with_name("")

        # =================================================================
        # ML ENHANCEMENT: All-Chinese Character Japanese Detection
        # =================================================================

        # Check if this is an all-Chinese character input that could be Japanese
        if (original_text and
            self._normalizer._text_preprocessor.is_all_chinese_input(original_text) and
            self._ml_classifier.is_available()):

            # Use ML classifier to check for Japanese names in Chinese characters
            ml_result = self._ml_classifier.classify_all_chinese_name(original_text)

            # If ML classifier confidently identifies it as Japanese, reject it
            if ml_result.success is False and "japanese" in ml_result.error_message:
                return ParseResult.failure("Japanese name detected by ML classifier")

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

        # Single loop for all definitive evidence checks (short-circuit optimization)
        for key in expanded_keys:
            # Check for Western names first (most common case, faster lookup)
            if key in WESTERN_NAMES:
                return ParseResult.failure("Western name detected")

            # Clean key once for multiple checks
            clean_key = StringManipulationUtils.remove_spaces(key)

            # Check Korean-only surnames (definitive Korean)
            if clean_key in KOREAN_ONLY_SURNAMES:
                return ParseResult.failure("Korean-only surname detected")

            # Check Japanese surnames (definitive Japanese)
            if clean_key in JAPANESE_SURNAMES:
                return ParseResult.failure("Japanese surname detected")

            # Check Vietnamese-only surnames (definitive Vietnamese)
            if clean_key in VIETNAMESE_ONLY_SURNAMES:
                return ParseResult.failure("appears to be Vietnamese name")

        # =================================================================
        # TIER 2: CULTURAL CONTEXT (Medium Confidence)
        # =================================================================

        # Optimized validation chain: calculate overlapping surname evidence once with reduced string ops
        def check_overlapping_surname(token):
            clean_token_lower = StringManipulationUtils.remove_spaces(token).lower()
            return (
                clean_token_lower in self._data.surnames and (
                    clean_token_lower in OVERLAPPING_KOREAN_SURNAMES or
                    clean_token_lower in OVERLAPPING_VIETNAMESE_SURNAMES
                )
            )

        has_overlapping_chinese_surname = any(check_overlapping_surname(token) for token in tokens)

        korean_structural_score = self._calculate_korean_structural_patterns(tokens, expanded_keys, normalized_cache)
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
        clean_first_token = StringManipulationUtils.remove_spaces(first_token).lower()

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

    def _calculate_korean_structural_patterns(
        self,
        tokens: tuple[str, ...],
        expanded_keys: list[str],
        normalized_cache: dict[str, str] | None = None,
    ) -> float:
        """Calculate Korean structural pattern score using simplified pattern analysis."""
        analysis = self._analyze_tokens_for_patterns(tokens)

        # Early returns for definitive cases
        if analysis["surname_type"] == "chinese_only":
            return 0.0  # Block Korean scoring for non-overlapping Chinese surnames
        if analysis["surname_type"] == "korean_only":
            return 10.0  # Definitive Korean evidence

        score = 0.0

        # Overlapping Korean surname anywhere in the name (given-first or surname-first)
        overlapping_any = any(
            StringManipulationUtils.remove_spaces(t).lower() in OVERLAPPING_KOREAN_SURNAMES
            for t in tokens
        )

        # 1. Hyphenated Korean patterns (strong signal) — Hyphen Gate
        for first, second in analysis["hyphenated_tokens"]:
            # Existing curated-list rule (definitive)
            if first in KOREAN_GIVEN_PATTERNS and second in KOREAN_GIVEN_PATTERNS:
                score += 3.0
                continue

            # Strict Chinese DB membership (no split-to-plausible fallback)
            def is_chinese_given_strict(tok: str) -> bool:
                if normalized_cache and tok in normalized_cache:
                    normalized = normalized_cache[tok]
                else:
                    normalized = self._normalizer.norm(tok)
                return (
                    normalized in self._data.given_names_normalized or tok.lower() in self._data.given_names_normalized
                )

            # Korean signature based on curated token sets only (no RR heuristics)
            def has_korean_signature(tok: str) -> bool:
                t = tok.lower()
                return t in KOREAN_GIVEN_PATTERNS or t in KOREAN_SPECIFIC_PATTERNS

            first_cn = is_chinese_given_strict(first)
            second_cn = is_chinese_given_strict(second)
            if overlapping_any:
                # With overlapping Korean surnames present, treat hyphenated given name
                # as Korean if EITHER half matches curated Korean tokens.
                if has_korean_signature(first) or has_korean_signature(second):
                    score += 3.0
            # General case (no overlapping surname): require that not both halves are
            # Chinese given tokens to avoid false positives on Chinese names.
            elif (not (first_cn and second_cn)) and (has_korean_signature(first) or has_korean_signature(second)):
                score += 3.0

        # 2. Known Korean name pairs (strong signal)
        score += len(analysis["korean_given_pairs"]) * 3.0

        # 3. Korean-specific patterns
        korean_specific_count = len(analysis["korean_specific_tokens"])
        if korean_specific_count >= 2:
            score += 2.0
        elif korean_specific_count == 1:
            score += 1.0

        # 3b. Overlapping-surname + space-separated Korean given token that fails Chinese validation
        # This captures cases like "Ha Young Lee" where:
        # - Surname is overlapping (e.g., Ha/Lee/Cho)
        # - One given token is a Korean-specific romanization (e.g., "young")
        # - That token is NOT a valid Chinese given token under our normalization
        # 4. Space-separated Korean signature with overlapping Korean surname anywhere
        #    Example: "Ha Young Lee" (overlapping surname 'Lee', token 'Young' is Korean-specific)
        overlapping_any = any(
            StringManipulationUtils.remove_spaces(t).lower() in OVERLAPPING_KOREAN_SURNAMES
            for t in tokens
        )
        if overlapping_any:
            def is_chinese_given_strict(tok: str) -> bool:
                if normalized_cache and tok in normalized_cache:
                    normalized = normalized_cache[tok]
                else:
                    normalized = self._normalizer.norm(tok)
                return (
                    normalized in self._data.given_names_normalized or tok.lower() in self._data.given_names_normalized
                )

            for tok in tokens:
                tok_clean = StringManipulationUtils.remove_spaces(tok).lower()
                if tok_clean in OVERLAPPING_KOREAN_SURNAMES:
                    continue
                t_lower = tok.lower()
                if (t_lower in KOREAN_GIVEN_PATTERNS or t_lower in KOREAN_SPECIFIC_PATTERNS) and not is_chinese_given_strict(tok):
                    score += 3.0
                    break

        # 5. Ambiguous patterns (only with Korean overlapping surname in first position)
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
                StringManipulationUtils.remove_spaces(key) in self._data.surnames for key in expanded_keys
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
            clean_key = StringManipulationUtils.remove_spaces(key)
            clean_key_lower = clean_key.lower()

            # Check if this is a Chinese surname
            normalized_key = StringManipulationUtils.remove_spaces(key_to_normalized.get(key, key))
            is_chinese_surname = self._data.is_surname(clean_key, normalized_key) or clean_key_lower in self._data.surnames

            if is_chinese_surname:
                # Get frequency
                surname_freq = self._data.get_surname_freq(clean_key_lower) or self._data.get_surname_freq(normalized_key)

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
            # Check for compact compound surnames in COMPOUND_VARIANTS
            elif clean_key_lower in COMPOUND_VARIANTS:
                # This is a compact compound surname - give it good strength
                target_compound = COMPOUND_VARIANTS[clean_key_lower]
                compound_parts = target_compound.split()

                # Verify that the target compound parts are valid Chinese surnames
                if len(compound_parts) == 2:
                    part1, part2 = compound_parts
                    if part1 in self._data.surnames_normalized and part2 in self._data.surnames_normalized:
                        # Both parts are valid Chinese surnames, give high confidence
                        chinese_surname_strength += 1.0
            else:
                # NEW: Check if this could be a compound Chinese given name
                split_result = StringManipulationUtils.split_concatenated_name(
                    clean_key_lower, normalized_cache, self._data, self._normalizer, self._config,
                )
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

"""
Ethnicity classification service for Chinese name processing.

This module provides sophisticated ethnicity classification to distinguish
Chinese names from Korean, Vietnamese, Japanese, and Western names using
linguistic patterns and cultural markers.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sinonym.chinese_names_data import (
    COMPOUND_VARIANTS,
    ETHNICITY_CHINESE_SURNAME_ROMANIZATION_ALIASES,
    JAPANESE_SURNAMES,
    KOREAN_AMBIGUOUS_PATTERNS,
    KOREAN_DIRECTIONAL_FAMILY_FIRST_SURNAMES,
    KOREAN_DIRECTIONAL_SINGLE_GIVEN_NAMES,
    KOREAN_GIVEN_PAIRS,
    KOREAN_GIVEN_PATTERNS,
    KOREAN_ONLY_SURNAMES,
    KOREAN_SPECIFIC_PATTERNS,
    NAME_ORDER_ROUTING_KOREAN_SURNAMES,
    OVERLAPPING_KOREAN_SURNAMES,
    OVERLAPPING_VIETNAMESE_SURNAMES,
    VIETNAMESE_GIVEN_PATTERNS,
    VIETNAMESE_ONLY_SURNAMES,
    WESTERN_NAMES,
)
from sinonym.coretypes import ParseResult
from sinonym.services.name_lookup import SurnameResolver
from sinonym.utils.string_manipulation import StringManipulationUtils
from sinonym.utils.thread_cache import ThreadLocalCache

MIN_COMPOUND_SURNAME_TOKEN_COUNT = 3
JAPANESE_CLASSIFIER_REJECTION = "japanese"
JAPANESE_CLASSIFIER_RUNTIME_ERROR = "ML Japanese classifier failed"
MIN_DIRECTIONAL_KOREAN_TOKENS = 2
MAX_DIRECTIONAL_KOREAN_TOKENS = 3
MIN_CONTEXTUAL_TAIWAN_SURNAME_FREQUENCY = 100.0
MIN_CHINESE_SURNAME_STRENGTH = 0.5
CONTEXTUAL_TAIWAN_GIVEN_PARTS = {
    "jungting": ("jung", "ting"),
    "tsung-jr": ("tsung", "jr"),
}
KOREAN_DIRECTIONAL_SURNAMES = frozenset(
    NAME_ORDER_ROUTING_KOREAN_SURNAMES | KOREAN_ONLY_SURNAMES | OVERLAPPING_KOREAN_SURNAMES,
)

# Optional ML Japanese classifier imports - consolidated from separate service
try:
    # Ensure custom model components are importable when deserializing
    import sinonym.ml_model_components  # noqa: F401

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class _MLJapaneseClassifier:
    """Consolidated ML Japanese classifier - moved from separate service for simplification."""

    def __init__(self, confidence_threshold: float = 0.8):
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._available = ML_AVAILABLE
        # Thread-local cache for ML classification results
        self._cache = ThreadLocalCache()

        if ML_AVAILABLE:
            try:
                # Prefer skops artifact; fall back to legacy joblib if needed
                from sinonym.resources import load_joblib, load_skops  # noqa: PLC0415

                try:
                    self._model = load_skops("chinese_japanese_classifier.skops")
                except Exception as skops_err:  # noqa: BLE001 - skops may raise several deserialization errors.
                    LOGGER.info(
                        "SKOPS model not available or failed to load (%s); falling back to legacy joblib artifact.",
                        skops_err,
                    )
                    self._model = load_joblib("chinese_japanese_classifier.joblib")
            except Exception as e:  # noqa: BLE001 - optional classifier load failure disables the ML path.
                LOGGER.warning("Failed to load ML Japanese classifier: %s", e)
                self._available = False

    def is_available(self) -> bool:
        """Check if ML classifier is available and loaded."""
        return self._available and self._model is not None

    def classify_all_chinese_name(self, name: str) -> ParseResult:
        """Classify an all-Chinese character name as Chinese or Japanese."""
        if not self.is_available():
            return ParseResult.success_with_name("")  # Default to allowing through

        cached = self._cache.get(name)
        if cached is not None:
            return cached

        try:
            # Get prediction and confidence (same as original)
            prediction = self._model.predict([name])[0]  # 'cn' or 'jp'
            probabilities = self._model.predict_proba([name])[0]
            confidence = max(probabilities)

            # Only reject as Japanese if we're very confident
            if prediction == "jp" and confidence >= self._confidence_threshold:
                result = ParseResult.failure(JAPANESE_CLASSIFIER_REJECTION)
            else:
                result = ParseResult.success_with_name("")
        except Exception as e:  # noqa: BLE001 - model-backed classifiers may raise arbitrary runtime errors.
            LOGGER.warning("ML Japanese classifier error for %r: %s", name, e, exc_info=True)
            return ParseResult.failure(JAPANESE_CLASSIFIER_RUNTIME_ERROR)
        else:
            self._cache.set(name, result)
            return result

    def japanese_probability(self, name: str) -> float:
        """Return the Japanese-class probability, raising on model runtime failures."""
        if not self.is_available():
            return 0.0

        try:
            probabilities = self._model.predict_proba([name])[0]
            classes = list(getattr(self._model, "classes_", ()))
            if "jp" in classes:
                return float(probabilities[classes.index("jp")])

            prediction = self._model.predict([name])[0]
            if prediction == "jp":
                return float(max(probabilities))
        except Exception as e:
            LOGGER.warning("ML Japanese classifier probability error for %r: %s", name, e, exc_info=True)
            message = "ML Japanese classifier probability failed"
            raise RuntimeError(message) from e
        return 0.0


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
        self._surname_resolver = SurnameResolver(self._data, self._normalizer)
        # Initialize consolidated ML Japanese classifier
        self._ml_classifier = _MLJapaneseClassifier(confidence_threshold=0.8)

    def japanese_probability(self, compact_chinese_text: str) -> float:
        """Return the Japanese-class probability for compact all-CJK text.

        Missing classifiers are treated as unavailable; loaded classifiers that
        fail at prediction time raise after logging context.
        """
        if not compact_chinese_text:
            return 0.0
        return self._ml_classifier.japanese_probability(compact_chinese_text)

    def _starts_with_chinese_compound_surname(
        self,
        tokens: tuple[str, ...],
        normalized_cache: dict[str, str],
    ) -> bool:
        """Return whether tokens begin with a recognized Chinese compound surname."""
        if len(tokens) < MIN_COMPOUND_SURNAME_TOKEN_COUNT:
            return False

        first_two = [self._normalizer.get_normalized(token, normalized_cache) for token in tokens[:2]]
        spaced = " ".join(first_two).lower()
        compact = StringManipulationUtils.remove_spaces(spaced)
        return (
            spaced in self._data.compound_surnames
            or spaced in self._data.compound_surnames_normalized
            or compact in self._data.compound_original_format_map
        )

    def classify_ethnicity(
        self,
        tokens: tuple[str, ...],
        normalized_cache: dict[str, str],
        original_text: str = "",
    ) -> ParseResult:
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
        compact_chinese_text = self._normalizer._text_preprocessor.compact_all_chinese_input(original_text)
        if compact_chinese_text and self._ml_classifier.is_available():
            # Use ML classifier to check for Japanese names in Chinese characters
            ml_result = self._ml_classifier.classify_all_chinese_name(compact_chinese_text)

            # If ML classifier confidently identifies it as Japanese, reject it
            if ml_result.success is False:
                if ml_result.error_message != JAPANESE_CLASSIFIER_REJECTION:
                    return ml_result
                if not self._starts_with_chinese_compound_surname(tokens, normalized_cache):
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

        # Create comprehensive key sets for pattern matching. Dedup while preserving
        # insertion order (NOT list(set(...)), whose order is PYTHONHASHSEED-dependent and
        # made the first-match ethnicity loop below non-deterministic across processes).
        original_keys_raw = [t.lower() for t in expanded_tokens]
        original_keys_normalized = [get_normalized(t) for t in expanded_tokens]
        expanded_keys = list(dict.fromkeys(original_keys_raw + original_keys_normalized))

        # Directional Korean structure must be evaluated before every
        # affirmative Chinese shortcut. Surname/given roles are essential:
        # many individual syllables and surnames overlap with Chinese.
        if self._has_directional_korean_structure(tokens):
            return ParseResult.failure("Korean structural patterns detected")

        if self.contextual_taiwan_given_parts(tokens) is not None:
            return ParseResult.success_with_name("")

        # =================================================================
        # TIER 1: DEFINITIVE EVIDENCE (High Confidence)
        # =================================================================

        if self._has_wade_giles_apostrophe_surname(tokens):
            return ParseResult.success_with_name("")

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

        if self._has_reviewed_chinese_surname_alias(tokens):
            return ParseResult.success_with_name("")
        if self._has_dominant_surname_compact_initial(tokens):
            return ParseResult.success_with_name("")

        # =================================================================
        # TIER 2: CULTURAL CONTEXT (Medium Confidence)
        # =================================================================

        # Optimized validation chain: calculate overlapping surname evidence once with reduced string ops
        def check_overlapping_surname(token):
            clean_token_lower = StringManipulationUtils.remove_spaces(token).lower()
            return clean_token_lower in self._data.surnames and (
                clean_token_lower in OVERLAPPING_KOREAN_SURNAMES or clean_token_lower in OVERLAPPING_VIETNAMESE_SURNAMES
            )

        has_overlapping_chinese_surname = any(check_overlapping_surname(token) for token in tokens)

        korean_score, vietnamese_score = self._calculate_non_chinese_patterns_unified(tokens, expanded_keys, normalized_cache)
        korean_threshold = 2.5 if has_overlapping_chinese_surname else 2.0

        if korean_score >= korean_threshold:
            return ParseResult.failure("Korean structural patterns detected")

        if vietnamese_score >= 2.0:
            return ParseResult.failure("appears to be Vietnamese name")

        # =================================================================
        # TIER 3: CHINESE DEFAULT (Low Confidence)
        # =================================================================

        # Chinese default: Accept if we have any reasonable Chinese surname evidence
        if self._has_sufficient_chinese_surname_strength(expanded_keys, normalized_cache):
            return ParseResult.success_with_name("")

        # No Chinese evidence found
        return ParseResult.failure("no Chinese evidence found")

    @staticmethod
    def _has_reviewed_chinese_surname_alias(tokens: tuple[str, ...]) -> bool:
        """Return whether a name edge is a reviewed Chinese surname spelling."""
        return bool(tokens) and (
            tokens[0].lower() in ETHNICITY_CHINESE_SURNAME_ROMANIZATION_ALIASES
            or tokens[-1].lower() in ETHNICITY_CHINESE_SURNAME_ROMANIZATION_ALIASES
        )

    def _has_wade_giles_apostrophe_surname(self, tokens: tuple[str, ...]) -> bool:
        """Return whether an exact name edge has backed Wade-Giles surname evidence."""
        return bool(tokens) and (
            self._surname_resolver.evidence_is_wade_giles_apostrophe_surname(tokens[0])
            or self._surname_resolver.evidence_is_wade_giles_apostrophe_surname(tokens[-1])
        )

    def _has_dominant_surname_compact_initial(self, tokens: tuple[str, ...]) -> bool:
        """Return dominant family-first surname plus a vowelless initial bundle."""
        if len(tokens) != 2:
            return False
        abbreviation = tokens[1]
        return bool(
            self._surname_resolver.evidence_is_dominant_surname(tokens[0])
            and abbreviation.isalpha()
            and 2 <= len(abbreviation) <= 3
            and not any(character.lower() in "aeiou" for character in abbreviation),
        )

    @staticmethod
    def _split_roman_components(tokens: tuple[str, ...]) -> list[str]:
        """Return lowercase Roman components from spaced or hyphenated tokens."""
        return [part.lower() for token in tokens for part in token.split("-") if part and part.isalpha()]

    @classmethod
    @lru_cache(maxsize=4096)
    def _has_directional_korean_structure(cls, tokens: tuple[str, ...]) -> bool:
        """Return whether surname and given evidence align as a Korean name."""
        if not MIN_DIRECTIONAL_KOREAN_TOKENS <= len(tokens) <= MAX_DIRECTIONAL_KOREAN_TOKENS:
            return False

        lowered = tuple(StringManipulationUtils.remove_spaces(token).lower() for token in tokens)
        given_first_parts = cls._split_roman_components(lowered[:-1])
        family_first_parts = cls._split_roman_components(lowered[1:])

        given_first = lowered[-1] in KOREAN_DIRECTIONAL_SURNAMES and tuple(given_first_parts) in KOREAN_GIVEN_PAIRS
        if given_first:
            return True

        directional_single = (
            lowered[0] in KOREAN_DIRECTIONAL_FAMILY_FIRST_SURNAMES
            and len(family_first_parts) == 1
            and family_first_parts[0] in KOREAN_DIRECTIONAL_SINGLE_GIVEN_NAMES
        )
        if directional_single:
            return True

        return bool(
            lowered[0] in KOREAN_DIRECTIONAL_SURNAMES
            and lowered[-1] not in KOREAN_DIRECTIONAL_SURNAMES
            and tuple(family_first_parts) in KOREAN_GIVEN_PAIRS,
        )

    def contextual_taiwan_given_parts(self, tokens: tuple[str, ...]) -> tuple[str, str] | None:
        """Return an exact contextual Taiwan given-name segmentation."""
        if len(tokens) != 2 or self._has_directional_korean_structure(tokens):
            return None
        given_key = StringManipulationUtils.remove_spaces(tokens[0]).lower()
        parts = CONTEXTUAL_TAIWAN_GIVEN_PARTS.get(given_key)
        if parts is None:
            return None
        if self._surname_resolver.evidence_frequency(tokens[-1]) < MIN_CONTEXTUAL_TAIWAN_SURNAME_FREQUENCY:
            return None
        return parts

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

        self._collect_single_token_patterns(tokens, analysis)
        analysis["korean_given_pairs"] = self._korean_given_pairs(tokens, analysis["surname_type"])
        return analysis

    @staticmethod
    def _collect_single_token_patterns(tokens: tuple[str, ...], analysis: dict) -> None:
        """Collect per-token ethnicity pattern signals."""
        for token in tokens:
            token_lower = token.lower()
            if "-" in token:
                parts = token.split("-")
                if len(parts) == 2:
                    analysis["hyphenated_tokens"].append((parts[0].lower(), parts[1].lower()))

            if token_lower in ["thi", "thị"]:
                analysis["has_thi_pattern"] = True
            if token_lower in KOREAN_SPECIFIC_PATTERNS:
                analysis["korean_specific_tokens"].append(token_lower)
            elif token_lower in KOREAN_AMBIGUOUS_PATTERNS:
                analysis["korean_ambiguous_tokens"].append(token_lower)
            if token_lower in VIETNAMESE_GIVEN_PATTERNS:
                analysis["vietnamese_tokens"].append(token_lower)

    def _korean_given_pairs(self, tokens: tuple[str, ...], surname_type: str) -> list[tuple[str, str]]:
        """Return Korean given-name pairs under plausible surname positions."""
        seen_pairs = set()
        pairs: list[tuple[str, str]] = []
        for given_tokens in self._candidate_korean_given_sequences(tokens, surname_type):
            for i in range(len(given_tokens) - 1):
                pair = (given_tokens[i], given_tokens[i + 1])
                if pair in KOREAN_GIVEN_PAIRS and pair not in seen_pairs:
                    pairs.append(pair)
                    seen_pairs.add(pair)
        return pairs

    @staticmethod
    def _candidate_korean_given_sequences(tokens: tuple[str, ...], surname_type: str) -> list[list[str]]:
        """Return plausible given-token spans for Korean pair detection."""

        def split_given_tokens(given_tokens: tuple[str, ...]) -> list[str]:
            split_tokens: list[str] = []
            for token in given_tokens:
                split_tokens.extend(part.lower() for part in token.split("-") if part)
            return split_tokens

        candidate_given_sequences = []
        if surname_type in {"korean_only", "korean_overlapping", "none"}:
            candidate_given_sequences.append(split_given_tokens(tokens[1:]))

        last_token = StringManipulationUtils.remove_spaces(tokens[-1]).lower() if tokens else ""
        if len(tokens) >= 2 and last_token in OVERLAPPING_KOREAN_SURNAMES:
            candidate_given_sequences.append(split_given_tokens(tokens[:-1]))

        return candidate_given_sequences

    @staticmethod
    def _has_trailing_overlapping_korean_surname(tokens: tuple[str, ...]) -> bool:
        """Return whether a final token can be a Korean surname."""
        return bool(
            tokens and StringManipulationUtils.remove_spaces(tokens[-1]).lower() in OVERLAPPING_KOREAN_SURNAMES,
        )

    def _first_token_has_dominant_chinese_surname(self, tokens: tuple[str, ...]) -> bool:
        """Return whether the first token has dominant as-written Chinese surname evidence."""
        if not tokens:
            return False
        return self._surname_resolver.evidence_is_dominant_surname(tokens[0])

    def _calculate_non_chinese_patterns_unified(
        self,
        tokens: tuple[str, ...],
        expanded_keys: list[str],
        normalized_cache: dict[str, str] | None = None,
    ) -> tuple[float, float]:
        """Calculate Korean and Vietnamese structural pattern scores with shared analysis."""
        # Single pattern analysis pass (eliminates duplicate work)
        analysis = self._analyze_tokens_for_patterns(tokens)

        # Calculate Korean score
        korean_score = self._calculate_korean_score_from_analysis(
            analysis,
            tokens,
            expanded_keys,
            normalized_cache,
        )

        # Calculate Vietnamese score
        vietnamese_score = self._calculate_vietnamese_score_from_analysis(
            analysis,
            tokens,
            expanded_keys,
        )

        return korean_score, vietnamese_score

    def _calculate_korean_score_from_analysis(
        self,
        analysis: dict,
        tokens: tuple[str, ...],
        expanded_keys: list[str],
        normalized_cache: dict[str, str] | None = None,
    ) -> float:
        """Calculate Korean score from pre-computed analysis."""
        # Early returns for definitive cases.
        if analysis["surname_type"] == "chinese_only":
            trailing_korean_pair = self._has_trailing_overlapping_korean_surname(tokens) and analysis["korean_given_pairs"]
            if not trailing_korean_pair or self._first_token_has_dominant_chinese_surname(tokens):
                return 0.0  # Block Korean scoring for non-overlapping Chinese surnames
        if analysis["surname_type"] == "korean_only":
            return 10.0  # Definitive Korean evidence

        score = 0.0

        # Overlapping Korean surname anywhere in the name
        overlapping_any = any(StringManipulationUtils.remove_spaces(t).lower() in OVERLAPPING_KOREAN_SURNAMES for t in tokens)

        # Helper functions (reused from original)
        def is_chinese_given_strict(tok: str) -> bool:
            normalized = normalized_cache[tok] if normalized_cache and tok in normalized_cache else self._normalizer.norm(tok)
            return normalized in self._data.given_names_normalized or tok.lower() in self._data.given_names_normalized

        def has_korean_signature(tok: str) -> bool:
            t = tok.lower()
            return t in KOREAN_GIVEN_PATTERNS or t in KOREAN_SPECIFIC_PATTERNS

        # 1. Hyphenated Korean patterns (strong signal)
        for first, second in analysis["hyphenated_tokens"]:
            # Existing curated-list rule (definitive)
            if first in KOREAN_GIVEN_PATTERNS and second in KOREAN_GIVEN_PATTERNS:
                score += 3.0
                continue

            first_cn = is_chinese_given_strict(first)
            second_cn = is_chinese_given_strict(second)
            if overlapping_any:
                if (has_korean_signature(first) or has_korean_signature(second)) and not (first_cn and second_cn):
                    score += 3.0
            elif (not (first_cn and second_cn)) and (has_korean_signature(first) or has_korean_signature(second)):
                score += 3.0

        # 2. Known Korean name pairs (strong signal)
        score += len(analysis["korean_given_pairs"]) * 3.0

        # 3. Korean-specific tokens (bounded signal to avoid over-rejection)
        korean_specific_count = len(analysis["korean_specific_tokens"])
        if korean_specific_count >= 2:
            score += 2.0
        elif korean_specific_count == 1:
            score += 1.0

        # 4. Ambiguous patterns (only with Korean overlapping surname in first position)
        if analysis["surname_type"] == "korean_overlapping":
            korean_ambiguous_count = len(analysis["korean_ambiguous_tokens"])
            if korean_ambiguous_count >= 2:
                score += 1.0

        return score

    def _calculate_vietnamese_score_from_analysis(
        self,
        analysis: dict,
        tokens: tuple[str, ...],
        expanded_keys: list[str],
    ) -> float:
        """Calculate Vietnamese score from pre-computed analysis."""
        score = 0.0

        # 1. Vietnamese "Thi" pattern (very strong indicator)
        if analysis["has_thi_pattern"]:
            score += 3.0

        # 2. Vietnamese surname + given name patterns (but only if no Chinese surname)
        vietnamese_surname_count = 1 if analysis["surname_type"] == "vietnamese_overlapping" else 0
        vietnamese_given_count = len(analysis["vietnamese_tokens"])

        if vietnamese_surname_count >= 1 and vietnamese_given_count >= 1:
            # Check if any token is a Chinese surname
            has_chinese_surname = any(StringManipulationUtils.remove_spaces(key) in self._data.surnames for key in expanded_keys)

            if not has_chinese_surname:
                score += 2.0  # Strong Vietnamese pattern

        # 3. Multiple Vietnamese given names
        if vietnamese_given_count >= 2:
            score += 1.5  # Medium Vietnamese pattern

        return score

    def _has_sufficient_chinese_surname_strength(
        self,
        expanded_keys: list[str],
        normalized_cache: dict[str, str],
    ) -> bool:
        """Return whether nonnegative surname evidence reaches the Chinese threshold."""
        chinese_surname_strength = 0.0

        # Local memoization for repeated split/component checks
        split_result_cache: dict[str, list[str] | None] = {}
        split_component_validity_cache: dict[tuple[str, ...], bool] = {}
        component_normalized_cache: dict[str, str] = {}

        for key in expanded_keys:
            clean_key = StringManipulationUtils.remove_spaces(key)
            clean_key_lower = clean_key.lower()

            # Check if this is a Chinese surname
            is_chinese_surname = self._surname_resolver.evidence_is_surname(clean_key)

            if is_chinese_surname:
                # Get frequency
                surname_freq = self._surname_resolver.evidence_frequency(clean_key)

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
                if chinese_surname_strength >= MIN_CHINESE_SURNAME_STRENGTH:
                    return True
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
                        return True
            else:
                # NEW: Check if this could be a compound Chinese given name
                if clean_key_lower in split_result_cache:
                    split_result = split_result_cache[clean_key_lower]
                else:
                    split_result = StringManipulationUtils.split_concatenated_name(
                        clean_key_lower,
                        normalized_cache,
                        self._data,
                        self._normalizer,
                        self._config,
                    )
                    split_result_cache[clean_key_lower] = split_result

                if split_result and len(split_result) >= 2:
                    split_result_key = tuple(split_result)
                    if split_result_key in split_component_validity_cache:
                        all_chinese_components = split_component_validity_cache[split_result_key]
                    else:
                        # Check if all components are valid Chinese given name components
                        all_chinese_components = True
                        for component in split_result:
                            if component in component_normalized_cache:
                                comp_normalized = component_normalized_cache[component]
                            else:
                                comp_normalized = self._normalizer.norm(component)
                                component_normalized_cache[component] = comp_normalized

                            if comp_normalized not in self._data.given_names_normalized:
                                all_chinese_components = False
                                break

                        split_component_validity_cache[split_result_key] = all_chinese_components

                    if all_chinese_components:
                        # Add modest boost for compound given names (helps cases like "Beining")
                        chinese_surname_strength += 0.3
                        if chinese_surname_strength >= MIN_CHINESE_SURNAME_STRENGTH:
                            return True

        return False

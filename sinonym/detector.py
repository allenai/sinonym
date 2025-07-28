"""
Chinese Name Detection and Normalization Module

This module provides sophisticated detection and normalization of Chinese names from various
romanization systems, with robust filtering to prevent false positives from Western, Korean,
Vietnamese, and Japanese names.

## Overview

The core functionality is provided by the `ChineseNameDetector` class, which uses a multi-stage
pipeline to process names:

1. **Input Preprocessing**: Handles mixed scripts, normalizes romanization variants
2. **Ethnicity Classification**: Filters non-Chinese names using linguistic patterns
3. **Probabilistic Parsing**: Identifies surname/given name boundaries using frequency data
4. **Compound Name Splitting**: Splits fused given names using tiered confidence system
5. **Output Formatting**: Produces standardized "Given-Name Surname" format

## Architecture

### Clean Service Separation
- **NormalizationService**: Pure centralized normalization with lazy computation
- **PinyinCacheService**: Isolated cache management with persistent storage
- **DataInitializationService**: Immutable data structure initialization
- **ChineseNameDetector**: Main detection engine with dependency injection

### Scala-Compatible Design
- **Immutable Data Structures**: All core data is frozen/immutable for thread safety
- **Functional Error Handling**: ParseResult with Either-like success/failure semantics
- **Pure Functions**: Side-effect free normalization suitable for Scala interop
- **Dependency Injection**: Clean separation of concerns, no circular dependencies

### Performance Optimizations
- **Lazy Normalization**: On-demand token processing reduces memory usage
- **Early Exit Patterns**: Non-Chinese names detected quickly without full processing
- **Persistent Caching**: Han→Pinyin mappings cached to disk for fast startup
- **Single-Pass Processing**: Minimized regex operations and string transformations

## Key Features

### Comprehensive Romanization Support
- **Pinyin**: Standard mainland Chinese romanization
- **Wade-Giles**: Traditional romanization system with aspirated consonants
- **Cantonese**: Hong Kong and southern Chinese romanizations
- **Mixed Scripts**: Handles names with both Han characters and Roman letters

### Advanced Name Splitting
The module uses a sophisticated **tiered confidence system** for splitting compound given names:

- **Gold Standard**: Both parts are high-confidence Chinese syllables (anchors)
- **Silver Standard**: One part is high-confidence, one is plausible
- **Bronze Standard**: Both parts are plausible with cultural validation

This prevents incorrect splitting of Western names (e.g., "Julian" → "Jul", "ian") while
correctly handling Chinese compounds (e.g., "Weiming" → "Wei", "Ming").

### Robust False Positive Prevention
- **Forbidden Phonetic Patterns**: Blocks Western consonant clusters (th, dr, br, gl, etc.)
- **Korean Name Detection**: Identifies Korean surnames and given name patterns
- **Vietnamese Name Detection**: Recognizes Vietnamese naming conventions
- **Cultural Validation**: Applies frequency analysis and phonetic rules

### Data-Driven Approach
- **Surname Database**: ~1400 Chinese surnames with frequency data
- **Given Name Database**: ~3000 Chinese given name syllables with probabilities
- **Compound Syllables**: ~400 valid Chinese syllable components for splitting
- **Ethnicity Markers**: Curated lists of non-Chinese name patterns

## Usage Examples

```python
# from s2and.chinese_names import ChineseNameDetector  # Original import - now internal

# Basic usage
detector = ChineseNameDetector()
result = detector.is_chinese_name("Zhang Wei")
# Returns: ParseResult(success=True, result="Wei Zhang")

# Compound given names
result = detector.is_chinese_name("Li Weiming")
# Returns: ParseResult(success=True, result="Wei-Ming Li")

# Mixed scripts
result = detector.is_chinese_name("张Wei Ming")
# Returns: ParseResult(success=True, result="Wei-Ming Zhang")

# Non-Chinese names (correctly rejected)
result = detector.is_chinese_name("John Smith")
# Returns: ParseResult(success=False, error_message="surname not recognised")

result = detector.is_chinese_name("Kim Min-jun")
# Returns: ParseResult(success=False, error_message="appears to be Korean name")

# Access result data
if result.success:
    print(f"Formatted name: {result.result}")
else:
    print(f"Error: {result.error_message}")

# Advanced usage - access normalization service directly
normalized_token = detector._normalizer.norm("wei")  # Returns: "wei"
normalized_token = detector._normalizer.norm("ts'ai")  # Returns: "cai" (Wade-Giles conversion)

# Get cache information
cache_info = detector.get_cache_info()
print(f"Cache size: {cache_info.cache_size} characters")
```

## Architecture

### Core Classes

- **ChineseNameDetector**: Main detection engine with caching and data management
- **PinyinCacheService**: Fast Han character to Pinyin conversion with disk caching
- **DataInitializationService**: Loads and processes surname/given name databases
- **ChineseNameConfig**: Configuration and regex patterns

### Data Sources

- **familyname_orcid.csv**: Chinese surnames with frequency data
- **givenname_orcid.csv**: Chinese given names with usage statistics
- **han_pinyin_cache.pkl**: Precomputed Han character to Pinyin mappings

### Processing Pipeline

1. **Preprocessing**: Clean input, normalize punctuation, handle compound surnames
2. **Tokenization**: Split into tokens, convert Han characters to Pinyin
3. **Ethnicity Check**: Score for Korean/Vietnamese/Japanese patterns vs Chinese evidence
4. **Parse Generation**: Create all valid (surname, given_name) combinations
5. **Scoring**: Rank parses using frequency data and cultural patterns
6. **Formatting**: Split compound names, capitalize, format as "Given-Name Surname"

## Error Handling

The module provides detailed error messages for debugging:
- `"surname not recognised"`: No valid Chinese surname found
- `"appears to be Korean name"`: Korean linguistic patterns detected
- `"appears to be Vietnamese name"`: Vietnamese naming conventions identified
- `"given name tokens are not plausibly Chinese"`: Given name validation failed

## Performance

- **Production Ready**: ~0.16ms average per name (comprehensive benchmark validated)
- **Cold start**: ~100ms (initial data loading with persistent cache)
- **Warm processing**: Sub-millisecond for most names with early exit optimization
- **Memory efficiency**: Lazy normalization reduces peak usage by ~60%
- **Cache optimization**: Persistent disk cache for Han→Pinyin mappings
- **Scalability**: Thread-safe design suitable for high-throughput processing

## API

The main class is `ChineseNameDetector`:
- `ChineseNameDetector()`: Main detector class
- `detector.is_chinese_name(name) -> ParseResult`: Returns structured result with success/error
- `ParseResult.success`: Boolean indicating if name was recognized as Chinese
- `ParseResult.result`: Formatted name if successful
- `ParseResult.error_message`: Error description if failed

## Thread Safety

The module is thread-safe after initialization. The caching layer uses immutable
data structures and the detector can be safely used from multiple threads.
"""

import logging
import math
import string

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
from sinonym.services import (
    CacheInfo,
    ChineseNameConfig,
    DataInitializationService,
    NameDataStructures,
    NormalizationService,
    ParseResult,
    PinyinCacheService,
)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN CHINESE NAME DETECTOR CLASS
# ════════════════════════════════════════════════════════════════════════════════


class ChineseNameDetector:
    """Main Chinese name detection and normalization service."""

    def __init__(self, config: ChineseNameConfig | None = None):
        self._config = config or ChineseNameConfig.create_default()
        self._cache_service = PinyinCacheService(self._config)
        self._normalizer = NormalizationService(self._config, self._cache_service)
        self._data_service = DataInitializationService(self._config, self._cache_service, self._normalizer)
        self._data: NameDataStructures | None = None

        # Initialize data structures
        self._initialize()

    def _initialize(self) -> None:
        """Initialize cache and data structures."""
        try:
            self._data = self._data_service.initialize_data_structures()
            # Inject data context into normalizer after initialization
            self._normalizer.set_data_context(self._data)
        except Exception as e:
            logging.warning(f"Failed to initialize at construction: {e}. Will initialize lazily.")

    def _ensure_initialized(self) -> None:
        """Ensure data is initialized (lazy initialization)."""
        if self._data is None:
            self._data = self._data_service.initialize_data_structures()
            # Inject data context into normalizer
            self._normalizer.set_data_context(self._data)

    # Public API methods
    def get_cache_info(self) -> CacheInfo:
        """Get cache information."""
        return self._cache_service.get_cache_info()

    def clear_pinyin_cache(self) -> None:
        """Clear the pinyin cache."""
        self._cache_service.clear_cache()

    def _remove_spaces(self, text: str) -> str:
        """Cache frequently used space removal operation - delegate to normalizer."""
        return self._normalizer.remove_spaces(text)

    def is_chinese_name(self, raw_name: str) -> ParseResult:
        """
        Main API method: Detect if a name is Chinese and normalize it.

        Returns ParseResult with:
        - success=True, result=formatted_name if Chinese name detected
        - success=False, error_message=reason if not Chinese name
        """
        # Input validation
        if not raw_name or len(raw_name) > 100:  # Reasonable name length limit
            return ParseResult.failure("invalid input length")

        if all(c in string.punctuation + string.whitespace for c in raw_name):
            return ParseResult.failure("name contains only punctuation/whitespace")

        self._ensure_initialized()

        # Use new normalization service for cleaner pipeline
        normalized_input = self._normalizer.apply(raw_name)

        if len(normalized_input.roman_tokens) < 2:
            return ParseResult.failure("needs at least 2 Roman tokens")

        # Check for non-Chinese ethnicity (optimized single-pass)
        non_chinese_result = self._single_pass_ethnicity_check(normalized_input.roman_tokens, normalized_input.norm_map)
        if non_chinese_result.success is False:
            return non_chinese_result

        # Try parsing in both orders
        for order in (normalized_input.roman_tokens, normalized_input.roman_tokens[::-1]):
            parse_result = self._parse_name_order(list(order), normalized_input.norm_map)
            if parse_result.success:
                return parse_result

        return ParseResult.failure("name not recognised as Chinese")

    def _single_pass_ethnicity_check(self, tokens: tuple[str, ...], normalized_cache: dict[str, str]) -> ParseResult:
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

        return chinese_surname_strength

    def _parse_name_order(self, order: list[str], normalized_cache: dict[str, str]) -> ParseResult:
        """Parse using probabilistic system with fallback - pattern matching style."""
        # Try probabilistic parsing first
        parse_result = self._best_parse(order, normalized_cache)

        # Pattern match on result type (Scala-like)
        if parse_result.success and isinstance(parse_result.result, tuple):
            surname_tokens, given_tokens = parse_result.result
            try:
                formatted_name = self._format_name_output(surname_tokens, given_tokens, normalized_cache)
                return ParseResult.success_with_name(formatted_name)
            except ValueError as e:
                return ParseResult.failure(str(e))

        # Fallback parsing - try different surname positions
        fallback_attempts = [
            (-1, slice(None, -1)),  # surname-last pattern
            (0, slice(1, None)),  # surname-first pattern
        ]

        for surname_pos, given_slice in fallback_attempts:
            result = self._try_fallback_parse(order, surname_pos, given_slice, normalized_cache)
            if result.success:
                return result

        return ParseResult.failure("surname not recognised")

    def _try_fallback_parse(
        self,
        order: list[str],
        surname_pos: int,
        given_slice: slice,
        normalized_cache: dict[str, str],
    ) -> ParseResult:
        """Try a single fallback parse configuration - pure function"""
        surname_token = order[surname_pos]
        normalized_surname = normalized_cache.get(surname_token, self._normalizer.norm(surname_token))

        if (
            len(surname_token) > 1  # Don't treat single letters as surnames
            and self._normalizer.remove_spaces(normalized_surname) in self._data.surnames_normalized
        ):
            surname_tokens = [surname_token]
            given_tokens = order[given_slice]
            if given_tokens:
                # Check if this parse would have a reasonable score
                score = self._calculate_parse_score(surname_tokens, given_tokens, order, normalized_cache)

                # Western name detection pattern
                has_single_letter_given = any(len(token) == 1 for token in given_tokens)
                has_multi_syllable_tokens = any(len(token) > 3 for token in order)

                # Check if any multi-syllable token is a known Chinese surname
                has_chinese_surname_in_tokens = any(
                    len(token) > 3
                    and (
                        self._normalizer.norm(token) in self._data.surnames
                        or self._normalizer.remove_spaces(normalized_cache.get(token, self._normalizer.norm(token)))
                        in self._data.surnames_normalized
                    )
                    for token in order
                )

                if (
                    has_single_letter_given
                    and has_multi_syllable_tokens
                    and score < -25.0
                    and not has_chinese_surname_in_tokens
                ):
                    # This looks like a Western name where single letters are initials
                    return ParseResult.failure("Western name pattern detected")

                try:
                    formatted_name = self._format_name_output(surname_tokens, given_tokens, normalized_cache)
                    return ParseResult.success_with_name(formatted_name)
                except ValueError:
                    return ParseResult.failure("Format validation failed")

        return ParseResult.failure("No valid surname found")

    def _best_parse(self, tokens: list[str], normalized_cache: dict[str, str]) -> ParseResult:
        """Find the best parse using probabilistic scoring."""
        if len(tokens) < 2:
            return ParseResult.failure("needs at least 2 tokens")

        parses = self._generate_all_parses(tokens, normalized_cache)
        if not parses:
            return ParseResult.failure("surname not recognised")

        # Score all parses
        scored_parses = []
        for surname_tokens, given_tokens in parses:
            score = self._calculate_parse_score(surname_tokens, given_tokens, tokens, normalized_cache)

            # Additional validation: reject parses where single letters are used as given names
            # when there are multi-syllable alternatives available (likely Western names)
            has_single_letter_given = any(len(token) == 1 for token in given_tokens)
            has_multi_syllable_tokens = any(len(token) > 3 for token in tokens)

            # Check if any multi-syllable token is a known Chinese surname
            has_chinese_surname_in_tokens = any(
                len(token) > 3
                and (
                    self._normalizer.norm(token) in self._data.surnames
                    or self._normalizer.remove_spaces(normalized_cache.get(token, self._normalizer.norm(token)))
                    in self._data.surnames_normalized
                )
                for token in tokens
            )

            if (
                has_single_letter_given
                and has_multi_syllable_tokens
                and score < -25.0
                and not has_chinese_surname_in_tokens
            ):
                # This looks like a Western name where single letters are initials
                continue

            if score > float("-inf"):
                scored_parses.append((surname_tokens, given_tokens, score))

        if not scored_parses:
            return ParseResult.failure("no valid parse found")

        # Find best score and handle ties
        best_score = max(scored_parses, key=lambda x: x[2])[2]
        best_parses = [p for p in scored_parses if abs(p[2] - best_score) <= 0.1]

        if len(best_parses) > 1:
            # Tie-break with surname frequency
            best_parse_result = max(
                best_parses,
                key=lambda x: self._data.surname_frequencies.get(
                    self._normalizer.remove_spaces(self._surname_key(x[0], normalized_cache)),
                    0,
                ),
            )
        else:
            best_parse_result = best_parses[0]

        return ParseResult.success_with_parse(best_parse_result[0], best_parse_result[1])

    def _generate_all_parses(
        self,
        tokens: list[str],
        normalized_cache: dict[str, str],
    ) -> list[tuple[list[str], list[str]]]:
        """Generate all possible (surname, given_name) parses for the tokens."""
        if len(tokens) < 2:
            return []

        parses = []

        # 1. Check compound surnames (2-token surnames)
        if len(tokens) >= 3:
            if self._is_compound_surname(tokens, 0, normalized_cache):
                parses.append((tokens[0:2], tokens[2:]))
            if self._is_compound_surname(tokens, 1, normalized_cache):
                parses.append((tokens[1:3], tokens[0:1]))

        # 2. Single-token surnames - only at beginning or end (contiguous sequences only)
        # Surname-first pattern: surname + given_names
        # Check both original and normalized forms, but exclude single letters
        first_token = tokens[0]
        first_normalized = normalized_cache.get(first_token, self._normalizer.norm(first_token))
        if len(first_token) > 1 and (  # Don't treat single letters as surnames
            self._normalizer.norm(first_token) in self._data.surnames
            or self._normalizer.remove_spaces(first_normalized) in self._data.surnames_normalized
        ):
            parses.append(([first_token], tokens[1:]))

        # Surname-last pattern: given_names + surname
        if len(tokens) >= 2:
            last_token = tokens[-1]
            last_normalized = normalized_cache.get(last_token, self._normalizer.norm(last_token))
            if len(last_token) > 1 and (  # Don't treat single letters as surnames
                self._normalizer.norm(last_token) in self._data.surnames
                or self._normalizer.remove_spaces(last_normalized) in self._data.surnames_normalized
            ):
                parses.append(([last_token], tokens[:-1]))

        # 3. Fallback: Check for hyphenated compound surnames at beginning or end
        # Beginning position
        if "-" in tokens[0]:
            lowercase_key = tokens[0].lower()  # Don't remove hyphens for compound_hyphen_map lookup
            if lowercase_key in self._data.compound_hyphen_map:
                space_form = self._data.compound_hyphen_map[lowercase_key]
                # Title-case the compound parts for output
                compound_parts = [part.title() for part in space_form.split()]
                if len(compound_parts) == 2 and len(tokens) > 1:
                    parses.append((compound_parts, tokens[1:]))

        # End position
        if len(tokens) >= 2 and "-" in tokens[-1]:
            lowercase_key = tokens[-1].lower()  # Don't remove hyphens for compound_hyphen_map lookup
            if lowercase_key in self._data.compound_hyphen_map:
                space_form = self._data.compound_hyphen_map[lowercase_key]
                # Title-case the compound parts for output
                compound_parts = [part.title() for part in space_form.split()]
                if len(compound_parts) == 2:
                    parses.append((compound_parts, tokens[:-1]))

        return parses

    def _is_compound_surname(self, tokens: list[str], start: int, normalized_cache: dict[str, str]) -> bool:
        """Check if tokens starting at 'start' form a compound surname."""
        if start + 1 >= len(tokens):
            return False

        # Use cached normalized values
        token1 = tokens[start]
        token2 = tokens[start + 1]
        keys = [
            normalized_cache.get(token1, self._normalizer.norm(token1)),
            normalized_cache.get(token2, self._normalizer.norm(token2)),
        ]
        compound_key = " ".join(keys)
        compound_original = " ".join(t.lower() for t in [token1, token2])

        return (
            compound_key in self._data.compound_surnames_normalized
            or compound_original in self._data.compound_surnames_normalized
            or (
                compound_original in COMPOUND_VARIANTS
                and COMPOUND_VARIANTS[compound_original] in self._data.compound_surnames_normalized
            )
        )

    def _calculate_parse_score(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        tokens: list[str],
        normalized_cache: dict[str, str],
    ) -> float:
        """Calculate unified score for a parse candidate."""
        if not given_tokens:
            return float("-inf")

        surname_key = self._surname_key(surname_tokens, normalized_cache)
        surname_logp = self._data.surname_log_probabilities.get(surname_key, self._config.default_surname_logp)

        # Handle compound surname mapping mismatches
        if surname_logp == self._config.default_surname_logp and len(surname_tokens) > 1:
            original_compound = " ".join(t.lower() for t in surname_tokens)
            surname_logp = self._data.surname_log_probabilities.get(
                original_compound,
                self._config.default_surname_logp,
            )

        given_logp_sum = sum(
            self._data.given_log_probabilities.get(
                self._given_name_key(g_token, normalized_cache),
                self._config.default_given_logp,
            )
            for g_token in given_tokens
        )

        validation_penalty = 0.0 if self._all_valid_given(given_tokens, normalized_cache) else -3.0

        compound_given_bonus = 0.0
        if len(given_tokens) == 2 and all(
            normalized_cache.get(t, self._normalizer.norm(t)) in self._data.given_names_normalized for t in given_tokens
        ):
            compound_given_bonus = 0.8

        cultural_score = self._cultural_plausibility_score(surname_tokens, given_tokens, normalized_cache)

        return surname_logp + given_logp_sum + validation_penalty + compound_given_bonus + cultural_score

    def _surname_key(self, surname_tokens: list[str], normalized_cache: dict[str, str]) -> str:
        """Convert surname tokens to lookup key, preferring original form when available."""
        if len(surname_tokens) == 1:
            # Try original form first (more likely to preserve correct romanization)
            original_key = self._normalizer.norm(surname_tokens[0])
            if original_key in self._data.surname_frequencies:
                return original_key
            # Fall back to normalized form
            return self._normalizer.remove_spaces(
                normalized_cache.get(surname_tokens[0], self._normalizer.norm(surname_tokens[0])),
            )
        # Compound surname - join with space
        return " ".join(normalized_cache.get(t, self._normalizer.norm(t)) for t in surname_tokens)

    def _given_name_key(self, given_token: str, normalized_cache: dict[str, str]) -> str:
        """Convert given name token to lookup key, preferring original form when available."""
        # Try original form first (more likely to preserve correct romanization)
        original_key = self._normalizer.norm(given_token)
        if original_key in self._data.given_log_probabilities:
            return original_key
        # Fall back to normalized form
        return self._normalizer.norm(normalized_cache.get(given_token, self._normalizer.norm(given_token)))

    def _all_valid_given(self, given_tokens: list[str], normalized_cache: dict[str, str]) -> bool:
        """Check if all given name tokens are valid - delegate to normalizer."""
        return self._normalizer.validate_given_tokens(given_tokens, normalized_cache)

    def _split_concat(self, token: str, normalized_cache: dict[str, str] | None = None) -> list[str] | None:
        """
        Try to split a fused or hyphenated given name - delegate to normalizer.
        """
        return self._normalizer.split_concat(token, normalized_cache)

    def _is_plausible_chinese_split(self, norm_a: str, norm_b: str, original_token: str) -> bool:
        """
        Check if a split represents an authentic Chinese name combination - delegate to normalizer.
        """
        return self._normalizer.is_plausible_chinese_split(norm_a, norm_b, original_token)

    def _cultural_plausibility_score(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        normalized_cache: dict[str, str],
    ) -> float:
        """Calculate cultural plausibility score for a Chinese name parse."""
        if not surname_tokens or not given_tokens:
            return -10.0

        score = 0.0
        surname_key = self._surname_key(surname_tokens, normalized_cache)

        # Surname frequency bonus
        surname_freq = self._data.surname_frequencies.get(self._normalizer.remove_spaces(surname_key), 0)
        if surname_freq == 0 and " " in surname_key:
            surname_freq = self._data.surname_frequencies.get(surname_key, 0)

        if surname_freq > 0:
            score += min(5.0, math.log10(surname_freq + 1) * 1.2)
        else:
            score -= 3.0

        # Compound surname validation
        if len(surname_tokens) == 2:
            compound_original = " ".join(t.lower() for t in surname_tokens)
            is_valid_compound = (
                surname_key in self._data.compound_surnames_normalized
                or self._normalizer.remove_spaces(surname_key) in self._data.compound_surnames_normalized
                or (
                    compound_original in COMPOUND_VARIANTS
                    and COMPOUND_VARIANTS[compound_original] in self._data.compound_surnames_normalized
                )
            )
            score += 5.0 if is_valid_compound else -2.0

        # Given name structure scoring
        if len(given_tokens) == 1:
            token = given_tokens[0]
            if len(token) > 6:
                score -= 1.0
            elif self._normalizer.split_concat(token, normalized_cache):
                score += 0.5
        elif len(given_tokens) == 2:
            score += 1.0
            if all(
                normalized_cache.get(t, self._normalizer.norm(t)) in self._data.given_names_normalized
                for t in given_tokens
            ):
                score += 1.5
        elif len(given_tokens) > 2:
            score -= 1.5

        # Avoid role confusion
        for token in surname_tokens:
            key = normalized_cache.get(token, self._normalizer.norm(token))
            if (
                key in self._data.given_names_normalized
                and self._normalizer.remove_spaces(normalized_cache.get(token, self._normalizer.norm(token)))
                not in self._data.surnames_normalized
            ):
                score -= 2.0

        for token in given_tokens:
            key = self._normalizer.remove_spaces(normalized_cache.get(token, self._normalizer.norm(token)))
            if key in self._data.surnames and self._data.surname_frequencies.get(key, 0) > 1000:
                score -= 1.5

        return score

    def _capitalize_name_part(self, part: str) -> str:
        """Properly capitalize a name part, handling apostrophes correctly.

        Standard .title() incorrectly capitalizes after apostrophes (ts'ai -> Ts'Ai).
        This function only capitalizes the first letter: ts'ai -> Ts'ai.
        """
        if not part:
            return part
        return part[0].upper() + part[1:].lower()

    def _format_name_output(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        normalized_cache: dict[str, str] | None = None,
    ) -> str:
        """Format parsed name components into final output string."""
        # First validate that given tokens could plausibly be Chinese
        if not self._normalizer.validate_given_tokens(given_tokens, normalized_cache):
            raise ValueError("given name tokens are not plausibly Chinese")

        parts = []
        for token in given_tokens:
            # If the token itself is a valid given name, don't try to split it.
            if normalized_cache and token in normalized_cache:
                normalized_token = normalized_cache[token]
            else:
                normalized_token = self._normalizer.norm(token)

            if normalized_token in self._data.given_names_normalized:
                parts.append(token)
                continue

            # NEW: Before trying to split, check if token is already a valid Chinese syllable
            if self._normalizer.is_valid_chinese_phonetics(token):
                # It's a valid syllable, don't split it
                parts.append(token)
                continue

            # Only try splitting if it's not already a valid syllable
            split = self._normalizer.split_concat(token, normalized_cache)
            if split:
                parts.extend(split)
            # Strict validation: only accept if it's a valid Chinese token
            elif self._normalizer.is_valid_given_name_token(token, normalized_cache):
                parts.append(token)
            else:
                raise ValueError(f"given name token '{token}' is not valid Chinese")

        if not parts:
            raise ValueError("given name invalid")

        # Capitalize each part properly, handling hyphens within parts
        formatted_parts = []
        for part in parts:
            # Clean up any leading/trailing hyphens that may have come from tokenization
            clean_part = part.strip("-")
            if not clean_part:  # Skip empty parts after stripping hyphens
                continue

            if "-" in clean_part:
                sub_parts = clean_part.split("-")
                formatted_part = "-".join(self._capitalize_name_part(sub) for sub in sub_parts)
                formatted_parts.append(formatted_part)
            else:
                formatted_parts.append(self._capitalize_name_part(clean_part))

        # Determine separator based on part lengths
        # Use spaces when we have mixed-length parts (some single chars, some multi-char)
        if len(formatted_parts) > 1:
            part_lengths = [
                len(part.replace("-", "")) for part in formatted_parts
            ]  # Count chars, ignoring internal hyphens
            has_single_char = any(length == 1 for length in part_lengths)
            has_multi_char = any(length > 1 for length in part_lengths)

            if has_single_char and has_multi_char:
                # Mixed lengths: use spaces (e.g., "Bin B" not "Bin-B")
                given_str = " ".join(formatted_parts)
            else:
                # All same length category: use hyphens (e.g., "Yu-Ming" or "A-B")
                given_str = "-".join(formatted_parts)
        else:
            given_str = formatted_parts[0] if formatted_parts else ""

        # Handle compound surnames properly
        if len(surname_tokens) > 1:
            surname_str = "-".join(self._capitalize_name_part(t) for t in surname_tokens)
        else:
            surname_str = self._capitalize_name_part(surname_tokens[0])

        return f"{given_str} {surname_str}"



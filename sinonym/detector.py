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
import string

from sinonym.services import (
    CacheInfo,
    ChineseNameConfig,
    DataInitializationService,
    EthnicityClassificationService,
    NameDataStructures,
    NameFormattingService,
    NameParsingService,
    NormalizationService,
    ParseResult,
    PinyinCacheService,
)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN CHINESE NAME DETECTOR CLASS
# ════════════════════════════════════════════════════════════════════════════════


class ChineseNameDetector:
    """Main Chinese name detection and normalization service."""

    def __init__(self, config: ChineseNameConfig | None = None, weights: list[float] | None = None):
        self._config = config or ChineseNameConfig.create_default()
        self._cache_service = PinyinCacheService(self._config)
        self._normalizer = NormalizationService(self._config, self._cache_service)
        self._data_service = DataInitializationService(self._config, self._cache_service, self._normalizer)
        self._data: NameDataStructures | None = None
        self._weights = weights  # Store weights to pass to parsing service

        # Service instances (initialized after data loading)
        self._ethnicity_service: EthnicityClassificationService | None = None
        self._parsing_service: NameParsingService | None = None
        self._formatting_service: NameFormattingService | None = None

        # Initialize data structures
        self._initialize()

    def _initialize(self) -> None:
        """Initialize cache and data structures."""
        try:
            self._data = self._data_service.initialize_data_structures()
            # Inject data context into normalizer after initialization
            self._normalizer.set_data_context(self._data)
            # Initialize services
            self._initialize_services()
        except Exception as e:
            logging.warning(f"Failed to initialize at construction: {e}. Will initialize lazily.")

    def _initialize_services(self) -> None:
        """Initialize service instances with data context."""
        if self._data is not None:
            self._ethnicity_service = EthnicityClassificationService(self._config, self._normalizer, self._data)
            self._parsing_service = NameParsingService(
                self._config,
                self._normalizer,
                self._data,
                weights=self._weights,
            )
            self._formatting_service = NameFormattingService(self._config, self._normalizer, self._data)

    def _ensure_initialized(self) -> None:
        """Ensure data is initialized (lazy initialization)."""
        if self._data is None:
            self._data = self._data_service.initialize_data_structures()
            # Inject data context into normalizer
            self._normalizer.set_data_context(self._data)
            # Initialize services
            self._initialize_services()

    # Public API methods
    def get_cache_info(self) -> CacheInfo:
        """Get cache information."""
        return self._cache_service.get_cache_info()

    def clear_pinyin_cache(self) -> None:
        """Clear the pinyin cache."""
        self._cache_service.clear_cache()

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

        # Early rejection for non-Chinese scripts
        if self._normalizer._text_preprocessor.contains_non_chinese_scripts(raw_name):
            return ParseResult.failure("contains non-Chinese characters")

        self._ensure_initialized()

        # Use new normalization service for cleaner pipeline
        normalized_input = self._normalizer.apply(raw_name)

        if len(normalized_input.roman_tokens) < 2:
            return ParseResult.failure("needs at least 2 Roman tokens")

        # Check if this is an all-Chinese input first
        is_all_chinese = self._normalizer._text_preprocessor.is_all_chinese_input(raw_name)

        # Check for non-Chinese ethnicity using normalized tokens (consistent for all inputs)
        non_chinese_result = self._ethnicity_service.classify_ethnicity(
            normalized_input.roman_tokens,
            normalized_input.norm_map,
            raw_name,
        )

        if non_chinese_result.success is False:
            return non_chinese_result

        # Try parsing in both orders - for all-Chinese inputs, choose best scoring parse

        if is_all_chinese and len(normalized_input.roman_tokens) == 2:
            # For all-Chinese 2-token inputs, manually create both parse candidates
            tokens = list(normalized_input.roman_tokens)
            token1, token2 = tokens[0], tokens[1]

            # Check if both tokens can be surnames
            token1_norm = normalized_input.norm_map.get(token1, self._normalizer.norm(token1))
            token2_norm = normalized_input.norm_map.get(token2, self._normalizer.norm(token2))

            token1_is_surname = (
                self._normalizer.norm(token1) in self._data.surnames or token1_norm in self._data.surnames_normalized
            )
            token2_is_surname = (
                self._normalizer.norm(token2) in self._data.surnames or token2_norm in self._data.surnames_normalized
            )

            best_result = None
            best_score = float("-inf")
            best_format_alignment = 0.0

            # Candidate 1: surname-first pattern (token1=surname, token2=given)
            if token1_is_surname:
                score1 = self._parsing_service.calculate_parse_score(
                    [token1],
                    [token2],
                    tokens,
                    normalized_input.norm_map,
                    is_all_chinese,
                )
                format_alignment1 = self._parsing_service._calculate_format_alignment_bonus([token1], [token2], tokens)

                if score1 > best_score or (score1 == best_score and format_alignment1 > best_format_alignment):
                    best_score = score1
                    best_format_alignment = format_alignment1
                    best_result = ([token1], [token2])

            # Candidate 2: surname-last pattern (token2=surname, token1=given)
            if token2_is_surname:
                score2 = self._parsing_service.calculate_parse_score(
                    [token2],
                    [token1],
                    tokens,
                    normalized_input.norm_map,
                    is_all_chinese,
                )
                format_alignment2 = self._parsing_service._calculate_format_alignment_bonus([token2], [token1], tokens)

                if score2 > best_score or (score2 == best_score and format_alignment2 > best_format_alignment):
                    best_score = score2
                    best_format_alignment = format_alignment2
                    best_result = ([token2], [token1])

            if best_result:
                surname_tokens, given_tokens = best_result
                try:
                    formatted_name = self._formatting_service.format_name_output(
                        surname_tokens,
                        given_tokens,
                        normalized_input.norm_map,
                        None,  # No compound surname format for simple cases
                        normalized_input.compound_metadata,
                    )
                    return ParseResult.success_with_name(formatted_name)
                except ValueError as e:
                    return ParseResult.failure(str(e))
        else:
            # Original logic for non-all-Chinese or multi-token inputs
            for order in (normalized_input.roman_tokens, normalized_input.roman_tokens[::-1]):
                parse_result = self._parsing_service.parse_name_order(
                    list(order),
                    normalized_input.norm_map,
                    normalized_input.compound_metadata,
                )
                if parse_result.success:
                    surname_tokens, given_tokens = parse_result.result
                    try:
                        formatted_name = self._formatting_service.format_name_output(
                            surname_tokens,
                            given_tokens,
                            normalized_input.norm_map,
                            parse_result.original_compound_surname,
                            normalized_input.compound_metadata,
                        )
                        return ParseResult.success_with_name(formatted_name)
                    except ValueError as e:
                        return ParseResult.failure(str(e))

        return ParseResult.failure("name not recognised as Chinese")

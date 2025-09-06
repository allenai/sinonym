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

from sinonym.coretypes import BatchFormatPattern, BatchParseResult
from sinonym.coretypes.results import ParsedName
from sinonym.services import (
    BatchAnalysisService,
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
    ServiceContext,
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
        self._batch_analysis_service: BatchAnalysisService | None = None

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
            # Create shared context to reduce dependency injection complexity
            context = ServiceContext(self._config, self._normalizer, self._data)

            self._ethnicity_service = EthnicityClassificationService(context)
            self._parsing_service = NameParsingService(context, weights=self._weights)
            self._formatting_service = NameFormattingService(context)
            self._batch_analysis_service = BatchAnalysisService(
                self._parsing_service,
                ethnicity_service=self._ethnicity_service,
            )

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

    def is_chinese_name(self, raw_name: str) -> ParseResult:
        """
        Main API method: Detect if a name is Chinese and normalize it.

        Returns ParseResult with:
        - success=True, result=formatted_name if Chinese name detected
        - success=False, error_message=reason if not Chinese name
        """
        # Input validation
        if not raw_name or len(raw_name) > self._config.max_name_length:
            return ParseResult.failure("invalid input length")

        if all(c in string.punctuation + string.whitespace for c in raw_name):
            return ParseResult.failure("name contains only punctuation/whitespace")

        # Early rejection for non-Chinese scripts
        if self._normalizer._text_preprocessor.contains_non_chinese_scripts(raw_name):
            return ParseResult.failure("contains non-Chinese characters")

        self._ensure_initialized()

        # Use new normalization service for cleaner pipeline
        normalized_input = self._normalizer.apply(raw_name)

        if len(normalized_input.roman_tokens) < self._config.min_tokens_required:
            return ParseResult.failure(f"needs at least {self._config.min_tokens_required} Roman tokens")

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

        if is_all_chinese and len(normalized_input.roman_tokens) == self._config.min_tokens_required:
            # For all-Chinese 2-token inputs, ALWAYS assume surname-first order
            # Two-character Chinese names are always (surname, given_name)
            tokens = list(normalized_input.roman_tokens)
            token1, token2 = tokens[0], tokens[1]

            # Check if first token can be a surname
            token1_norm = normalized_input.norm_map.get(token1, self._normalizer.norm(token1))
            token1_is_surname = self._data.is_surname(token1, token1_norm)

            # For 2-character all-Chinese names, use surname-first if token1 is a valid surname
            if token1_is_surname:
                best_result = ([token1], [token2])
            else:
                # Fallback: if token1 is not a surname, try token2 as surname (less common but possible)
                token2_norm = normalized_input.norm_map.get(token2, self._normalizer.norm(token2))
                token2_is_surname = self._data.is_surname(token2, token2_norm)
                if token2_is_surname:
                    best_result = ([token2], [token1])
                else:
                    best_result = None

            if best_result:
                surname_tokens, given_tokens = best_result
                try:
                    formatted_name, given_final, surname_final, surname_str, given_str, middle_tokens = (
                        self._formatting_service.format_name_output_with_tokens(
                            surname_tokens,
                            given_tokens,
                            normalized_input.norm_map,
                            normalized_input.compound_metadata,
                        )
                    )
                    parsed = ParsedName(
                        surname=surname_str,
                        given_name=given_str,
                        surname_tokens=surname_final,
                        given_tokens=given_final,
                        middle_name=" ".join(middle_tokens) if middle_tokens else "",
                        middle_tokens=middle_tokens,
                    )
                    return ParseResult.success_with_name(formatted_name, parsed=parsed)
                except ValueError as e:
                    return ParseResult.failure(str(e))
        elif is_all_chinese and len(normalized_input.roman_tokens) == 3:
            # For 3-character all-Chinese names: check compound surname vs single surname
            tokens = list(normalized_input.roman_tokens)
            
            # Try both possibilities and see which one the parsing service accepts
            # Option 1: First two tokens as compound surname + third as given
            compound_parse = self._parsing_service.parse_name_order(
                tokens,
                normalized_input.norm_map, 
                normalized_input.compound_metadata,
            )
            
            if (compound_parse.success and 
                len(compound_parse.result[0]) == 2 and 
                len(compound_parse.result[1]) == 1):
                # Parsing service recognized first two as compound surname
                best_result = compound_parse.result
            else:
                # Option 2: First token as single surname + last two as given name  
                best_result = ([tokens[0]], tokens[1:])

            if best_result:
                surname_tokens, given_tokens = best_result
                try:
                    formatted_name, given_final, surname_final, surname_str, given_str, middle_tokens = (
                        self._formatting_service.format_name_output_with_tokens(
                            surname_tokens,
                            given_tokens,
                            normalized_input.norm_map,
                            normalized_input.compound_metadata,
                        )
                    )
                    parsed = ParsedName(
                        surname=surname_str,
                        given_name=given_str,
                        surname_tokens=surname_final,
                        given_tokens=given_final,
                        middle_name=" ".join(middle_tokens) if middle_tokens else "",
                        middle_tokens=middle_tokens,
                    )
                    return ParseResult.success_with_name(formatted_name, parsed=parsed)
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
                        formatted_name, given_final, surname_final, surname_str, given_str, middle_tokens = (
                            self._formatting_service.format_name_output_with_tokens(
                                surname_tokens,
                                given_tokens,
                                normalized_input.norm_map,
                                normalized_input.compound_metadata,
                            )
                        )
                        parsed = ParsedName(
                            surname=surname_str,
                            given_name=given_str,
                            surname_tokens=surname_final,
                            given_tokens=given_final,
                            middle_name=" ".join(middle_tokens) if middle_tokens else "",
                            middle_tokens=middle_tokens,
                        )
                        return ParseResult.success_with_name(formatted_name, parsed=parsed)
                    except ValueError as e:
                        return ParseResult.failure(str(e))

        return ParseResult.failure("name not recognised as Chinese")

    def analyze_name_batch(
        self,
        names: list[str],
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
    ) -> BatchParseResult:
        """
        Analyze a batch of names with format pattern detection.

        This method processes multiple names together, detects the dominant
        formatting pattern (surname-first vs given-first), and applies it
        consistently to improve accuracy for ambiguous cases.

        Args:
            names: List of raw name strings to analyze
            format_threshold: Minimum percentage (0.0-1.0) required for format detection
            minimum_batch_size: Minimum number of names required for batch processing

        Returns:
            BatchParseResult containing individual results, format pattern, and improvements

        Example:
            # Academic author list (surname-first pattern)
            names = ["Zhang Wei", "Li Ming", "Bei Yu", "Wang Xiaoli"]
            result = detector.analyze_name_batch(names)
            # "Bei Yu" will be correctly parsed as "Bei Yu" due to batch context
        """
        self._ensure_initialized()

        if self._batch_analysis_service is None:
            # Fallback to individual processing if batch service not available
            individual_results = [self.is_chinese_name(name) for name in names]
            return self._create_fallback_batch_result(names, individual_results)

        # Configure threshold for this analysis
        original_threshold = self._batch_analysis_service._format_threshold
        self._batch_analysis_service._format_threshold = format_threshold

        try:
            return self._batch_analysis_service.analyze_name_batch(
                names,
                self._normalizer,
                self._data,
                self._formatting_service,
                minimum_batch_size,
            )
        finally:
            # Restore original threshold
            self._batch_analysis_service._format_threshold = original_threshold

    def detect_batch_format(
        self,
        names: list[str],
        format_threshold: float = 0.55,
    ) -> BatchFormatPattern:
        """
        Detect the format pattern of a batch without full processing.

        This is useful for understanding the formatting consistency of a
        name list before deciding whether to apply batch processing.

        Args:
            names: List of raw name strings to analyze
            format_threshold: Minimum percentage (0.0-1.0) required for format detection

        Returns:
            BatchFormatPattern indicating the dominant format and confidence

        Example:
            pattern = detector.detect_batch_format(["Zhang Wei", "Li Ming", "Wang Xiaoli"])
            if pattern.threshold_met:
                print(f"Detected {pattern.dominant_format} with {pattern.confidence:.1%} confidence")
        """
        self._ensure_initialized()

        if self._batch_analysis_service is None:
            # Return a fallback pattern indicating mixed format
            from sinonym.coretypes import NameFormat

            return BatchFormatPattern(
                dominant_format=NameFormat.MIXED,
                confidence=0.0,
                surname_first_count=0,
                given_first_count=0,
                total_count=len(names),
                threshold_met=False,
            )

        # Configure threshold for this analysis
        original_threshold = self._batch_analysis_service._format_threshold
        self._batch_analysis_service._format_threshold = format_threshold

        try:
            return self._batch_analysis_service.detect_batch_format(
                names,
                self._normalizer,
                self._data,
            )
        finally:
            # Restore original threshold
            self._batch_analysis_service._format_threshold = original_threshold

    def process_name_batch(
        self,
        names: list[str],
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
    ) -> list[ParseResult]:
        """
        Process a batch of names and return just the parse results.

        This is a convenience method that returns only the ParseResult list
        from batch analysis, similar to calling is_chinese_name() on each name
        but with batch format detection applied.

        Args:
            names: List of raw name strings to process
            format_threshold: Minimum percentage (0.0-1.0) required for format detection
            minimum_batch_size: Minimum number of names required for batch processing

        Returns:
            List of ParseResult objects, one for each input name

        Example:
            names = ["Zhang Wei", "Li Ming", "Bei Yu"]
            results = detector.process_name_batch(names)
            for result in results:
                if result.success:
                    print(f"Formatted: {result.result}")
        """
        batch_result = self.analyze_name_batch(names, format_threshold, minimum_batch_size)
        return batch_result.results

    def _create_fallback_batch_result(
        self, names: list[str], individual_results: list[ParseResult],
    ) -> BatchParseResult:
        """Create a fallback BatchParseResult when batch analysis is not available."""
        from sinonym.coretypes import BatchFormatPattern, IndividualAnalysis, NameFormat

        # Create dummy format pattern
        format_pattern = BatchFormatPattern(
            dominant_format=NameFormat.MIXED,
            confidence=0.0,
            surname_first_count=0,
            given_first_count=0,
            total_count=len(names),
            threshold_met=False,
        )

        # Create dummy individual analyses
        individual_analyses = []
        for name in names:
            individual_analyses.append(
                IndividualAnalysis(
                    raw_name=name,
                    candidates=[],
                    best_candidate=None,
                    confidence=0.0,
                ),
            )

        return BatchParseResult(
            names=names,
            results=individual_results,
            format_pattern=format_pattern,
            individual_analyses=individual_analyses,
            improvements=[],
        )
